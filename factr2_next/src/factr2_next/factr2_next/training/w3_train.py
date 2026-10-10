"""Manifest-only W3 training; validation selects best, test remains untouched."""
import copy
from datetime import datetime
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from factr2_next.data_collection.quality import validate_manifest, sha256, write_json, commit_at
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.models import build_model

FEATURE_ORDER = ['q', 'qdot', 'q_cmd-q']


def train(cfg):
    from factr2_next.training.train import set_seed, run_epoch
    cfg = copy.deepcopy(cfg)
    manifest_path = str(Path(cfg['data']['manifest']).resolve())
    expected = {'side':cfg['side'], 'joint_order':cfg['joint_names'], 'contract_version':cfg['contract_version'], 'sample_hz':cfg['data'].get('sample_hz',50)}
    manifest, reports = validate_manifest(manifest_path, expected)
    cfg['data']['manifest'] = manifest_path
    seed = int(cfg['train'].get('seed',0));set_seed(seed)
    torch.set_num_threads(int(cfg['train'].get('cpu_threads',1)))
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['train'].get('device','cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested_cuda_unavailable')
    cfg['train']['resolved_device'] = str(device)
    train_ds, val_ds = [ManifestDataset(manifest['splits'][s],cfg['data']['keys'],50,
         cfg['data'].get('memory_budget_mb',512)) for s in ('train','val')]
    norm = train_ds.fit_normalization()
    train_ds.norm = val_ds.norm = norm
    model = build_model(cfg['model'],21,7,50).to(device)
    optimizer=cfg['train'].get('optimizer','adam')
    if optimizer not in ('adam','adamw'): raise ValueError('Unsupported optimizer')
    cls=torch.optim.AdamW if optimizer=='adamw' else torch.optim.Adam
    opt = cls(model.parameters(),lr=float(cfg['train']['learning_rate']),weight_decay=float(cfg['train'].get('weight_decay',0)))
    batch = int(cfg['train']['batch_size']);epochs=int(cfg['train']['epochs'])
    if batch < 1 or epochs < 1:
        raise ValueError('batch_epochs_positive')
    loaders = [DataLoader(ds,batch_size=batch,shuffle=(i==0)) for i,ds in enumerate((train_ds,val_ds))]
    metrics = {'train_loss':[], 'val_loss':[], 'selection_rule':'minimum validation normalized MSE; first epoch on ties',
               'best_epoch':None, 'seed':seed, 'device':str(device), 'window_counts':{'train':len(train_ds),'val':len(val_ds)},
               'dataset_step_bytes':train_ds.step_bytes+val_ds.step_bytes,
               'eager_window_bytes':(len(train_ds)+len(val_ds))*50*21*4}
    best, state = float('inf'), None
    early=cfg['train'].get('early_stopping',{})
    patience=int(early.get('patience',20)); warmup=int(early.get('warmup',10)); delta=float(early.get('min_delta',1e-5))
    if patience<1 or warmup<0 or delta<0 or not np.isfinite(delta):raise ValueError('Invalid early stopping')
    early_best=float('inf'); stale=0
    started=time.monotonic()
    for epoch in range(1,epochs+1):
        extra={'gradient_clip':cfg['train']['gradient_clip']} if 'gradient_clip' in cfg['train'] else {}
        tr=run_epoch(model,loaders[0],device,opt,**extra)
        with torch.inference_mode():va=run_epoch(model,loaders[1],device)
        if not np.isfinite([tr,va]).all():
            raise ValueError('nonfinite_loss')
        metrics['train_loss'].append(tr);metrics['val_loss'].append(va)
        if va < best:
            best=va;metrics['best_epoch']=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"{cfg['side']} epoch {epoch}: train={tr:.6f} val={va:.6f}",flush=True)
        if va < early_best-delta: early_best=va; stale=0
        else: stale+=1
        if early.get('enabled',False) and epoch>=warmup and stale>=patience:
            metrics['early_stopped_epoch']=epoch
            break
    model.load_state_dict(state);model.eval()
    x,_=train_ds[0]
    with torch.inference_mode():
        before=(model(x.unsqueeze(0).to(device)).cpu().numpy()[0]*norm['y_std']+norm['y_mean'])
    metrics.update(training_seconds=time.monotonic()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    run = Path(cfg['save']['output_dir'])/(cfg['save']['run_name']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    torch.save(dict(model_state_dict=state,input_size=21,output_size=7,history=50,model=cfg['model']),run/'model.pt')
    (run/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    np.savez(run/'normalization.npz',**norm)
    write_json(run/'metrics.json',metrics)
    source=reports['train'][0]['metadata']
    w3_workspace=Path(source.get('w3_workspace',
        Path(__file__).resolve().parents[5].parent/'dual_arm_robot')).expanduser().resolve()
    meta={k:copy.deepcopy(source[k]) for k in ('source','tool','gripper','load','calibration','control')}
    meta.update(schema='w3_checkpoint_v1',contract_version=cfg['contract_version'],side=cfg['side'],
        joint_order=cfg['joint_names'],feature_order=FEATURE_ORDER,history=50,input_size=21,output_size=7,sample_hz=cfg['data'].get('sample_hz',50),
        dataset_id=manifest['dataset_id'],manifest_sha256=sha256(manifest_path),seed=seed,resolved_device=str(device),
        software_versions={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},
        w3_workspace=str(w3_workspace),
        evaluation_scope=manifest.get('evaluation_scope','manifest-defined splits'),
        training_source_sha256=sha256(Path(__file__)),
        software_commits={'factr2':commit_at(Path(__file__).resolve().parents[5]),'w3':commit_at(w3_workspace)},
        baseline='train eligible-label mean Nm',artifact_sha256={f:sha256(run/f) for f in
        ('model.pt','config.yaml','normalization.npz','metrics.json')})
    write_json(run/'metadata.json',meta)
    from factr2_next.inference.checkpoint import load_checkpoint
    loaded=load_checkpoint(run,device,expected={'side':cfg['side'],'joint_order':cfg['joint_names']})
    with torch.inference_mode():after=loaded.model(x.unsqueeze(0).to(device)).cpu().numpy()[0]*norm['y_std']+norm['y_mean']
    np.testing.assert_allclose(after,before,atol=1e-5,rtol=1e-5)
    print('saved '+str(run.resolve()),flush=True)
    return run
