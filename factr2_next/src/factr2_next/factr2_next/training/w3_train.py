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


def persist_best_snapshot(run, state, norm, cfg, metrics, source, manifest_path):
    """Keep an atomic recovery copy before another optimization epoch starts."""
    path = run / f"best_epoch_{metrics['best_epoch']:04d}.pt"
    temporary = path.with_suffix('.pending')
    torch.save(dict(model_state_dict=state, normalization=norm, config=cfg,
        metrics=copy.deepcopy(metrics), source_metadata=source,
        manifest_sha256=sha256(manifest_path), input_size=21, output_size=7,
        history=50, model=cfg['model']), temporary)
    temporary.replace(path)
    write_json(run/'best_snapshot.json',dict(path=str(path.resolve()),
        sha256=sha256(path), best_epoch=metrics['best_epoch'], complete=True))
    return path


def train(cfg):
    from factr2_next.training.train import set_seed, run_epoch
    cfg = copy.deepcopy(cfg)
    if cfg['model']['type'] == 'joint_lstm':
        from factr2_next.w3_config import validate_w3_config
        validate_w3_config(cfg, 'train')
    manifest_path = str(Path(cfg['data']['manifest']).resolve())
    expected = {'side':cfg['side'], 'joint_order':cfg['joint_names'], 'contract_version':cfg['contract_version'], 'sample_hz':cfg['data'].get('sample_hz',50)}
    manifest, reports = validate_manifest(manifest_path, expected)
    if cfg['data'].get('require_independent_sessions', False):
        from factr2_next.data_collection.campaign import validate_independent_manifest
        validate_independent_manifest(manifest, reports)
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
    coverage = None
    if cfg['data'].get('coverage_requirements') is not None:
        from factr2_next.training.coverage_audit import audit_coverage
        coverage = audit_coverage(train_ds, val_ds, cfg['data']['coverage_requirements'])
        if not coverage['passed']:
            raise ValueError('measured_coverage_failed: '+', '.join(coverage['reasons']))
    train_ds.norm = val_ds.norm = norm
    loss_units = cfg['train'].get('loss_units', 'normalized')
    if loss_units not in ('normalized', 'physical_nm'):
        raise ValueError('Invalid loss_units')
    output_scale = norm['y_std'] if loss_units == 'physical_nm' else None
    model = build_model(cfg['model'],21,7,50).to(device)
    optimizer=cfg['train'].get('optimizer','adam')
    if optimizer not in ('adam','adamw'): raise ValueError('Unsupported optimizer')
    cls=torch.optim.AdamW if optimizer=='adamw' else torch.optim.Adam
    opt = (None if cfg['model']['type'] == 'joint_lstm' else
           cls(model.parameters(),lr=float(cfg['train']['learning_rate']),weight_decay=float(cfg['train'].get('weight_decay',0))))
    batch = int(cfg['train']['batch_size']);epochs=int(cfg['train']['epochs'])
    if batch < 1 or epochs < 1:
        raise ValueError('batch_epochs_positive')
    loaders = [DataLoader(ds,batch_size=batch,shuffle=(i==0)) for i,ds in enumerate((train_ds,val_ds))]
    metrics = {'train_loss':[], 'val_loss':[], 'selection_rule':'minimum validation normalized MSE; first epoch on ties',
               'best_epoch':None, 'seed':seed, 'device':str(device), 'window_counts':{'train':len(train_ds),'val':len(val_ds)},
               'dataset_step_bytes':train_ds.step_bytes+val_ds.step_bytes,
               'eager_window_bytes':(len(train_ds)+len(val_ds))*50*21*4}
    metrics['loss_units'] = loss_units
    if coverage is not None:
        metrics['coverage_audit'] = coverage
    diagnostics_enabled = cfg['train'].get('validation_diagnostics', False)
    if diagnostics_enabled:
        metrics['validation_diagnostics'] = []
    if loss_units == 'physical_nm':
        metrics['selection_rule'] = 'minimum validation physical Nm squared MSE; first epoch on ties'
    best, state = float('inf'), None
    early=cfg['train'].get('early_stopping',{})
    patience=int(early.get('patience',20)); warmup=int(early.get('warmup',10)); delta=float(early.get('min_delta',1e-5))
    if patience<1 or warmup<0 or delta<0 or not np.isfinite(delta):raise ValueError('Invalid early stopping')
    early_best=float('inf'); stale=0
    started=time.monotonic()
    run = Path(cfg['save']['output_dir'])/(cfg['save']['run_name']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    if cfg['model']['type'] == 'joint_lstm':
        from factr2_next.training.joint_lstm import fit_joint_lstm
        def save_joint_best(joint_state):
            persist_best_snapshot(run, joint_state, norm, cfg, metrics,
                                  reports['train'][0]['metadata'], manifest_path)
        if cfg['train'].get('parallel_joints'):
            from factr2_next.training.parallel_joint_lstm import fit_parallel
            state = fit_parallel(model, loaders, device, norm, cfg, metrics, save_joint_best, run)
        else:
            state = fit_joint_lstm(model, loaders, device, norm, cfg, metrics, save_joint_best)
    else:
        for epoch in range(1,epochs+1):
            extra={'gradient_clip':cfg['train']['gradient_clip']} if 'gradient_clip' in cfg['train'] else {}
            scale_args = {'output_scale':output_scale} if output_scale is not None else {}
            tr=run_epoch(model,loaders[0],device,opt,**scale_args,**extra)
            if diagnostics_enabled:
                from factr2_next.training.validation_metrics import validation_epoch
                va, diagnostic = validation_epoch(model, loaders[1], device, norm, loss_units)
                metrics['validation_diagnostics'].append(diagnostic)
            else:
                with torch.inference_mode():va=run_epoch(model,loaders[1],device,**scale_args)
            if not np.isfinite([tr,va]).all():
                raise ValueError('nonfinite_loss')
            metrics['train_loss'].append(tr);metrics['val_loss'].append(va)
            if va < best:
                best=va;metrics['best_epoch']=epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                persist_best_snapshot(run,state,norm,cfg,metrics,reports['train'][0]['metadata'],manifest_path)
            print(f"{cfg['side']} epoch {epoch}: train={tr:.6f} val={va:.6f}",flush=True)
            if diagnostics_enabled:
                for group, values in diagnostic['groups'].items():
                    print(f"  val {group}: max joint RMSE={max(values['per_joint_rmse_nm']):.4f} Nm "
                          f"max abs bias={max(map(abs, values['signed_bias_nm'])):.4f} Nm", flush=True)
            if va < early_best-delta: early_best=va; stale=0
            else: stale+=1
            stop_file=cfg['train'].get('stop_file')
            if stop_file and Path(stop_file).exists():
                metrics.update(stop_reason='user_requested',stopped_epoch=epoch)
                break
            if early.get('enabled',False) and epoch>=warmup and stale>=patience:
                metrics['early_stopped_epoch']=epoch
                break
    model.load_state_dict(state);model.eval()
    x,_=train_ds[0]
    with torch.inference_mode():
        before=(model(x.unsqueeze(0).to(device)).cpu().numpy()[0]*norm['y_std']+norm['y_mean'])
    metrics.update(training_seconds=time.monotonic()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    torch.save(dict(model_state_dict=state,input_size=21,output_size=7,history=50,model=cfg['model']),run/'model.pt')
    (run/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    np.savez(run/'normalization.npz',**norm)
    write_json(run/'metrics.json',metrics)
    source=reports['train'][0]['metadata']
    w3_workspace=Path(source.get('w3_workspace',
        Path(__file__).resolve().parents[5].parent/'dual_arm_robot')).expanduser().resolve()
    meta={k:copy.deepcopy(source[k]) for k in ('source','tool','gripper','load','calibration','control')}
    if 'control_profile_sha256' in source:
        meta['control_profile_sha256'] = source['control_profile_sha256']
    if 'control_profile_scope' in source:
        meta['control_profile_scope'] = source['control_profile_scope']
    for field in ('operating_envelope', 'task_coverage'):
        if field in source:
            meta[field] = copy.deepcopy(source[field])
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
    source_dir = Path(__file__).resolve().parent
    meta['training_source_files_sha256'] = {name:sha256(source_dir/name) for name in
        ('w3_train.py', 'train.py', 'dataset.py', 'models.py', 'coverage_audit.py', 'validation_metrics.py')}
    meta['data_contract_source_files_sha256'] = {name:sha256(source_dir.parent/'data_collection'/name)
        for name in ('quality.py','campaign.py')}
    if cfg['model']['type'] == 'joint_lstm':
        meta['architecture'] = 'independent_joint_lstm_v1'
        meta['per_joint_best_epoch'] = metrics['per_joint_best_epoch']
        meta['training_source_files_sha256']['joint_lstm.py'] = sha256(source_dir/'joint_lstm.py')
        if cfg['train'].get('parallel_joints'):
            meta['training_source_files_sha256']['parallel_joint_lstm.py'] = sha256(source_dir/'parallel_joint_lstm.py')
    if cfg.get('trial'):
        meta.update(trial=copy.deepcopy(cfg['trial']),deployment_ready=False,automatic_deployment=False)
    write_json(run/'metadata.json',meta)
    from factr2_next.inference.checkpoint import load_checkpoint
    loaded=load_checkpoint(run,device,expected={'side':cfg['side'],'joint_order':cfg['joint_names']})
    with torch.inference_mode():after=loaded.model(x.unsqueeze(0).to(device)).cpu().numpy()[0]*norm['y_std']+norm['y_mean']
    np.testing.assert_allclose(after,before,atol=1e-5,rtol=1e-5)
    print('saved '+str(run.resolve()),flush=True)
    return run
