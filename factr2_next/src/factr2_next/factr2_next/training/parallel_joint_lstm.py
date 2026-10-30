"""Spawn independent scalar learners; no shared weights or optimizers."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from factr2_next.data_collection.quality import write_json
from factr2_next.training.models import LSTMRegressor


def training_contract(cfg, norm):
    """Pin all optimization/data settings before accepting another run's state."""
    from factr2_next.data_collection.quality import sha256
    settings = copy.deepcopy(cfg['train'])
    settings.pop('resume_from_run', None)
    return dict(schema='parallel_joint_training_contract_v1', model=cfg['model'],
        data=cfg['data'], train=settings, side=cfg['side'], joint_names=cfg['joint_names'],
        manifest_sha256=sha256(cfg['data']['manifest']), torch_version=str(torch.__version__),
        normalization={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in norm.items()})


def atomic_save(value, path):
    path=Path(path);temporary=path.with_suffix('.pending')
    torch.save(value,temporary);temporary.replace(path)


def prepare_recovery(source, work, contract):
    """Reuse complete branches; replay legacy unfinished branches if no optimizer exists."""
    source=Path(source).resolve()
    if json.loads((source/'training_contract.json').read_text()) != contract:
        raise ValueError('parallel_joint_recovery_contract_mismatch')
    audit=[]
    for joint in range(7):
        origin=source/'joint_training'/f'joint_{joint+1}'
        target=work/f'joint_{joint+1}'
        if not origin.is_dir():continue
        if (origin/'final.pt').exists():
            progress=json.loads((origin/'progress.json').read_text())
            if progress['status']!='complete':raise ValueError('recovery_final_not_complete')
            shutil.copytree(origin,target)
            mode='reuse_complete'
        elif (origin/'resume.pt').exists():
            shutil.copytree(origin,target);mode='exact_epoch_resume'
        else:
            # Older versions saved only selected weights. An Adam restart is not
            # equivalent: replay from the same initialization and verify history.
            target.mkdir();reference=target/'replay_reference';reference.mkdir()
            for name in ('progress.json','best.pt'):shutil.copy2(origin/name,reference/name)
            mode='deterministic_replay_from_seed'
        audit.append(dict(joint=joint+1,mode=mode,source=str(origin)))
    return audit


class JointTargets(Dataset):
    def __init__(self, source, joint):
        self.source, self.joint = source, joint

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, y = self.source[index]
        return x, y[self.joint:self.joint+1]


def scalar_epoch(model, loader, scale, optimizer=None, clip=1., mixed=False):
    model.train(optimizer is not None)
    total = 0.; count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, target in loader:
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=mixed and optimizer is not None):
                pred = model(x)
            loss = ((pred.float()-target)*scale).square().mean()
            if not torch.isfinite(loss): raise ValueError('parallel_joint_nonfinite_loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach())*len(x); count += len(x)
    if not count: raise ValueError('parallel_joint_empty_dataset')
    return total/count


def worker(joint, states, datasets, norm, cfg, directory):
    from factr2_next.training.train import set_seed
    folder=Path(directory)/f'joint_{joint+1}';folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():
        print(f'J{joint+1}: preserved completed branch',flush=True);return
    settings = cfg['train']; set_seed(int(settings.get('seed',0))+joint)
    torch.set_num_threads(int(settings.get('worker_threads',2)))
    torch.use_deterministic_algorithms(True)
    model_cfg = cfg['model']
    model = LSTMRegressor(21,1,**{k:model_cfg[k] for k in
        ('hidden_size','num_layers','head_hidden','head_layers','bidirectional','dropout','state_mode')})
    model.load_state_dict(states[joint])
    # Common window order, independent dropout RNG/optimizer state.
    loaders = [DataLoader(JointTargets(ds,joint),batch_size=settings['batch_size'],shuffle=(i==0),
        generator=torch.Generator().manual_seed(settings.get('seed',0))) for i,ds in enumerate(datasets)]
    cls = torch.optim.AdamW if settings.get('optimizer','adam')=='adamw' else torch.optim.Adam
    optimizer=cls(model.parameters(),lr=float(settings['learning_rate']),weight_decay=float(settings.get('weight_decay',0)))
    scale=float(norm['y_std'][joint]) if settings.get('loss_units')=='physical_nm' else 1.
    early=settings.get('early_stopping',{});best=early_best=float('inf');stale=0
    history=[];best_epoch=None;best_state=None;started=time.monotonic()
    initial_epoch=1;elapsed_before=0.
    reference=None
    if (folder/'replay_reference/progress.json').exists():
        reference=json.loads((folder/'replay_reference/progress.json').read_text())
    if (folder/'resume.pt').exists():
        saved=torch.load(folder/'resume.pt',weights_only=True)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best,early_best,stale=saved['best'],saved['early_best'],saved['stale']
        history,best_epoch,best_state=saved['history'],saved['best_epoch'],saved['best_state']
        initial_epoch=saved['epoch']+1;elapsed_before=saved['elapsed_seconds']
        torch.set_rng_state(saved['torch_rng'])
        for loader,rng in zip(loaders,saved['loader_rng']):loader.generator.set_state(rng)
        print(f'J{joint+1}: exact resume after epoch {saved["epoch"]}',flush=True)
    stop_reason='epoch_budget'
    epoch=initial_epoch-1
    for epoch in range(initial_epoch,int(settings['epochs'])+1):
        # A crash after saving a stopping epoch must not run an extra epoch.
        if early.get('enabled',False) and epoch-1>=early.get('warmup',10) and stale>=early.get('patience',10):
            epoch-=1;stop_reason='validation_patience';break
        tr=scalar_epoch(model,loaders[0],scale,optimizer,settings.get('gradient_clip',1.),
                        settings.get('autocast_bfloat16',False))
        # Validation matches FP32 online inference, irrespective of training precision.
        va=scalar_epoch(model,loaders[1],scale)
        if reference and epoch<=len(reference['history']):
            expected=reference['history'][epoch-1]
            np.testing.assert_allclose([tr,va],[expected['train_mse'],expected['validation_mse']],rtol=1e-6,atol=1e-10)
        history.append(dict(epoch=epoch,train_mse=tr,validation_mse=va))
        if va < best:
            best,best_epoch=va,epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            atomic_save(dict(state=best_state,epoch=epoch,validation_mse=va),folder/'best.pt')
        if reference and epoch==reference['best_epoch']:
            expected=torch.load(folder/'replay_reference/best.pt',weights_only=True)
            for key,value in best_state.items():
                torch.testing.assert_close(value,expected['state'][key],rtol=0,atol=0)
        if va < early_best-float(early.get('min_delta',1e-5)):early_best,stale=va,0
        else:stale+=1
        elapsed=elapsed_before+time.monotonic()-started
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            best=best,early_best=early_best,stale=stale,history=history,best_epoch=best_epoch,
            best_state=best_state,epoch=epoch,elapsed_seconds=elapsed,torch_rng=torch.get_rng_state(),
            loader_rng=[loader.generator.get_state() for loader in loaders]),folder/'resume.pt')
        write_json(folder/'progress.json',dict(joint=joint+1,epoch=epoch,best_epoch=best_epoch,
            best_validation_mse=best,elapsed_seconds=elapsed,status='training',history=history))
        print(f'J{joint+1} epoch {epoch}: validation MSE={va:.7g}, best={best_epoch}',flush=True)
        if settings.get('stop_file') and Path(settings['stop_file']).exists():
            stop_reason='stop_file';break
        if early.get('enabled',False) and epoch>=early.get('warmup',10) and stale>=early.get('patience',10):
            stop_reason='validation_patience';break
    atomic_save(dict(state=best_state,best_epoch=best_epoch,selected_validation_mse=best,
        history=history,stop_reason=stop_reason),folder/'final.pt')
    write_json(folder/'progress.json',dict(joint=joint+1,epoch=epoch,best_epoch=best_epoch,
        best_validation_mse=best,elapsed_seconds=elapsed_before+time.monotonic()-started,
        status='complete',stop_reason=stop_reason,history=history))


def fit_parallel(model, loaders, device, norm, cfg, metrics, on_best, run):
    if str(device)!='cpu':raise ValueError('parallel_joint_cpu_only')
    if int(cfg['train'].get('parallel_joints',0)) != len(model.joints):
        raise ValueError('parallel_joints_must_equal_joint_count')
    stop_file=cfg['train'].get('stop_file')
    if stop_file and Path(stop_file).exists():raise ValueError('parallel_joint_stop_file_exists')
    states=[copy.deepcopy(branch.state_dict()) for branch in model.joints]
    work=Path(run)/'joint_training';work.mkdir()
    contract=training_contract(cfg,norm)
    write_json(Path(run)/'training_contract.json',contract)
    recovery=cfg['train'].get('resume_from_run')
    if recovery:
        metrics['recovery']=prepare_recovery(recovery,work,contract)
        write_json(Path(run)/'recovery.json',metrics['recovery'])
    # Spawn rather than fork: parent has already initialized threaded PyTorch.
    torch.multiprocessing.spawn(worker,args=(states,[loader.dataset for loader in loaders],norm,cfg,str(work)),
                                nprocs=len(model.joints),join=True)
    results=[torch.load(work/f'joint_{j+1}/final.pt',weights_only=True) for j in range(len(model.joints))]
    state={f'joints.{j}.{k}':v for j,result in enumerate(results) for k,v in result['state'].items()}
    model.load_state_dict(state);model.eval()
    count=max(len(r['history']) for r in results)
    metrics.update(selection_rule='independent per-joint minimum FP32 validation MSE; first on ties',
        optimizer_scope='spawned independent optimizer and clip per joint',
        execution='seven CPU processes; independent branch seeds; common shuffled window order',
        autocast_bfloat16=bool(cfg['train'].get('autocast_bfloat16',False)),
        validation_precision='float32',weight_precision='float32',
        per_joint_best_epoch=[r['best_epoch'] for r in results],
        selected_validation_mse=[r['selected_validation_mse'] for r in results],
        per_joint_best_validation_mse=[r['selected_validation_mse'] for r in results],
        per_joint_stop_reason=[r['stop_reason'] for r in results],
        per_joint_stopped_epoch=[len(r['history']) for r in results],
        best_epoch=max(r['best_epoch'] for r in results),test_used_for_fit=False,
        stop_reason='all_joint_workers_complete',per_joint_train_loss=[],per_joint_val_loss=[])
    for epoch in range(count):
        tr=[r['history'][epoch]['train_mse'] if epoch<len(r['history']) else None for r in results]
        va=[r['history'][min(epoch,len(r['history'])-1)]['validation_mse'] for r in results]
        metrics['per_joint_train_loss'].append(tr);metrics['per_joint_val_loss'].append(va)
        metrics['train_loss'].append(float(np.mean([v for v in tr if v is not None])))
        metrics['val_loss'].append(float(np.mean(va)))
    from .joint_lstm import validate_joints
    scale=torch.as_tensor(norm['y_std'] if cfg['train'].get('loss_units')=='physical_nm' else np.ones(7),dtype=torch.float32)
    actual=validate_joints(model,loaders[1],device,scale)
    np.testing.assert_allclose(actual,metrics['selected_validation_mse'],rtol=1e-5,atol=1e-8)
    if cfg['train'].get('validation_diagnostics',False):
        from .validation_metrics import validation_epoch
        _,diagnostic=validation_epoch(model,loaders[1],device,norm,cfg['train'].get('loss_units','normalized'))
        metrics['validation_diagnostics']=[dict(selected_composite=True,**diagnostic)]
    on_best(state)
    return state
