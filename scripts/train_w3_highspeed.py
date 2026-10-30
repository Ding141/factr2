"""Single-seed warm-start NEXT mean, with independent high-speed holdouts.

No ROS/control imports. Old low-speed split identities are retained. Normalizing
statistics remain those of the incumbent's training data, preserving warm-start
units. A fixed balanced subset of train windows reduces redundant overlapping
windows; each input still has 50 consecutive samples at 100 Hz.
"""
import argparse
import copy
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader,Subset,Dataset

import run_w3_next_u_ablation as core
from factr2_next.data_collection.campaign import arm_control_profile
from factr2_next.data_collection.quality import sha256,sidecar_path,write_json,make_manifest,require
from factr2_next.training.next_u import SCHEMA,error_metrics,predict_dataset,train_epoch
from factr2_next.training.train import set_seed

ROOT=Path(__file__).resolve().parents[1]


def groups(ds):
    out={False:[],True:[]};start=0
    for end,info in zip(ds.ends,ds.episode_info):
        high=str(info.get('speed','')).startswith('continuous_v')
        out[high].extend(range(start,end));start=end
    if any(not v for v in out.values()):raise ValueError('Both old and new data required in each role')
    return {k:np.asarray(v,np.int64) for k,v in out.items()}


def subset_indices(ds,cap=12000):
    g=groups(ds);rng=np.random.default_rng(0)
    # Five-frame spacing affects target selection, never input sampling/history.
    pools={k:v[::5] for k,v in g.items()};n=min(cap,*(len(v) for v in pools.values()))
    return {k:np.sort(rng.choice(v,n,replace=False)) for k,v in pools.items()}


def prepare(campaign,parent,incumbent,output,scheme='physical_mse'):
    campaign,parent,incumbent,output=map(lambda p:Path(p).resolve(),(campaign,parent,incumbent,output))
    if output.exists():raise FileExistsError(output)
    oldcfg,oldmanifest=core.verify(parent);core.load_artifact(incumbent)
    entries={role:[Path(e['path']) for e in oldmanifest['splits'][role]] for role in ('train','val','test')}
    pending=[]
    for role in entries:
        stage=campaign/'plans'/role
        quality_path=stage/'continuous_quality.json'
        quality=json.loads(quality_path.read_text())
        if not any(b['accepted'] for b in quality['blocks']):raise ValueError('No quality-passing new blocks')
        source=campaign/'derived'/role/'samples.h5';meta=copy.deepcopy(require(source)['metadata'])
        if meta['derivation'].get('quality_audit_sha256')!=sha256(quality_path):raise ValueError('Changed quality selection')
        if meta['derivation']['role']!=role or meta['targeted_collection']['role']!=role:raise ValueError('Prospective role changed')
        meta['control_profile_scope']='arm_v2';meta['control_profile_sha256']=arm_control_profile(meta)
        raw=require(Path(meta['derivation']['source_h5']))
        meta['derivation'].update(parent_session_id=raw['metadata']['session_id'],
            parent_h5_sha256=raw['h5_sha256'])
        meta['training_split']=dict(role=role,assignment='prospective independent whole recording; approved before capture')
        pending.append((role,source,meta))
    # Check actual gravity, friction, calibration, gains and robot model, not paths.
    profiles={arm_control_profile(require(p)['metadata']) for ps in entries.values() for p in ps}
    profiles.update(m['control_profile_sha256'] for _,_,m in pending)
    if len(profiles)!=1:raise ValueError('Old/new physical control profiles differ; do not mix')
    output.mkdir(parents=True)
    for role,source,meta in pending:
        p=output/f'highspeed_{role}.h5';shutil.copy2(source,p);write_json(sidecar_path(p),meta);entries[role].append(p)
    make_manifest(output/'split.json',entries,output.name)
    manifest=json.loads((output/'split.json').read_text())
    manifest.update(evaluation_scope='independent_recording_sessions',
        split_protocol='Retain old session roles; prospective independent high-speed train/val/test recordings',
        prospective_validation=True,signal_time_shift_seconds=0.,delay_estimated=False)
    write_json(output/'split.json',manifest)
    cfg=copy.deepcopy(oldcfg);cfg['data']['manifest']=str(output/'split.json')
    cfg['trial']=dict(scope='Retained old session splits plus prospective new high-speed session holdouts; failed whole motion blocks excluded',
        deployment_ready=False,collection_frequency_hz=100,
        pending=['known physical contact/release validation','independent cold/warm comparison',
                 'poses and measured speeds outside observed training coverage'])
    cfg['train'].update(epochs=20,batch_size=512,learning_rate=.0001,cpu_threads=8,seed=0,
                        stop_file=str(output/'STOP_TRAINING'))
    cfg['train']['early_stopping']=dict(enabled=True,patience=4,warmup=3,min_delta=1e-5)
    if scheme not in ('physical_mse','anchored_head','anchored_full'):raise ValueError('Unknown adaptation scheme')
    if scheme=='anchored_full':cfg['train']['learning_rate']=.00003
    cfg['adaptation']=dict(scheme=scheme,seed=0,maximum_epochs=20,
        anchored_head=dict(freeze_lstm=scheme!='anchored_full',dropout_during_adaptation=False,
            objective='0.5*(old physical MSE + 2*old teacher distillation MSE)/old baseline train MSE + 0.5*new physical MSE/new baseline train MSE',
            rationale='Old/new validation error scales differ; preserve old mapping while learning new motion',
            test_used_for_scheme_selection=False))
    cfg['comparison']=dict(schema=SCHEMA,arms=['NEXT_MSE'],seeds=[0],fresh_paired_training=False,
        historical_checkpoint_reused=True,warm_start=str(incumbent),
        selection='equal-weight old/new validation MSE; old <=110% incumbent; new <= incumbent; per-joint increase <=max(20%,0.02Nm); epoch0 included',
        training_subset='fixed seed0, stride5 target pool, balanced old/new, max12000 per stratum',
        normalization='frozen incumbent train-only statistics; no refit on holdout data',
        test_scope='New independent whole-session high-speed test from '+str(campaign)+'; retain all historical session roles',
        deployment_ready=False,automatic_deployment=False)
    core.validate_config(cfg,[0]);manifest,reports,independent=core.validate_data(output/'split.json',cfg)
    shutil.copytree(incumbent,output/'incumbent')
    shutil.copy2(incumbent/'normalization.npz',output/'normalization.npz')
    (output/'experiment.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_json(output/'deployment_criteria.json',dict(
        scope='Read-only external joint torque observation; never motor feedforward',
        minimum_new_highspeed_test_mse_improvement_fraction=.05,
        maximum_old_test_mse_ratio=1.15,
        minimum_final_highspeed_test_mse_improvement_fraction=.05,
        maximum_final_old_test_mse_ratio=1.15,
        minimum_highspeed_test_position_support_fraction=.98,
        per_joint_rmse_allowance='no increase greater than max(20% incumbent RMSE, 0.02 Nm)',
        require_completed_independent_recordings=True,require_clean_CAN=True,
        test_fail_action='Keep incumbent; freeze failed candidate and do not tune on this test recording',
        real_contact_validation=False))
    for role in ('train','val'):
        inds=subset_indices(core.dataset(manifest,cfg,role))
        np.savez(output/(role+'_selection.npz'),old=inds[False],high=inds[True])
    frozen=output/'frozen_runtime';package=Path(__import__('factr2_next').__file__).resolve().parent
    shutil.copytree(package,frozen/'factr2_next',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train_w3_highspeed.py','run_w3_next_u_ablation.py'):
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    (output/'start.py').write_text("from pathlib import Path\nimport sys,runpy\nr=Path(__file__).resolve().parent\nsys.path.insert(0,str(r/'frozen_runtime'))\nsys.argv=[str(r/'frozen_runtime/train_w3_highspeed.py'),'run','--experiment',str(r)]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
    locked={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    data={str(p):sha256(p) for ps in entries.values() for f in ps for p in (f,sidecar_path(f))}
    write_json(output/'lock.json',dict(schema=SCHEMA,files=locked,data_files=data,independent=independent,
        software=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),h5py=h5py.__version__)))
    write_json(output/'state.json',dict(status='prepared',test_used=False))
    print('Prepared '+str(output),flush=True)


def val_score(model,norm,ds,selection,batch):
    result={}
    for key in ('old','high'):
        sub=Subset(ds,selection[key].tolist());sub.norm=norm;ds.norm=norm
        result[key]=error_metrics(predict_dataset(model,norm,sub,batch)['error_nm'])
    return result


def validation_eligible(val,baseline):
    if (val['old']['physical_mse_nm2']>1.10*baseline['old']['physical_mse_nm2'] or
            val['high']['physical_mse_nm2']>baseline['high']['physical_mse_nm2']):return False
    for role in ('old','high'):
        before=np.asarray(baseline[role]['per_joint_rmse_nm']);after=np.asarray(val[role]['per_joint_rmse_nm'])
        if np.any(after>before+np.maximum(.2*before,.02)):return False
    return True


class TaggedWindows(Dataset):
    def __init__(self,ds,selection):
        self.ds=ds;self.indices=np.r_[selection['old'],selection['high']]
        self.high=np.r_[np.zeros(len(selection['old']),bool),np.ones(len(selection['high']),bool)]
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return (*self.ds[int(self.indices[i])],bool(self.high[i]))


def anchored_epoch(model,teacher,loader,opt,norm,baseline,clip):
    # eval disables dropout only; autograd and head optimization remain active.
    model.eval();teacher.eval();scale=torch.as_tensor(norm['y_std']);total=0.
    lo=max(baseline['old']['physical_mse_nm2'],1e-6)
    hi=max(baseline['high']['physical_mse_nm2'],1e-6)
    for x,y,high in loader:
        mean,_=model(x);error=((y-mean)*scale).square().mean(1)
        old=~high
        with torch.no_grad():target=teacher(x)[0]
        drift=((mean-target)*scale).square().mean(1)
        loss=.5*((error[old]+2*drift[old]).mean()/lo+error[high].mean()/hi)
        if not torch.isfinite(loss):raise ValueError('Nonfinite anchored loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip,error_if_nonfinite=True)
        opt.step();total+=float(loss.detach())*len(x)
    return total/len(loader.dataset)


def run(folder):
    folder=Path(folder).resolve();cfg,manifest=core.verify(folder)
    if json.loads((folder/'state.json').read_text())['status']!='prepared':raise ValueError('No automatic training restart')
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);set_seed(0)
    model,norm,_=core.load_artifact(folder/'incumbent');model.configure_arm('NEXT_MSE')
    anchored=cfg.get('adaptation',{}).get('scheme') in ('anchored_head','anchored_full')
    teacher=None
    if anchored:
        teacher=copy.deepcopy(model).requires_grad_(False).eval()
        if cfg['adaptation']['scheme']=='anchored_head':model.base.lstm.requires_grad_(False)
    ds={r:core.dataset(manifest,cfg,r) for r in ('train','val')}
    sel={r:dict(np.load(folder/(r+'_selection.npz'))) for r in ds}
    for v in ds.values():v.norm=norm
    indices=np.concatenate(list(sel['train'].values())).tolist()
    train_data=TaggedWindows(ds['train'],sel['train']) if anchored else Subset(ds['train'],indices)
    loader=DataLoader(train_data,batch_size=cfg['train']['batch_size'],shuffle=True,
                      generator=torch.Generator().manual_seed(0))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['train']['learning_rate'],weight_decay=1e-6)
    baseline=val_score(model,norm,ds['val'],sel['val'],512)
    train_baseline=val_score(model,norm,ds['train'],sel['train'],512) if anchored else None
    best=np.mean([m['physical_mse_nm2'] for m in baseline.values()]);stale=0;start=time.monotonic()
    record=dict(best_epoch=0,best_validation_mse_nm2=float(best),baseline_validation=baseline,
                baseline_training=train_baseline,epochs=[],
                selection=cfg['comparison']['selection'],test_used_for_fit=False)
    path=folder/'results/seed_0/NEXT_MSE';core.save_artifact(path,model,cfg,norm,0,'NEXT_MSE',record,folder)
    for epoch in range(1,21):
        if (folder/'STOP_TRAINING').exists():raise RuntimeError('Training stopped; no automatic export')
        write_json(folder/'state.json',dict(status='training',epoch=epoch,test_used=False))
        loss=(anchored_epoch(model,teacher,loader,opt,norm,train_baseline,cfg['train']['gradient_clip']) if anchored
              else train_epoch(model,loader,opt,norm,'NEXT_MSE',cfg['train']['gradient_clip']))
        val=val_score(model,norm,ds['val'],sel['val'],512)
        score=float(np.mean([m['physical_mse_nm2'] for m in val.values()]))
        accepted=validation_eligible(val,baseline)
        improved=accepted and score<best-1e-5
        stale=0 if improved else stale+1
        record['epochs'].append(dict(epoch=epoch,loss=loss,validation=val,selection_score=score,eligible=accepted,
                                     elapsed_seconds=time.monotonic()-start))
        if improved:
            best=score;record.update(best_epoch=epoch,best_validation_mse_nm2=best)
            core.save_artifact(path,model,cfg,norm,0,'NEXT_MSE',record,folder)
        write_json(folder/'history.json',record)
        print(f'epoch={epoch} loss={loss:.7f} val_old={val["old"]["physical_mse_nm2"]:.7f} val_high={val["high"]["physical_mse_nm2"]:.7f} best={record["best_epoch"]}',flush=True)
        if epoch>=(10 if anchored else 3) and stale>=4:break
    selected,_,_=core.load_artifact(path);old,_,_=core.load_artifact(folder/'incumbent')
    evaluation=folder/'evaluation';evaluation.mkdir();report=dict(selected_epoch=record['best_epoch'],splits={})
    # Selection is irrevocably closed before any test errors are read here.
    write_json(folder/'state.json',dict(status='evaluating',selected_epoch=record['best_epoch'],selection_closed=True))
    for role in ('val','test'):
        data=ds['val'] if role=='val' else core.dataset(manifest,cfg,role);g=groups(data)
        report['splits'][role]={}
        for name,m in (('INCUMBENT',old),('NEXT_MSE',selected)):
            a=predict_dataset(m,norm,data,512)
            np.savez_compressed(evaluation/f'seed_0_{role}_{name}.npz',**a)
            report['splits'][role][name]={('high' if k else 'old'):error_metrics(a['error_nm'][idx]) for k,idx in g.items()}
    write_json(evaluation/'comparison.json',report)
    write_json(folder/'state.json',dict(status='trained',selected_epoch=record['best_epoch'],
        completed_epochs=len(record['epochs']),training_seconds=time.monotonic()-start,test_used_for_fit=False))
    core.verify(folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for k in ('campaign','parent','incumbent','output'):q.add_argument('--'+k,type=Path,required=True)
    q.add_argument('--scheme',choices=['physical_mse','anchored_head','anchored_full'],default='physical_mse')
    sub.add_parser('run').add_argument('--experiment',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.campaign,a.parent,a.incumbent,a.output,a.scheme)
    else:run(a.experiment)
