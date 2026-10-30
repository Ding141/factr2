"""Offline matched variance head and kinematic residual refit for a new mean.

Never reuse old mean-dependent caches/compensation. Validation selects heads and
correction; test errors are evaluated after selection. No actuation APIs.
"""
import argparse
import json
import platform
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

import run_w3_next_u_ablation as core
import run_w3_frozen_sigma as sigma
import fit_w3_kinematic_compensation as residual
from factr2_next.data_collection.quality import sha256,write_json
from factr2_next.training.frozen_variance import SCHEMA,cache_mean,fit_head,head_variance,variance_metrics
from train_w3_highspeed import groups


def matched_sigma(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cfg,manifest=core.verify(parent)
    if json.loads((parent/'state.json').read_text())['status']!='trained':raise ValueError('Mean not finished')
    model,norm,_=core.load_artifact(parent/'results/seed_0/NEXT_MSE')
    output.mkdir(parents=True);shutil.copytree(parent/'results/seed_0/NEXT_MSE',output/'mean_checkpoint')
    recipe=dict(sigma.DEFAULT_RECIPE,test_scope=cfg['comparison']['test_scope'])
    write_json(output/'recipe.json',recipe)
    shutil.copy2(__file__,output/'source_snapshot.py')
    files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    lock=dict(schema=SCHEMA,parent=str(parent),files=files,
        external_files={str(parent/'lock.json'):sha256(parent/'lock.json'),
                        str(parent/'evaluation/comparison.json'):sha256(parent/'evaluation/comparison.json')},
        mean_state_sha256=sigma.mean_digest(model),
        source_checkpoint_sha256=sha256(parent/'results/seed_0/NEXT_MSE/model.pt'),
        source_selected_epoch=json.loads((output/'mean_checkpoint/metrics.json').read_text())['best_epoch'],
        software=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),h5py=h5py.__version__),
        changes='Matching variance head for newly selected mean; no joint NEXT-U comparison required')
    write_json(output/'lock.json',lock);torch.set_num_threads(8)
    caches={};datasets={};(output/'cache').mkdir()
    for role in ('train','val'):
        write_json(output/'state.json',dict(status='caching_'+role))
        datasets[role]=core.dataset(manifest,cfg,role)
        caches[role]=cache_mean(model,norm,datasets[role],512)
        np.savez_compressed(output/'cache'/f'{role}.npz',**caches[role]);print('Matched cache '+role,flush=True)
    def progress(record,best):
        write_json(output/'progress.json',dict(epoch=record['epoch'],best_epoch=best))
        print(f'Matched sigma epoch={record["epoch"]} best={best}',flush=True)
    head,hidden_norm,fixed,history=fit_head(caches['train'],caches['val'],norm['y_std'],recipe,on_epoch=progress,
        should_stop=lambda:(output/'STOP_TRAINING').exists())
    if history['stop_reason']=='user_stop':
        write_json(output/'state.json',dict(status='stopped',best_epoch=history['best_epoch']))
        raise RuntimeError('User stopped variance training; no export or followup fitting')
    sigma.save_head(output,head,hidden_norm,fixed,recipe,history,lock['source_checkpoint_sha256'])
    datasets['test']=core.dataset(manifest,cfg,'test')
    caches['test']=cache_mean(model,norm,datasets['test'],512)
    np.savez_compressed(output/'cache/test.npz',**caches['test'])
    report=dict(mean_selected_epoch=lock['source_selected_epoch'],best_epoch=history['best_epoch'],
                test_used_for_fit=False,splits={})
    for role in ('val','test'):
        cache=caches[role];learned=head_variance(head,cache,hidden_norm)
        report['splits'][role]=dict(arms={})
        for name,v in (('FIXED_SIGMA',fixed),('LEARNED_SIGMA',learned)):
            report['splits'][role]['arms'][name]=dict(overall=variance_metrics(cache,v,norm['y_std']))
    (output/'evaluation').mkdir();write_json(output/'evaluation/comparison.json',report)
    sigma.verify(output)
    write_json(output/'state.json',dict(status='complete',mean_unchanged=True,best_epoch=history['best_epoch'],test_used_for_fit=False))
    return output


def matching_series(parent,sigma_dir,roles=('train','val','test')):
    cfg,manifest=core.verify(parent);_,norm,_=core.load_artifact(Path(parent)/'results/seed_0/NEXT_MSE')
    result={}
    for role in roles:
        if role not in ('train','val','test'):raise ValueError('Unknown series split')
        ds=core.dataset(manifest,cfg,role)
        with np.load(Path(sigma_dir)/'cache'/f'{role}.npz') as c:
            mu=c['mean_free_nm'];tau=mu+c['error_z']*norm['y_std']
        offset=0;items=[]
        for (x,_),info in zip(ds.episodes,ds.episode_info):
            n=len(x)-49
            with h5py.File(info['path']) as h:stamps=h[info['episode']+'/joint_pos/timestamps'][49:]
            item=residual.series(stamps,x[49:],tau[offset:offset+n],mu[offset:offset+n],np.full(n,info['path']))
            item['high']=str(info.get('speed','')).startswith('continuous_v');items.append(item);offset+=n
        if offset!=len(mu):raise ValueError('Wrong mean cache alignment')
        result[role]=items
    return result


def fit_compensation(parent,sigma_dir,base,output):
    parent,sigma_dir,base,output=map(lambda p:Path(p).resolve(),(parent,sigma_dir,base,output))
    if output.exists():raise FileExistsError(output)
    if sha256(base/'mean_checkpoint/model.pt')!=sha256(parent/'results/seed_0/NEXT_MSE/model.pt'):
        raise ValueError('New residual must match the new deployed mean')
    output.mkdir(parents=True);identity=sha256(base/'metadata.json')
    data=matching_series(parent,sigma_dir)
    def pairs(role,high=None):return [(x,np.ones(len(x['t']),bool)) for x in data[role] if high is None or x['high']==high]
    baseline={k:residual.mse(pairs('val',k)) for k in (False,True)}
    candidates=[]
    recipe=dict(base=str(base),base_metadata_sha256=identity,signal=residual.DEFAULT_SIGNAL,
        source_mean_experiment=str(parent),source_sigma_experiment=str(sigma_dir),
        mean_cache_sha256={r:sha256(sigma_dir/'cache'/f'{r}.npz') for r in data},
        train_role='train only; old low-speed recordings retained',selection_role='val',
        candidate_ridge=[.001,.01,.1],modes=['velocity','velocity_acceleration'],scopes=['pooled','new_highspeed'],
        selection='equal old/new validation normalized MSE, joint enable only if new improves >=2% and old <=105%',
        test_used_for_fit=False,no_torque_feature=True,no_online_zeroing=True)
    write_json(output/'recipe.json',recipe)
    for mode in recipe['modes']:
        for scope in recipe['scopes']:
            for lam in recipe['candidate_ridge']:
                artifact=residual.train_artifact(pairs('train',None if scope=='pooled' else True),mode,lam,identity,scope,.005)
                va={k:residual.mse(pairs('val',k),artifact) for k in baseline}
                enabled=(va[True]<=.98*baseline[True])&(va[False]<=1.05*baseline[False])
                artifact['enabled_joints']=enabled.tolist()
                # Recompute with disabled joints and support fallback applied.
                va={k:residual.mse(pairs('val',k),artifact) for k in baseline}
                score=float(np.mean([np.mean(va[k]/np.maximum(baseline[k],1e-8)) for k in baseline]))
                candidates.append((score,artifact,dict(mode=mode,scope=scope,ridge=lam,score=score,enabled=enabled.tolist())))
    score,artifact,selected=min(candidates,key=lambda x:x[0])
    if not any(artifact['enabled_joints']):raise RuntimeError('No correction passed validation; do not activate one')
    artifact.update(selection=selected,validation_only_selection=True);write_json(output/'model.json',artifact)
    comparison=dict(selected=selected,candidates=[c[2] for c in candidates],contact_transfer=residual.contact_transfer(),splits={})
    for role in data:
        comparison['splits'][role]=[]
        for item in data[role]:
            report,_,_,_=residual.metrics(item,np.ones(len(item['t']),bool),artifact)
            report.update(highspeed=item['high'],source=str(item['groups'][0]));comparison['splits'][role].append(report)
    write_json(output/'comparison.json',comparison)
    write_json(output/'manifest.json',dict(schema=residual.SCHEMA,base_metadata_sha256=identity,deployment_ready=False,
        files={f:sha256(output/f) for f in ('model.json','recipe.json','comparison.json')}))
    print(json.dumps(dict(compensation=str(output),selected=selected)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('sigma');q.add_argument('--parent',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('compensation')
    for n in ('parent','sigma-dir','base','output'):q.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.command=='sigma':matched_sigma(a.parent,a.output)
    else:fit_compensation(a.parent,a.sigma_dir,a.base,a.output)
