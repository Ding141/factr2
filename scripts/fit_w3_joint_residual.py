"""Fit on original train, select on original val; do not refit on live tests."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from factr2_next.training.dataset import ManifestDataset
from factr2_next.inference.joint_lstm_observer import load_bundle,ResidualCascade
from factr2_next.inference.kinematic_compensation import SignalProcessor,features,feature_names,SCHEMA,DEFAULT_SIGNAL,ResidualModel,digest
from factr2_next.data_collection.quality import write_json


def collect(ds,loaded,stride,cached=None):
    items=[];offset=0
    bounds=np.deg2rad(loaded.metadata['operating_envelope']['position_bounds_deg'])
    for i,((x,y),info) in enumerate(zip(ds.episodes,ds.episode_info)):
        with h5py.File(info['path']) as h:stamp=np.array(h[info['episode']+'/joint_pos/timestamps'],np.int64)
        processor=SignalProcessor();xx=[];eligible=[]
        outside=np.any((x[:,:7]<bounds[:,0])|(x[:,:7]>bounds[:,1]),axis=1).astype(int)
        cs=np.r_[0,np.cumsum(outside)]
        for k,(t,row) in enumerate(zip(stamp,x)):
            f=processor.update(int(t),row[:7],row[7:14],row[:7]+row[14:],np.zeros(7),np.zeros(7))
            xx.append(features(f['actual'],f['command'],'position_dynamic_coupling'))
            eligible.append(k>=99 and f['ready'] and cs[k+1]-cs[max(0,k-49)]==0)
        indices=np.arange(49,len(x),stride)
        if cached is None:
            norm=loaded.normalization;pred=[]
            for start in range(0,len(indices),256):
                windows=np.stack([x[k-49:k+1] for k in indices[start:start+256]])
                z=torch.from_numpy((windows-norm['x_mean'])/norm['x_std'])
                with torch.inference_mode():p=loaded.model(z).numpy()
                pred.append(p*norm['y_std']+norm['y_mean'])
            mu=np.concatenate(pred).astype(np.float32)
        else:
            n=len(x)-49;mu=cached[offset:offset+n:stride];offset+=n
            norm=loaded.normalization
            sentinel=np.stack([x[k-49:k+1] for k in (indices[0],indices[-1])])
            with torch.inference_mode():check=loaded.model(torch.from_numpy((sentinel-norm['x_mean'])/norm['x_std'])).numpy()
            np.testing.assert_allclose(check*norm['y_std']+norm['y_mean'],mu[[0,-1]],rtol=1e-4,atol=2e-5)
        target=y[indices]-mu
        items.append(dict(x=np.asarray(xx)[indices],q=x[indices,:7],target=target,
                          keep=np.asarray(eligible)[indices],stamp=stamp[indices],
                          high=Path(info['path']).parent.name=='wide_highspeed_mean_20261026_v1',
                          group=info['path'],episode=info['episode']))
        if (i+1)%20==0:print('prepared',i+1,'/',len(ds.episodes),flush=True)
    if cached is not None:assert offset==len(cached)
    return items


def evaluate(items,artifact=None):
    result={}
    model=ResidualModel(artifact) if artifact else None
    for high in (False,True):
        errors=[];supports=[]
        for item in items:
            if item['high']!=high:continue
            corr,support=model.predict_features(item['x'],item['q'],item['keep']) if model else (np.zeros_like(item['target']),np.zeros(len(item['target']),bool))
            filt=ResidualCascade();output=[]
            for value in item['target']-corr:output.append(filt.update(value.astype(np.float32)))
            errors.append(np.asarray(output)[item['keep']]);supports.append(support[item['keep']])
        a=np.concatenate(errors)
        result['high' if high else 'low']=dict(rows=len(a),mse_nm2=np.mean(a*a,axis=0).tolist(),
                 rms_nm=np.sqrt(np.mean(a*a,axis=0)).tolist(),support_fraction=float(np.concatenate(supports).mean()))
    return result


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    loaded,meta=load_bundle(a.bundle);torch.set_num_threads(4)
    manifest=json.loads(a.manifest.read_text());keys=loaded.config['data']['keys']
    recipe=dict(mean_model_sha256=digest(a.bundle/'mean_checkpoint/model.pt'),manifest_sha256=digest(a.manifest),
        source_manifest=str(a.manifest.resolve()),training_stride=5,validation_stride=1,
        fit='original train recordings only; no Oct30 live observation fitting',selection='original validation only; per joint, equal low/high normalized error',
        prior_live_data_role='retrospective evaluation only',test_used_for_fit=False,
        modes=['velocity_acceleration','position_velocity_acceleration','position_dynamic_coupling'],ridge=[.001,.01,.1,1.,10.],
        application='subtract_before_identical_RC12_RC5',no_intercept=True,no_torque_features=True,no_online_zeroing=True)
    write_json(out/'recipe.json',recipe)
    train=collect(ManifestDataset(manifest['splits']['train'],keys,50,512),loaded,5)
    assert a.val_cache.resolve()==Path(meta['evaluation'])/'val_predictions.npz'
    assert json.loads((Path(meta['evaluation'])/'comparison.json').read_text())['mean_sha256']==recipe['mean_model_sha256']
    with np.load(a.val_cache) as f:cache=f['mean_free_nm']
    val=collect(ManifestDataset(manifest['splits']['val'],keys,50,512),loaded,1,cache)
    recipe['validation_cache_sha256']=digest(a.val_cache);write_json(out/'recipe.json',recipe)
    x=np.concatenate([i['x'][i['keep']] for i in train]);y=np.concatenate([i['target'][i['keep']] for i in train]);q=np.concatenate([i['q'][i['keep']] for i in train])
    weights=[]
    for high in (False,True):
        groups={i['group'] for i in train if i['high']==high}
        counts={g:sum(int(i['keep'].sum()) for i in train if i['group']==g) for g in groups}
        for g,n in counts.items():assert n>0
    for item in train:
        group_items=[i for i in train if i['high']==item['high']]
        groups={i['group'] for i in group_items};n=sum(int(i['keep'].sum()) for i in group_items if i['group']==item['group'])
        weights.append(np.full(int(item['keep'].sum()),.5/(len(groups)*n)))
    w=np.concatenate(weights);np.testing.assert_allclose(w.sum(),1.)
    n=x.shape[1];floors=np.full(n,.001);floors[:70]=np.repeat([.005,.005,.1,.1,.1,.1,.005,.005,.01,.01],7)
    scale=np.maximum(np.sqrt(np.average(x*x,axis=0,weights=w)),floors);active=np.ptp(x,axis=0)>=2*floors
    z=x/scale;z[:,~active]=0;gram=(z*w[:,None]).T@z;rhs=(z*w[:,None]).T@y
    template=dict(schema=SCHEMA,feature_mode='position_dynamic_coupling',feature_names=feature_names('position_dynamic_coupling'),signal=DEFAULT_SIGNAL,
        feature_scale=scale.tolist(),feature_min=x.min(0).tolist(),feature_max=x.max(0).tolist(),
        position_bounds_rad=np.c_[q.min(0)-np.deg2rad(1),q.max(0)+np.deg2rad(1)].tolist(),
        correction_cap_nm=np.maximum(.03,2*np.percentile(abs(y),99,axis=0)).tolist(),enabled_joints=[True]*7,
        base_metadata_sha256=digest(a.bundle/'mean_checkpoint/metadata.json'),mean_model_sha256=recipe['mean_model_sha256'],
        application=recipe['application'],uses_torque_as_feature=False,online_learning=False,intercept=False)
    base=evaluate(val);candidates=[]
    for mode in recipe['modes']:
        used=active.copy();used[len(feature_names(mode)):]=False;ids=np.flatnonzero(used)
        for ridge in recipe['ridge']:
            coef=np.zeros((n,7));coef[ids]=np.linalg.solve(gram[np.ix_(ids,ids)]+ridge*np.eye(len(ids)),rhs[ids])
            artifact=dict(template,weights=coef.tolist(),active_features=used.tolist())
            score=evaluate(val,artifact);ratio={k:np.asarray(score[k]['mse_nm2'])/np.maximum(base[k]['mse_nm2'],1e-10) for k in base}
            feasible=(ratio['low']<=1.05)&(ratio['high']<=.98)
            candidates.append(dict(mode=mode,ridge=ridge,score=(.5*(ratio['low']+ratio['high'])).tolist(),feasible=feasible.tolist(),artifact=artifact,validation=score))
            print(mode,ridge,'validation ratio',.5*(ratio['low']+ratio['high']),flush=True)
    coef=np.zeros((n,7));enabled=[];selection=[]
    for j in range(7):
        viable=[c for c in candidates if c['feasible'][j]]
        c=min(viable,key=lambda c:c['score'][j]) if viable else None
        enabled.append(c is not None)
        if c:coef[:,j]=np.asarray(c['artifact']['weights'])[:,j]
        selection.append(dict(joint=j+1,enabled=c is not None,mode=c['mode'] if c else None,ridge=c['ridge'] if c else None))
    assert any(enabled),'No correction passed validation; do not deploy'
    final=dict(template,weights=coef.tolist(),active_features=np.any(abs(coef)>1e-12,axis=1).tolist(),enabled_joints=enabled,selection=selection)
    write_json(out/'model.json',final)
    write_json(out/'validation.json',dict(baseline=base,compensated=evaluate(val,final),selection=selection,
                  candidates=[{k:v for k,v in c.items() if k!='artifact'} for c in candidates],model_sha256=digest(out/'model.json')))
    write_json(out/'manifest.json',dict(files={f:digest(out/f) for f in ('model.json','recipe.json','validation.json')}))
    print('frozen selection',selection,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','manifest','val-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
