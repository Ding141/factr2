"""Fit frozen posture/dynamics residuals on recording-separated no-load data.

No ROS or actuation. Validation selects each joint's regularization; the model
is written and hashed before opening the retrospective test split. Recent
physical observation recordings are evaluation only, never fitting inputs.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from finish_w3_highspeed import matching_series
from fit_w3_kinematic_compensation import predict as old_predict, smooth, contact_transfer
from factr2_next.data_collection.quality import write_json, sha256
from factr2_next.inference.kinematic_compensation import (
    SCHEMA, POSE_MODES, DEFAULT_SIGNAL, feature_names, augment_position_features, ResidualModel,
)

ROOT = Path(__file__).resolve().parents[1]


def position_gate(item, bounds):
    bad = np.any((item['x'][:, :7] < bounds[:, 0]) |
                 (item['x'][:, :7] > bounds[:, 1]), axis=1).astype(int)
    c = np.r_[0, np.cumsum(bad)]; n = len(bad)
    return ((c[1:]-c[np.maximum(np.arange(n)+1-50, 0)]) == 0) & (np.arange(n) >= 49)


def prepare(items, bounds):
    for item in items:
        item['position_gate'] = position_gate(item, bounds)
        item['keep'] = item['eligible'] & (item['t'] >= item['t'][0]+1_000_000_000)
        item['fit_keep'] = item['keep'] & item['position_gate']
        item['pose_features'] = augment_position_features(
            item['features'], item['position_filtered'], POSE_MODES[-1])


def predict(item, artifact):
    value,supported=ResidualModel(artifact).predict_features(
        item['pose_features'],item['position_filtered'],item['eligible'])
    supported &= item['position_gate'];value[~supported]=0
    return value, supported


def evaluate(items, artifact=None, incumbent=None):
    accum = {key:dict(filter_only=[],incumbent=[],final=[],support=[]) for key in ('low','high')}
    for item in items:
        zero = np.zeros_like(item['target']); before = zero
        if incumbent is not None:
            before, _ = old_predict(item, incumbent); before[~item['position_gate']] = 0
        correction, support = (zero,np.zeros(len(zero),bool)) if artifact is None else predict(item, artifact)
        hz = DEFAULT_SIGNAL['output_hz']; keep = item['keep']; key = 'high' if item['high'] else 'low'
        values = dict(filter_only=smooth(item['target'],item['t'],hz),
                      incumbent=smooth(item['target']-before,item['t'],hz),
                      final=smooth(item['target']-correction,item['t'],hz),support=support)
        for k,v in values.items():accum[key][k].append(v[keep])
    result = {}
    for key, values in accum.items():
        a = {k:np.concatenate(v) for k,v in values.items()}
        result[key] = dict(rows=len(a['final']),support_fraction=float(a['support'].mean()),variants={})
        for k,v in a.items():
            if k == 'support':continue
            result[key]['variants'][k] = dict(mse_nm2=np.mean(v*v,axis=0).tolist(),
                rms_nm=np.sqrt(np.mean(v*v,axis=0)).tolist(),overall_rms_nm=float(np.sqrt(np.mean(v*v))),
                bias_nm=v.mean(0).tolist(),abs_p95_nm=np.percentile(abs(v),95,axis=0).tolist())
    return result


def class_recording_weights(items):
    """Equal total mass for low/high; equal recording mass within each class."""
    counts = {key:{} for key in (False,True)}
    for item in items:
        group = str(item['groups'][0]); count = int(item['fit_keep'].sum())
        counts[item['high']][group] = counts[item['high']].get(group,0)+count
    result = []
    for item in items:
        count = int(item['fit_keep'].sum()); group = str(item['groups'][0]); c = counts[item['high']]
        result.append(np.full(count, .5/(len(c)*c[group])))
    weights = np.concatenate(result)
    np.testing.assert_allclose(weights.sum(),1.)
    return weights,counts


def run(args):
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    base = args.base.resolve();inc_path = args.incumbent.resolve()
    incumbent = json.loads((inc_path/'model.json').read_text());identity=sha256(base/'metadata.json')
    if identity!=incumbent['base_metadata_sha256']:raise ValueError('Incumbent mean identity mismatch')
    bounds=np.deg2rad(json.loads((base/'metadata.json').read_text())['operating_envelope']['position_bounds_deg'])
    recipe=dict(base=str(base),base_metadata_sha256=identity,signal=DEFAULT_SIGNAL,
        source_mean_experiment=str(args.parent.resolve()),source_sigma_experiment=str(args.sigma_dir.resolve()),
        incumbent=str(inc_path),incumbent_sha256=sha256(inc_path/'model.json'),
        modes=list(POSE_MODES),candidate_ridge=[.001,.01,.1,1.],
        train_role='original recording-separated train only; no observation/test refit',selection_role='val only',
        weighting='equal low/high mass; equal recording mass within each class',
        selection='per-joint minimum equal-class normalized final-output validation MSE; prefer low MSE <=105% and high <=98% of filter-only',
        test_used_for_fit=False,test_previously_inspected=True,no_torque_feature=True,no_online_zeroing=True,
        no_output_deadband=True,no_output_clipping=True,all_seven_joints=True,
        contact_validation='fixed-kinematics synthetic transfer only; not physical contact accuracy')
    write_json(output/'recipe.json',recipe)
    print('Loading train/val only with matched frozen NEXT predictions',flush=True)
    data=matching_series(args.parent,args.sigma_dir,roles=('train','val'))
    for role in data:prepare(data[role],bounds)
    train=data['train'];weights,counts=class_recording_weights(train)
    x=np.concatenate([i['pose_features'][i['fit_keep']] for i in train])
    y=np.concatenate([i['target'][i['fit_keep']] for i in train])
    q=np.concatenate([i['x'][i['fit_keep'],:7] for i in train])
    # Fixed physical floors; acceleration floors from the original train-only recipe.
    floors=np.r_[np.asarray(incumbent['excitation_floor']),np.full(70,.001),
                 np.full(14,.000025),np.full(28,.0001),np.full(28,.001),
                 .1*np.asarray(incumbent['excitation_floor'])]
    assert x.shape[1]==len(floors)==len(feature_names(POSE_MODES[-1]))
    scale=np.maximum(np.sqrt(np.average(x*x,axis=0,weights=weights)),floors)
    active=np.ptp(x,axis=0)>=2*floors
    z=x/scale;z[:,~active]=0
    gram=(z*weights[:,None]).T@z;rhs=(z*weights[:,None]).T@y
    template=copy.deepcopy(incumbent)
    for k in ('selection','activation_override','validation_only_weight_selection'):template.pop(k,None)
    template.update(feature_mode=POSE_MODES[-1],feature_names=feature_names(POSE_MODES[-1]),
        feature_scale=scale.tolist(),feature_min=x.min(0).tolist(),feature_max=x.max(0).tolist(),
        active_features=active.tolist(),excitation_floor=floors.tolist(),enabled_joints=[True]*7,
        position_bounds_rad=np.stack([q.min(0)-np.deg2rad(1),q.max(0)+np.deg2rad(1)],axis=1).tolist(),
        scope='balanced_low_high',intercept=False,validation_only_selection=True)
    baseline=evaluate(data['val'],incumbent=incumbent)
    before={key:np.array(v['variants']['filter_only']['mse_nm2']) for key,v in baseline.items()}
    candidates=[]
    for mode in POSE_MODES:
        n=len(feature_names(mode));nactive=active.copy();nactive[n:]=False
        ids=np.flatnonzero(nactive)
        for ridge in recipe['candidate_ridge']:
            coef=np.zeros((len(scale),7));coef[ids]=np.linalg.solve(gram[np.ix_(ids,ids)]+ridge*np.eye(len(ids)),rhs[ids])
            a=copy.deepcopy(template);a['weights']=coef.tolist()
            # Unused basis dimensions may not restrict the simpler candidate's support.
            a['active_features']=nactive.tolist()
            score=evaluate(data['val'],a,incumbent)
            ratios={key:np.array(v['variants']['final']['mse_nm2'])/np.maximum(before[key],1e-10) for key,v in score.items()}
            candidates.append(dict(mode=mode,ridge=ridge,artifact=a,validation=score,
                joint_score=(.5*(ratios['low']+ratios['high'])).tolist(),
                feasible=((ratios['low']<=1.05)&(ratios['high']<=.98)).tolist()))
            print(mode,ridge,'val rms',{k:round(v['variants']['final']['overall_rms_nm'],5) for k,v in score.items()},flush=True)
    selected=[];final=copy.deepcopy(template);coef=np.zeros((len(scale),7));used=np.zeros(len(scale),bool)
    for joint in range(7):
        feasible=[c for c in candidates if c['feasible'][joint]]
        c=min(feasible or candidates,key=lambda c:c['joint_score'][joint])
        coef[:,joint]=np.array(c['artifact']['weights'])[:,joint]
        used |= (np.abs(coef[:,joint])>1e-12)
        selected.append(dict(joint=joint+1,mode=c['mode'],ridge=c['ridge'],validation_feasible=bool(c['feasible'][joint])))
    final['weights']=coef.tolist();final['active_features']=(active & used).tolist()
    # Preserve conservative support for each selected candidate, including all nonzero columns.
    final['selection']=dict(per_joint=selected,role='val',test_used=False)
    write_json(output/'model.json',final);model_hash=sha256(output/'model.json')
    write_json(output/'selection_frozen.json',dict(model_sha256=model_hash,selection=selected,
        test_loaded=False,training_rows=len(x),class_recording_counts=counts))
    print('Selection frozen before opening test:',selected,flush=True)
    report=dict(selected=selected,model_sha256=model_hash,splits=dict(val=evaluate(data['val'],final,incumbent)),
        candidates=[{k:v for k,v in c.items() if k!='artifact'} for c in candidates],
        contact_transfer=contact_transfer(),test_used_for_fit=False,test_previously_inspected=True)
    data.update(matching_series(args.parent,args.sigma_dir,roles=('test',)))
    prepare(data['test'],bounds);report['splits']['test']=evaluate(data['test'],final,incumbent)
    assert sha256(output/'model.json')==model_hash
    passed=True;checks={}
    for key,result in report['splits']['test'].items():
        b=np.array(result['variants']['filter_only']['rms_nm']);f=np.array(result['variants']['final']['rms_nm'])
        limit=1.15 if key=='low' else .95
        aggregate=result['variants']['final']['overall_rms_nm']**2 <= limit*result['variants']['filter_only']['overall_rms_nm']**2
        per=(f<=b+np.maximum(.2*b,.02));checks[key]=dict(aggregate_pass=bool(aggregate),per_joint_pass=per.tolist())
        passed &= aggregate and bool(per.all())
    report.update(passed=bool(passed),checks=checks,elapsed_seconds=time.monotonic()-started)
    write_json(output/'comparison.json',report)
    write_json(output/'manifest.json',dict(schema=SCHEMA,base_metadata_sha256=identity,deployment_ready=False,
        files={f:sha256(output/f) for f in ('model.json','recipe.json','comparison.json','selection_frozen.json')}))
    print(json.dumps(dict(passed=bool(passed),test={k:{v:r['overall_rms_nm'] for v,r in s['variants'].items()} for k,s in report['splits']['test'].items()}),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent','sigma-dir','base','incumbent','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
