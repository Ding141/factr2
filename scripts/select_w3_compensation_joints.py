"""Select per-joint updates against the incumbent on validation only.

The same frozen mean is retained. Unqualified joints and unsupported nonlinear
features use the incumbent correction, not an artificial zero residual.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from finish_w3_highspeed import matching_series
from fit_w3_pose_dynamic_compensation import prepare,evaluate
from fit_w3_kinematic_compensation import contact_transfer
from factr2_next.data_collection.quality import sha256,write_json


def embed_incumbent(incumbent,template):
    a=copy.deepcopy(template)
    for k in ('nonlinear_residual','motion_gate_deg_s','speed_blend','support_fallback'):a.pop(k,None)
    n=len(incumbent['feature_scale']);w=np.zeros_like(a['weights'])
    w[:n]=np.array(incumbent['weights'])*np.array(a['feature_scale'])[:n,None]/np.array(incumbent['feature_scale'])[:,None]
    a['weights']=w.tolist();a['active_features']=incumbent['active_features']+[False]*(len(w)-n)
    a['feature_min'][:n]=incumbent['feature_min'];a['feature_max'][:n]=incumbent['feature_max']
    # Use the original normalization for support margins, preserving exact
    # physical range semantics as well as the numerical linear correction.
    a['feature_scale'][:n]=incumbent['feature_scale']
    a['weights'][:n]=incumbent['weights']
    a['position_bounds_rad']=incumbent['position_bounds_rad']
    a['selection']=dict(source='frozen incumbent; representation padded without numerical changes')
    return a


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    candidate=json.loads((a.candidate/'model.json').read_text());inc=json.loads((a.incumbent/'model.json').read_text())
    c=json.loads((a.candidate/'comparison.json').read_text());identity=sha256(a.base/'metadata.json')
    assert candidate['base_metadata_sha256']==inc['base_metadata_sha256']==identity
    ratios={key:np.array(value['variants']['final']['mse_nm2'])/np.maximum(value['variants']['incumbent']['mse_nm2'],1e-10)
            for key,value in c['splits']['val'].items()}
    scores={name:np.mean([np.array(v['variants'][name]['mse_nm2'])/np.maximum(v['variants']['filter_only']['mse_nm2'],1e-10)
                for v in c['splits']['val'].values()],axis=0) for name in ('incumbent','final')}
    selected=(ratios['low']<=1.05)&(ratios['high']<=1.05)&(scores['final']<=.98*scores['incumbent'])
    recipe=dict(base=str(a.base.resolve()),base_metadata_sha256=identity,signal=candidate['signal'],
        source_mean_experiment=str(a.parent.resolve()),source_sigma_experiment=str(a.sigma_dir.resolve()),
        nonlinear_candidate=str(a.candidate.resolve()),incumbent=str(a.incumbent.resolve()),
        candidate_sha256=sha256(a.candidate/'model.json'),incumbent_sha256=sha256(a.incumbent/'model.json'),
        selection_role='val only',criterion='each low/high validation MSE <=105% incumbent AND equal-class normalized score improves >=2%',
        fallback='original incumbent for unselected joints and outside nonlinear support',
        no_torque_feature=True,no_online_zeroing=True,no_output_deadband=True,no_observation_refit=True,
        test_used_for_fit=False,test_previously_inspected=True)
    write_json(out/'recipe.json',recipe)
    final=copy.deepcopy(candidate);legacy=embed_incumbent(inc,final);n=len(inc['feature_scale'])
    for j in range(7):
        if selected[j]:continue
        scale=np.array(final['feature_scale'])[:n]/np.array(inc['feature_scale'])
        w=np.array(final['weights']);w[:n,j]=np.array(inc['weights'])[:,j]*scale;final['weights']=w.tolist()
        final['motion_gate_deg_s'][j]=0.
        last=final['nonlinear_residual']['layers'][-1]
        w=np.array(last['weight']);w[:,j]=0.;last['weight']=w.tolist();last['bias'][j]=0.
    final['support_fallback']=legacy
    final['selection']=dict(role='val',nonlinear_joints=selected.tolist(),test_used=False,
        criterion=recipe['criterion'],low_validation_mse_ratio_to_incumbent=ratios['low'].tolist(),
        high_validation_mse_ratio_to_incumbent=ratios['high'].tolist())
    write_json(out/'model.json',final);h=sha256(out/'model.json')
    write_json(out/'selection_frozen.json',dict(model_sha256=h,selection=final['selection'],test_loaded=False))
    print('Frozen validation-selected update',selected.tolist(),flush=True)
    meta=json.loads((a.base/'metadata.json').read_text());bounds=np.deg2rad(meta['operating_envelope']['position_bounds_deg'])
    report=dict(model_sha256=h,selected=final['selection'],splits={},test_used_for_fit=False,
                test_previously_inspected=True,contact_transfer=contact_transfer())
    for role in ('val','test'):
        items=matching_series(a.parent,a.sigma_dir,roles=(role,))[role];prepare(items,bounds)
        report['splits'][role]=evaluate(items,final,inc)
    passed=True;checks={}
    for key,result in report['splits']['test'].items():
        b=np.array(result['variants']['filter_only']['rms_nm']);f=np.array(result['variants']['final']['rms_nm'])
        limit=1.15 if key=='low' else .95
        aggregate=result['variants']['final']['overall_rms_nm']**2<=limit*result['variants']['filter_only']['overall_rms_nm']**2
        per=f<=b+np.maximum(.2*b,.02);checks[key]=dict(aggregate=bool(aggregate),per_joint=per.tolist());passed &= aggregate and bool(per.all())
    assert sha256(out/'model.json')==h
    report.update(passed=bool(passed),checks=checks);write_json(out/'comparison.json',report)
    write_json(out/'manifest.json',dict(schema=final['schema'],base_metadata_sha256=identity,deployment_ready=False,
        files={f:sha256(out/f) for f in ('model.json','recipe.json','comparison.json','selection_frozen.json')}))
    print(json.dumps(dict(passed=bool(passed),test={k:{n:v['overall_rms_nm'] for n,v in s['variants'].items()} for k,s in report['splits']['test'].items()}),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent','sigma-dir','base','incumbent','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
