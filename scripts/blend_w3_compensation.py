"""Freeze a smooth slow/fast correction blend using validation only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from finish_w3_highspeed import matching_series
from fit_w3_pose_dynamic_compensation import prepare, evaluate
from fit_w3_kinematic_compensation import smooth, contact_transfer
from factr2_next.inference.kinematic_compensation import ResidualModel
from factr2_next.data_collection.quality import write_json,sha256


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    slow=json.loads((a.slow/'model.json').read_text());fast=json.loads((a.fast/'model.json').read_text())
    incumbent=json.loads((a.incumbent/'model.json').read_text());meta=json.loads((a.base/'metadata.json').read_text())
    identity=sha256(a.base/'metadata.json');assert slow['base_metadata_sha256']==fast['base_metadata_sha256']==identity
    choices=[.1,.25,.5,1.,2.,4.,8.,16.]
    recipe=dict(base=str(a.base.resolve()),base_metadata_sha256=identity,signal=slow['signal'],
        source_mean_experiment=str(a.parent.resolve()),source_sigma_experiment=str(a.sigma_dir.resolve()),
        slow_component=str(a.slow.resolve()),fast_component=str(a.fast.resolve()),
        component_sha256=dict(slow=sha256(a.slow/'model.json'),fast=sha256(a.fast/'model.json')),
        selection_role='val only',transition_candidates_deg_s=choices,
        objective='per joint equal low/high normalized final MSE, require low<=105% and high<=98% where feasible',
        reason='validation shows complementary slow-speed neural and high-speed posture ridge errors',
        test_used_for_fit=False,test_previously_inspected=True,no_online_zeroing=True,no_torque_feature=True,
        no_output_deadband=True,no_observation_refit=True,blend='convex continuous kinematics-only correction gain; not external-output suppression')
    write_json(out/'recipe.json',recipe)
    data=matching_series(a.parent,a.sigma_dir,roles=('val',));items=data['val'];prepare(items,np.deg2rad(meta['operating_envelope']['position_bounds_deg']))
    baseline=evaluate(items,incumbent=incumbent)
    base={key:np.array(v['variants']['filter_only']['mse_nm2']) for key,v in baseline.items()}
    final=copy.deepcopy(fast);final.pop('nonlinear_residual',None);final.pop('motion_gate_deg_s',None)
    final['weights']=np.zeros_like(final['weights']).tolist()
    final['active_features']=(np.array(slow['active_features'])|np.array(fast['active_features'])).tolist()
    final['speed_blend']=dict(slow=slow,fast=fast,transition_deg_s=[1.]*7)
    scores=[]
    for transition in choices:
        model=copy.deepcopy(final);model['speed_blend']['transition_deg_s']=[transition]*7
        metrics=evaluate(items,model,incumbent)
        ratio={key:np.array(v['variants']['final']['mse_nm2'])/np.maximum(base[key],1e-10) for key,v in metrics.items()}
        score=.5*(ratio['low']+ratio['high']);feasible=(ratio['low']<=1.05)&(ratio['high']<=.98)
        scores.append(dict(transition=transition,score=score,feasible=feasible))
        print('transition',transition,'val',{k:round(v['variants']['final']['overall_rms_nm'],5) for k,v in metrics.items()},flush=True)
    selected=[]
    for j in range(7):
        feasible=[s for s in scores if s['feasible'][j]]
        best=min(feasible or scores,key=lambda s:s['score'][j])
        selected.append(dict(joint=j+1,transition_deg_s=best['transition'],validation_feasible=bool(best['feasible'][j])))
    final['speed_blend']['transition_deg_s']=[s['transition_deg_s'] for s in selected]
    final['selection']=dict(role='val',per_joint=selected,test_used=False)
    write_json(out/'model.json',final);model_hash=sha256(out/'model.json')
    write_json(out/'selection_frozen.json',dict(model_sha256=model_hash,test_loaded=False,selection=selected))
    print('Blend frozen',selected,flush=True)
    report=dict(selected=selected,model_sha256=model_hash,splits=dict(val=evaluate(items,final,incumbent)),
                contact_transfer=contact_transfer(),test_used_for_fit=False,test_previously_inspected=True)
    test=matching_series(a.parent,a.sigma_dir,roles=('test',))['test'];prepare(test,np.deg2rad(meta['operating_envelope']['position_bounds_deg']))
    report['splits']['test']=evaluate(test,final,incumbent);passed=True;checks={}
    for key,result in report['splits']['test'].items():
        b=np.array(result['variants']['filter_only']['rms_nm']);f=np.array(result['variants']['final']['rms_nm'])
        limit=1.15 if key=='low' else .95
        aggregate=result['variants']['final']['overall_rms_nm']**2<=limit*result['variants']['filter_only']['overall_rms_nm']**2
        per=f<=b+np.maximum(.2*b,.02);checks[key]=dict(aggregate=bool(aggregate),per_joint=per.tolist());passed &= aggregate and bool(per.all())
    assert sha256(out/'model.json')==model_hash
    report.update(passed=bool(passed),checks=checks);write_json(out/'comparison.json',report)
    write_json(out/'manifest.json',dict(schema=final['schema'],base_metadata_sha256=identity,deployment_ready=False,
        files={f:sha256(out/f) for f in ('model.json','recipe.json','comparison.json','selection_frozen.json')}))
    print(json.dumps(dict(passed=bool(passed),test={k:{n:v['overall_rms_nm'] for n,v in s['variants'].items()} for k,s in report['splits']['test'].items()}),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent','sigma-dir','base','incumbent','slow','fast','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
