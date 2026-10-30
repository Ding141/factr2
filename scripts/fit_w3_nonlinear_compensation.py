"""Train/validate a frozen kinematics-only residual MLP; no robot control.

The previous posture ridge was rejected on validation compatibility. A fixed
seed, finite epoch budget, class-normalized loss and per-joint continuous
correction gates are selected on validation only. Tests open after freezing.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from finish_w3_highspeed import matching_series
from fit_w3_pose_dynamic_compensation import prepare, class_recording_weights, evaluate
from fit_w3_kinematic_compensation import smooth, contact_transfer
from factr2_next.data_collection.quality import write_json, sha256
from factr2_next.inference.kinematic_compensation import DEFAULT_SIGNAL, ResidualModel


def export(template, model, center, scale, indices, gates):
    a=copy.deepcopy(template)
    a['weights']=np.zeros_like(a['weights']).tolist()
    a['active_features']=[False]*len(a['feature_scale'])
    for i in indices:a['active_features'][int(i)]=True
    a['motion_gate_deg_s']=gates
    a['nonlinear_residual']=dict(activation='tanh',input_indices=indices.tolist(),
        input_center=center.tolist(),input_scale=scale.tolist(),
        fixed_bias_from_train=True,online_bias_adaptation=False,
        layers=[dict(weight=l.weight.detach().cpu().numpy().T.astype(float).tolist(),
                     bias=l.bias.detach().cpu().numpy().astype(float).tolist())
                for l in model if isinstance(l,nn.Linear)])
    a['selection']=dict(role='val',test_used=False)
    return a


def validation_choice(items, artifact, baseline, gates):
    # Compute the same frozen nonlinear correction once, then select smooth
    # speed gains per output. Every candidate has identical causal filters.
    raw_model=ResidualModel(artifact);raw_model.speed_gate=np.asarray(0.)
    data=[]
    for item in items:
        x=item['pose_features'];raw=raw_model.correction_from_features(x)
        active=raw_model.active;lo,hi=raw_model.low,raw_model.high
        margin=np.maximum(.2*(hi-lo),.2*raw_model.scale)
        q=item['position_filtered'];bounds=raw_model.position_bounds
        supported=(np.all(((x>=lo-margin)&(x<=hi+margin))[:,active],axis=1)
            &np.all((q>=bounds[:,0])&(q<=bounds[:,1]),axis=1)
            &item['eligible']&item['position_gate'])
        raw[~supported]=0
        data.append((item,raw,np.mean(x[:,7:14]**2,axis=1,keepdims=True)))
    scores=[]
    for gate in gates:
        accum={False:[],True:[]}
        for item,raw,speed2 in data:
            c=raw*speed2/(speed2+np.deg2rad(gate)**2)
            out=smooth(item['target']-c,item['t'],DEFAULT_SIGNAL['output_hz'])
            accum[item['high']].append(out[item['keep']])
        mse={key:np.mean(np.concatenate(v)**2,axis=0) for key,v in accum.items()}
        ratios={key:mse[key]/np.maximum(baseline[key],1e-10) for key in mse}
        scores.append(dict(gate=gate,score=.5*(ratios[False]+ratios[True]),
            feasible=(ratios[False]<=1.05)&(ratios[True]<=.98),mse=mse))
    selected=[];score=[];feasible=[]
    for j in range(7):
        choices=[c for c in scores if c['feasible'][j]]
        c=min(choices or scores,key=lambda c:c['score'][j])
        selected.append(c['gate']);score.append(float(c['score'][j]));feasible.append(bool(c['feasible'][j]))
    return selected,float(np.mean(score)),feasible


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    template=json.loads((args.template/'model.json').read_text())
    incumbent=json.loads((args.incumbent/'model.json').read_text())
    identity=sha256(args.base/'metadata.json');assert template['base_metadata_sha256']==identity
    recipe=dict(base=str(args.base.resolve()),base_metadata_sha256=identity,signal=DEFAULT_SIGNAL,
        source_mean_experiment=str(args.parent.resolve()),source_sigma_experiment=str(args.sigma_dir.resolve()),
        architecture=[140,64,64,7],seed=0,epochs=30,batch_size=4096,learning_rate=.001,
        weight_decay=.001,training_motion_gate_deg_s=2.,candidate_motion_gates_deg_s=[1.,2.,4.,8.],
        selection='validation only: each joint gate preferring low MSE<=105%, high<=98%; common epoch minimum equal-class normalized final-output MSE',
        loss='equal class and recording mass, per-joint class-RMS normalization (floor .02Nm)',
        train_role='original train recordings only',test_used_for_fit=False,test_previously_inspected=True,
        no_observation_refit=True,no_torque_feature=True,no_online_zeroing=True,no_output_deadband=True,
        nonlinear_bias='fixed train parameters; never zeroed from live torque',
        motivation='posture-ridge validation low-speed compatibility failed for J2/J6/J7')
    write_json(output/'recipe.json',recipe);torch.manual_seed(0);torch.set_num_threads(1)
    bounds=np.deg2rad(json.loads((args.base/'metadata.json').read_text())['operating_envelope']['position_bounds_deg'])
    print('Loading train/val only',flush=True)
    data=matching_series(args.parent,args.sigma_dir,roles=('train','val'))
    for role in data:prepare(data[role],bounds)
    items=data['train'];w,counts=class_recording_weights(items)
    features=np.concatenate([i['pose_features'][i['fit_keep']] for i in items])
    y=np.concatenate([i['target'][i['fit_keep']] for i in items])
    high=np.concatenate([np.full(i['fit_keep'].sum(),i['high'],bool) for i in items])
    indices=np.arange(140);center=np.average(features[:,indices],axis=0,weights=w)
    scale=np.maximum(np.sqrt(np.average((features[:,indices]-center)**2,axis=0,weights=w)),
                     np.array(template['excitation_floor'])[indices])
    x=(features[:,indices]-center)/scale
    loss_scale=np.stack([np.maximum(np.sqrt(np.mean(y[high==h]**2,axis=0)),.02) for h in (False,True)])
    gate=np.mean(features[:,7:14]**2,axis=1,keepdims=True)
    gate=gate/(gate+np.deg2rad(recipe['training_motion_gate_deg_s'])**2)
    tx,ty,tw,tg,ts=[torch.tensor(a,dtype=torch.float32) for a in
        (x,y,w*len(w),gate,loss_scale[high.astype(int)])]
    caps=torch.tensor(template['correction_cap_nm'],dtype=torch.float32)
    model=nn.Sequential(nn.Linear(140,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,7))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    baseline=evaluate(data['val'],incumbent=incumbent)
    base={h:np.array(baseline['high' if h else 'low']['variants']['filter_only']['mse_nm2']) for h in (False,True)}
    best=None;history=[];rng=np.random.default_rng(0)
    for epoch in range(1,recipe['epochs']+1):
        model.train();order=rng.permutation(len(x));total=0.
        for offset in range(0,len(x),recipe['batch_size']):
            ids=order[offset:offset+recipe['batch_size']]
            correction=torch.maximum(torch.minimum(model(tx[ids]),caps),-caps)*tg[ids]
            loss=((((correction-ty[ids])/ts[ids])**2).mean(1)*tw[ids]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(ids)
        if epoch%2 and epoch!=1:continue
        a=export(template,model,center,scale,indices,recipe['training_motion_gate_deg_s'])
        gates,score,feasible=validation_choice(data['val'],a,base,recipe['candidate_motion_gates_deg_s'])
        record=dict(epoch=epoch,train_loss=total/len(x),val_score=score,gates=gates,feasible=feasible)
        history.append(record);print(json.dumps(record),flush=True)
        # A valid joint-wise gate is preferred to a lower score that regresses.
        rank=(sum(not f for f in feasible),score)
        if best is None or rank<best[0]:
            a['motion_gate_deg_s']=gates;a['selection'].update(epoch=epoch,validation_feasible=feasible)
            best=(rank,a,record)
        write_json(output/'progress.json',dict(epoch=epoch,best=best[2]))
    final=best[1];write_json(output/'model.json',final);model_hash=sha256(output/'model.json')
    write_json(output/'selection_frozen.json',dict(model_sha256=model_hash,test_loaded=False,
        selection=best[2],training_rows=len(x),class_recording_counts=counts))
    write_json(output/'history.json',history)
    print('Model frozen; loading retrospective test only now',flush=True)
    report=dict(model_sha256=model_hash,selected=best[2],splits=dict(val=evaluate(data['val'],final,incumbent)),
        contact_transfer=contact_transfer(),test_used_for_fit=False,test_previously_inspected=True)
    test=matching_series(args.parent,args.sigma_dir,roles=('test',))['test'];prepare(test,bounds)
    report['splits']['test']=evaluate(test,final,incumbent)
    checks={};passed=True
    for key,result in report['splits']['test'].items():
        b=np.array(result['variants']['filter_only']['rms_nm']);f=np.array(result['variants']['final']['rms_nm'])
        limit=1.15 if key=='low' else .95
        aggregate=result['variants']['final']['overall_rms_nm']**2<=limit*result['variants']['filter_only']['overall_rms_nm']**2
        per=f<=b+np.maximum(.2*b,.02);checks[key]=dict(aggregate=bool(aggregate),per_joint=per.tolist())
        passed &= aggregate and bool(per.all())
    assert sha256(output/'model.json')==model_hash
    report.update(passed=bool(passed),checks=checks,elapsed_seconds=time.monotonic()-start)
    write_json(output/'comparison.json',report)
    write_json(output/'manifest.json',dict(schema=template['schema'],base_metadata_sha256=identity,deployment_ready=False,
        files={f:sha256(output/f) for f in ('model.json','recipe.json','comparison.json','selection_frozen.json','history.json')}))
    print(json.dumps(dict(passed=bool(passed),selected=best[2],test={k:{n:v['overall_rms_nm'] for n,v in s['variants'].items()} for k,s in report['splits']['test'].items()}),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent','sigma-dir','base','incumbent','template','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
