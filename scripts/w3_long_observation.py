"""Freeze a bounded, long unloaded NEXT observation; never transmit motion.

Every motion block is one complete native trajectory. Explicit endpoint holds
are labelled separately; continuous blocks retain nonzero interior velocities.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from w3_coverage_plan import ArmModel
from w3_baseline_protocol import conditions
from w3_feedback_protocol import plateau_stroke
from w3_targeted_protocol import SCHEMA,validate,knot,sample,bounds,oscillation,thermal_declaration

ROOT=Path(__file__).resolve().parents[1]
PROFILES=[dict(id='observe_v2p4_a3',velocity_deg_s=2.4,acceleration_deg_s2=3.,jerk_deg_s3=15.),
          dict(id='observe_v4_a6',velocity_deg_s=4.,acceleration_deg_s2=6.,jerk_deg_s3=25.),
          dict(id='observe_v6_a10',velocity_deg_s=6.,acceleration_deg_s2=10.,jerk_deg_s3=40.)]


def build_plan(description,initial,left,envelope):
    cfg=yaml.safe_load((ROOT/'config/w3/motions/right_targeted_free_v1.yaml').read_text())
    bound=np.array(envelope['position_bounds_deg']);home=np.asarray(initial,float)
    cfg.update(name='right_large_long_next_observation_20261023',profiles=copy.deepcopy(PROFILES),
        home_deg=home.tolist(),anchors_deg=[home.tolist()],continuous_cycles=3,
        working_bounds_deg=np.c_[bound[:,0]+2.,bound[:,1]-2.].tolist(),
        collection_selection=dict(single_joints=list(range(7)),anchor_indices=[0],coupled_patterns=1,profiles=[p['id'] for p in PROFILES]))
    working=np.array(cfg['working_bounds_deg']);model=ArmModel(description,'right',cfg['tool_frame'])
    current=home.copy();steps=[];rng=np.random.default_rng(102306)
    def add(knots,kind,family,profile,active,dwell=None):
        nonlocal current
        duration=knots[-1]['time_seconds'];t=np.arange(round(duration/.01)+1)*.01;q,_,_=sample(knots,t)
        low,high,peak=bounds(knots)
        if max(abs(current-q[0]))>1e-7:raise ValueError('Discontinuous block')
        if np.any(low<working[:,0]) or np.any(high>working[:,1]):raise ValueError('Outside 2 degree training margin: '+kind)
        cond=float(max(conditions(model,q)))
        if cond>cfg['max_condition']:raise ValueError('Ill-conditioned motion '+kind)
        steps.append(dict(name=f'{len(steps)+1:04d}_{kind}',family=family,kind=kind,phase=family,speed=profile['id'],
            profile=copy.deepcopy(profile),active_joints=active,pose_index=0,approach_direction=None,
            dwell_seconds=dwell,duration_seconds=float(duration),hold_seconds=0.,times_seconds=t.tolist(),
            positions_deg=q.tolist(),knots=knots,peak_bounds={k:peak[i].tolist() for i,k in enumerate(('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3'))},
            max_condition=cond,continuous_internal_dwell=False if family=='continuous' else None,
            annotations=[dict(start_seconds=0.,end_seconds=duration,state='stationary_hold' if dwell else kind)]))
        current=q[-1].copy()
    def hold(duration,kind,p):add([knot(0,current),knot(duration,current)],kind,'transition',p,[],duration)
    def straight(target,kind,p):
        delta=np.asarray(target)-current;distance=max(abs(delta))
        if distance<1e-7:return
        # Choose a whole 10ms duration while retaining the exact target. A
        # slightly smaller velocity fits the same acceleration/jerk profile.
        v=p['velocity_deg_s'];a=p['acceleration_deg_s2'];j=p['jerk_deg_s3']
        tj=a/j
        duration=np.ceil((distance/v+v/a+tj)*100-1e-8)/100
        # Solve distance/v + v/a + tj = duration for the smaller root.
        v=(a*(duration-tj)-np.sqrt((a*(duration-tj))**2-4*a*distance))/2
        ks,_=plateau_stroke(current,delta/distance,distance,float(v),a,j)
        add(ks,kind,'continuous',p,np.flatnonzero(abs(delta)>1e-7).tolist())
    def wave(amplitude,kind,p,phase=None):
        amp=np.asarray(amplitude,float)
        ks=oscillation(current,amp,p,cfg,rng.uniform(-np.pi,np.pi,7) if phase is None else phase)
        add(ks,kind,'continuous',p,np.flatnonzero(amp).tolist())
    hold(20.,'initial_settle',PROFILES[0])
    center=home+np.array([9.6,5.,-4.,-4.,3.,-5.,0.])
    budgets=(180.,380.,620.)
    for index,p in enumerate(PROFILES):
        for joint in range(7):
            amp=np.zeros(7);amp[joint]=.6
            ks=oscillation(current,amp,p,cfg,np.zeros(7));add(ks,f'probe_{index}_j{joint+1}','speed_probe',p,[joint])
        # This first continuous block forces seven-joint promotion BEFORE any
        # larger movement at this profile.
        wave([.6]*7,f'gentle_coupled_verification_{index}',p,np.zeros(7))
        if index==0:
            for joint,delta in [(0,16.8),(1,9.6),(3,-7.2)]:
                target=home.copy();target[joint]+=delta
                straight(target,f'large_j{joint+1}_out',p);hold(3.,f'hold_large_j{joint+1}_out',p)
                straight(home,f'large_j{joint+1}_return',p);hold(3.,f'hold_large_j{joint+1}_return',p)
            straight(center,'coupled_center_approach',p)
        amplitude=([6.,2.,2.,2.,2.,2.,.8] if index==0 else
                   [7.,3.,3.,3.,2.,3.,1.] if index==1 else [8.,3.,4.,3.,3.,4.,1.2])
        repetition=0
        while sum(s['duration_seconds'] for s in steps)<budgets[index]:
            wave(amplitude,f'coupled_profile{index}_cycle{repetition}',p);repetition+=1
        hold(10.,f'profile{index}_settle',p)
    # Slow return uses a trajectory whose bounds remain below the final
    # declared profile; the frozen profile order never regresses.
    straight(home,'return_home',PROFILES[-1]);hold(30.,'final_settle',PROFILES[-1])
    plan=dict(schema=SCHEMA,name=cfg['name'],side='right',family='continuous',role='test',config=cfg,
        robot_description=description,model_sha256=model.sha256,joint_names=model.names,
        initial_deg=home.tolist(),left_pose_deg=np.asarray(left,float).tolist(),thermal=thermal_declaration('unclassified'),steps=steps)
    validate(plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    stage=args.output.resolve();assert stage.is_dir() and not (stage/'motion.yaml').exists()
    from w3_targeted_capture import live_state
    description,initial,left=live_state()
    meta=json.loads((ROOT/'models/w3/right/targeted_candidate_20261016/metadata.json').read_text())
    plan=build_plan(description,initial,left,meta['operating_envelope'])
    raw=yaml.dump(plan,Dumper=yaml.CSafeDumper,sort_keys=False);(stage/'motion.yaml').write_text(raw);(stage/'robot.urdf').write_text(description)
    q=np.concatenate([np.asarray(s['positions_deg']) for s in plan['steps']])
    spec=dict(purpose='Long, larger-amplitude unloaded observation of latest joint torque dashboard',role='test',
        model_dir=str(ROOT/'models/w3/right/targeted_candidate_20261016'),model_hash=meta['artifact_sha256']['model.pt'],
        plan_sha256=hashlib.sha256(raw.encode()).hexdigest(),initial_command_deg=initial,left_pose_deg=left,
        training_envelope=meta['operating_envelope'],command_seconds=sum(s['duration_seconds'] for s in plan['steps']),steps=len(plan['steps']),
        planned_position_range_deg=np.c_[q.min(0),q.max(0)].tolist(),velocity_profiles_deg_s=[p['velocity_deg_s'] for p in PROFILES],
        expected_units='joint external torque, Nm; no endpoint wrench conversion',
        ground_truth=dict(source='operator declares unloaded',assumed_external_joint_torque_nm=[0.]*7,independently_measured=False),
        no_model_or_control_changes=True,no_time_shift_fit=True,automatic_retry=False)
    (stage/'test_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n');print(json.dumps(spec,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
