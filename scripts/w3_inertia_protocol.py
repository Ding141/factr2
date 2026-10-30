"""Offline frozen acceleration experiment; live preparation is read-only.

One complete native trajectory per move. Same geometric stroke, velocity and
jerk; acceleration plateaus change. Execution uses the existing guarded capture.
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
from w3_targeted_protocol import knot, sample, bounds, oscillation, validate, thermal_declaration

ROOT=Path(__file__).resolve().parents[1]
VELOCITY=3.84
JERK=25.
DISTANCE=6.144
ACCELERATIONS=(4.,6.,8.)
PATTERNS={'J1':[1,0,0,0,0,0,0], 'J2':[0,1,0,0,0,0,0],
          'J4':[0,0,0,-1,0,0,0], 'coupled':[1,.5,0,-.5,0,0,0]}


def matched_stroke(start,direction,acceleration):
    """Seven-phase S-curve; all durations exactly on a centisecond grid."""
    if acceleration not in ACCELERATIONS:raise ValueError('Unknown acceleration condition')
    start=np.asarray(start,float);direction=np.asarray(direction,float)
    if start.shape!=(7,) or direction.shape!=(7,) or not np.isfinite(np.r_[start,direction]).all() or max(abs(direction))!=1:
        raise ValueError('Invalid stroke vectors')
    tj=acceleration/JERK;ta=VELOCITY/acceleration-tj
    tc=DISTANCE/VELOCITY-(2*tj+ta)
    if min(tj,ta,tc)<.05:raise ValueError('Insufficient stroke for this velocity/acceleration/jerk')
    segments=[(JERK,tj),(0.,ta),(-JERK,tj),(0.,tc),(-JERK,tj),(0.,ta),(JERK,tj)]
    position=velocity=acc=elapsed=0.;knots=[knot(0,start)]
    for jerk,duration in segments:
        position+=velocity*duration+acc*duration**2/2+jerk*duration**3/6
        velocity+=acc*duration+jerk*duration**2/2;acc+=jerk*duration;elapsed+=duration
        knots.append(knot(elapsed,start+direction*position,direction*velocity,direction*acc))
    if abs(position-DISTANCE)>1e-9 or abs(velocity)>1e-9 or abs(acc)>1e-9:raise ValueError('S-curve integration')
    if abs(elapsed/.01-round(elapsed/.01))>1e-7:raise ValueError('Non-grid final time')
    knots[-1]=knot(round(elapsed,2),start+DISTANCE*direction)
    return knots


def build_plan(description,initial,left,envelope,name):
    cfg=yaml.safe_load((ROOT/'config/w3/motions/right_targeted_free_v1.yaml').read_text())
    bound=np.array(envelope['position_bounds_deg']);initial=np.asarray(initial,float)
    if np.any(initial<bound[:,0]+.6) or np.any(initial>bound[:,1]-.6):raise ValueError('Start near collection boundary')
    profile=dict(id='paired_v3p84_a4to8_j25',velocity_deg_s=VELOCITY,acceleration_deg_s2=8.,jerk_deg_s3=JERK)
    cfg.update(name=name,profiles=[profile],home_deg=initial.tolist(),anchors_deg=[initial.tolist()],
        working_bounds_deg=np.c_[bound[:,0]+.6,bound[:,1]-.6].tolist(),continuous_cycles=2,
        collection_selection=dict(single_joints=[0,1,3],anchor_indices=[0],coupled_patterns=1,profiles=[profile['id']]))
    model=ArmModel(description,'right',cfg['tool_frame']);steps=[];current=initial.copy()
    def add(knots,kind,family,active,annotations,dwell=None,condition=None):
        nonlocal current
        duration=knots[-1]['time_seconds'];times=np.arange(round(duration/.01)+1)*.01
        q,_,_=sample(knots,times);low,high,peak=bounds(knots)
        if max(abs(q[0]-current))>1e-8:raise ValueError('Discontinuous move')
        if np.any(low<bound[:,0]+.6) or np.any(high>bound[:,1]-.6):raise ValueError('Outside training margin: '+kind)
        step=dict(name=f'{len(steps)+1:04d}_{kind}',family=family,kind=kind,phase=family,speed=profile['id'],
            profile=copy.deepcopy(profile),active_joints=list(active),pose_index=0,approach_direction=None,
            dwell_seconds=dwell,duration_seconds=duration,hold_seconds=0.,times_seconds=times.tolist(),
            positions_deg=q.tolist(),knots=knots,
            peak_bounds={k:peak[j].tolist() for j,k in enumerate(('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3'))},
            max_condition=float(max(conditions(model,q))),annotations=annotations)
        if condition is not None:step['inertia_condition']=condition
        steps.append(step);current=q[-1].copy()
    def hold(duration,kind,family='transition',condition=None):
        annotation=dict(start_seconds=0.,end_seconds=duration,state='stationary_hold')
        if condition is not None:annotation.update(condition)
        add([knot(0,current),knot(duration,current)],kind,family,[],[annotation],duration,condition)
    hold(10.,'initial_settle')
    # Required seven-joint feedback probes use a lower acceleration bound.
    probe_profile=dict(profile,acceleration_deg_s2=4.)
    for j in range(7):
        amplitude=np.zeros(7);amplitude[j]=.6
        ks=oscillation(current,amplitude,probe_profile,cfg,np.zeros(7))
        add(ks,f'probe_j{j+1}','speed_probe',[j],
            [dict(start_seconds=0.,end_seconds=ks[-1]['time_seconds'],state='excluded_safety_probe')])
    hold(2.,'hold_after_probes')
    # Three Latin-square condition orders; pattern order also alternates.
    orders=[(4.,6.,8.),(6.,8.,4.),(8.,4.,6.)]
    pattern_orders=[list(PATTERNS),list(reversed(PATTERNS)),['J2','J4','coupled','J1']]
    for repeat,order in enumerate(orders,1):
        for pattern in pattern_orders[repeat-1]:
            direction=np.array(PATTERNS[pattern],float);active=np.flatnonzero(direction).tolist()
            for acceleration in order:
                condition=dict(pattern=pattern,repeat=repeat,acceleration_deg_s2=acceleration,
                    velocity_deg_s=VELOCITY,jerk_deg_s3=JERK,path_distance_deg=DISTANCE,
                    path_direction=direction.tolist(),pair_id=f'{pattern}_r{repeat}')
                for sign,label in [(1,'out'),(-1,'return')]:
                    c=dict(condition,direction=label)
                    ks=matched_stroke(current,sign*direction,acceleration)
                    annotation=dict(start_seconds=0.,end_seconds=ks[-1]['time_seconds'],state='inertia_comparison',**c)
                    add(ks,f'inertia_{pattern}_r{repeat}_a{int(acceleration)}_{label}','hysteresis',active,[annotation],condition=c)
                    hold(2.,f'hold_{pattern}_r{repeat}_a{int(acceleration)}_{label}','hysteresis',c)
    hold(10.,'final_settle')
    plan=dict(schema='w3_targeted_free_motion_v1',name=name,side='right',family='hysteresis',role='test',
        thermal=thermal_declaration('unclassified'),config=cfg,robot_description=description,
        model_sha256=hashlib.sha256(description.encode()).hexdigest(),joint_names=model.names,
        initial_deg=initial.tolist(),left_pose_deg=np.asarray(left,float).tolist(),steps=steps,
        inertia_protocol=dict(schema='w3_inertia_comparison_v1',velocity_deg_s=VELOCITY,
            acceleration_deg_s2=list(ACCELERATIONS),jerk_deg_s3=JERK,distance_deg=DISTANCE,
            patterns=PATTERNS,repeats=3,condition_order=[list(x) for x in orders],
            explicit_endpoint_dwell_seconds=2.,data_role='independent diagnostic only; not training',
            matching_criteria=dict(actual_peak_speed_max_to_min_ratio=1.15,
                same_direction_start_pose_span_deg=.3,same_direction_end_pose_span_deg=.3,
                acceleration_order='strictly increasing median achieved peak, and high/low >=1.4',
                derivative_windows_samples=[21,31],delay_fitting=False,
                unmatched_trials='report separately; do not claim isolated acceleration causality')))
    validate(plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stage=args.output.resolve()
    if stage.exists():raise ValueError('Use a new output directory')
    from w3_targeted_capture import live_state
    description,initial,left=live_state()
    meta=json.loads((ROOT/'models/w3/right/targeted_candidate_20261016/metadata.json').read_text())
    if hashlib.sha256(description.encode()).hexdigest()!='e3a8e37ee96fe8ff435af115e63aa788b4aed3e7b9c1dd2f65de9bcd9948e345':
        raise ValueError('Robot description changed')
    plan=build_plan(description,initial,left,meta['operating_envelope'],stage.name)
    encoded=yaml.dump(plan,Dumper=yaml.CSafeDumper,sort_keys=False)
    validate(yaml.safe_load(encoded))
    stage.mkdir(parents=True);path=stage/'motion.yaml';path.write_text(encoded)
    (stage/'robot.urdf').write_text(description)
    spec=dict(purpose='matched-path, matched-target-peak-velocity acceleration test; NEXT only',role='test',
        model_dir=str(ROOT/'models/w3/right/targeted_candidate_20261016'),model_hash=meta['artifact_sha256']['model.pt'],
        endpoint_config=yaml.safe_load((ROOT/'config/w3/right/endpoint_wrench_targeted_20261016.yaml').read_text()),
        plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),initial_command_deg=initial,left_pose_deg=plan['left_pose_deg'],
        training_envelope=meta['operating_envelope'],command_seconds=sum(s['duration_seconds'] for s in plan['steps']),
        steps=len(plan['steps']),inertia_protocol=plan['inertia_protocol'],automatic_retry=False,
        ground_truth=dict(source='operator requests unloaded no-contact test',assumed_external_wrench=[0.]*6,independently_measured=False))
    (stage/'test_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ('command_seconds','steps','initial_command_deg','left_pose_deg','inertia_protocol')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
