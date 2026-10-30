"""Freeze a no-contact feedback/tracking diagnostic; never transmit motion here.

Every stroke is one native trajectory. Endpoint dwells are separate labelled
blocks. This diagnostic never estimates or changes a compensation time shift.
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

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = [1., 1., -1., -1., 1., -1., 1.]
RECIPE = dict(
    role='independent diagnostic; never training or delay calibration',
    position_derivatives=dict(sample_period_seconds=.01, polynomial_order=3,
        centered_windows_samples=[11, 21, 31, 51], include_naive_difference=True,
        edge_policy='exclude half-window at record boundaries; no differentiation across gaps >40ms'),
    plateau_guard_seconds=.25,
    plateau_checks=['encoder displacement divided by elapsed time',
        'trapezoidal integral of reported velocity against encoder displacement',
        'unshifted raw velocity and fixed-window position derivatives'],
    raw_can_matching='latest preceding kernel RX per joint; report ages and disagreement, never shift',
    uncertainty='method sensitivity and quantization bounds, not calibrated physical confidence intervals',
    latency='do not estimate, fit or apply any new time shift on these independent data',
    static_analysis='last 8s of initial/final settle; all 7 feedback channels',
    controller_changes=False, model_changes=False,
)


def plateau_stroke(start, direction, distance, velocity, acceleration, jerk):
    """Integrate seven constant-jerk phases, retaining exact interior q/v/a."""
    start=np.asarray(start,float); direction=np.asarray(direction,float)
    if start.shape!=(7,) or direction.shape!=(7,) or not np.isfinite(np.r_[start,direction]).all() or max(abs(direction))!=1:
        raise ValueError('Invalid stroke vectors')
    if not np.isfinite([distance,velocity,acceleration,jerk]).all() or min(distance,velocity,acceleration,jerk)<=0:
        raise ValueError('Invalid positive motion limits')
    tj=acceleration/jerk; ta=velocity/acceleration-tj
    tc=distance/velocity-(2*tj+ta)
    if min(tj,ta,tc)<.05: raise ValueError('Insufficient constant-velocity stroke')
    pieces=[(jerk,tj),(0.,ta),(-jerk,tj),(0.,tc),(-jerk,tj),(0.,ta),(jerk,tj)]
    x=v=a=t=0.; knots=[knot(0,start)]
    for j,dt in pieces:
        x+=v*dt+a*dt**2/2+j*dt**3/6
        v+=a*dt+j*dt**2/2; a+=j*dt; t+=dt
        knots.append(knot(t,start+direction*x,direction*v,direction*a))
    if abs(x-distance)>1e-9 or abs(v)>1e-9 or abs(a)>1e-9: raise ValueError('Integration failure')
    if abs(t/.01-round(t/.01))>1e-7: raise ValueError('Duration must lie on acquisition grid')
    knots[-1]=knot(round(t,2),start+distance*direction)
    return knots, [2*tj+ta,2*tj+ta+tc]


def build_plan(description, initial, left, envelope, name):
    cfg=yaml.safe_load((ROOT/'config/w3/motions/right_targeted_free_v1.yaml').read_text())
    initial=np.asarray(initial,float); bound=np.asarray(envelope['position_bounds_deg'])
    profile=dict(id='feedback_v2p4_a3_j15',velocity_deg_s=2.4,acceleration_deg_s2=3.,jerk_deg_s3=15.)
    cfg.update(name=name, profiles=[profile],home_deg=initial.tolist(),anchors_deg=[initial.tolist()],
        working_bounds_deg=np.c_[bound[:,0]+.6,bound[:,1]-.6].tolist(), continuous_cycles=2,
        collection_selection=dict(single_joints=list(range(7)),anchor_indices=[0],coupled_patterns=0,profiles=[profile['id']]))
    model=ArmModel(description,'right',cfg['tool_frame']); steps=[]; current=initial.copy()
    def add(knots,kind,family,active,condition=None,dwell=None):
        nonlocal current
        duration=knots[-1]['time_seconds']; times=np.arange(round(duration/.01)+1)*.01
        q,_,_=sample(knots,times); low,high,peak=bounds(knots)
        if max(abs(q[0]-current))>1e-8: raise ValueError('Discontinuous move')
        if np.any(low<bound[:,0]+.6) or np.any(high>bound[:,1]-.6): raise ValueError('Outside training margin: '+kind)
        annotation=dict(start_seconds=0.,end_seconds=duration,state=kind)
        if condition is not None: annotation.update(condition)
        step=dict(name=f'{len(steps)+1:04d}_{kind}', family=family,kind=kind,phase=family,speed=profile['id'],
            profile=copy.deepcopy(profile),active_joints=list(active),pose_index=0,approach_direction=None,
            dwell_seconds=dwell,duration_seconds=duration,hold_seconds=0.,times_seconds=times.tolist(),
            positions_deg=q.tolist(),knots=knots,
            peak_bounds={key:peak[i].tolist() for i,key in enumerate(('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3'))},
            max_condition=float(max(conditions(model,q))),annotations=[annotation])
        if condition is not None: step['feedback_condition']=condition
        steps.append(step); current=q[-1].copy()
    def hold(duration,kind,family='transition',condition=None):
        add([knot(0,current),knot(duration,current)],kind,family,[],condition,duration)
    hold(15.,'initial_settle')
    for joint in range(7):
        amplitude=np.zeros(7); amplitude[joint]=.6
        add(oscillation(current,amplitude,profile,cfg,np.zeros(7)),f'probe_j{joint+1}','speed_probe',[joint])
    hold(10.,'hold_after_probes')
    # All axes: long low-speed plateau, with both directions at the same pose.
    # Priority axes: two further velocity levels, without increasing acceleration.
    trials=[(j,2.4,.6,1.5,7.5,'all_axes') for j in range(7)]
    trials += [(j,6.,v,3.,15.,'velocity_ladder') for v in (1.2,2.4) for j in (0,1,3)]
    for joint,distance,velocity,acceleration,jerk,group in trials:
        for sign,label in [(1.,'out'),(-1.,'return')]:
            direction=np.zeros(7); direction[joint]=sign*DIRECTIONS[joint]
            knots,plateau=plateau_stroke(current,direction,distance,velocity,acceleration,jerk)
            condition=dict(group=group,joint=joint+1,direction=label, signed_direction=float(direction[joint]),
                distance_deg=distance,velocity_deg_s=velocity,acceleration_deg_s2=acceleration,
                jerk_deg_s3=jerk,plateau_seconds=plateau)
            key=f'j{joint+1}_v{velocity:g}_{label}'
            add(knots,'diagnostic_'+key,'hysteresis',[joint],condition)
            hold(3.,'hold_'+key,'hysteresis',condition)
    hold(15.,'final_settle')
    plan=dict(schema='w3_targeted_free_motion_v1',name=name,side='right',family='hysteresis',role='test',
        thermal=thermal_declaration('unclassified'),config=cfg,robot_description=description,
        model_sha256=hashlib.sha256(description.encode()).hexdigest(),joint_names=model.names,
        initial_deg=initial.tolist(),left_pose_deg=np.asarray(left,float).tolist(),steps=steps,
        feedback_protocol=copy.deepcopy(RECIPE))
    validate(plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); stage=args.output.resolve()
    if stage.exists(): raise ValueError('Use a new output directory')
    from w3_targeted_capture import live_state
    description,initial,left=live_state()
    meta=json.loads((ROOT/'models/w3/right/targeted_candidate_20261016/metadata.json').read_text())
    if hashlib.sha256(description.encode()).hexdigest()!='e3a8e37ee96fe8ff435af115e63aa788b4aed3e7b9c1dd2f65de9bcd9948e345':
        raise ValueError('Robot description changed')
    plan=build_plan(description,initial,left,meta['operating_envelope'],stage.name)
    encoded=yaml.dump(plan,Dumper=yaml.CSafeDumper,sort_keys=False)
    validate(yaml.load(encoded,Loader=yaml.CSafeLoader))
    spec=dict(purpose='position/velocity feedback, derivative uncertainty, native and CAN tracking; NEXT only',role='test',
        model_dir=str(ROOT/'models/w3/right/targeted_candidate_20261016'),model_hash=meta['artifact_sha256']['model.pt'],
        endpoint_config=yaml.safe_load((ROOT/'config/w3/right/endpoint_wrench_targeted_20261016.yaml').read_text()),
        plan_sha256=hashlib.sha256(encoded.encode()).hexdigest(),initial_command_deg=np.asarray(initial,float).tolist(),
        left_pose_deg=plan['left_pose_deg'],training_envelope=meta['operating_envelope'],
        command_seconds=sum(s['duration_seconds'] for s in plan['steps']),steps=len(plan['steps']),
        feedback_protocol=RECIPE,automatic_retry=False,
        ground_truth=dict(source='operator requests unloaded no-contact diagnostic',assumed_external_wrench=[0.]*6,independently_measured=False))
    stage.mkdir(parents=True);(stage/'motion.yaml').write_text(encoded);(stage/'robot.urdf').write_text(description)
    (stage/'test_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ('command_seconds','steps','initial_command_deg','feedback_protocol')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
