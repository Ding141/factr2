"""Offline bounded continuous high-speed paths; no ROS or actuation APIs.

The formal segment is one native quintic-Hermite trajectory. Only its first and
last states stop. Interior knots retain analytic velocity and acceleration.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from w3_targeted_protocol import (ArmModel, bounds, coefficients, conditions,
    knot, move, oscillation, retime, sample, thermal_declaration, validate)

ROOT=Path(__file__).resolve().parents[1]
PROFILES=[dict(id='continuous_v06',velocity_deg_s=6.,acceleration_deg_s2=15.,jerk_deg_s3=60.),
          dict(id='continuous_v12',velocity_deg_s=12.,acceleration_deg_s2=30.,jerk_deg_s3=120.)]
CENTER=np.array([1.,55.,-4.,-58.,1.,-2.,0.])
AMPLITUDE=np.array([6.,6.,5.,5.,4.,3.,2.])


def window(t,duration,ramp_fraction):
    """C3 endpoint ramp and a constant interior; derivatives in real seconds."""
    ramp=duration*ramp_fraction
    u=np.minimum(t,duration-t)/ramp
    u=np.clip(u,0.,1.)
    p=np.array([0.,0.,0.,0.,35.,-84.,70.,-20.])
    w=np.polynomial.polynomial.polyval(u,p)
    dw=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(p))/ramp
    ddw=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(p,2))/ramp**2
    plateau=(t>=ramp)&(t<=duration-ramp)
    w[plateau]=1.;dw[plateau]=0.;ddw[plateau]=0.
    dw[t>duration/2]*=-1
    return w,dw,ddw


def continuous(center,amplitude,profile,seed,nominal_seconds=90.,knot_seconds=.25):
    if nominal_seconds<30 or not .1<=knot_seconds<=.5:raise ValueError('continuous duration/knot spacing')
    center,amplitude=np.asarray(center,float),np.asarray(amplitude,float)
    if center.shape!=(7,) or amplitude.shape!=(7,) or np.any(amplitude<=0):raise ValueError('seven positive amplitudes')
    rng=np.random.default_rng(seed);phase=rng.uniform(-np.pi,np.pi,7)
    factors=np.array([.91,1.,.86,.97,.88,.93,.82])*rng.uniform(.95,1.05,7)
    omega=np.minimum.reduce([.82*profile['velocity_deg_s']/amplitude,
        np.sqrt(.55*profile['acceleration_deg_s2']/amplitude),
        np.cbrt(.5*profile['jerk_deg_s3']/amplitude)])*factors
    cycles=omega*nominal_seconds/(2*np.pi)
    fraction=8./nominal_seconds
    def make(duration):
        n=math.ceil(duration/knot_seconds);t=np.linspace(0.,duration,n+1)
        w,dw,ddw=window(t,duration,fraction);freq=2*np.pi*cycles/duration
        s=np.sin(t[:,None]*freq+phase);c=np.cos(t[:,None]*freq+phase)
        q=center+amplitude*w[:,None]*s
        v=amplitude*(dw[:,None]*s+w[:,None]*freq*c)
        a=amplitude*(ddw[:,None]*s+2*dw[:,None]*freq*c-w[:,None]*freq**2*s)
        q[[0,-1]]=center;v[[0,-1]]=0.;a[[0,-1]]=0.
        return [knot(ti,qi,vi,ai) for ti,qi,vi,ai in zip(t,q,v,a)]
    knots=retime(make,profile,.01,nominal_seconds)
    duration=knots[-1]['time_seconds']
    return knots,dict(generator='analytic_multisine_with_C3_endpoint_window',seed=seed,
        cycles_per_joint=cycles.tolist(),phase_rad=phase.tolist(),
        nominal_seconds=nominal_seconds,actual_seconds=duration,
        entry_ramp_seconds=duration*fraction,exit_ramp_seconds=duration*fraction,
        plateau_seconds=duration*(1-2*fraction),one_native_trajectory=True,
        interior_arrival_waits=0,interior_intentional_holds=0)


def smoothness(knots):
    times=np.arange(round(knots[-1]['time_seconds']/.01)+1)*.01
    q,v,a=sample(knots,times);duration=times[-1]
    active=(times>=.12*duration)&(times<=.88*duration)
    speed=np.linalg.norm(v,axis=1)
    jump=[]
    def jerk(c,u,dt):return sum(c[k]*math.factorial(k)/math.factorial(k-3)*u**(k-3)/dt**3 for k in range(3,6))
    for before,middle,after in zip(knots,knots[1:],knots[2:]):
        left=coefficients(before,middle);right=coefficients(middle,after)
        jump.append(abs(jerk(left,1.,middle['time_seconds']-before['time_seconds'])-
                        jerk(right,0.,after['time_seconds']-middle['time_seconds'])))
    return dict(interior_min_vector_speed_deg_s=float(speed[active].min()),
        interior_stopped_samples=int((speed[active]<.05).sum()),
        interior_knot_stops=int(sum(np.linalg.norm(k['velocity_deg_s'])<.05 for k in knots[1:-1]
                               if .12*duration<=k['time_seconds']<=.88*duration)),
        maximum_knot_jerk_jump_deg_s3=float(np.max(jump)) if jump else 0.,
        sampled_velocity_peak_deg_s=abs(v).max(0).tolist(),sampled_acceleration_peak_deg_s2=abs(a).max(0).tolist())


def build(description,initial,left,role,seed,seconds=90.,*,center=None,amplitude=None,
          profiles=None,excitation=None,always_pilot=False):
    if role not in ('train','val','test'):raise ValueError('explicit recording role')
    center=np.asarray(CENTER if center is None else center,float)
    amplitude=np.asarray(AMPLITUDE if amplitude is None else amplitude,float)
    profiles=copy.deepcopy(PROFILES if profiles is None else profiles)
    excitation=continuous if excitation is None else excitation
    cfg=yaml.safe_load((ROOT/'config/w3/motions/right_targeted_free_v1.yaml').read_text())
    cfg.update(name='right_continuous_highspeed_20261025',home_deg=center.tolist(),
        anchors_deg=[center.tolist()],working_bounds_deg=np.c_[center-amplitude-1.,center+amplitude+1.].tolist(),
        profiles=copy.deepcopy(profiles),trajectory_generator='continuous_multisine_v1',
        continuous_amplitude_cap_deg=amplitude.tolist(),continuous_cycles=2)
    model=ArmModel(description,'right',cfg['tool_frame']);steps=[];current=np.asarray(initial,float)
    def add(knots,name,kind,profile,family='transition',active=None,dwell=None,definition=None):
        nonlocal current
        duration=knots[-1]['time_seconds'];t=np.arange(round(duration/.01)+1)*.01
        q=sample(knots,t)[0];low,high,peak=bounds(knots)
        if np.max(abs(q[0]-current))>1e-7:raise ValueError('discontinuous initial state')
        c=max(conditions(model,q))
        if kind!='initial_approach' and c>cfg['max_condition']:raise ValueError('ill-conditioned formal path')
        step=dict(name=name,family=family,kind=kind,phase=family,speed=profile['id'],profile=copy.deepcopy(profile),
            active_joints=active or [],pose_index=0,approach_direction=None,dwell_seconds=dwell,
            duration_seconds=float(duration),hold_seconds=0.,times_seconds=t.tolist(),positions_deg=q.tolist(),
            knots=knots,peak_bounds={key:peak[i].tolist() for i,key in enumerate(
                ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3'))},max_condition=float(c),
            continuous_internal_dwell=False if family=='continuous' else None,
            annotations=[dict(start_seconds=0.,end_seconds=float(duration),state='stationary_hold' if dwell else family)])
        if definition:
            ramp=definition['entry_ramp_seconds'];step['continuous_definition']=definition
            step['annotations']=[dict(start_seconds=0.,end_seconds=ramp,state='entry_ramp'),
                dict(start_seconds=ramp,end_seconds=duration-ramp,state='continuous_excitation'),
                dict(start_seconds=duration-ramp,end_seconds=duration,state='exit_ramp')]
            step['smoothness']=smoothness(knots)
            if step['smoothness']['interior_stopped_samples']:raise ValueError('planned interior stop')
        steps.append(step);current=q[-1].copy()
    approach=dict(profiles[0],id='approach_v02p4',velocity_deg_s=2.4,acceleration_deg_s2=3.,jerk_deg_s3=15.)
    # Initial movement remains separate from formal training and speed promotion.
    # This is a distinct slow profile, with no requirement to collect new low-speed data.
    cfg['profiles'].insert(0,approach)
    if always_pilot or np.max(abs(current-center))>.01:
        target=current.copy();target[0]+=.5
        add([knot(0,current),knot(5,target)],'startup_pilot_out','initial_approach',approach)
        add([knot(0,current),knot(5,initial)],'startup_pilot_return','initial_approach',approach)
        add(move(current,center,approach,.01),'approach_working_pose','initial_approach',approach)
    add([knot(0,current),knot(5,current)],'baseline_before','settle',profiles[0],dwell=5.)
    for pi,profile in enumerate(profiles):
        for j in range(7):
            amp=np.zeros(7);amp[j]=min(3.,profile['velocity_deg_s']**2/profile['acceleration_deg_s2'])
            ks=oscillation(center,amp,profile,cfg,np.zeros(7))
            add(ks,f'{profile["id"]}_probe_j{j+1}',f'probe_j{j+1}',profile,family='speed_probe',active=[j])
        if profile.get('probe_only'):continue
        ks,definition=excitation(center,amplitude,profile,seed+pi*1009,seconds)
        add(ks,f'{profile["id"]}_{role}_continuous','continuous_multisine',profile,
            family='continuous',active=list(range(7)),definition=definition)
    add([knot(0,current),knot(5,current)],'baseline_after','settle',profiles[-1],dwell=5.)
    plan=dict(schema='w3_targeted_free_motion_v1',name=cfg['name']+'_'+role,side='right',family='continuous',role=role,
        config=cfg,robot_description=description,model_sha256=model.sha256,joint_names=model.names,
        initial_deg=np.asarray(initial,float).tolist(),left_pose_deg=np.asarray(left,float).tolist(),
        thermal=thermal_declaration('unclassified'),steps=steps)
    validate(plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--check-collision',action='store_true')
    parser.add_argument('--seconds',type=float,default=90.);args=parser.parse_args()
    from w3_collision_check import check
    report=json.loads((args.preflight/'summary.json').read_text());description=(args.preflight/'robot.urdf').read_text()
    initial=[report['last_command_deg'][f'right_joint_{j}'] for j in range(7)]
    left=[report['last_actual_deg'][f'left_joint_{j}'] for j in range(7)]
    args.output.mkdir(parents=True,exist_ok=False);summary=[]
    for index,role in enumerate(('train','val','test')):
        folder=args.output/role;folder.mkdir();start=initial if role=='train' else CENTER
        plan=build(description,start,left,role,202610250+index,args.seconds)
        path=folder/'motion.yaml';path.write_text(yaml.dump(plan,Dumper=yaml.CSafeDumper,sort_keys=False))
        collision=check(path,ROOT.parent/'dual_arm_robot',left,folder/'motion.collision.json') if args.check_collision else None
        item=dict(role=role,steps=len(plan['steps']),seconds=sum(s['duration_seconds'] for s in plan['steps']),
            plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),collision_accepted=collision['accepted'] if collision else None,
            formal=[dict(name=s['name'],definition=s['continuous_definition'],smoothness=s['smoothness'])
                    for s in plan['steps'] if s['family']=='continuous'])
        summary.append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
    (args.output/'plans.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
