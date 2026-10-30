#!/usr/bin/env python3
"""Freeze separate continuous/hysteresis/warmup experiments; never sends motion.

All dense samples are collision audit data only. One native JointTrajectory is
sent per block. Interior knots retain their velocity and acceleration.
"""
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from w3_coverage_plan import ArmModel
from w3_baseline_protocol import conditions

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'w3_targeted_free_motion_v1'
FAMILIES = ('continuous', 'hysteresis', 'warmup')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def vector(x, length=7):
    a = np.asarray(x, float)
    if a.shape != (length,) or not np.isfinite(a).all(): raise ValueError('Invalid finite vector')
    return a


def coefficients(a, b):
    """Same normalized quintic Hermite polynomial as TrajectorySpline::make."""
    t = b['time_seconds'] - a['time_seconds']
    if not np.isfinite(t) or t < .049: raise ValueError('Native knots too close')
    q0, q1 = vector(a['position_deg']), vector(b['position_deg'])
    x, y = vector(a['velocity_deg_s'])*t, vector(b['velocity_deg_s'])*t
    z, w = vector(a['acceleration_deg_s2'])*t*t, vector(b['acceleration_deg_s2'])*t*t
    d = q1-q0
    return np.array([q0, x, z/2, 10*d-6*x-4*y-1.5*z+.5*w,
                     -15*d+8*x+7*y+1.5*z-w, 6*d-3*x-3*y-.5*z+.5*w])


def bernstein(c, derivative, duration):
    n = 5-derivative
    p = np.array([c[i+derivative]*math.factorial(i+derivative)/math.factorial(i)/duration**derivative
                  for i in range(n+1)])
    return np.array([sum(p[i]*math.comb(k,i)/math.comb(n,i) for i in range(k+1)) for k in range(n+1)])


def sample(knots, times):
    times = np.asarray(times, float)
    q, v, acc = [np.empty((len(times), 7)) for _ in range(3)]
    indices=range(len(knots)-1)
    if len(times)==1:
        i=min(max(int(np.searchsorted([k['time_seconds'] for k in knots],times[0],side='right'))-1,0),len(knots)-2)
        indices=[i]
    for i in indices:
        a,b=knots[i:i+2]
        mask = (times >= a['time_seconds']-1e-9) & ((times < b['time_seconds']-1e-9) if i+2 < len(knots) else (times <= b['time_seconds']+1e-9))
        c = coefficients(a, b); t = b['time_seconds']-a['time_seconds']
        u = np.clip((times[mask]-a['time_seconds'])/t, 0, 1)
        for order, dest in enumerate((q,v,acc)):
            p = np.array([c[k]*math.factorial(k)/math.factorial(k-order)/t**order for k in range(order,6)])
            dest[mask] = sum(u[:,None]**k*p[k] for k in range(len(p)))
    return q, v, acc


def bounds(knots):
    upper = np.zeros((3,7)); low = np.full(7, np.inf); high = -low
    for a, b in zip(knots, knots[1:]):
        c = coefficients(a,b); duration = b['time_seconds']-a['time_seconds']
        qb = bernstein(c,0,duration); low = np.minimum(low,qb.min(axis=0)); high = np.maximum(high,qb.max(axis=0))
        for d in (1,2,3): upper[d-1] = np.maximum(upper[d-1],abs(bernstein(c,d,duration)).max(axis=0))
    return low, high, upper


def knot(t, q, v=None, a=None):
    return dict(time_seconds=float(t), position_deg=vector(q).tolist(),
                velocity_deg_s=(np.zeros(7) if v is None else vector(v)).tolist(),
                acceleration_deg_s2=(np.zeros(7) if a is None else vector(a)).tolist())


def retime(make, profile, dt, initial_duration=4.):
    duration = initial_duration
    limits = np.array([profile[k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])
    for _ in range(12):
        duration = math.ceil(duration/dt)*dt
        knots = make(duration)
        _, _, maximum = bounds(knots)
        ratio = max((maximum[d]/limits[d]).max()**(1/(d+1)) for d in range(3))
        if ratio <= .98: return knots
        duration *= max(1.03, ratio*1.03)
    raise ValueError('Cannot retime within v/a/jerk limits')


def move(start, end, profile, dt):
    # A fixed four-second move hides velocity/acceleration differences for small
    # friction probes. Set time from this profile, with conservative utilization
    # of the same whole-polynomial bounds enforced by the native controller.
    peak=bounds([knot(0,start),knot(1,end)])[2].max(axis=1)
    limits=np.array([profile[k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])
    duration=max(.25,max((peak[d]/(limits[d]*fraction))**(1/(d+1))
                         for d,fraction in enumerate((.75,.5,.8))))
    return retime(lambda duration:[knot(0,start),knot(duration,end)], profile, dt,duration)


def oscillation(anchor, amplitude, profile, cfg, phase):
    cycles = cfg['continuous_cycles']; dt = cfg['sample_period_seconds']
    def make(duration):
        n = max(16, math.ceil(duration/cfg['knot_period_seconds']))
        # Uniform normalized knots; only the final duration must lie on 100 Hz.
        t = np.linspace(0,duration,n+1); s=t/duration
        sin, cos = np.sin(np.pi*s), np.cos(np.pi*s)
        w=sin**4; dw=4*np.pi*sin**3*cos
        ddw=12*np.pi**2*sin**2*cos**2-4*np.pi**2*sin**4
        omega=2*np.pi*cycles
        f=np.sin(omega*s[:,None]+phase); df=omega*np.cos(omega*s[:,None]+phase)
        q=anchor+amplitude*w[:,None]*f
        v=amplitude*(dw[:,None]*f+w[:,None]*df)/duration
        a=amplitude*(ddw[:,None]*f+2*dw[:,None]*df-w[:,None]*omega**2*f)/duration**2
        q[[0,-1]]=anchor;v[[0,-1]]=0;a[[0,-1]]=0
        return [knot(ti,qi,vi,ai) for ti,qi,vi,ai in zip(t,q,v,a)]
    return retime(make,profile,dt)


def validate_recipe(cfg):
    if cfg['version'] != 1 or cfg['side'] != 'right': raise ValueError('Only right-arm v1 is supported')
    if cfg['sample_period_seconds'] != .01 or not .1 <= cfg['knot_period_seconds'] <= .5: raise ValueError('Require 100Hz audit and sparse knots')
    for k, low, high in [('tracking_limit_deg',.5,3),('promotion_tracking_deg',.5,2.5),
                         ('arrival_tolerance_deg',.05,1.5),('left_stationary_tolerance_deg',.05,1),
                         ('hardware_margin_deg',3,15),('max_condition',1.1,20)]:
        if not isinstance(cfg[k],(float,int)) or not np.isfinite(cfg[k]) or not low <= cfg[k] <= high: raise ValueError(k)
    if cfg['promotion_tracking_deg'] > cfg['tracking_limit_deg']: raise ValueError('Promotion must be stricter than stop')
    if not 2 <= cfg['continuous_cycles'] <= 8 or not all(1 <= x <= 30 for x in cfg['dwell_seconds']): raise ValueError('Invalid cycles/dwell')
    previous=0.; seen=set()
    for p in cfg['profiles']:
        if p['id'] in seen: raise ValueError('Duplicate profile')
        seen.add(p['id'])
        for k, maximum in [('velocity_deg_s',20),('acceleration_deg_s2',90),('jerk_deg_s3',180)]:
            if not np.isfinite(p[k]) or not .1 <= p[k] <= maximum: raise ValueError('Conservative profile bound: '+k)
        if p['velocity_deg_s'] < previous: raise ValueError('Profiles must increase progressively')
        previous=p['velocity_deg_s']
    for x in cfg['anchors_deg']: vector(x)
    vector(cfg['home_deg']);vector(cfg['return_offset_deg']);vector(cfg['continuous_amplitude_cap_deg'])
    selection=cfg.get('collection_selection')
    if selection:
        for key,n in [('single_joints',7),('anchor_indices',len(cfg['anchors_deg']))]:
            values=selection[key]
            if len(set(values))!=len(values) or any(type(j)!=int or not 0<=j<n for j in values):raise ValueError('Invalid collection subset')
        if not selection['anchor_indices'] or not 0<=selection['coupled_patterns']<=2:raise ValueError('Invalid anchor/coupled selection')
        if not selection['single_joints'] and not selection['coupled_patterns']:raise ValueError('No excitation selected')
        if not set(selection['profiles'])<={p['id'] for p in cfg['profiles']}:raise ValueError('Unknown selected profile')


def thermal_declaration(condition, idle_minutes=0, warmup_session=None):
    if condition == 'cold_start':
        if not np.isfinite(idle_minutes) or idle_minutes < 30: raise ValueError('Cold start needs operator-declared idle >=30 minutes')
    elif condition == 'warm_start':
        if not warmup_session: raise ValueError('Warm start requires a completed warmup session')
        folder=Path(warmup_session).resolve(); run=json.loads((folder/'audit/motion_run.json').read_text())
        p,_,digest=load(folder/'audit/motion.yaml')
        if p['family']!='warmup' or run['status']!='completed' or run['motion_sha256']!=digest:
            raise ValueError('Warmup evidence mismatch')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(run['completed_utc'])).total_seconds()
        duration=sum(s['duration_seconds'] for s in p['steps'] if s['family']=='warmup')
        if age<0 or age>300 or duration<300 or len(run['events'])!=len(p['steps']):
            raise ValueError('Warm start needs >=5min completed warmup within the last 5min')
        warmup_session=str(folder)
    elif condition != 'unclassified': raise ValueError('Invalid thermal condition')
    return dict(initial_condition=condition, source='operator_declaration_and_motion_history',
                declared_idle_minutes=float(idle_minutes), warmup_session=warmup_session,
                temperature_measured=False, temperature_c=None,
                steady_temperature_verified=False,
                label_scope='initial condition only; temperature may change during capture')


def build(cfg, description, initial, left, role, family, thermal, max_velocity=12., warmup_minutes=5.):
    validate_recipe(cfg)
    if role not in cfg['seeds'] or family not in FAMILIES: raise ValueError('Invalid role/family')
    if not np.isfinite(max_velocity) or not .1<=max_velocity<=20:raise ValueError('Invalid maximum speed')
    if not np.isfinite(warmup_minutes) or not 5<=warmup_minutes<=30:raise ValueError('Warmup must be 5..30 minutes')
    if thermal['initial_condition']=='warm_start':
        warm,_,_=load(Path(thermal['warmup_session'])/'audit/motion.yaml')
        if warm['model_sha256']!=sha(description.encode()):raise ValueError('Warmup and collection robot models differ')
    profiles=[p for p in cfg['profiles'] if p['velocity_deg_s'] <= max_velocity]
    if not profiles: raise ValueError('No speed profile selected')
    if family=='warmup': profiles=profiles[:1]
    selection=cfg.get('collection_selection',dict(single_joints=list(range(7)),anchor_indices=list(range(len(cfg['anchors_deg']))),coupled_patterns=2,profiles=[p['id'] for p in profiles]))
    selected_profiles=set(selection['profiles'])
    if family!='warmup':
        if not selected_profiles or not selected_profiles<={p['id'] for p in profiles}:raise ValueError('Selected profile exceeds --max-velocity')
        last=max(i for i,p in enumerate(profiles) if p['id'] in selected_profiles)
        profiles=profiles[:last+1]  # retain earlier profile probes for progressive validation
    m=ArmModel(description,'right',cfg['tool_frame']); current=vector(initial); dt=cfg['sample_period_seconds']
    working=np.asarray(cfg['working_bounds_deg'],float); hardware=m.limits_deg
    if working.shape != (7,2) or np.any(working[:,0]>=working[:,1]) or np.any(working[:,0]<hardware[:,0]+cfg['hardware_margin_deg']) or np.any(working[:,1]>hardware[:,1]-cfg['hardware_margin_deg']): raise ValueError('Invalid working/hardware bounds')
    rng=np.random.default_rng(cfg['seeds'][role]); steps=[]
    def add(knots, kind, group, profile, active=None, pose=None, direction=None, dwell=None):
        nonlocal current
        duration=knots[-1]['time_seconds']; t=np.arange(round(duration/dt)+1)*dt
        q,_,_=sample(knots,t)
        low, high, peak=bounds(knots)
        if np.max(abs(q[0]-current))>1e-7: raise ValueError('Discontinuous block')
        limit=hardware if kind=='initial_approach' else working
        if np.any(low<limit[:,0]+(cfg['hardware_margin_deg'] if kind=='initial_approach' else 0)) or np.any(high>limit[:,1]-(cfg['hardware_margin_deg'] if kind=='initial_approach' else 0)): raise ValueError('Path exceeds envelope: '+kind)
        cond=conditions(m,q)
        if kind!='initial_approach' and max(cond)>cfg['max_condition']: raise ValueError('Ill-conditioned path: '+kind)
        steps.append(dict(name=f'{len(steps)+1:04d}_{profile["id"]}_{kind}', family=group,
            kind=kind, phase=group, speed=profile['id'], profile=copy.deepcopy(profile),
            active_joints=active or [], pose_index=pose, approach_direction=direction,
            dwell_seconds=dwell, duration_seconds=float(duration), hold_seconds=0.,
            times_seconds=t.tolist(),positions_deg=q.tolist(),knots=knots,
            peak_bounds=dict(velocity_deg_s=peak[0].tolist(),acceleration_deg_s2=peak[1].tolist(),jerk_deg_s3=peak[2].tolist()),
            max_condition=float(max(cond)), continuous_internal_dwell=False if group=='continuous' else None,
            annotations=[dict(start_seconds=0.,end_seconds=duration,state='stationary_hold' if dwell else group)]))
        current=q[-1].copy()
    def go(goal,kind,profile,group='transition',**kw):
        if np.max(abs(vector(goal)-current))>1e-8: add(move(current,goal,profile,dt),kind,group,profile,**kw)
    def hold(seconds,kind,profile,group,**kw):
        add([knot(0,current),knot(seconds,current)],kind,group,profile,dwell=seconds,**kw)
    home=vector(cfg['home_deg']); go(home,'initial_approach',profiles[0])
    hold(3.,'initial_settle',profiles[0],'transition')
    for profile in profiles:
        # Every velocity/acceleration profile is probed before its collection.
        for j in range(7):
            amp=np.zeros(7);amp[j]=min(3.,profile['velocity_deg_s']**2/profile['acceleration_deg_s2'])
            ks=oscillation(current,amp,profile,cfg,np.zeros(7))
            add(ks,f'probe_j{j+1}','speed_probe',profile,active=[j])
        if family!='warmup' and profile['id'] not in selected_profiles:continue
        if family in ('continuous','warmup'):
            repetitions=1 if family=='continuous' else math.ceil(warmup_minutes*60/4)
            for rep in range(repetitions):
                anchors=[(pi,cfg['anchors_deg'][pi]) for pi in selection['anchor_indices']] if family=='continuous' else [(0,cfg['home_deg'])]
                for pi, anchor in anchors:
                    go(anchor,f'anchor{pi}_approach',profile)
                    patterns=[[j] for j in selection['single_joints']]+[list(range(7))]*selection['coupled_patterns'] if family=='continuous' else [list(range(7))]
                    for pattern, active in enumerate(patterns):
                        cap=vector(cfg['continuous_amplitude_cap_deg']);amp=np.minimum(cap,profile['velocity_deg_s']**2/profile['acceleration_deg_s2'])
                        amp=np.minimum(amp,.7*np.minimum(current-working[:,0],working[:,1]-current))
                        amp[[j for j in range(7) if j not in active]]=0
                        phase=rng.uniform(-np.pi,np.pi,7)
                        ks=oscillation(current,amp,profile,cfg,phase)
                        add(ks,f'anchor{pi}_cycle{rep}_pattern{pattern}',family,profile,active=active,pose=pi)
                        if family=='continuous':
                            duration=steps[-1]['duration_seconds']
                            steps[-1]['annotations']=[dict(start_seconds=0.,end_seconds=.2*duration,state='entry_ramp'),
                                dict(start_seconds=.2*duration,end_seconds=.8*duration,state='continuous_excitation'),
                                dict(start_seconds=.8*duration,end_seconds=duration,state='exit_ramp')]
                if family=='warmup' and sum(s['duration_seconds'] for s in steps if s['family']=='warmup')>=warmup_minutes*60:
                    break
        else:
            for pi in selection['anchor_indices']:
                anchor=cfg['anchors_deg'][pi]
                go(anchor,f'anchor{pi}_approach',profile)
                for pattern, active in enumerate([[j] for j in selection['single_joints']]+([list(range(7))] if selection['coupled_patterns'] else [])):
                    offset=np.zeros(7);offset[active]=vector(cfg['return_offset_deg'])[active]
                    dwell=cfg['dwell_seconds'][(pi+pattern)%len(cfg['dwell_seconds'])]
                    for direction,sign in [('from_low',-1.),('from_high',1.)]:
                        go(vector(anchor)+sign*offset,f'anchor{pi}_condition_{pattern}_{direction}',profile,
                           group='hysteresis',active=active,pose=pi,direction=direction)
                        go(anchor,f'anchor{pi}_return_{pattern}_{direction}',profile,
                           group='hysteresis',active=active,pose=pi,direction=direction)
                        hold(dwell,f'static_anchor{pi}_pattern{pattern}_{direction}',profile,'hysteresis',
                             active=active,pose=pi,direction=direction)
    go(home,'return_home',profiles[-1]);hold(3.,'final_settle',profiles[-1],'transition')
    plan=dict(schema=SCHEMA,name=cfg['name']+'_'+family+'_'+role,side='right',family=family,role=role,
              config=copy.deepcopy(cfg),robot_description=description,model_sha256=sha(description.encode()),
              joint_names=m.names,initial_deg=vector(initial).tolist(),left_pose_deg=vector(left).tolist(),
              thermal=thermal,steps=steps)
    validate(plan)
    return plan


def validate(plan):
    if plan.get('schema')!=SCHEMA or plan['family'] not in FAMILIES: raise ValueError('Invalid targeted schema')
    cfg=plan['config'];validate_recipe(cfg)
    m=ArmModel(plan['robot_description'],'right',cfg['tool_frame'])
    if plan['model_sha256']!=m.sha256 or plan['joint_names']!=m.names: raise ValueError('Model mismatch')
    current=vector(plan['initial_deg']);dt=cfg['sample_period_seconds']; working=np.array(cfg['working_bounds_deg'])
    seen=set();last_profile=-1;probes=set()
    ids={p['id']:i for i,p in enumerate(cfg['profiles'])}
    if not 1<=len(plan['steps'])<=2000: raise ValueError('Invalid step count')
    for step in plan['steps']:
        if step['name'] in seen: raise ValueError('Duplicate event name')
        seen.add(step['name']);p=step['profile'];idx=ids[p['id']]
        if p!=cfg['profiles'][idx] or idx<last_profile: raise ValueError('Changed or regressed profile')
        last_profile=idx
        if step['family'] not in ('transition','speed_probe',plan['family']): raise ValueError('Mixed experiment families')
        active=step['active_joints']
        if any(type(j)!=int or not 0<=j<7 for j in active): raise ValueError('Invalid active joint')
        if step['family']=='speed_probe':probes.update((p['id'],j) for j in active)
        elif step['family']==plan['family'] and not all((p['id'],j) in probes for j in range(7)): raise ValueError('Missing speed probe')
        knots=step['knots'];t=np.asarray(step['times_seconds']); q=np.asarray(step['positions_deg'])
        if len(knots)<2 or len(knots)>2000 or step['duration_seconds']>600 or not np.isfinite(t).all(): raise ValueError('Invalid duration/knots')
        if abs(t[0])>1e-9 or abs(t[-1]-step['duration_seconds'])>1e-8 or np.max(abs(np.diff(t)-dt))>1e-8: raise ValueError('Invalid audit grid')
        expected,_,_=sample(knots,t)
        if q.shape!=expected.shape or not np.allclose(q,expected,atol=1e-8,rtol=0) or np.max(abs(q[0]-current))>1e-7: raise ValueError('Native/audit path mismatch')
        if any(np.max(abs(vector(k[field])))>1e-8 for k in (knots[0],knots[-1]) for field in ('velocity_deg_s','acceleration_deg_s2')): raise ValueError('Block boundary not at rest')
        low,high,peak=bounds(knots)
        lim=np.array([p[k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])[:,None]
        if np.any(peak>lim+1e-7):raise ValueError('Native polynomial exceeds profile limits')
        for i,k in enumerate(('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')):
            if not np.allclose(step['peak_bounds'][k],peak[i],atol=1e-7,rtol=0):raise ValueError('Incorrect declared derivative peaks')
        if step['speed']!=p['id'] or step['phase']!=step['family'] or abs(knots[-1]['time_seconds']-t[-1])>1e-8:raise ValueError('Incorrect labels/time')
        envelope=m.limits_deg if step['kind']=='initial_approach' else working
        margin=cfg['hardware_margin_deg'] if step['kind']=='initial_approach' else 0.
        if np.any(low<envelope[:,0]+margin) or np.any(high>envelope[:,1]-margin): raise ValueError('Path envelope')
        if step['kind']!='initial_approach' and max(conditions(m,q))>cfg['max_condition']:raise ValueError('Ill-conditioned path')
        if step['family'] in ('continuous','speed_probe','warmup'):
            inactive=[j for j in range(7) if j not in active]
            if inactive and np.max(np.ptp(q[:,inactive],axis=0))>1e-7:raise ValueError('Moves undeclared joints')
        if step['family']=='continuous':
            if step.get('dwell_seconds') or len(knots)<=2:raise ValueError('Continuous block has intentional dwell')
            if not any(np.max(abs(np.asarray(k['velocity_deg_s'])))>.01 for k in knots[1:-1]):raise ValueError('All interior knots stop')
        current=q[-1]
    return m


def load(path):
    raw=Path(path).read_bytes();plan=yaml.load(raw,Loader=getattr(yaml,'CSafeLoader',yaml.SafeLoader));validate(plan)
    return plan,raw,sha(raw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/w3/motions/right_targeted_free_v1.yaml')
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--live',action='store_true');source.add_argument('--urdf',type=Path)
    p.add_argument('--initial-deg',nargs=7,type=float);p.add_argument('--left-pose-deg',nargs=7,type=float)
    p.add_argument('--role',required=True,choices=['train','val','test']);p.add_argument('--family',required=True,choices=FAMILIES)
    p.add_argument('--thermal',required=True,choices=['cold_start','warm_start','unclassified']);p.add_argument('--idle-minutes',type=float,default=0)
    p.add_argument('--warmup-session',type=Path);p.add_argument('--warmup-minutes',type=float,default=5.)
    p.add_argument('--max-velocity',type=float,default=12.,help='Only includes profiles up to this limit; native/online probes still gate execution')
    p.add_argument('--collect-profiles',nargs='+',help='Collect only these profiles while retaining all preceding seven-joint probes')
    p.add_argument('--single-joints',nargs='*',type=int,help='Zero-based joints to collect, default all seven')
    p.add_argument('--anchor-indices',nargs='+',type=int,help='Subset of fixed shared anchors')
    p.add_argument('--coupled-patterns',type=int,choices=[0,1,2],default=2)
    p.add_argument('--check-collision',action='store_true',help='Run offline URDF mesh checks with this plan\'s frozen left pose')
    p.add_argument('--workspace',type=Path,default=Path('/home/venom/dual_arm_robot'),help='W3 workspace containing description meshes')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Never overwrite a frozen plan')
    cfg=yaml.safe_load(args.config.read_text())
    if args.collect_profiles or args.single_joints is not None or args.anchor_indices or args.coupled_patterns!=2:
        cfg['collection_selection']=dict(profiles=args.collect_profiles or [p['id'] for p in cfg['profiles'] if p['velocity_deg_s']<=args.max_velocity],
            single_joints=args.single_joints if args.single_joints is not None else list(range(7)),
            anchor_indices=args.anchor_indices or list(range(len(cfg['anchors_deg']))),coupled_patterns=args.coupled_patterns)
    thermal=thermal_declaration(args.thermal,args.idle_minutes,args.warmup_session)
    if args.live:
        from w3_targeted_capture import live_state
        description,initial,left=live_state()
    else:
        if args.initial_deg is None or args.left_pose_deg is None:p.error('Offline planning requires both arm poses')
        description=args.urdf.read_text();initial=args.initial_deg;left=args.left_pose_deg
    plan=build(cfg,description,initial,left,args.role,args.family,thermal,args.max_velocity,args.warmup_minutes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(yaml.dump(plan,Dumper=getattr(yaml,'CSafeDumper',yaml.SafeDumper),sort_keys=False))
    collision=None
    if args.check_collision:
        from w3_collision_check import check
        collision=args.output.with_suffix('.collision.json')
        check(args.output,args.workspace.resolve(),plan['left_pose_deg'],collision)
    print(json.dumps(dict(plan=str(args.output),motion_sent=False,blocks=len(plan['steps']),
          seconds=sum(s['duration_seconds'] for s in plan['steps']),family=args.family,thermal=thermal,
          collision_report=str(collision) if collision else None),ensure_ascii=False,indent=2))
    print('Required next: '+('C recording, then D --send' if collision else 'offline mesh check with the frozen left pose, then C recording and D --send'))


if __name__=='__main__':main()
