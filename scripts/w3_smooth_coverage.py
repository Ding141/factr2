"""Continuous C2 coverage paths; no ROS or hardware operations."""
import numpy as np
from scipy.interpolate import CubicSpline
from w3_coverage_plan import ArmModel, SCHEMA, validate_config, validate_plan, vector


def phase_progress(u):
    # Integral of quintic velocity ramps, then constant phase velocity.
    # Position, velocity and acceleration meet continuously at ramp boundaries.
    ramp=.1
    def integral(z): return 2.5*z**4-3*z**5+z**6
    result=np.empty_like(u);first=u<ramp;last=u>1-ramp;middle=~(first|last)
    result[first]=ramp*integral(u[first]/ramp)/(1-ramp)
    result[middle]=(u[middle]-ramp/2)/(1-ramp)
    result[last]=1-ramp*integral((1-u[last])/ramp)/(1-ramp)
    return result


def retime(curve, speed, dt, jerk_limit=120., uniform_phase=False, pieces=1):
    velocity = np.asarray(speed['velocity_deg_s'])
    acceleration = np.asarray(speed['acceleration_deg_s2'])
    duration = 2.
    for _ in range(30):
        count = int(np.ceil(duration / dt / pieces))*pieces; duration = count * dt
        t = np.arange(count + 1) * dt; u = t / duration
        s = phase_progress(u) if uniform_phase else u**3 * (10 + u * (-15 + 6*u))
        q = curve(s)
        # Match actual linear position interpolation, including stationary boundaries.
        v = np.diff(q, axis=0) / dt
        a = np.diff(np.vstack([np.zeros(7), v, np.zeros(7)]), axis=0) / dt
        j = np.diff(np.vstack([np.zeros(7), a, np.zeros(7)]), axis=0) / dt
        # The sender supplies derivatives of a C2 cubic spline. Check that
        # interpolation as well as discrete differences, which can miss local jerk peaks.
        interpolant=CubicSpline(t,q,axis=0,bc_type=((1,np.zeros(7)),(1,np.zeros(7))))
        continuous_jerk=np.max(np.abs(interpolant(t,3)))
        ratio = max(np.max(np.abs(v)/velocity), np.sqrt(np.max(np.abs(a)/acceleration)),
                    np.cbrt(max(np.max(np.abs(j)),continuous_jerk)/jerk_limit))
        if ratio <= 1.000001:
            return t, q
        duration *= max(1.05, ratio*1.03)
    raise ValueError('Unable to retime smooth trajectory')


def build_smooth_plan(config, description, initial_deg):
    model = ArmModel(description, config['side'], config['tool_frame'])
    home, bounds = validate_config(config, model)
    initial = vector(initial_deg, 7, 'initial')
    steps=[]; current=initial.copy(); dt=config['sample_period_seconds']
    def append(curve, phase, kind, speed, repetition, joint=None):
        nonlocal current
        t,q=retime(curve,config['speeds'][speed],dt)
        if not np.isfinite(q).all(): raise ValueError('Nonfinite smooth path')
        steps.append(dict(name=f'{len(steps)+1:03d}_{phase}_{kind}_{speed}_{repetition}',
            phase=phase,kind=kind,speed=speed,repetition=repetition,joint=joint,
            duration_seconds=float(t[-1]),hold_seconds=0.,times_seconds=t.tolist(),positions_deg=q.tolist()))
        current=q[-1].copy()
    def transition(goal,kind,speed='slow',repetition=0):
        start=current.copy(); goal=np.asarray(goal)
        if np.max(np.abs(goal-start)) < 1e-8: return
        append(lambda s: start[None,:]+s[:,None]*(goal-start),'transition',kind,speed,repetition)
    transition(home,'approach')
    if config.get('continuous_repetitions',False):
        # Repeat the same joint without parking at the center between repetitions.
        for speed in ('slow','fast'):
            for joint in range(7):
                center_joint=np.mean(bounds[joint]);amplitude=np.ptp(bounds[joint])/2
                if not bounds[joint,0]<home[joint]<bounds[joint,1]:raise ValueError('Interior home required')
                phase=np.arcsin((home[joint]-center_joint)/amplitude)
                repetitions=config['repeats']
                def curve(s,joint=joint,amplitude=amplitude,center_joint=center_joint,phase=phase):
                    q=np.repeat(home[None,:],len(s),axis=0)
                    q[:,joint]=center_joint+amplitude*np.sin(phase+2*np.pi*repetitions*s)
                    return q
                t,q=retime(curve,config['speeds'][speed],dt,uniform_phase=True,pieces=repetitions)
                size=(len(t)-1)//repetitions
                for repeat in range(1,repetitions+1):
                    lo=(repeat-1)*size;hi=repeat*size
                    part_t=t[lo:hi+1]-t[lo];part_q=q[lo:hi+1]
                    steps.append(dict(name=f'{len(steps)+1:03d}_single_joint_sine_cycle_{speed}_{repeat}_j{joint+1}',
                        phase='single_joint',kind='sine_cycle',speed=speed,repetition=repeat,joint=joint,
                        duration_seconds=float(part_t[-1]),hold_seconds=0.,
                        times_seconds=part_t.tolist(),positions_deg=part_q.tolist()))
                current=q[-1].copy()
    else:
        for speed in ('slow','fast'):
            for repeat in range(1,config['repeats']+1):
                for joint in range(7):
                    amplitude=min(home[joint]-bounds[joint,0],bounds[joint,1]-home[joint])
                    if amplitude <= 0: raise ValueError('Sine sweep requires interior home')
                    def curve(s,joint=joint,amplitude=amplitude):
                        q=np.repeat(home[None,:],len(s),axis=0)
                        q[:,joint]+=amplitude*np.sin(2*np.pi*s)
                        return q
                    append(curve,'single_joint','sine_cycle',speed,repeat,joint)
    center=model.fk(np.deg2rad(home))[:3,3]
    extent=config['cartesian']['line_half_extent_m']; radius=config['cartesian']['circle_radius_m']
    u=np.linspace(0,1,801); theta=2*np.pi*u
    paths=[]
    for axis in range(3):
        xyz=np.repeat(center[None,:],len(u),axis=0); xyz[:,axis]+=extent*np.sin(theta)
        paths.append((f'line_{"xyz"[axis]}',xyz))
    xyz=np.repeat(center[None,:],len(u),axis=0)
    xyz[:,0]+=radius*(np.cos(theta)-1); xyz[:,1]+=radius*np.sin(theta)
    paths.append(('circle_xy',xyz))
    solutions=[]
    for kind,xyz in paths:
        seed=np.deg2rad(home); q=[home.copy()]
        for point in xyz[1:]:
            seed=model.ik_position(point,seed,np.deg2rad(bounds));q.append(np.rad2deg(seed))
        solutions.append((kind,CubicSpline(u,np.asarray(q),axis=0)))
    for speed in ('slow','fast'):
        for repeat in range(1,config['repeats']+1):
            for kind,curve in solutions:
                append(curve,'cartesian',kind,speed,repeat)
                transition(home,'reset_home',speed,repeat)
    transition(home if config.get('finish_at_home',False) else initial,'finish_home' if config.get('finish_at_home',False) else 'return_initial')
    plan=dict(schema=SCHEMA,name=config['name'],side=config['side'],config=config,
        robot_description=description,model_sha256=model.sha256,joint_names=model.names,
        initial_deg=initial.tolist(),steps=steps)
    validate_plan(plan)
    return plan
