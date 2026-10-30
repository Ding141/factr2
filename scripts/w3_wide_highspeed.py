"""Offline broad-pose continuous excitation. No ROS or robot command API."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from w3_continuous_highspeed import build, window
from w3_targeted_protocol import knot, retime

ROOT=Path(__file__).resolve().parents[1]
CENTER=np.array([1.,50.,-4.,-62.,1.,-2.,0.])
SWEEP=np.array([20.,10.,14.,15.,15.,10.,12.])
CARRIER=np.array([6.,4.,5.,5.,4.,4.,3.])
AMPLITUDE=SWEEP+CARRIER
PROFILES=[dict(id='continuous_v06_probe',velocity_deg_s=6.,acceleration_deg_s2=15.,jerk_deg_s3=60.,probe_only=True),
          dict(id='continuous_v12_wide',velocity_deg_s=12.,acceleration_deg_s2=30.,jerk_deg_s3=120.),
          dict(id='continuous_v18_wide',velocity_deg_s=18.,acceleration_deg_s2=45.,jerk_deg_s3=150.)]


def wide(center,amplitude,profile,seed,nominal_seconds=160.,knot_seconds=.25):
    if not 90<=nominal_seconds<=300:raise ValueError('90..300 seconds for broad pose coverage')
    if not np.allclose(amplitude,AMPLITUDE):raise ValueError('Frozen sweep and carrier envelope')
    rng=np.random.default_rng(seed)
    phase=rng.uniform(-np.pi,np.pi,(2,7))
    slow_cycles=np.array([1.11,1.37,1.61,1.23,1.83,2.03,1.49])*rng.uniform(.95,1.05,7)
    slow_omega=2*np.pi*slow_cycles/nominal_seconds
    # A slow, large pose sweep plus a small, fast carrier excites both posture
    # and dynamics. Analytic derivatives include the envelope product terms.
    budget=np.maximum(.25,profile['velocity_deg_s']*.83-SWEEP*slow_omega)
    omega=np.minimum.reduce([budget/CARRIER,
        np.sqrt(.55*profile['acceleration_deg_s2']/CARRIER),
        np.cbrt(.45*profile['jerk_deg_s3']/CARRIER)])*rng.uniform(.90,1.,7)
    cycles=np.stack([slow_cycles,omega*nominal_seconds/(2*np.pi)])
    amplitudes=np.stack([SWEEP,CARRIER])
    fraction=10./nominal_seconds
    def make(duration):
        t=np.linspace(0.,duration,int(np.ceil(duration/knot_seconds))+1)
        w,dw,ddw=window(t,duration,fraction)
        freq=2*np.pi*cycles/duration
        theta=t[:,None,None]*freq[None]+phase[None]
        s,c=np.sin(theta),np.cos(theta)
        x=(amplitudes*s).sum(1)
        dx=(amplitudes*freq*c).sum(1)
        ddx=(-amplitudes*freq**2*s).sum(1)
        q=center+w[:,None]*x
        v=dw[:,None]*x+w[:,None]*dx
        a=ddw[:,None]*x+2*dw[:,None]*dx+w[:,None]*ddx
        q[[0,-1]]=center;v[[0,-1]]=0.;a[[0,-1]]=0.
        return [knot(ti,qi,vi,ai) for ti,qi,vi,ai in zip(t,q,v,a)]
    ks=retime(make,profile,.01,nominal_seconds)
    duration=ks[-1]['time_seconds']
    return ks,dict(generator='broad_pose_sweep_plus_fast_carrier_C3',seed=seed,
        pose_sweep_amplitude_deg=SWEEP.tolist(),carrier_amplitude_deg=CARRIER.tolist(),
        cycles_per_component=cycles.tolist(),phase_rad=phase.tolist(),nominal_seconds=nominal_seconds,
        actual_seconds=duration,entry_ramp_seconds=duration*fraction,
        exit_ramp_seconds=duration*fraction,plateau_seconds=duration*(1-2*fraction),
        one_native_trajectory=True,interior_arrival_waits=0,interior_intentional_holds=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-collision',action='store_true')
    a=p.parse_args()
    from w3_collision_check import check
    report=json.loads((a.preflight/'summary.json').read_text())
    description=(a.preflight/'robot.urdf').read_text()
    initial=[report['last_command_deg'][f'right_joint_{j}'] for j in range(7)]
    left=[report['last_actual_deg'][f'left_joint_{j}'] for j in range(7)]
    a.output.mkdir(parents=True,exist_ok=False);summary=[]
    for index,label in enumerate(('train','val','test','observation')):
        role='test' if label=='observation' else label
        folder=a.output/label;folder.mkdir()
        profiles=[dict(p,probe_only=True) if p['velocity_deg_s']==12 else p for p in PROFILES] if label=='observation' else PROFILES
        plan=build(description,initial if index==0 else CENTER,left,role,202610260+index,160.,
            center=CENTER,amplitude=AMPLITUDE,profiles=profiles,excitation=wide,always_pilot=True)
        plan['name']='right_wide_continuous_20261026_'+label
        plan['config']['name']='right_wide_continuous_20261026'
        path=folder/'motion.yaml'
        path.write_text(yaml.dump(plan,Dumper=yaml.CSafeDumper,sort_keys=False))
        collision=check(path,ROOT.parent/'dual_arm_robot',left,folder/'motion.collision.json') if a.check_collision else None
        q=np.vstack([s['positions_deg'] for s in plan['steps']])
        item=dict(role=role,label=label,steps=len(plan['steps']),seconds=sum(s['duration_seconds'] for s in plan['steps']),
            plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),collision_accepted=collision['accepted'] if collision else None,
            commanded_bounds_deg=np.c_[q.min(0),q.max(0)].tolist(),
            formal=[dict(name=s['name'],definition=s['continuous_definition'],smoothness=s['smoothness'])
                    for s in plan['steps'] if s['family']=='continuous'])
        summary.append(item);print(json.dumps(item),flush=True)
    (a.output/'plans.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
