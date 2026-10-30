"""Offline quality audit of commanded continuity and measured tracking/stalls.

Encoder feedback velocity is quantized. The stall screen uses a 200 ms position
change; a single joint's normal direction reversal is not a whole-arm pause.
Screen thresholds are frozen before viewing the new records, not fit to torque.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def longest_run(mask,dt):
    edge=np.diff(np.r_[False,mask,False].astype(int))
    return float(max(np.flatnonzero(edge==-1)-np.flatnonzero(edge==1),default=0)*dt)


def audit(stage):
    stage=Path(stage).resolve()
    state=json.loads((stage/'test_state.json').read_text())
    if state['status']!='completed':raise ValueError('Incomplete collection: never train on a failed trajectory')
    session=Path(state['session'])
    run=json.loads((session/'audit/motion_run.json').read_text())
    plan=yaml.load((stage/'motion.yaml').read_text(),Loader=yaml.CSafeLoader)
    if run['status']!='completed':raise ValueError('Motion incomplete')
    with (stage/'samples.csv').open() as f:rows=list(csv.DictReader(f))
    t=np.array([int(r['source_timestamp_ns']) for r in rows],np.int64)
    if np.any(np.diff(t)<=0):raise ValueError('Nonmonotonic recorded source stamps')
    def values(prefix):return np.array([[float(r[f'{prefix}_j{j+1}']) for j in range(7)] for r in rows])
    q,cmd,tau=map(values,('q_deg','command_deg','base_external_raw_nm'))
    results=[]
    for step,event in zip(plan['steps'],run['events']):
        if step['family']!='continuous':continue
        mask=(t>=event['command_start_ros_ns'])&(t<event['command_end_ros_ns'])
        tt=t[mask];qq=q[mask];cc=cmd[mask]
        if len(tt)<100:raise ValueError('Too few formal rows')
        gap=np.max(np.diff(tt))*1e-9
        coverage=(tt[-1]-tt[0])*1e-9+.01
        if gap>.0201 or coverage<step['duration_seconds']-.03:raise ValueError('Discontinuous/incomplete formal recording')
        # Reject true flat tracking under sustained motion, while keeping normal
        # turnarounds and the smooth eight-second entry/exit ramp.
        width=20
        dq=qq[width:]-qq[:-width];dc=cc[width:]-cc[:-width]
        mid=(tt[width:]+tt[:-width])/2
        interior=(mid>=event['command_start_ros_ns']+10e9)&(mid<=event['command_end_ros_ns']-10e9)
        stuck=(abs(dc)>.30)&(abs(dq)<.04)&interior[:,None]
        durations=[longest_run(stuck[:,j],.01) for j in range(7)]
        whole=(np.linalg.norm(dc,axis=1)>.6)&(np.linalg.norm(dq,axis=1)<.06)&interior
        item=dict(name=step['name'],rows=len(tt),coverage_seconds=coverage,max_gap_ms=gap*1000,
            command_velocity_peak_deg_s=event['telemetry']['native_velocity_peak_deg_s'],
            tracking_max_deg=abs(qq-cc).max(0).tolist(),
            measured_position_span_deg=np.ptp(qq,axis=0).tolist(),
            commanded_position_span_deg=np.ptp(cc,axis=0).tolist(),
            suspected_joint_stall_longest_seconds=durations,
            suspected_whole_arm_pause_longest_seconds=longest_run(whole,.01),
            accepted=bool(max(durations)<.30 and longest_run(whole,.01)<.10))
        results.append(item)
    report=dict(schema='w3_continuous_recording_audit_v1',role=plan['role'],source=str(session),
        accepted=all(r['accepted'] for r in results) and len(results)==2,
        criteria=dict(position_change_window_seconds=.20,command_delta_threshold_deg=.30,
            measured_flat_threshold_deg=.04,max_joint_stall_seconds=.30,
            maximum_formal_gap_ms=20.1),blocks=results,
        limitation='Position-change stall screen; no external camera or independent torque ground truth')
    (stage/'continuous_quality.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(7,2,figsize=(15,17),sharex=True)
    seconds=(t-t[0])*1e-9
    for j in range(7):
        axes[j,0].plot(seconds,cmd[:,j],lw=.7,label='command')
        axes[j,0].plot(seconds,q[:,j],lw=.7,label='measured')
        axes[j,0].set_ylabel(f'J{j+1} (deg)')
        axes[j,1].plot(seconds,tau[:,j],lw=.5)
        axes[j,1].set_ylabel('Old NEXT residual (Nm)')
    axes[0,0].legend();axes[-1,0].set_xlabel('s');axes[-1,1].set_xlabel('s')
    fig.suptitle(plan['role']+' | raw positions and unchanged pre-training observer')
    fig.tight_layout();fig.savefig(stage/'recording_waveforms.png',dpi=130);plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    r=audit(p.parse_args().stage);print(json.dumps(r,indent=2))
    if not r['accepted']:raise SystemExit('Quality failed: keep raw data; do not train/deploy from this campaign')
