#!/usr/bin/env python3
"""Read-only physical coverage diagnostics; write reports outside the raw H5."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from factr2_next.data_collection.quality import sha256, write_json
from w3_coverage_plan import load_coverage


def analyze(session):
    files = sorted(session.glob('*.h5'))
    if len(files) != 1:
        raise ValueError('Expected one closed capture H5')
    source = files[0]; digest = sha256(source)
    plan, _, plan_digest = load_coverage(session/'audit/motion.yaml')
    motion = json.loads((session/'audit/motion_run.json').read_text())
    quality = json.loads((session/'quality.json').read_text())
    episodes = []
    with h5py.File(source, 'r') as h:
        for name, group in h.items():
            if not len(group):
                continue
            t = group['joint_pos/timestamps'][:].reshape(-1).astype(float)*1e-9
            q = np.rad2deg(group['joint_pos/data'][:].astype(float))
            cmd = np.rad2deg(group['joint_cmd/data'][:].astype(float))
            tau = group['measured_joint_torque/data'][:].astype(float)
            if len(t) < 2 or np.any(np.diff(t) <= 0):
                raise ValueError('Invalid timestamps')
            # Estimate physical velocity from a 200 ms secant, not the motor's
            # quantized instantaneous velocity, which can be nonzero at rest.
            half = max(1, round(.1/np.median(np.diff(t))))
            v = np.full_like(q, np.nan); vc = v.copy()
            if len(t) > 2*half:
                dt = t[2*half:]-t[:-2*half]
                v[half:-half] = (q[2*half:]-q[:-2*half])/dt[:, None]
                vc[half:-half] = (cmd[2*half:]-cmd[:-2*half])/dt[:, None]
            episodes.append(dict(name=name,t=t,q=q,cmd=cmd,tau=tau,v=v,vc=vc))
    if not episodes:
        raise ValueError('No recorded frames')
    q, cmd, tau = [np.vstack([e[k] for e in episodes]) for k in ('q','cmd','tau')]
    report = dict(source_h5=str(source.resolve()),source_sha256=digest,
        rows=len(q),side=plan['side'],quality_accepted=quality['accepted'],
        motion_completed=motion['status']=='completed' and motion['motion_sha256']==plan_digest,
        tracking_rmse_deg=np.sqrt(np.mean((q-cmd)**2,axis=0)).tolist(),
        max_tracking_error_deg=np.max(abs(q-cmd),axis=0).tolist(),
        actual_range_deg=np.ptp(q,axis=0).tolist(),command_range_deg=np.ptp(cmd,axis=0).tolist(),
        peak_abs_measured_torque_nm=np.max(abs(tau),axis=0).tolist(),events=[],
        velocity_method='200 ms centered secant of measured position; ignores episode edges',
        low_motion_method='Fraction of samples with |command secant velocity| >= 1 deg/s and |measured secant velocity| < 0.2 deg/s',
        interpretation='Descriptive friction/tracking diagnostic; not an external-force accuracy or independent-session training acceptance test.')
    selected_j6 = None
    for event in motion['events']:
        start = datetime.fromisoformat(event['utc']).timestamp()
        end = datetime.fromisoformat(event.get('completed_utc',motion.get('completed_utc',event['utc']))).timestamp()
        rows = [(e, (e['t']>=start)&(e['t']<=end)) for e in episodes]
        rows = [(e,m) for e,m in rows if np.any(m)]
        entry = {k:event[k] for k in ('name','phase','kind','speed','joint','repetition','status')}
        if not rows:
            report['events'].append(entry); continue
        a,b,v,vc = [np.vstack([e[k][m] for e,m in rows]) for k in ('q','cmd','v','vc')]
        entry.update(rows=len(a),actual_range_deg=np.ptp(a,axis=0).tolist(),
            command_range_deg=np.ptp(b,axis=0).tolist(),
            tracking_rmse_deg=np.sqrt(np.mean((a-b)**2,axis=0)).tolist())
        if event['phase']=='single_joint':
            j=event['joint']; moving=np.isfinite(v[:,j]) & (abs(vc[:,j])>=1.)
            entry.update(range_fraction=float(np.ptp(a[:,j])/np.ptp(b[:,j])),
                measured_speed_p95_deg_s=float(np.nanpercentile(abs(v[:,j]),95)),
                measured_speed_p99_deg_s=float(np.nanpercentile(abs(v[:,j]),99)),
                command_speed_peak_deg_s=float(np.nanmax(abs(vc[:,j]))),
                moving_samples=int(moving.sum()),
                low_motion_fraction=float(np.mean(abs(v[moving,j])<.2)) if moving.any() else None)
            if j==5 and event['speed']=='fast' and selected_j6 is None:
                times=np.concatenate([e['t'][m] for e,m in rows]);selected_j6=(times-times[0],a[:,j],b[:,j],v[:,j],vc[:,j])
        report['events'].append(entry)
    report['source_unchanged']=sha256(source)==digest
    out=session/'analysis';out.mkdir(exist_ok=True)
    write_json(out/'capture_diagnostics.json',report)
    fig,axes=plt.subplots(2,2,figsize=(12,8)); x=np.arange(7)
    for speed,color in [('slow','tab:blue'),('fast','tab:orange')]:
        events=[e for e in report['events'] if e['phase']=='single_joint' and e['speed']==speed and 'range_fraction' in e]
        coverage=[np.mean([e['range_fraction'] for e in events if e['joint']==j]) if any(e['joint']==j for e in events) else np.nan for j in x]
        error=[np.mean([e['tracking_rmse_deg'][j] for e in events if e['joint']==j]) if any(e['joint']==j for e in events) else np.nan for j in x]
        axes[0,0].plot(x,coverage,'o-',label=speed,color=color);axes[0,1].plot(x,error,'o-',label=speed,color=color)
    axes[0,0].axhline(1,ls='--',color='gray');axes[0,0].set_ylabel('Actual / commanded joint span')
    axes[0,1].set_ylabel('Tracking RMSE (deg)')
    for ax in axes[0]:ax.set_xticks(x,[f'J{i+1}' for i in x]);ax.legend();ax.grid(alpha=.3)
    if selected_j6 is not None:
        t,a,b,v,vc=selected_j6
        axes[1,0].plot(t,b,label='command');axes[1,0].plot(t,a,label='measured');axes[1,0].set_ylabel('J6 position (deg)')
        axes[1,1].plot(t,vc,label='command');axes[1,1].plot(t,v,label='measured');axes[1,1].set_ylabel('J6 secant velocity (deg/s)')
    for ax in axes[1]:ax.set_xlabel('Time in first fast J6 cycle (s)');ax.legend();ax.grid(alpha=.3)
    fig.suptitle('Measured coverage and tracking diagnostics');fig.tight_layout()
    fig.savefig(out/'capture_diagnostics.png',dpi=150);fig.savefig(out/'capture_diagnostics.pdf');plt.close(fig)
    print(json.dumps({k:report[k] for k in ('rows','quality_accepted','motion_completed','tracking_rmse_deg','actual_range_deg','max_tracking_error_deg','source_unchanged')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    analyze(p.parse_args().session.resolve())
