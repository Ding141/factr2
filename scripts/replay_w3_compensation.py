"""Counterfactual replay of frozen compensators; never publishes robot commands."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from fit_w3_kinematic_compensation import series, smooth
from fit_w3_pose_dynamic_compensation import prepare,predict
from factr2_next.inference.kinematic_compensation import ResidualModel,Compensator
from factr2_next.data_collection.quality import write_json,sha256


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    artifact=json.loads((a.experiment/'model.json').read_text());recipe=json.loads((a.experiment/'recipe.json').read_text())
    base=Path(recipe['base']);meta=json.loads((base/'metadata.json').read_text())
    assert sha256(base/'metadata.json')==artifact['base_metadata_sha256']
    with (a.recording/'samples.csv').open() as f:rows=list(csv.DictReader(f))
    def array(key):return np.array([[float(r[f'{key}_j{j+1}']) for j in range(7)] for r in rows])
    q,v,cmd=[np.deg2rad(array(k)) for k in ('q_deg','velocity_deg_s','command_deg')]
    t=np.array([int(r['source_timestamp_ns']) for r in rows],np.int64)
    measured=array('measured_tau_nm');free=array('base_free_pred_nm')
    item=series(t,np.c_[q,v,cmd-q],measured,free,np.full(len(t),str(a.recording)))
    prepare([item],np.deg2rad(meta['operating_envelope']['position_bounds_deg']))
    source_valid=np.array([r['next_state']=='valid' for r in rows])
    correction,support=predict(item,artifact);support &= source_valid;correction[~support]=0
    final=smooth(item['target']-correction,t,artifact['signal']['output_hz'])
    filter_only=smooth(item['target'],t,artifact['signal']['output_hz'])
    previous=array('external_filtered_nm');raw=array('base_external_raw_nm')
    state=json.loads((a.recording/'test_state.json').read_text())
    run=json.loads((Path(state['session'])/'audit/motion_run.json').read_text())
    formal=[e for e in run['events'] if e['family']=='continuous' and e['status']=='completed']
    mask=np.zeros(len(t),bool)
    for e in formal:mask |= (t>=e['command_start_ros_ns']+10e9)&(t<e['command_end_ros_ns']-10e9)
    mask &= source_valid
    model=ResidualModel(artifact);pipeline=Compensator(model);step=[];cost=[]
    # Rebuild filters from the same first input as the batch recipe. This checks
    # full online/offline mathematics, not only a vectorized approximation.
    for k in range(len(t)):
        now=time.perf_counter()
        value=pipeline.update(int(t[k]),q[k],v[k],cmd[k],measured[k],free[k],
            allow_compensation=bool(item['position_gate'][k] and source_valid[k]))
        cost.append(time.perf_counter()-now);step.append(value['final'])
    step=np.array(step);parity=float(abs(step-final).max())
    np.testing.assert_allclose(step,final,atol=1e-9,rtol=1e-9)
    # Synthetic sustained torque at exactly the same kinematics must pass
    # through the two causal low-pass filters with unity steady gain.
    one,two=[Compensator(model) for _ in range(2)];delta=[]
    center=np.mean(q[mask],axis=0)
    for k in range(400):
        pos=center+.02*np.sin(k*.03);z=np.zeros(7);stamp=1_000_000_000+k*10_000_000
        r=one.update(stamp,pos,z,pos,z,z)
        twin=two.update(stamp,pos,z,pos,np.full(7,.3 if k>=100 else 0),z)
        np.testing.assert_array_equal(r['correction'],twin['correction']);delta.append(twin['final']-r['final'])
    delta=np.array(delta);np.testing.assert_allclose(delta[-1],.3,atol=1e-10)
    metrics={}
    for name,values in dict(raw=raw,filter_only=filter_only,previous=previous,final=final).items():
        x=values[mask];metrics[name]=dict(overall_rms_nm=float(np.sqrt(np.mean(x*x))),
            rms_nm=np.sqrt(np.mean(x*x,axis=0)).tolist(),bias_nm=x.mean(0).tolist(),
            abs_p95_nm=np.percentile(abs(x),95,axis=0).tolist())
    report=dict(experiment=str(a.experiment.resolve()),model_sha256=sha256(a.experiment/'model.json'),
        source_recording=str(a.recording.resolve()),recorded_frames=len(t),evaluated_frames=int(mask.sum()),
        offline_replay_only=True,physical_motion_performed=False,observation_used_for_fit=False,
        previously_observed_data=True,metrics=metrics,online_offline_max_error_nm=parity,
        support_fraction=float(support[mask].mean()),
        compute_time_ms=dict(median=float(np.median(cost)*1000),p99=float(np.percentile(cost,99)*1000)),
        fixed_kinematics_contact_test=dict(injection_nm=.3,settled_gain=float(delta[-1,0]/.3),
            rise_90pct_ms=int(np.flatnonzero(delta[100:,0]>=.27)[0])*10,physical_contact_validation=False))
    write_json(out/'summary.json',report)
    np.savez_compressed(out/'waveforms.npz',source_timestamp_ns=t,raw_nm=raw,filter_only_nm=filter_only,
        previous_nm=previous,final_nm=final,correction_nm=correction,support=support,evaluated=mask)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e=formal[0];show=(t>=e['command_start_ros_ns'])&(t<e['command_end_ros_ns']);tt=(t-e['command_start_ros_ns'])*1e-9
    fig,axes=plt.subplots(7,1,figsize=(14,14),sharex=True)
    for j,ax in enumerate(axes):
        for label,y,color in [('Filter only',filter_only,'#bbbbbb'),('Previous compensation',previous,'#d47a1f'),('New compensation (replay)',final,'#1670ca')]:
            ax.plot(tt[show],np.where(source_valid[show],y[show,j],np.nan),lw=.8,label=label,color=color)
        ax.axhline(0,color='black',lw=.5);ax.set_ylabel(f'J{j+1} (Nm)');ax.grid(alpha=.2)
    axes[0].legend(ncol=3);axes[0].set_title('Frozen model replay on earlier no-load motion | no new robot motion')
    axes[-1].set_xlabel('Seconds');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('experiment','recording','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
