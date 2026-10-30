#!/usr/bin/env python3
"""Derive labeled whole-block episodes without mixing experiments or sessions."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'factr2_next/src/factr2_next'))
from factr2_next.data_collection.quality import require,sha256,sidecar_path,write_json
from w3_targeted_protocol import load

STATES={'entry_ramp':0,'continuous_excitation':1,'exit_ramp':2,'hysteresis':3,'stationary_hold':4}


def temperature_streams(session):
    path=session/'audit/motor_temperature.csv'
    with path.open() as h:rows=list(csv.DictReader(h))
    result={}
    for j in range(7):
        selected=[r for r in rows if int(r['joint'])==j and r['temperature_valid']=='1']
        if not selected:continue
        times=np.array([int(r['kernel_receive_ns']) for r in selected],dtype=np.int64)
        if np.any(np.diff(times)<=0):raise ValueError('Temperature source clock not increasing')
        result[j]=(times,np.array([[float(r['mos_c']),float(r['rotor_c'])] for r in selected]))
    return result


def temperatures(session,stamps,cache=None):
    cache=temperature_streams(session) if cache is None else cache
    mos=np.full((len(stamps),7),np.nan);rotor=mos.copy();valid=np.zeros_like(mos,dtype=bool)
    for j,(times,values) in cache.items():
        idx=np.searchsorted(times,stamps,side='right')-1
        ok=(idx>=0)&((stamps-times[np.maximum(idx,0)])<=500_000_000)
        # Causal sample-and-hold with an explicit 500ms freshness mask. These
        # values are labels, not interpolated NEXT torque supervision.
        mos[ok,j]=values[idx[ok],0];rotor[ok,j]=values[idx[ok],1];valid[ok,j]=True
    return dict(mos_temperature_c=mos,rotor_temperature_c=rotor,temperature_valid=valid)


def labels(step,stamps,event,source_start):
    offset=(stamps-int(event['command_start_ros_ns']))*1e-9
    state=np.full(len(stamps),-1,np.int8)
    for i,a in enumerate(step['annotations']):
        mask=(offset>=a['start_seconds']-1e-9)&((offset<a['end_seconds']) if i+1<len(step['annotations']) else (offset<=a['end_seconds']+1e-9))
        state[mask]=STATES[a['state']]
    if np.any(state<0):raise ValueError('Unlabeled sample interval')
    return dict(motion_state=state,intentional_stationary_hold=state==STATES['stationary_hold'],
                phase_offset_seconds=offset,elapsed_since_first_source_seconds=(stamps-source_start)*1e-9)


def prepare(session,output,completed_only=False,quality_audit=None):
    session=Path(session).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Never overwrite derived data')
    if (session/'CONTACT_DETECTED').exists():raise ValueError('Operator marked contact; forbidden as free-motion training data')
    plan,_,digest=load(session/'audit/motion.yaml')
    if plan['family']=='warmup':raise ValueError('Warmup is recorded history, not a training experiment')
    run=json.loads((session/'audit/motion_run.json').read_text())
    if run['motion_sha256']!=digest or run['role']!=plan['role'] or run['family']!=plan['family']:raise ValueError('Run/plan mismatch')
    if completed_only and plan['role']!='train':raise ValueError('Evaluation sessions must complete the frozen protocol')
    if not completed_only and (run['status']!='completed' or len(run['events'])!=len(plan['steps'])):raise ValueError('Require completed session or explicit training --completed-only')
    collision=json.loads((session/'audit/collision.json').read_text())
    if collision.get('accepted') is not True or collision['plan_sha256']!=digest:raise ValueError('Collision audit mismatch')
    files=list(session.glob('*.h5'))
    if len(files)!=1:raise ValueError('Require one closed raw H5')
    source=files[0];raw_report=require(source);meta=raw_report['metadata']
    if meta['source']!='real' or meta['sample_hz']!=100 or meta['contact'].get('present') is not False:raise ValueError('Require actual real 100Hz contact-free data')
    if meta.get('targeted_collection',{}).get('family')!=plan['family']:raise ValueError('Missing raw family provenance')
    can_state=json.loads((session/'audit/can_record_state.json').read_text())
    if can_state['status']!='closed' or can_state['socket_dropped_frames'] or can_state['kernel_timestamps_missing']:
        raise ValueError('Require cleanly closed raw CAN timing record')
    accepted_names=None;quality=None
    if quality_audit is not None:
        quality_audit=Path(quality_audit).resolve();quality=json.loads(quality_audit.read_text())
        if (Path(quality['source']).resolve()!=session or quality['role']!=plan['role']
                or quality['schema']!='w3_continuous_recording_audit_v1'
                or quality['criteria']!=dict(position_change_window_seconds=.2,command_delta_threshold_deg=.3,
                    measured_flat_threshold_deg=.04,max_joint_stall_seconds=.3,maximum_formal_gap_ms=20.1)):
            raise ValueError('Motion-quality identity/criteria changed')
        expected={s['name'] for s in plan['steps'] if s['family']==plan['family']}
        if {b['name'] for b in quality['blocks']}!=expected:raise ValueError('Incomplete quality audit')
        accepted_names={b['name'] for b in quality['blocks'] if b['accepted']}
        if not accepted_names:raise ValueError('No complete quality-passing blocks')
    selected=[];steps={s['name']:s for s in plan['steps']};last_stop=-1
    for e in run['events']:
        if e['name'] not in steps:raise ValueError('Unknown executed event')
        s=steps[e['name']]
        if any(e[k]!=s[k] for k in ('family','speed','kind','active_joints','approach_direction')):raise ValueError('Changed event label')
        if e['status']!='completed':
            if not completed_only:raise ValueError('Incomplete event')
            continue
        if s['family']!=plan['family']:continue  # probes/transitions never silently enter either family
        if accepted_names is not None and s['name'] not in accepted_names:continue
        a,b=int(e['command_start_ros_ns']),int(e['command_end_ros_ns'])
        if a<last_stop or abs((b-a)*1e-9-s['duration_seconds'])>1e-7:raise ValueError('Overlapping or altered event times')
        last_stop=b;selected.append((s,e,a,b))
    if not selected:raise ValueError('No completed experiment blocks')
    output.mkdir(parents=True)
    target=output/'samples.h5';derived=copy.deepcopy(meta)
    derived.update(session_id=output.name,episodes={},derivation=dict(source_h5=str(source),source_sha256=sha256(source),
        source_session=str(session),plan_sha256=digest,role=plan['role'],family=plan['family'],
        completed_only=completed_only,synthetic_samples_added=0,
        discarded_families=['transition','speed_probe','warmup'],
        history_policy='For long-memory features, compute causally on full raw source before selecting ranges; reset after source gaps. Never combine independent sessions.'))
    derived['temperature'].update(observation_source='passive nominal CAN MOS/rotor Celsius',
        motor_temperature_csv_sha256=sha256(session/'audit/motor_temperature.csv'),
        raw_can_sha256=sha256(session/'audit/raw_can.bin'),
        alignment='causal kernel software receive stamp <= ROS source header; age <=500ms; invalid/mismatched clocks stored as NaN plus mask')
    if quality is not None:
        derived['derivation'].update(quality_audit=str(quality_audit),quality_audit_sha256=sha256(quality_audit),
            accepted_motion_events=sorted(accepted_names),
            excluded_motion_events=sorted(b['name'] for b in quality['blocks'] if not b['accepted']),
            selection='Drop entire blocks using fixed encoder-motion thresholds before fitting; never filter by torque/model error')
    ranges=[]
    temperature_cache=temperature_streams(session)
    with h5py.File(source,'r') as raw,h5py.File(target,'w') as dest:
        keys=('joint_pos','joint_vel','joint_cmd','measured_joint_torque')
        streams={ep:raw[ep+'/joint_pos/timestamps'][:] for ep in raw if len(raw[ep+'/joint_pos/timestamps'])>=50}
        source_start=min(int(t[0]) for t in streams.values())
        dest.attrs.update(schema='factr2_next_h5_v1',session_name=output.name,created_at=meta.get('created_at',run['completed_utc']))
        index=0
        for s,e,a,b in selected:
            observed=0.;pieces=0
            for ep,t in streams.items():
                lo,hi=np.searchsorted(t,[a,b],side='left')
                if hi-lo<50:continue
                stamps=t[lo:hi]
                if np.max(np.diff(stamps))>20_100_000:raise ValueError('A source gap lies inside an experiment; keep raw data and exclude/recollect it')
                observed+=(int(stamps[-1])-int(stamps[0]))*1e-9+.01;pieces+=1
                name=f'ep_{index:04d}';index+=1;g=dest.create_group(name)
                for key in keys:
                    src=raw[ep+'/'+key]
                    if not np.array_equal(src['timestamps'][lo:hi],stamps):raise ValueError('Streams differ')
                    grp=g.create_group(key);grp.create_dataset('data',data=src['data'][lo:hi]);grp.create_dataset('timestamps',data=stamps)
                lab=g.create_group('labels')
                for key,value in labels(s,stamps,e,source_start).items():lab.create_dataset(key,data=value)
                for key,value in temperatures(session,stamps,temperature_cache).items():lab.create_dataset(key,data=value)
                info={k:s[k] for k in ('family','phase','kind','speed','active_joints','pose_index','approach_direction','dwell_seconds','profile','peak_bounds')}
                info.update(rows=len(stamps),event=s['name'],source_episode=ep,source_row_start=int(lo),source_row_stop_exclusive=int(hi),
                            state_codes=STATES,thermal_initial_condition=plan['thermal']['initial_condition'],
                            measured_position_mean_deg=np.rad2deg(g['joint_pos/data'][:]).mean(axis=0).tolist(),
                            native_goal_deg=s['positions_deg'][-1])
                derived['episodes'][name]=info;ranges.append(dict(episode=name,**info))
            # A completed event may not be reconstructed from multiple discontinuous
            # raw episodes. No filling and no accepting short prefixes as full data.
            if pieces!=1 or observed<(b-a)*1e-9-.03:raise ValueError('Incomplete whole-event data: '+s['name'])
    derived['h5_sha256']=sha256(target);write_json(sidecar_path(target),derived)
    result=require(target);write_json(output/'quality.json',result)
    write_json(output/'source_ranges.json',dict(role=plan['role'],family=plan['family'],ranges=ranges))
    return dict(path=str(target),episodes=len(ranges),rows=sum(x['rows'] for x in ranges),role=plan['role'],family=plan['family'],accepted=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--completed-only',action='store_true');p.add_argument('--quality-audit',type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.session,a.output,a.completed_only,a.quality_audit),indent=2))


if __name__=='__main__':main()
