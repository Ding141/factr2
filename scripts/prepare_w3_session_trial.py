#!/usr/bin/env python3
"""Derive purged, whole-motion splits from one coverage capture; never alter raw H5."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from factr2_next.data_collection.quality import require, sha256, sidecar_path, write_json, make_manifest
from factr2_next.w3_config import SIGNALS

ROOT=Path(__file__).resolve().parents[1]


def role(event, previous=None):
    if event['phase']=='transition':
        return previous if event['kind']=='reset_home' else 'train'
    if event['repetition']==1:
        return 'train'
    if event['repetition']!=2:
        raise ValueError('This trial protocol requires exactly two motion repetitions')
    if event['phase']=='single_joint':
        index=event['joint']
    elif event['phase']=='cartesian':
        index=next((i for i,k in enumerate(('line_x','line_y','line_z','circle_xy')) if event['kind'].startswith(k)),None)
        if index is None:raise ValueError('Unknown Cartesian path')
    else:raise ValueError('Unknown motion phase')
    return 'val' if (index+int(event['speed']=='fast'))%2==0 else 'test'


def guarded_range(stamps, group, purge):
    a=round(datetime.fromisoformat(group['start_utc']).timestamp()*1e9)
    b=round(datetime.fromisoformat(group['end_utc']).timestamp()*1e9)
    lo=int(np.searchsorted(stamps,a))+purge
    hi=int(np.searchsorted(stamps,b,side='right'))-purge
    if hi-lo<50:
        # Pure setup/return motion is optional training data. A complete
        # excitation block must still retain a full history after purging.
        if all(phase=='transition' for phase in group['phases']):
            return None
        raise ValueError('Motion block too short after 1 second guard on both ends')
    return lo,hi


def completed_events(run, allow_prefix=False, continuity=None):
    events=run.get('events',[])
    if run.get('status')=='completed' and events and all(e['status']=='completed' for e in events):
        return events,[]
    # Explicit recovery only for the diagnosed moving-waypoint arrival bug.
    # Do not salvage a hardware, communication or motion tracking failure.
    if (not allow_prefix or not run.get('error','').startswith('Continuous block endpoint tracking error ')
            or not continuity or not continuity.get('accepted') or len(events)<2
            or any(e['status']!='completed' for e in events[:-1])
            or events[-1]['status']!='sending' or events[-1]['phase']!='single_joint'
            or events[-1]['repetition']!=1):
        raise ValueError('Require a completed motion report or explicitly audited completed prefix')
    return events[:-1],[events[-1]['name']]


def prepare(session, output, allow_completed_prefix=False):
    if output.exists():raise ValueError('Output already exists; choose a new trial directory')
    files=sorted(session.glob('*.h5'))
    if len(files)!=1:raise ValueError('Require exactly one capture H5')
    source=files[0];checked=require(source,['ep_0000']);meta=checked['metadata']
    run=json.loads((session/'audit/motion_run.json').read_text())
    continuity=json.loads((session/'analysis/controller_smoothness.json').read_text()) if allow_completed_prefix else None
    events,excluded=completed_events(run,allow_completed_prefix,continuity)
    groups=[];previous=None
    for event in events:
        split=role(event,previous)
        if split is None:raise ValueError('Reset without preceding path')
        if not groups or groups[-1]['split']!=split:
            groups.append({'split':split,'start_utc':event['utc'],'end_utc':event['completed_utc'],
                'events':[event['name']],'phases':[event['phase']]})
        else:
            groups[-1]['end_utc']=event['completed_utc'];groups[-1]['events'].append(event['name']);groups[-1]['phases'].append(event['phase'])
        previous=split
    with h5py.File(source,'r') as h:
        stamps=h['ep_0000/joint_pos/timestamps'][:]
        arrays={k:h[f'ep_0000/{k}/data'][:] for k in SIGNALS}
    used=np.full(len(stamps),-1,dtype=np.int8);blocks=[];skipped=[];purge=int(meta['sample_hz'])
    for group in groups:
        accepted=guarded_range(stamps,group,purge)
        if accepted is None:
            skipped.append({**group,'reason':'pure transition too short for a 50-frame history after 1-second trim at both ends'})
            continue
        lo,hi=accepted
        if np.any(used[lo:hi]!=-1):raise ValueError('Source row overlap')
        used[lo:hi]=('train','val','test').index(group['split'])
        blocks.append({**group,'source_episode':'ep_0000','source_row_start':lo,'source_row_stop_exclusive':hi,'rows':hi-lo})
    # Full histories are materialized only within these accepted continuous blocks.
    ranges=sorted(blocks,key=lambda x:x['source_row_start'])
    for left,right in zip(ranges,ranges[1:]):
        if left['source_row_stop_exclusive']+2*purge>right['source_row_start']:
            raise ValueError('Missing two-second guard between derived blocks')
    output.mkdir(parents=True)
    entries={};stats={}
    for split in ('train','val','test'):
        path=output/(split+'.h5');selected=[b for b in blocks if b['split']==split]
        if not selected:raise ValueError('Empty split')
        derived=copy.deepcopy(meta);derived['episodes']={};derived['session_id']=output.name+'_'+split
        derived['derivation']={'scope':'within-session repeated-motion trial, not independent sessions',
            'parent_h5':str(source.resolve()),'parent_h5_sha256':checked['h5_sha256'],
            'parent_session_id':meta['session_id'],'guard_frames_each_end':purge,'source_ranges':[]}
        with h5py.File(path,'w') as h:
            h.attrs.update(schema='factr2_next_h5_v1',session_name=derived['session_id'],created_at=datetime.now().astimezone().isoformat())
            for index,block in enumerate(selected):
                name=f'ep_{index:04d}';lo=block['source_row_start'];hi=block['source_row_stop_exclusive'];ep=h.create_group(name)
                for key in SIGNALS:
                    g=ep.create_group(key);g.create_dataset('data',data=arrays[key][lo:hi],compression='gzip');g.create_dataset('timestamps',data=stamps[lo:hi])
                derived['episodes'][name]={'rows':hi-lo,'source_row_start':lo,'source_row_stop_exclusive':hi,'source_episode':'ep_0000'}
                derived['derivation']['source_ranges'].append({**block,'derived_episode':name})
        derived['h5_sha256']=sha256(path);write_json(sidecar_path(path),derived)
        require(path);entries[split]=[path]
        stats[split]={'episodes':len(selected),'rows':sum(b['rows'] for b in selected),'windows':sum(b['rows']-49 for b in selected)}
    manifest_path=output/'split.json'
    manifest=make_manifest(manifest_path,entries,output.name)
    manifest.update(evaluation_scope='within one real session; repeated trajectories and shared hardware',
        split_protocol=f'first repetition train; second repetition whole joint/path cycles alternate val/test; fast assignment inverted; {purge}-frame (1-second) trim at both ends',
        parent_h5_sha256=checked['h5_sha256'])
    write_json(manifest_path,manifest)
    audit={'parent_h5':str(source.resolve()),'parent_sha256_before':checked['h5_sha256'],
        'parent_sha256_after':sha256(source),'source_rows':len(stamps),'retained_rows':int(np.count_nonzero(used!=-1)),
        'guard_frames_each_end':purge,'source_ranges_disjoint':True,'scope':manifest['evaluation_scope'],
        'splits':stats,'blocks':blocks,'skipped_transition_groups':skipped,
        'completed_prefix_only':bool(excluded),'excluded_incomplete_events':excluded}
    assert audit['parent_sha256_before']==audit['parent_sha256_after']
    write_json(output/'split_audit.json',audit)
    cfg=yaml.safe_load((ROOT/f'config/w3/{meta["side"]}/train.yaml').read_text())
    cfg['data']['sample_hz']=meta['sample_hz']
    cfg['data']['manifest']=str(manifest_path.resolve())
    cfg['train'].update(batch_size=256,epochs=20,device='cpu',cpu_threads=1)
    if meta['sample_hz']==100:
        cfg['train'].update(optimizer='adamw',weight_decay=1e-6,gradient_clip=1.0,epochs=80,
            cpu_threads=min(8,os.cpu_count() or 1),
            early_stopping=dict(enabled=True,patience=20,warmup=10,min_delta=1e-5))
    cfg['save'].update(output_dir=str(ROOT/'runs'),run_name='next_'+meta['side']+'_'+output.name)
    (output/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(json.dumps({'output':str(output),'scope':audit['scope'],'splits':stats,'retained_rows':audit['retained_rows']},indent=2))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--completed-prefix',action='store_true',help='Only audited complete blocks before the diagnosed moving-waypoint check failure')
    args=p.parse_args();prepare(args.session.resolve(),args.output.resolve(),args.completed_prefix)
