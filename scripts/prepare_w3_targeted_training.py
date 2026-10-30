#!/usr/bin/env python3
"""Freeze whole-session holdouts for a preliminary NEXT training run."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'factr2_next/src/factr2_next'))
from factr2_next.data_collection.quality import require,sha256,sidecar_path,write_json,make_manifest,validate_manifest
from factr2_next.data_collection.campaign import arm_control_profile,validate_independent_manifest
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.coverage_audit import audit_coverage
from factr2_next.w3_config import validate_w3_config


def check_assignment(seen_sessions,seen_hashes,identity,digest,role):
    for key,seen in ((identity,seen_sessions),(digest,seen_hashes)):
        if key in seen:
            raise ValueError('Repeated parent recording in dataset assignments: '+str(key))
        seen[key]=role


def prepare(spec_path,output):
    spec_path=Path(spec_path).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Never overwrite a frozen training dataset')
    spec=yaml.safe_load(spec_path.read_text())
    if spec['version']!=1 or set(spec['sessions'])!={'train','val','test'}:raise ValueError('Require three explicit session lists')
    pending=[];seen_sessions={};seen_hashes={};bounds_low=np.full(7,np.inf);bounds_high=-bounds_low
    for role in ('train','val','test'):
        if not spec['sessions'][role]:raise ValueError('Empty split')
        for name in spec['sessions'][role]:
            path=Path(name).expanduser().resolve();report=require(path);meta=report['metadata'];d=meta['derivation']
            if meta['source']!='real' or meta['sample_hz']!=100 or d['family'] not in ('continuous','hysteresis'):
                raise ValueError('Require real targeted 100Hz data')
            parent=Path(d['source_h5']);session=Path(d['source_session']);raw=require(parent)
            if sha256(parent)!=d['source_sha256'] or (session/'CONTACT_DETECTED').exists():raise ValueError('Changed parent or contact marker')
            motion=json.loads((session/'audit/motion_run.json').read_text())
            can=json.loads((session/'audit/can_record_state.json').read_text())
            if (motion['status']!='completed' or any(e['status']!='completed' for e in motion['events'])
                    or motion['motion_sha256']!=d['plan_sha256'] or sha256(session/'audit/motion.yaml')!=d['plan_sha256']):
                raise ValueError('Incomplete or changed source protocol')
            if can['status']!='closed' or can['socket_dropped_frames'] or can['kernel_timestamps_missing']:
                raise ValueError('Incomplete CAN log')
            identity=raw['metadata']['session_id'];digest=raw['h5_sha256']
            check_assignment(seen_sessions,seen_hashes,identity,digest,role)
            m=copy.deepcopy(meta);m['derivation'].update(parent_session_id=identity,parent_h5=str(parent),parent_h5_sha256=digest)
            m['control_profile_scope']='arm_v2';m['control_profile_sha256']=arm_control_profile(m)
            m['training_split']=dict(role=role,collection_role=meta['targeted_collection']['role'],
                assignment='post_hoc_whole_recording_session_holdout',source_samples=str(path),source_samples_sha256=report['h5_sha256'])
            if role=='train':
                with h5py.File(path) as h:
                    for ep in h:
                        q=np.rad2deg(h[ep+'/joint_pos/data'][:]);bounds_low=np.minimum(bounds_low,q.min(0));bounds_high=np.maximum(bounds_high,q.max(0))
            pending.append((role,path,m,report))
    if len({m['control_profile_sha256'] for _,_,m,_ in pending})!=1:raise ValueError('Physical control profile mismatch')
    output.mkdir(parents=True);shutil.copy2(spec_path,output/'session_assignments.yaml')
    entries={r:[] for r in ('train','val','test')};audit=[]
    envelope=dict(position_bounds_deg=np.c_[bounds_low,bounds_high].tolist(),position_margin_deg=0.,
                  source='observed training positions only; rectangular bounds do not certify all pose combinations')
    for index,(role,source,meta,report) in enumerate(pending):
        target=output/f'{role}_{index:02d}.h5';shutil.copy2(source,target)
        if sha256(target)!=report['h5_sha256']:raise ValueError('Copy changed samples')
        meta['operating_envelope']=envelope
        meta['task_coverage']=dict(scope='targeted free motion; post-hoc whole-session holdouts',
            prospective_independent_protocol_completed=False,temperature_conditions='unclassified; nominal measurements retained')
        write_json(sidecar_path(target),meta);require(target);entries[role].append(target)
        audit.append(dict(role=role,parent_session_id=meta['derivation']['parent_session_id'],
            parent_h5_sha256=meta['derivation']['parent_h5_sha256'],source_samples=str(source),
            frozen_samples=str(target),h5_sha256=report['h5_sha256'],collection_role=meta['targeted_collection']['role']))
    manifest_path=output/'split.json';manifest=make_manifest(manifest_path,entries,output.name)
    manifest.update(evaluation_scope='independent_recording_sessions',
        split_protocol='Frozen post-hoc whole C-session holdouts; original collection roles preserved; no window/random/event split',
        prospective_validation=False,signal_time_shift_seconds=0.,delay_estimated=False)
    write_json(manifest_path,manifest);manifest,reports=validate_manifest(manifest_path)
    independent=validate_independent_manifest(manifest,reports)
    cfg=yaml.safe_load((ROOT/'config/w3/right/train_next_pose_v3.yaml').read_text())
    datasets={r:ManifestDataset(manifest['splits'][r],cfg['data']['keys'],50,512) for r in ('train','val','test')}
    formal_requirements=cfg['data'].pop('coverage_requirements')
    formal=audit_coverage(datasets['train'],datasets['val'],formal_requirements)
    write_json(output/'production_coverage_audit.json',formal)
    stats={r:dict(sessions=len(manifest['splits'][r]),episodes=len(ds.episodes),windows=len(ds),
            rows=sum(len(x) for x,_ in ds.episodes),groups=sorted(set(i['family'] for i in ds.episode_info)),
            speed_profiles=sorted(set(i['speed'] for i in ds.episode_info))) for r,ds in datasets.items()}
    write_json(output/'split_audit.json',dict(assignments=audit,independence=independent,splits=stats,
        normalization='fit only on train inside training process',
        time_alignment='raw streams unchanged; no delay estimation, rescaling, filtering, or bias subtraction',
        history='50-frame stateless NEXT; histories never cross original episode/session boundaries',
        evaluation_limitations=['held-out recordings were originally collected as train; assignments now frozen before fitting',
            'validation and test cover 6 deg/s cap at the reference anchor; 10 deg/s and second-anchor evaluation not independently covered',
            'dedicated cold/warm and contact-release experiments pending']))
    cfg['data'].update(manifest=str(manifest_path),require_independent_sessions=True)
    cfg['train'].update(device='cpu',cpu_threads=8,stop_file=str(output/'STOP_TRAINING'))
    cfg['save'].update(output_dir=str(ROOT/'runs'),run_name='targeted_next_'+output.name)
    cfg['trial']=dict(scope='post-hoc disjoint whole recording sessions; validation/test limited to 6 deg/s',
        deployment_ready=False,production_coverage_audit=str(output/'production_coverage_audit.json'),
        pending=['prospective matching multi-speed independent validation/test','dedicated cold/warm comparison',
                 '12 deg/s and higher measured coverage','contact/release validation'])
    validate_w3_config(cfg,'train');(output/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(json.dumps(dict(output=str(output),splits=stats,production_coverage_passed=formal['passed'],
        deployment_ready=False,independent_session_overlap=False),ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sessions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.sessions,args.output)


if __name__=='__main__':main()
