#!/usr/bin/env python3
"""Prepare whole-event episodes from independent, complete, contact-free sessions."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from factr2_next.data_collection.quality import require, sha256, sidecar_path, write_json, make_manifest, validate_manifest
from factr2_next.data_collection.campaign import control_profile, arm_control_profile, geometry_sha256, validate_independent_manifest
from factr2_next.w3_config import SIGNALS

ROOT = Path(__file__).resolve().parents[1]


def collect_source(session, role, selected_events=None, allow_evaluation_resume=False):
    files = sorted(session.glob('*.h5'))
    if len(files) != 1:
        raise ValueError('Require exactly one raw H5: ' + str(session))
    source = files[0]; report = require(source)
    meta = report['metadata']
    if meta.get('derivation') or meta['source'] != 'real' or meta['sample_hz'] != 100 or meta['contact'].get('present') is not False:
        raise ValueError('Require raw real 100Hz explicitly contact-free recording')
    run = json.loads((session/'audit/motion_run.json').read_text())
    from w3_coverage_plan import load_coverage
    plan, _, _ = load_coverage(session/'audit/motion.yaml')
    protocol = plan['config'].get('baseline_protocol', {})
    if protocol.get('role') != role or run.get('motion_sha256') != sha256(session/'audit/motion.yaml'):
        raise ValueError('Require matching baseline protocol for role ' + role)
    if selected_events is None:
        if (run.get('status') != 'completed' or len(run.get('events', [])) != len(plan['steps'])
                or any(e.get('status') != 'completed' or any(e.get(k) != s[k]
                       for k in ('name', 'phase', 'kind', 'speed'))
                       for e, s in zip(run['events'], plan['steps']))):
            raise ValueError('Require complete matching baseline protocol for role ' + role)
    else:
        if ((role != 'train' and not (allow_evaluation_resume and role in ('val','test')))
                or not isinstance(selected_events, list) or not selected_events
                or any(not isinstance(x, str) for x in selected_events)
                or len(set(selected_events)) != len(selected_events)):
            raise ValueError('Explicit unique completed event selection is allowed only for training')
        planned = {s['name']: s for s in plan['steps']}
        actual = {e['name']: e for e in run.get('events', [])}
        for name in selected_events:
            if (name not in planned or name not in actual or actual[name].get('status') != 'completed'
                    or any(actual[name].get(k) != planned[name][k] for k in ('phase', 'kind', 'speed'))):
                raise ValueError('Selected event did not complete: ' + name)
        run = copy.deepcopy(run)
        run['events'] = [e for e in run['events'] if e['name'] in selected_events]
    collision = json.loads((session/'audit/collision.json').read_text())
    if (collision.get('accepted') is not True or collision.get('plan_sha256') != sha256(session/'audit/motion.yaml')
            or collision.get('robot_description_sha256') != plan['model_sha256']):
        raise ValueError('Frozen collision audit missing or mismatched')
    return source, meta, report, run, control_profile(meta)


def event_streams(source):
    """Read closed raw position streams once for whole-event recovery checks."""
    streams = {}
    with h5py.File(source, 'r') as h:
        for ep in h:
            if f'{ep}/joint_pos/timestamps' not in h or len(h[f'{ep}/joint_pos/timestamps']) < 50:
                continue  # Final empty/short episodes are excluded by raw quality checks.
            streams[ep] = (h[f'{ep}/joint_pos/timestamps'][:],
                           np.rad2deg(h[f'{ep}/joint_pos/data'][:]))
    return streams


def event_blocks(source, meta, run, plan, streams=None):
    """Preserve each raw episode boundary; never bridge a feedback timeout."""
    if streams is None:
        streams = event_streams(source)
    blocks = []
    steps = {s['name']: s for s in plan['steps']}
    previous_ns = -1
    for event in run['events']:
        step = steps[event['name']]
        guard = int(meta['sample_hz']) if event['kind'].startswith('static_') else 0
        lo_ns = round(datetime.fromisoformat(event['utc']).timestamp()*1e9) + guard*10_000_000
        hi_ns = round(datetime.fromisoformat(event['completed_utc']).timestamp()*1e9) - guard*10_000_000
        if lo_ns < previous_ns:
            raise ValueError('Overlapping event ranges')
        previous_ns = hi_ns
        pieces = []; measured = []; observed_seconds = 0.
        for ep, (stamps, positions) in streams.items():
            lo, hi = int(np.searchsorted(stamps, lo_ns)), int(np.searchsorted(stamps, hi_ns))
            if hi-lo < 50:
                continue
            observed_seconds += (int(stamps[hi-1])-int(stamps[lo]))*1e-9 + .01
            measured.append(positions[lo:hi])
            pieces.append(dict(source_episode=ep, source_row_start=lo, source_row_stop_exclusive=hi,
                kind=event['kind'], phase=event['phase'], speed=event['speed'], event=event['name'],
                guard_frames_each_end=guard))
        expected_seconds = (hi_ns-lo_ns)*1e-9
        if (not pieces or expected_seconds <= 0 or observed_seconds < .995*expected_seconds
                or (guard and len(pieces) != 1)):
            if event['phase'] == 'transition':
                continue
            raise ValueError('Incomplete continuous event coverage: ' + event['name'])
        if event['phase'] == 'single_joint':
            joint = step['joint']; bounds = plan['config']['working_bounds_deg'][joint]
            actual = np.concatenate(measured)[:, joint]; tolerance = .1*(bounds[1]-bounds[0])
            if actual.min() > bounds[0]+tolerance or actual.max() < bounds[1]-tolerance:
                raise ValueError('Insufficient measured single-joint coverage: ' + event['name'])
        blocks.extend(pieces)
    return blocks


def prepare(spec, output):
    if output.exists():
        raise ValueError('Output already exists; never replace raw data or a frozen split')
    if set(spec['sessions']) != {'train', 'val', 'test'}:
        raise ValueError('Specify three independent session roles')
    scope = spec.get('control_profile_scope')
    if scope not in (None, 'arm_v2'):
        raise ValueError('Unknown control profile scope')
    resumed_roles = spec.get('resumed_evaluation_roles', [])
    if (not isinstance(resumed_roles,list) or any(r not in ('val','test') for r in resumed_roles)
            or len(set(resumed_roles)) != len(resumed_roles)
            or (resumed_roles and set(spec.get('protocol_reference', {})) != {'train','val','test'})):
        raise ValueError('Evaluation resumption requires explicit roles and all frozen protocol references')
    sources = {}
    seen = set(); ids = set(); profiles = set()
    for role, folders in spec['sessions'].items():
        if not isinstance(folders, list) or not folders:
            raise ValueError('Each role requires at least one independent session')
        sources[role] = []
        for folder in folders:
            selection = None
            if isinstance(folder, dict):
                if set(folder) != {'session', 'events'} or 'protocol_reference' not in spec:
                    raise ValueError('Resumed selections require an explicit frozen protocol reference')
                selection = folder['events']; folder = folder['session']
            source, meta, report, run, profile = collect_source(Path(folder).resolve(), role, selection,
                allow_evaluation_resume=role in resumed_roles)
            if scope == 'arm_v2':
                profile = arm_control_profile(meta)
            if report['h5_sha256'] in seen or meta['session_id'] in ids:
                raise ValueError('Session reused across campaign roles')
            seen.add(report['h5_sha256']); ids.add(meta['session_id']); profiles.add(profile)
            sources[role].append((source, meta, report, run, profile))
    if len(profiles) != 1:
        raise ValueError('Freeze identical control gains, gravity and friction across sessions')
    # Validate all input and build ranges before creating any derived output.
    prepared = []
    protocols = []
    working_envelopes = []
    for role, records in sources.items():
        for source, meta, report, run, profile in records:
            plan = yaml.load((source.parent/'audit/motion.yaml').read_text(),
                             Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
            protocols.append(plan['config']['baseline_protocol'])
            working_envelopes.append(plan['config']['working_bounds_deg'])
            blocks = event_blocks(source, meta, run, plan)
            prepared.append((role, source, meta, report, profile, blocks))
    versions = {p['version'] for p in protocols}
    if len(versions) != 1:
        raise ValueError('Do not mix baseline protocol revisions in one campaign')
    if versions == {2}:
        for field in ('recipe_sha256', 'shared_pose_seed', 'task_poses_deg'):
            if any(p[field] != protocols[0][field] for p in protocols):
                raise ValueError('Independent sessions require the same task coverage: ' + field)
        if any(b != working_envelopes[0] for b in working_envelopes):
            raise ValueError('Independent sessions require identical working envelopes')
    completion_audit = {}
    for role in sources:
        selected = [b for r, *_, blocks in prepared if r == role for b in blocks]
        if not any(b['kind'].startswith('static_') for b in selected):
            raise ValueError('No static baseline blocks')
        if 'protocol_reference' in spec:
            from w3_coverage_plan import load_coverage
            reference, _, _ = load_coverage(spec['protocol_reference'][role])
            if reference['config']['baseline_protocol']['role'] != role:
                raise ValueError('Frozen reference role mismatch')
            signature = lambda b:(b['phase'], b['kind'], b['speed'])
            required = {signature(s) for s in reference['steps'] if s['phase'] != 'transition'}
            if not required <= {signature(b) for b in selected}:
                raise ValueError('Incomplete resumed protocol: ' + role)
            for source, _, _, run, _ in sources[role]:
                original, _, _ = load_coverage(source.parent/'audit/motion.yaml')
                for field in ('home_deg', 'working_bounds_deg', 'tool_frame', 'baseline_protocol'):
                    if original['config'][field] != reference['config'][field]:
                        raise ValueError('Resumed reference/config mismatch: ' + field)
                if (original['model_sha256'] != reference['model_sha256'] and not
                        (scope == 'arm_v2' and geometry_sha256(original['robot_description']) ==
                         geometry_sha256(reference['robot_description']))):
                    raise ValueError('Resumed reference robot mismatch')
                for speed in {e['speed'] for e in run['events']}:
                    if original['config']['speeds'][speed] != reference['config']['speeds'][speed]:
                        raise ValueError('Resumed selected speed limits mismatch')
            completion_audit[role] = dict(reference=str(Path(spec['protocol_reference'][role]).resolve()),
                reference_sha256=sha256(spec['protocol_reference'][role]),
                control_profile_scope=scope or 'full_snapshot_v1',
                geometry_sha256=geometry_sha256(reference['robot_description']),
                required_signatures=[list(s) for s in sorted(required)],
                observed_signatures=[list(s) for s in sorted({signature(b) for b in selected})],
                sources=[dict(h5=str(source), h5_sha256=report['h5_sha256'],
                              selected_completed_events=[e['name'] for e in run['events']])
                         for source, _, report, run, _ in sources[role]], complete=True)
    output.mkdir(parents=True)
    if completion_audit:
        write_json(output/'protocol_completeness.json', completion_audit)
    splits = {s:[] for s in ('train', 'val', 'test')}
    for index, (role, source, meta, report, profile, blocks) in enumerate(prepared):
        path = output/f'{role}_{index:02d}.h5'
        derived = copy.deepcopy(meta); derived['session_id'] = output.name+'_'+path.stem
        if versions == {2}:
            derived['operating_envelope'] = dict(position_bounds_deg=working_envelopes[0],
                                                position_margin_deg=3.)
            derived['task_coverage'] = {k:copy.deepcopy(protocols[0][k]) for k in
                                       ('recipe_sha256', 'shared_pose_seed', 'task_poses_deg', 'max_condition')}
        derived['episodes'] = {}; derived['control_profile_sha256'] = profile
        if scope:
            derived['control_profile_scope'] = scope
        derived['derivation'] = dict(parent_session_id=meta['session_id'],
            parent_h5=str(source), parent_h5_sha256=report['h5_sha256'],
            scope='independent_recording_sessions', guard_frames_each_end={'static':100, 'dynamic':0})
        with h5py.File(source, 'r') as raw, h5py.File(path, 'w') as h:
            h.attrs.update(schema='factr2_next_h5_v1', session_name=derived['session_id'],
                           created_at=datetime.now().astimezone().isoformat())
            for i, block in enumerate(blocks):
                ep = f'ep_{i:04d}'; dest = h.create_group(ep)
                lo, hi = block['source_row_start'], block['source_row_stop_exclusive']
                for key in SIGNALS:
                    group = dest.create_group(key)
                    source_ep = block['source_episode']
                    group.create_dataset('data', data=raw[f'{source_ep}/{key}/data'][lo:hi], compression='gzip')
                    group.create_dataset('timestamps', data=raw[f'{source_ep}/{key}/timestamps'][lo:hi])
                derived['episodes'][ep] = dict(rows=hi-lo, **block)
        derived['h5_sha256'] = sha256(path); write_json(sidecar_path(path), derived)
        require(path); splits[role].append(path)
    manifest_path = output/'split.json'
    manifest = make_manifest(manifest_path, splits, output.name)
    manifest['evaluation_scope'] = 'independent_recording_sessions'
    write_json(manifest_path, manifest)
    manifest, reports = validate_manifest(manifest_path)
    audit = validate_independent_manifest(manifest, reports)
    write_json(output/'split_audit.json', audit)
    template = 'train_next_v2.yaml' if versions == {2} else 'train_baseline.yaml'
    cfg = yaml.safe_load((ROOT/'config/w3/right'/template).read_text())
    cfg['data']['manifest'] = str(manifest_path.resolve())
    cfg['save']['run_name'] = 'next_right_'+output.name
    (output/'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(prepare(yaml.safe_load(args.sessions.read_text()), args.output.resolve()))
