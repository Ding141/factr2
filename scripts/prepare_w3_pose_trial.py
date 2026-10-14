#!/usr/bin/env python3
"""Exploratory whole-pose holdout from ONE completed direct-goal recording.

This does not qualify as independent-session or high-speed acceptance.
The original campaign preparation and its acceptance gates remain unchanged.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import re

import h5py
import numpy as np
import yaml

from prepare_w3_baseline_campaign import collect_source
from factr2_next.data_collection.quality import make_manifest, require, sha256, sidecar_path, write_json
from factr2_next.training.coverage_audit import audit_coverage
from factr2_next.training.dataset import ManifestDataset
from factr2_next.w3_config import SIGNALS, validate_w3_config

ROOT = Path(__file__).resolve().parents[1]


def event_role(kind):
    match = re.search(r'(?:^|_)pose(\d+)_', kind)
    if not match:
        return 'train'
    pose = int(match[1])
    if not 0 <= pose <= 7:
        raise ValueError('Expected eight task poses')
    return 'val' if pose == 6 else 'test' if pose == 7 else 'train'


def grouped_bounds(events):
    groups = []
    for event in events:
        role = event_role(event['kind'])
        lo = round(datetime.fromisoformat(event['utc']).timestamp()*1e9)
        hi = round(datetime.fromisoformat(event['completed_utc']).timestamp()*1e9)
        if not groups or groups[-1]['role'] != role:
            groups.append(dict(role=role, start=lo, stop=hi, events=[event['name']]))
        else:
            groups[-1]['stop'] = hi
            groups[-1]['events'].append(event['name'])
    # Purge one second at every contiguous split block's endpoints. Histories
    # remain inside an episode, never spanning a role change or receive gap.
    return {name:(g['role'], g['start']+1_000_000_000, g['stop']-1_000_000_000)
            for g in groups for name in g['events']}


def trial_event_blocks(source, run):
    """Keep measured rows; allow at most two boundary sample ticks, no filling.

    Short completed goals can have a one-tick difference between event-clock
    duration and sample-clock duration. This explicit trial allowance does not
    modify the stricter production campaign preparation.
    """
    blocks, skipped = [], []
    with h5py.File(source, 'r') as raw:
        streams = {ep:raw[ep+'/joint_pos/timestamps'][:] for ep in raw
                   if len(raw[ep+'/joint_pos/timestamps']) >= 50}
    for event in run['events']:
        guard = 1_000_000_000 if event['kind'].startswith('static_') else 0
        a = round(datetime.fromisoformat(event['utc']).timestamp()*1e9)+guard
        b = round(datetime.fromisoformat(event['completed_utc']).timestamp()*1e9)-guard
        pieces, observed = [], 0.
        for ep, stamps in streams.items():
            lo, hi = int(np.searchsorted(stamps, a)), int(np.searchsorted(stamps, b))
            if hi <= lo:
                continue
            observed += (int(stamps[hi-1])-int(stamps[lo]))*1e-9+.01
            if hi-lo >= 50:
                pieces.append(dict(source_episode=ep, source_row_start=lo,
                    source_row_stop_exclusive=hi, event=event['name'], kind=event['kind'],
                    phase=event['phase'], speed=event['speed']))
        expected = (b-a)*1e-9
        if expected <= 0 or expected-observed > .020001:
            raise ValueError('Trial event missing more than two sample ticks: '+event['name'])
        if not pieces:
            skipped.append(dict(event=event['name'], reason='fewer than 50 real samples',
                                expected_seconds=expected, observed_seconds=observed))
        blocks.extend(pieces)
    return blocks, skipped


def prepare(session, output):
    if output.exists():
        raise ValueError('Never replace a frozen dataset')
    source, meta, checked, run, profile = collect_source(session, 'train')
    plan = yaml.load((session/'audit/motion.yaml').read_text(), Loader=yaml.CSafeLoader)
    if plan['config'].get('execution_mode') != 'direct_pose_goals':
        raise ValueError('Require the completed direct-goal protocol')
    bounds = grouped_bounds(run['events'])
    blocks, skipped = trial_event_blocks(source, run)
    accepted = []
    used = {}
    with h5py.File(source, 'r') as raw:
        for block in blocks:
            role, a, b = bounds[block['event']]
            stamps = raw[f"{block['source_episode']}/joint_pos/timestamps"][:]
            lo = max(block['source_row_start'], int(np.searchsorted(stamps, a)))
            hi = min(block['source_row_stop_exclusive'], int(np.searchsorted(stamps, b)))
            if hi-lo < 50:
                continue
            markers = used.setdefault(block['source_episode'], np.full(len(stamps), -1, dtype=np.int8))
            if np.any(markers[lo:hi] != -1):
                raise ValueError('Raw source row overlap across episodes/splits')
            markers[lo:hi] = ('train', 'val', 'test').index(role)
            accepted.append(dict(block, role=role, source_row_start=lo,
                                 source_row_stop_exclusive=hi, rows=hi-lo))
        for markers in used.values():
            for left, right in zip(np.flatnonzero(markers >= 0), np.flatnonzero(markers >= 0)[1:]):
                if markers[left] != markers[right] and right-left < 200:
                    raise ValueError('Insufficient purge at a split boundary')
        if {b['role'] for b in accepted} != {'train', 'val', 'test'}:
            raise ValueError('Empty role')
        output.mkdir(parents=True)
        splits = {}
        stats = {}
        for role in ('train', 'val', 'test'):
            chosen = [b for b in accepted if b['role'] == role]
            path = output/(role+'.h5')
            derived = copy.deepcopy(meta)
            derived.update(session_id=output.name+'_'+role, episodes={}, control_profile_sha256=profile)
            derived['operating_envelope'] = dict(position_bounds_deg=plan['config']['working_bounds_deg'],
                                                 position_margin_deg=3.)
            derived['derivation'] = dict(scope='within_one_session_whole_pose_holdout_trial',
                parent_session_id=meta['session_id'], parent_h5=str(source),
                parent_h5_sha256=checked['h5_sha256'], guard_frames_split_boundaries=100)
            with h5py.File(path, 'w') as dest:
                dest.attrs.update(schema='factr2_next_h5_v1', session_name=derived['session_id'],
                                  created_at=datetime.now().astimezone().isoformat())
                for i, block in enumerate(chosen):
                    ep = f'ep_{i:04d}'
                    lo, hi = block['source_row_start'], block['source_row_stop_exclusive']
                    for key in SIGNALS:
                        g = dest.create_group(ep+'/'+key)
                        origin = raw[block['source_episode']+'/'+key]
                        g.create_dataset('data', data=origin['data'][lo:hi], compression='gzip')
                        g.create_dataset('timestamps', data=origin['timestamps'][lo:hi])
                    derived['episodes'][ep] = {k:v for k,v in block.items() if k != 'role'}
            derived['h5_sha256'] = sha256(path)
            write_json(sidecar_path(path), derived)
            require(path)
            splits[role] = [path]
            stats[role] = dict(episodes=len(chosen), rows=sum(b['rows'] for b in chosen),
                              windows=sum(b['rows']-49 for b in chosen))
    manifest_path = output/'split.json'
    manifest = make_manifest(manifest_path, splits, output.name)
    manifest.update(evaluation_scope='single real session; task poses 6/7 held out as val/test',
                    independent_sessions=False, deployment_ready=False,
                    split_protocol='train: joint sweeps and poses 0..5; val: pose6; test: pose7; 1s role-boundary purge')
    write_json(manifest_path, manifest)
    if sha256(source) != checked['h5_sha256']:
        raise ValueError('Raw source changed')
    write_json(output/'split_audit.json', dict(source=str(source), source_sha256=checked['h5_sha256'],
        source_unchanged=True, source_ranges_disjoint=True, independent_sessions=False,
        splits=stats, blocks=accepted, skipped_short_events=skipped,
        event_boundary_sample_allowance_seconds=.02, synthetic_samples_added=0,
        deployment_ready=False))
    cfg = yaml.safe_load((ROOT/'config/w3/right/train_next_pose_v3.yaml').read_text())
    requirements = copy.deepcopy(cfg['data']['coverage_requirements'])
    ds = [ManifestDataset(manifest['splits'][role], cfg['data']['keys'], 50, 512)
          for role in ('train', 'val')]
    strict = audit_coverage(*ds, requirements)
    strict.update(independent_sessions=False, deployment_ready=False,
                  interpretation='Frozen production requirements evaluated unchanged; this trial does not override acceptance')
    write_json(output/'production_coverage_audit.json', strict)
    # Explicit trial config, isolated from the production training entry point.
    cfg['data'].pop('coverage_requirements')
    cfg['data'].update(manifest=str(manifest_path), require_independent_sessions=False)
    cfg['trial'] = dict(scope=manifest['evaluation_scope'], deployment_ready=False,
        production_coverage_audit=str(output/'production_coverage_audit.json'),
        pending=['independent matching val/test recordings', 'high-speed coverage', 'contact/release validation'])
    cfg['train']['stop_file'] = str(output/'STOP_TRAINING')
    cfg['save'].update(output_dir=str(ROOT/'runs'), run_name='trial_'+output.name)
    validate_w3_config(cfg, 'train')
    (output/'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(dict(output=str(output), splits=stats, production_coverage_passed=strict['passed'],
                          production_coverage_reasons=strict['reasons']), ensure_ascii=False), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.session.resolve(), args.output.resolve())
