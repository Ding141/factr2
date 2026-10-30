"""Verify one completed identification recording and attach frozen block roles.

Reads original ROS precision. This module never controls hardware or learns a
model. It refuses partial motion, mismatched plan/model identities and bad joins.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from factr2_next.inference.kinematic_compensation import digest


def label(folder):
    spec = json.loads((folder/'test_spec.json').read_text())
    state = json.loads((folder/'test_state.json').read_text())
    recorder = json.loads((folder/'recorder_state.json').read_text())
    if state['status'] != 'completed' or recorder['status'] != 'closed':
        raise ValueError('require completed motion and closed recorder before labeling')
    session = Path(state['session'])
    run = json.loads((session/'audit/motion_run.json').read_text())
    if (run['status'] != 'completed' or run['motion_sha256'] != spec['plan_sha256']
            or digest(folder/'motion.yaml') != spec['plan_sha256']
            or len(run['events']) != spec['steps']
            or any(e['status'] != 'completed' for e in run['events'])):
        raise ValueError('incomplete or changed frozen motion plan')
    if digest(Path(spec['model_dir'])/'model.json') != spec['model_hash']:
        raise ValueError('observer model changed during collection')
    if digest(Path(spec['base_model_dir'])/'metadata.json') != spec['base_metadata_sha256']:
        raise ValueError('base model changed during collection')
    quality = json.loads((session/'quality.json').read_text())
    if not quality['accepted']:
        raise ValueError('raw H5 quality check failed')
    with (folder/'samples.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    stamps = np.asarray([int(r['source_timestamp_ns']) for r in rows], dtype=np.int64)
    if len(stamps)<1000 or np.any(np.diff(stamps)<=0):
        raise ValueError('insufficient or nonmonotonic synchronized frames')
    phase = np.full(len(rows), 'unclassified', dtype=object)
    roles = np.full(len(rows), 'unused', dtype=object)
    for event in run['events']:
        name = event['name']
        if name not in spec['block_roles']:
            raise ValueError('event absent from predeclared split: '+name)
        mask = (stamps >= event['command_start_ros_ns']) & (stamps < event['command_end_ros_ns'])
        if np.any(phase[mask] != 'unclassified'):
            raise ValueError('overlapping event intervals')
        phase[mask], roles[mask] = name, spec['block_roles'][name]
    def array(key):
        return np.array([[float(r[f'{key}_j{j}']) for j in range(1,8)] for r in rows])
    identity = float(np.max(abs(array('measured_tau_nm')-array('base_free_pred_nm')-
                                array('base_external_raw_nm'))))
    if not np.isfinite(identity) or identity>1e-5:
        raise ValueError('measured minus free torque does not match recorded residual')
    aliases = [f'free_pred_tau_nm_j{j}' for j in range(1,8)]
    with (folder/'labeled_samples.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields+aliases+['exact_event','predeclared_role'])
        writer.writeheader()
        for row, name, role in zip(rows, phase, roles):
            writer.writerow(dict(row, **{alias:row[f'base_free_pred_nm_j{j}']
                                          for j,alias in enumerate(aliases,1)},
                                 exact_event=name, predeclared_role=role))
    gaps = np.diff(stamps)*1e-9
    valid = np.array([r['next_state']=='valid' for r in rows])
    report = dict(status='labeled', rows=len(rows), duration_seconds=float((stamps[-1]-stamps[0])*1e-9),
                  maximum_joined_gap_ms=float(gaps.max()*1000), gaps_above_40ms=int((gaps>.04).sum()),
                  residual_identity_error_nm=identity, raw_h5_quality_accepted=True,
                  predeclared_valid_rows={role:int(((roles==role)&valid).sum()) for role in ('train','val','test')},
                  source_sha256=digest(folder/'samples.csv'), labeled_sha256=digest(folder/'labeled_samples.csv'),
                  plan_sha256=spec['plan_sha256'], split_manifest_sha256=digest(folder/'test_spec.json'),
                  no_load_provenance='human operator confirmation; no independent external torque sensor',
                  input_precision='full ROS precision; dashboard rounding excluded',
                  filters_reset_on_gaps=True)
    (folder/'label_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(label(args.observation.resolve()),ensure_ascii=False,indent=2))
