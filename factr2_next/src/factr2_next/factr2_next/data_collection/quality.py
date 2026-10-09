"""Strict W3 H5/sidecar checks and content-addressed episode splits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from factr2_next.data_collection.h5_writer import SCHEMA
from factr2_next.w3_config import SIGNALS


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def commit_at(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def sidecar_path(path):
    return Path(path).with_suffix('.metadata.json')


def session_metadata(cfg, config_path):
    supplied = dict(cfg.get('metadata', {}))
    required = ('source', 'tool', 'gripper', 'load', 'calibration', 'control',
                'trajectory_id', 'contact', 'temperature', 'audit_paths', 'health_gate_enabled')
    missing = [k for k in required if k not in supplied]
    if missing:
        raise ValueError(f'metadata_missing:{missing}')
    if supplied['source'] not in ('synthetic', 'real'):
        raise ValueError('metadata_source')
    if supplied['source'] == 'real' and not supplied['health_gate_enabled']:
        raise ValueError('real_health_required')
    calibration = supplied['calibration']
    if calibration.get('path'):
        calibration['sha256'] = sha256(calibration['path'])
    elif supplied['source'] == 'real':
        raise ValueError('real_calibration_required')
    factr_root = Path(__file__).resolve().parents[5]
    w3_workspace = Path(supplied.get('w3_workspace', factr_root.parent / 'dual_arm_robot')).expanduser().resolve()
    return {**supplied, 'contract_version': cfg['contract_version'], 'side': cfg['side'],
            'joint_order': cfg['joint_names'], 'sample_hz': cfg['recording']['target_hz'],
            'topics': cfg['topics'], 'session_id': cfg['session_name'],
            'time_source': 'ROS header ns; recorder watchdog monotonic',
            'config_sha256': sha256(config_path),
            'w3_workspace': str(w3_workspace),
            'software_commits': {'factr2': commit_at(factr_root),
                                 'w3': commit_at(w3_workspace)},
            'episodes': {}}


def _nonfinite_json(value):
    raise ValueError('nonfinite_json:' + value)


def check(path, episodes=None, expected=None):
    path = Path(path).resolve()
    result = {'path': str(path), 'errors': [], 'episodes': {}, 'excluded': {}}
    err = result['errors']
    try:
        meta = json.loads(sidecar_path(path).read_text(), parse_constant=_nonfinite_json)
        result['metadata'] = meta
        result['h5_sha256'] = sha256(path)
        if meta.get('h5_sha256') != result['h5_sha256']:
            err.append('sidecar_hash')
        side = meta.get('side')
        if meta.get('contract_version') != 'w3_next_v1' or side not in ('left', 'right'):
            err.append('metadata_contract_side')
        if meta.get('joint_order') != [f'{side}_joint_{i}' for i in range(7)]:
            err.append('metadata_joint_order')
        if meta.get('sample_hz') != 50:
            err.append('metadata_sample_hz')
        for key in ('session_id', 'topics', 'software_commits', 'config_sha256', 'tool', 'gripper',
                    'load', 'calibration', 'control', 'trajectory_id', 'contact', 'temperature',
                    'time_source', 'audit_paths', 'health_gate_enabled', 'episodes'):
            if key not in meta:
                err.append('metadata_missing:' + key)
        for key, (suffix, field) in SIGNALS.items():
            spec = meta.get('topics', {}).get(key, {})
            topic = str(spec.get('topic', '')).format(robot_topic_root=f'/factr2/{side}')
            if spec.get('field') != field or topic != f'/factr2/{side}/{suffix}':
                err.append('metadata_topic:' + key)
        if meta.get('source') not in ('synthetic', 'real'):
            err.append('metadata_source')
        if meta.get('source') == 'real':
            if not meta.get('health_gate_enabled'):
                err.append('real_health_required')
            for key in ('tool', 'load', 'calibration', 'contact'):
                if not meta.get(key):
                    err.append('real_metadata:' + key)
            cal = meta.get('calibration', {})
            if not cal.get('path') or not cal.get('sha256'):
                err.append('real_calibration_required')
            elif sha256(cal['path']) != cal['sha256']:
                err.append('calibration_hash')
        if expected:
            for k, v in expected.items():
                if meta.get(k) != v:
                    err.append('metadata_mismatch:' + k)
        with h5py.File(path, 'r') as h5:
            if h5.attrs.get('schema') != SCHEMA:
                err.append('schema')
            for k in ('session_name', 'created_at'):
                if not h5.attrs.get(k):
                    err.append('root_attribute:' + k)
            selected = sorted(h5) if episodes is None else list(episodes)
            if not selected:
                err.append('no_episodes')
            for ep in selected:
                reasons = []
                info = {'errors': reasons}
                result['episodes'][ep] = info
                if ep not in h5:
                    reasons.append('missing_episode')
                    continue
                if ep not in meta.get('episodes', {}):
                    reasons.append('episode_metadata_missing')
                if meta.get('episodes', {}).get(ep, {}).get('excluded_reason'):
                    reasons.append('excluded_episode')
                    result['excluded'][ep] = meta['episodes'][ep]['excluded_reason']
                lengths, times = [], []
                for key in SIGNALS:
                    if key not in h5[ep] or not all(k in h5[ep][key] for k in ('data', 'timestamps')):
                        reasons.append('missing_stream:' + key)
                        continue
                    a, t = h5[ep][key]['data'][:], h5[ep][key]['timestamps'][:]
                    lengths.append(len(a))
                    if a.ndim != 2 or a.shape[1:] != (7,):
                        reasons.append('data_shape:' + key)
                    if a.dtype.kind != 'f' or not np.isfinite(a).all():
                        reasons.append('data_finite_float:' + key)
                    if t.dtype != np.dtype('int64') or t.shape != (len(a),):
                        reasons.append('timestamp_dtype_shape:' + key)
                    elif not len(t) or np.any(t <= 0) or np.any(np.diff(t) <= 0):
                        reasons.append('timestamp_positive_increasing:' + key)
                    times.append(t)
                if len(lengths) != 4 or len(set(lengths)) != 1:
                    reasons.append('stream_lengths')
                if times and any(not np.array_equal(times[0], t) for t in times[1:]):
                    reasons.append('stream_stamp_equality')
                n = lengths[0] if lengths else 0
                info['rows'] = n
                if n < 50:
                    reasons.append('too_short')
                if times and times[0].dtype == np.dtype('int64') and times[0].ndim == 1 and len(times[0]) >= 2:
                    gaps_ns = np.diff(times[0])
                    dt = gaps_ns * 1e-9
                    duration = (int(times[0][-1]) - int(times[0][0])) * 1e-9
                    hz = (len(times[0]) - 1) / duration if duration > 0 else 0
                    info.update(duration_seconds=duration, hz=hz, max_gap_seconds=float(dt.max()))
                    if not 45 <= hz <= 55:
                        reasons.append('frequency')
                    if gaps_ns.max() > 40_000_000:
                        reasons.append('gap')
                if reasons:
                    err.extend(ep + ':' + r for r in reasons)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, IndexError) as exc:
        err.append('read_error:' + str(exc))
    result['accepted'] = not err
    return result


def require(path, episodes=None, expected=None):
    report = check(path, episodes, expected)
    if not report['accepted']:
        raise ValueError('; '.join(report['errors']))
    return report


def validate_manifest(path, expected=None):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') != 'w3_split_v1' or not manifest.get('dataset_id'):
        raise ValueError('manifest_schema_dataset_id')
    seen, reports, signatures = set(), {}, []
    for split in ('train', 'val', 'test'):
        entries = manifest.get('splits', {}).get(split)
        if not entries:
            raise ValueError('manifest_empty:' + split)
        reports[split] = []
        for entry in entries:
            report = require(entry['path'], entry['episodes'], expected)
            if entry.get('h5_sha256') != report['h5_sha256']:
                raise ValueError('manifest_hash')
            meta = report['metadata']
            signatures.append({k: meta[k] for k in ('side', 'joint_order', 'contract_version',
                                                   'tool', 'gripper', 'load', 'calibration')})
            if meta['contact'].get('present') is not False:
                raise ValueError('free_motion_contact_required')
            for ep in entry['episodes']:
                identity = (report['h5_sha256'], ep)
                if identity in seen:
                    raise ValueError('split_overlap:' + str(identity))
                seen.add(identity)
            reports[split].append(report)
    if any(s != signatures[0] for s in signatures):
        raise ValueError('dataset_profile_mismatch')
    return manifest, reports


def make_manifest(output, splits, dataset_id):
    manifest = {'schema': 'w3_split_v1', 'dataset_id': dataset_id, 'splits': {}}
    for split, paths in splits.items():
        entries = []
        for path in paths:
            meta = json.loads(sidecar_path(path).read_text())
            eps = [ep for ep, v in meta['episodes'].items() if not v.get('excluded_reason')]
            report = require(path, eps)
            entries.append({'path': str(Path(path).resolve()), 'episodes': eps,
                            'h5_sha256': report['h5_sha256'], 'audit_paths': meta['audit_paths']})
        manifest['splits'][split] = entries
    write_json(output, manifest)
    try:
        validate_manifest(output)
    except Exception:
        Path(output).unlink()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    checker = sub.add_parser('check')
    checker.add_argument('path')
    checker.add_argument('--json', dest='output')
    checker.add_argument('--episodes', nargs='+')
    maker = sub.add_parser('manifest')
    maker.add_argument('--output', required=True)
    maker.add_argument('--dataset-id', required=True)
    for split in ('train', 'val', 'test'):
        maker.add_argument('--' + split, nargs='+', required=True)
    verify = sub.add_parser('verify-manifest')
    verify.add_argument('path')
    args = parser.parse_args()
    if args.command == 'check':
        report = check(args.path, args.episodes)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report['accepted'] else 1)
    if args.command == 'manifest':
        make_manifest(args.output, {s: getattr(args, s) for s in ('train', 'val', 'test')}, args.dataset_id)
    else:
        validate_manifest(args.path)
    print('PASS')


if __name__ == '__main__':
    main()
