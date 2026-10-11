#!/usr/bin/env python3
"""Free-space baseline and manually annotated load/release evaluation; never tare."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np
import yaml
from factr2_next.data_collection.quality import require, validate_manifest, sha256
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.offline_eval import read_episode, predict_sequence, physical_metrics
from factr2_next.data_collection.campaign import validate_independent_manifest, control_profile
from w3_coverage_plan import ArmModel
from w3_baseline_protocol import conditions
from factr2_next.inference.endpoint_wrench import estimate_wrench, rotate_wrench_at_same_origin


def validate_thresholds(cfg):
    for key in ('per_joint_abs_bias_nm', 'per_joint_abs_p95_nm', 'per_joint_rmse_nm', 'per_joint_return_gap_nm',
                'release_recovery_nm', 'maximum_post_release_drift_nm'):
        a = np.asarray(cfg[key], dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError('Freeze seven positive thresholds: ' + key)
    for key in ('static_pose_tolerance_deg', 'static_measured_span_deg', 'static_speed_rad_s',
                'release_settle_seconds', 'minimum_free_window_seconds', 'minimum_contact_norm_nm'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('Invalid threshold: ' + key)
    for key in ('endpoint_abs_bias', 'endpoint_abs_p95'):
        a = np.asarray(cfg[key], dtype=float)
        if a.shape != (6,) or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError('Freeze six positive endpoint thresholds: ' + key)
    if type(cfg['minimum_release_cases']) is not int or cfg['minimum_release_cases'] < 3:
        raise ValueError('Require at least three contact/release cases')
    estimate_wrench(np.c_[np.eye(6), np.zeros(6)], np.zeros(7), **cfg['endpoint_solver'])


def endpoint_free_metrics(residual, q, model, cfg):
    values = []; reasons = {}; total = 0
    # Same equations/frame/solver as the 10 Hz dashboard, without hiding invalid rows.
    for tau, position in zip(residual[::10], q[::10]):
        pose, jac = model.fk(position, True)
        wrench, diag = estimate_wrench(jac, tau, **cfg['endpoint_solver'])
        total += 1
        if wrench is None:
            reasons[diag['reason']] = reasons.get(diag['reason'], 0) + 1
        else:
            values.append(rotate_wrench_at_same_origin(wrench, pose[:3, :3]))
    rate = len(values)/total if total else 0.
    if not values:
        return dict(valid_fraction=rate, invalid_reasons=reasons, passed=False)
    a = np.asarray(values); bias = a.mean(axis=0); p95 = np.percentile(abs(a), 95, axis=0)
    return dict(valid_fraction=rate, invalid_reasons=reasons,
        axes=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], units=['N']*3+['Nm']*3,
        signed_bias=bias.tolist(), abs_p95=p95.tolist(),
        passed=bool(rate >= .99 and np.all(abs(bias) <= cfg['endpoint_abs_bias'])
                    and np.all(p95 <= cfg['endpoint_abs_p95'])))


def free_metrics(residual, q, vel, cfg):
    metrics = physical_metrics(residual, np.zeros_like(residual))
    pose = np.rad2deg(q.mean(axis=0))
    span = np.rad2deg(np.ptp(q, axis=0))
    stable = bool(np.max(span) <= cfg['static_measured_span_deg']
                  and np.max(abs(vel)) <= cfg['static_speed_rad_s'])
    ok = bool(stable and np.all(abs(np.asarray(metrics['signed_bias_nm'])) <= cfg['per_joint_abs_bias_nm'])
              and np.all(np.asarray(metrics['abs_residual_p95_nm']) <= cfg['per_joint_abs_p95_nm']))
    return dict(metrics, pose_deg=pose.tolist(), measured_span_deg=span.tolist(),
                stationary=stable, passed=ok)


def evaluate_free(loaded, manifest_path, split, cfg):
    validate_thresholds(cfg)
    manifest, reports = validate_manifest(manifest_path, {'side':'right', 'sample_hz':100})
    validate_independent_manifest(manifest, reports)
    blocks = []; residuals = []; geometry = []; pairs = {}
    for entry, source_report in zip(manifest['splits'][split], reports[split]):
        meta = source_report['metadata']
        source = Path(meta['derivation']['parent_h5'])
        plan = yaml.load((source.parent/'audit/motion.yaml').read_text(),
                         Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
        model = ArmModel(plan['robot_description'], 'right', plan['config']['tool_frame'])
        for ep in entry['episodes']:
            arrays = read_episode(entry['path'], ep)
            pred, measured = predict_sequence(loaded, arrays)
            residual = measured-pred; residuals.append(residual)
            info = meta['episodes'][ep]; kind = info['kind']
            q = arrays['joint_pos'][49:]; vel = arrays['joint_vel'][49:]
            if info['phase'] == 'cartesian':
                geometry.extend(conditions(model, np.rad2deg(q[::10])))
            if not kind.startswith('static_'):
                continue
            report = dict(session=meta['derivation']['parent_session_id'], episode=ep, kind=kind,
                          **free_metrics(residual, q, vel, cfg))
            report['endpoint'] = endpoint_free_metrics(residual, q, model, cfg)
            report['passed'] = report['passed'] and report['endpoint']['passed']
            blocks.append(report)
            match = re.fullmatch(r'static_pose(\d+)_from_(low|high)', kind)
            if match:
                pairs.setdefault((report['session'], match[1]), {})[match[2]] = report
    if not residuals or not blocks or not pairs:
        raise ValueError('Missing dynamic/static/matched-return evaluation blocks')
    returns = []
    for (session, pose_id), pair in pairs.items():
        if set(pair) != {'low', 'high'}:
            raise ValueError('Incomplete matched directional return pair')
        a, b = pair['low'], pair['high']
        matched = bool(np.max(abs(np.asarray(a['pose_deg'])-b['pose_deg'])) <= cfg['static_pose_tolerance_deg'])
        gap = abs(np.asarray(a['signed_bias_nm'])-b['signed_bias_nm'])
        endpoint_gap = (abs(np.asarray(a['endpoint']['signed_bias'])-b['endpoint']['signed_bias'])
                        if a['endpoint'].get('signed_bias') and b['endpoint'].get('signed_bias') else None)
        returns.append(dict(session=session, pose_id=pose_id, matched_pose=matched,
            baseline_gap_nm=gap.tolist(), endpoint_baseline_gap=endpoint_gap.tolist() if endpoint_gap is not None else None,
            passed=bool(matched and np.all(gap <= cfg['per_joint_return_gap_nm'])
                        and endpoint_gap is not None and np.all(endpoint_gap <= cfg['endpoint_abs_bias']))))
    geometry_rate = float(np.mean([c is not None and c <= 100 for c in geometry])) if geometry else None
    overall = physical_metrics(np.vstack(residuals), np.zeros_like(np.vstack(residuals)))
    return dict(split=split, overall=overall,
        static_blocks=blocks, matched_returns=returns, task_geometry_valid_fraction=geometry_rate,
        passed=bool(np.all(np.asarray(overall['per_joint_rmse_nm']) <= cfg['per_joint_rmse_nm'])
                    and all(b['passed'] for b in blocks) and all(b['passed'] for b in returns)
                    and geometry_rate is not None and geometry_rate >= .99),
        automatic_rezero=False, accuracy_scope='external joint torque in operator-confirmed free space')


def release_case(residual, stamps, q, vel, case, cfg):
    """Window times are seconds since FIRST H5 ROS stamp, not wall-clock time."""
    if case.get('operator_verified') is not True:
        raise ValueError('Manual contact/free-space annotations are required')
    spans = case['spans']
    if set(spans) != {'before_free', 'contact', 'after_free'}:
        raise ValueError('Require before_free/contact/after_free spans')
    indices = {}; previous = -float('inf')
    for label in ('before_free', 'contact', 'after_free'):
        a, b = np.asarray(spans[label], dtype=float)
        if not np.isfinite([a, b]).all() or a < 0 or a >= b or a < previous:
            raise ValueError('Invalid, overlapping or unordered release annotations')
        previous = b
        start = a + (cfg['release_settle_seconds'] if label == 'after_free' else 0)
        selected = np.flatnonzero((stamps >= start) & (stamps < b))
        needed = int(100*cfg['minimum_free_window_seconds']) if label != 'contact' else 50
        if len(selected) < needed:
            raise ValueError('Release annotation has insufficient post-history samples: ' + label)
        indices[label] = selected
    pre, post = indices['before_free'], indices['after_free']
    before = free_metrics(residual[pre], q[pre], vel[pre], cfg)
    after = free_metrics(residual[post], q[post], vel[post], cfg)
    matched = bool(np.max(abs(np.asarray(before['pose_deg'])-after['pose_deg'])) <= cfg['static_pose_tolerance_deg'])
    gap = abs(np.asarray(before['signed_bias_nm'])-after['signed_bias_nm'])
    mid = len(post)//2
    drift = abs(residual[post[:mid]].mean(axis=0)-residual[post[mid:]].mean(axis=0))
    contact_score = float(np.percentile(np.linalg.norm(residual[indices['contact']], axis=1), 90))
    recovered = bool(np.all(abs(residual[post].mean(axis=0)) <= cfg['release_recovery_nm']))
    # Never let a constant-zero estimator pass by merely returning to zero.
    passed = bool(before['passed'] and after['passed'] and recovered
                  and np.all(drift <= cfg['maximum_post_release_drift_nm'])
                  and (not matched or np.all(gap <= cfg['per_joint_return_gap_nm']))
                  and contact_score >= cfg['minimum_contact_norm_nm'])
    return dict(case_id=case['id'], before=before, after=after, matched_pose=matched,
        baseline_gap_nm=gap.tolist() if matched else None,
        post_release_drift_nm=drift.tolist(), contact_response_norm_p90_nm=contact_score,
        recovered=recovered, passed=passed,
        interpretation='Different poses must each be near zero; do not subtract the first pose baseline')


def evaluate_release(loaded, path, annotations, cfg):
    validate_thresholds(cfg)
    report = require(path, ['ep_0000'], {'side':'right', 'sample_hz':100})
    meta = report['metadata']
    if meta['contact'].get('present') is not True:
        raise ValueError('Use --validation-only recording; never train on contact/release data')
    if annotations.get('h5_sha256') != report['h5_sha256'] or not annotations.get('cases'):
        raise ValueError('Annotations must match finalized H5 hash and contain cases')
    for key in ('tool', 'gripper', 'load', 'calibration'):
        if meta[key] != loaded.metadata[key]:
            raise ValueError('Release/model profile mismatch: ' + key)
    model_profile = control_profile(loaded.metadata)
    if loaded.metadata.get('control_profile_sha256') != model_profile:
        raise ValueError('Candidate frozen control profile changed or is missing')
    if control_profile(meta) != model_profile:
        raise ValueError('Release/model control profile mismatch')
    arrays = read_episode(path, 'ep_0000')
    pred, measured = predict_sequence(loaded, arrays)
    with h5py.File(path, 'r') as h:
        times = h['ep_0000/joint_pos/timestamps'][:]
    stamps = (times[49:]-times[0])*1e-9
    identifiers = [c['id'] for c in annotations['cases']]
    if len(identifiers) < cfg['minimum_release_cases'] or len(set(identifiers)) != len(identifiers):
        raise ValueError('Require distinct contact/release cases; repeated failure must not be hidden')
    description_dump = yaml.safe_load((Path(path).parent/'audit/robot_description.yaml').read_text())
    description = next(iter(description_dump.values()))['ros__parameters']['robot_description']
    model = ArmModel(description, 'right', 'right_attachment_point')
    residual = measured-pred; q = arrays['joint_pos'][49:]; vel = arrays['joint_vel'][49:]
    cases = []
    for c in annotations['cases']:
        result = release_case(residual, stamps, q, vel, c, cfg)
        result['endpoint'] = {}
        for label in ('before_free', 'after_free'):
            a, b = c['spans'][label]
            start = a + (cfg['release_settle_seconds'] if label == 'after_free' else 0.)
            idx = np.flatnonzero((stamps >= start) & (stamps < b))
            result['endpoint'][label] = endpoint_free_metrics(residual[idx], q[idx], model, cfg)
        result['passed'] = result['passed'] and all(e['passed'] for e in result['endpoint'].values())
        if result['matched_pose'] and all(e.get('signed_bias') is not None for e in result['endpoint'].values()):
            a = result['endpoint']['before_free']; b = result['endpoint']['after_free']
            gap = abs(np.asarray(a['signed_bias'])-b['signed_bias'])
            result['endpoint_baseline_gap'] = gap.tolist()
            result['passed'] = result['passed'] and bool(np.all(gap <= cfg['endpoint_abs_bias']))
        cases.append(result)
    return dict(cases=cases, passed=all(c['passed'] for c in cases), automatic_rezero=False,
                scope='Contact response and empty post-release baseline; no calibrated contact accuracy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--acceptance', required=True, type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--release-h5', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if (bool(args.manifest) == bool(args.release_h5) or bool(args.release_h5) != bool(args.annotations)):
        p.error('Choose one manifest or one release H5 with annotations')
    loaded = load_checkpoint(args.checkpoint, expected={'side':'right', 'sample_hz':100})
    cfg = yaml.safe_load(args.acceptance.read_text())
    if args.manifest:
        result = evaluate_free(loaded, args.manifest, args.split, cfg)
    elif args.release_h5 and args.annotations:
        result = evaluate_release(loaded, args.release_h5, yaml.safe_load(args.annotations.read_text()), cfg)
    else:
        p.error('Choose manifest or annotated release H5')
    result['acceptance_sha256'] = sha256(args.acceptance)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], report=str(args.output)), indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
