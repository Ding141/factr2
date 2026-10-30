"""Read-only audit of recorded NEXT residuals; never initializes ROS or moves a robot."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'factr2_next/src/factr2_next')]
from w3_coverage_plan import ArmModel
from factr2_next.inference.endpoint_wrench import estimate_wrench


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(stage, formal=False):
    with (stage / 'samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selected = np.array([r['phase'] == 'random_motion' for r in rows])
    if formal:
        selected[:] = False
        report = json.loads((stage / 'feedback_analysis.json').read_text())
        stamps = np.array([int(r['source_timestamp_ns']) for r in rows], dtype=np.int64)
        for trial in report['trials']:
            selected |= (stamps >= trial['start_ros_ns']) & (stamps <= trial['end_ros_ns'])
    rows = [r for r, keep in zip(rows, selected) if keep]
    q = np.deg2rad([[float(r[f'q_deg_j{i}']) for i in range(1, 8)] for r in rows])
    model = ArmModel((stage / 'robot.urdf').read_text(), 'right', 'right_attachment_point')
    jacobians = model.jacobians(q)
    results = dict(stage=str(stage), rows=len(rows),
        recorded_reasons=dict(Counter(r['endpoint_reason'] for r in rows)),
        recorded_next_states=dict(Counter(r['next_state'] for r in rows)),
        recorded_contact_states=dict(Counter(r['contact_state'] for r in rows)),
        source_sha256={p: digest(stage / p) for p in ('samples.csv', 'robot.urdf')},
        modes={})
    for mode in ('full_wrench', 'force_only'):
        scaled = jacobians.copy()
        scaled[:, :3] /= .4
        if mode == 'force_only':
            scaled = scaled[:, :3]
        singular = np.linalg.svd(scaled, compute_uv=False)
        condition = singular[:, 0] / singular[:, -1]
        mode_results = dict(condition_quantiles=dict(zip(('min', 'p50', 'p95', 'max'),
            map(float, np.quantile(condition, [0, .5, .95, 1])))),
            geometry_rejected_rows=int(np.sum(condition > 100)), signals={})
        for key in ('external_raw_nm', 'external_filtered_nm'):
            residual = np.array([[float(r[f'{key}_j{i}']) for i in range(1, 8)] for r in rows])
            reasons, norms, errors = [], [], []
            for j, tau in zip(jacobians, residual):
                wrench, info = estimate_wrench(j, tau, mode=mode)
                reasons.append(info['reason'])
                if 'torque_fit_error_nm' in info:
                    errors.append(info['torque_fit_error_nm'])
                if wrench is not None:
                    norms.append(float(np.linalg.norm(wrench[:3])))
            values = np.array(norms)
            mode_results['signals'][key] = dict(solver_reasons=dict(Counter(reasons)),
                solver_valid_fraction=len(norms) / len(rows),
                force_norm_p95_valid_n=float(np.percentile(values, 95)) if len(values) else None,
                joint_mae_nm=np.mean(np.abs(residual), axis=0).tolist(),
                joint_averaged_mae_nm=float(np.mean(np.abs(residual))),
                fit_error_p95_nm=float(np.percentile(errors, 95)) if errors else None)
        results['modes'][mode] = mode_results
    return results


if __name__ == '__main__':
    result = dict(date='2026-10-21', scope='offline solver replay of recorded residuals',
        limitations=['No new model fit, inference, timestamp shift, or parameter tuning.',
            'Solver validity is not accuracy; force statistics include only solver-valid frames.',
            'force_only assumes zero moment about right_attachment_point.',
            'Runtime freshness/envelope gates are not re-executed; their recorded states are reported.',
            'These are already-inspected diagnostic datasets, not a new blind test.'],
        code_sha256={str(p): digest(p) for p in (Path(__file__).resolve(),
            ROOT / 'scripts/w3_coverage_plan.py',
            ROOT / 'factr2_next/src/factr2_next/factr2_next/inference/endpoint_wrench.py')},
        stages=[audit(ROOT / 'log/random_free_test_20261015/attempt_03'),
            audit(ROOT / 'log/next_feedback_trial_20261019_v1', formal=True)])
    target = Path(__file__).with_name('endpoint_replay.json')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
