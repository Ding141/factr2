#!/usr/bin/env python3
"""Offline audit. Fit reads train only; validate cannot fit or change parameters.

No ROS node is created. ROS message classes are used only to deserialize a
read-only sqlite bag. Positive lag means response(t + lag) follows reference(t).
Signal phase lag is deliberately NOT called a hardware sampling delay.
"""
import argparse
import csv
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import h5py
import numpy as np
from scipy import signal
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'factr2_next/src/factr2_next'))
KEYS = ('joint_pos', 'joint_vel', 'joint_cmd', 'measured_joint_torque')
SETTINGS = dict(dt_s=.01, smooth_samples=21, polynomial_order=3,
                max_lag_ms=150, tracking_max_lag_ms=500,
                lag_step_ms=10, active_speed_deg_s=2.,
                torque_active_slope_nm_s=.05, min_pair_rows=200,
                max_gap_s=.0201, static_trim_s=1., bootstrap_repeats=200,
                bootstrap_seed=20261016)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    def convert(v):
        if isinstance(v, dict): return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [convert(x) for x in v]
        if isinstance(v, np.ndarray): return convert(v.tolist())
        if isinstance(v, np.generic): return convert(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(convert(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x): return {'n': 0}
    return dict(n=len(x), mean=x.mean(), std=x.std(), minimum=x.min(),
                p05=np.percentile(x, 5), p50=np.median(x),
                p95=np.percentile(x, 95), p99=np.percentile(x, 99), maximum=x.max())


def contains(stamps, intervals):
    starts = np.array([a for a, _ in intervals], dtype=np.int64)
    stops = np.array([b for _, b in intervals], dtype=np.int64)
    idx = np.searchsorted(starts, stamps, side='right') - 1
    return (idx >= 0) & (stamps <= stops[np.maximum(idx, 0)])


def load_train(h5, audit):
    blocks = [b for b in audit['blocks'] if b['role'] == 'train']
    episodes = []
    with h5py.File(h5, 'r') as h:
        assert len(h) == len(blocks), 'split audit/train episode mismatch'
        for name, block in zip(sorted(h), blocks):
            ep = h[name]
            t = ep['joint_pos/timestamps'][:]
            assert len(t) == block['rows'] and np.all(np.diff(t) > 0)
            arrays = {k: ep[k + '/data'][:].astype(float) for k in KEYS}
            assert all(np.array_equal(t, ep[k + '/timestamps'][:]) for k in KEYS)
            assert all(v.shape == (len(t), 7) and np.isfinite(v).all() for v in arrays.values())
            episodes.append(dict(t=t, q=arrays[KEYS[0]], v=arrays[KEYS[1]],
                                 c=arrays[KEYS[2]], tau=arrays[KEYS[3]],
                                 kind=block['kind'], event=block['event'], episode=name))
    return episodes


DIAG_KEYS = ('state_source_stamp_ns', 'command_source_stamp_ns',
             'state_source_age_seconds', 'command_source_age_seconds',
             'state_receive_age_seconds', 'command_receive_age_seconds',
             'source_skew_seconds')


def read_bag(bag, intervals):
    """Read selected topics only. libzstd avoids installing new dependencies."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    from diagnostic_msgs.msg import DiagnosticArray
    lib = ctypes.CDLL(ctypes.util.find_library('zstd'))
    lib.ZSTD_getFrameContentSize.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    lib.ZSTD_decompress.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
    lib.ZSTD_decompress.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = (ctypes.c_size_t,)
    lib.ZSTD_isError.restype = ctypes.c_uint
    native, diag = [], []
    starts = np.array([a for a, _ in intervals], dtype=np.int64)
    stops = np.array([b for _, b in intervals], dtype=np.int64)
    def in_scope(stamp):
        i = int(np.searchsorted(starts, stamp, side='right')) - 1
        return i >= 0 and stamp <= stops[i]
    files = sorted(Path(bag).glob('*.db3'))
    for path in files:
        con = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        topics = {i: n for i, n in con.execute('select id,name from topics')}
        ids = [i for i, n in topics.items() if n in
               ('/joint_position_controller/command_state', '/factr2/right/adapter_status')]
        query = 'select timestamp,topic_id,data from messages where topic_id in (' + ','.join('?' for _ in ids) + ') order by timestamp'
        for received, tid, data in con.execute(query, ids):
            if data[:4] == b'\x28\xb5\x2f\xfd':
                size = lib.ZSTD_getFrameContentSize(data, len(data))
                if not 0 < size < 2**20: raise ValueError('unsupported zstd frame size')
                dest = ctypes.create_string_buffer(size)
                n = lib.ZSTD_decompress(dest, size, data, len(data))
                if lib.ZSTD_isError(n) or n != size: raise ValueError('corrupt zstd')
                data = dest.raw
            if topics[tid].endswith('/command_state'):
                m = deserialize_message(data, JointState)
                stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
                if not in_scope(stamp): continue
                indices = [m.name.index(f'right_joint_{j}') for j in range(7)]
                native.append((stamp, received, [m.position[i] for i in indices],
                               [m.velocity[i] for i in indices]))
            else:
                m = deserialize_message(data, DiagnosticArray)
                for s in m.status:
                    if s.name != 'factr2/adapter/right': continue
                    values = {v.key: json.loads(v.value) for v in s.values}
                    stamp = values.get('state_source_stamp_ns', 0)
                    if not in_scope(stamp): continue
                    header = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
                    diag.append((header, received, s.message, values))
        con.close()
        print('read-only bag extracted:', path.name, len(native), len(diag), flush=True)
    nt = np.array([x[0] for x in native], np.int64)
    order = np.argsort(nt, kind='stable')
    return dict(t=nt[order], received=np.array([x[1] for x in native], np.int64)[order],
                q=np.array([x[2] for x in native])[order],
                v=np.array([x[3] for x in native])[order]), diag


def diagnostics_summary(entries):
    values = [e[3] for e in entries]
    result = {k: stats([v[k] * 1000 for v in values if v.get(k) is not None])
              for k in DIAG_KEYS if not k.endswith('_ns')}
    result['units'] = 'all timing statistics in milliseconds'
    result['state_minus_command_ms'] = stats([(v['state_source_stamp_ns'] - v['command_source_stamp_ns']) / 1e6
                                               for v in values if v.get('state_source_stamp_ns') and v.get('command_source_stamp_ns')])
    result['bag_or_recorder_minus_diag_header_ms'] = stats([(r - h) / 1e6 for h, r, _, _ in entries])
    result['messages'] = {key: sum(e[2] == key for e in entries) for key in sorted(set(e[2] for e in entries))}
    result['age_identity_max_error_ns'] = max((abs((h - v['state_source_stamp_ns']) - v['state_source_age_seconds'] * 1e9)
                                               for h, _, _, v in entries if v.get('state_source_age_seconds') is not None), default=None)
    for source in ('state', 'command'):
        result[source + '_header_to_callback_approx_ms'] = stats([
            (v[source + '_source_age_seconds'] - v[source + '_receive_age_seconds']) * 1000
            for v in values if v.get(source + '_source_age_seconds') is not None
            and v.get(source + '_receive_age_seconds') is not None])
    return result


def add_native(episodes, native, diag):
    # Select the exact command source stamp recorded by the adapter, not nearest
    # position values and not the sparse final trajectory target.
    mapping = {}
    for _, _, _, v in diag:
        mapping.setdefault(int(v['state_source_stamp_ns']), int(v['command_source_stamp_ns']))
    residual, stamp_skew = [], []
    matches = missing = 0
    for ep in episodes:
        cmd_stamps = np.array([mapping.get(int(t), -1) for t in ep['t']], np.int64)
        idx = np.searchsorted(native['t'], cmd_stamps)
        good = (idx < len(native['t'])) & (cmd_stamps > 0)
        good &= native['t'][np.minimum(idx, len(native['t']) - 1)] == cmd_stamps
        ep['native_v'] = np.full_like(ep['c'], np.nan)
        ep['native_v'][good] = native['v'][idx[good]]
        residual.extend((ep['c'][good] - native['q'][idx[good]]).tolist())
        stamp_skew.extend(((ep['t'][good] - cmd_stamps[good]) / 1e6).tolist())
        matches += int(good.sum()); missing += int((~good).sum())
    return dict(exact_native_matches=matches, missing_native_matches=missing,
                max_abs_qcmd_rounding_error_rad=np.max(abs(np.array(residual)), axis=0),
                state_minus_selected_native_stamp_ms=stats(stamp_skew))


def regular_runs(episodes, cfg):
    """Interpolation is for analysis only; never bridges a gap or event boundary."""
    out = []
    for ep in episodes:
        cuts = np.r_[0, np.where(np.diff(ep['t']) * 1e-9 > cfg['max_gap_s'])[0] + 1, len(ep['t'])]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi - lo < 70: continue
            t = (ep['t'][lo:hi] - ep['t'][lo]) * 1e-9
            grid = np.arange(0, t[-1] + 1e-9, cfg['dt_s'])
            item = dict(t=grid, kind=ep['kind'], event=ep['event'])
            for key in ('q', 'v', 'c', 'tau', 'pred', 'native_v'):
                if key not in ep: continue
                item[key] = np.stack([np.interp(grid, t, ep[key][lo:hi, j]) for j in range(7)], axis=1)
            n = cfg['smooth_samples']
            for key in ('q', 'v', 'c', 'tau', 'pred', 'native_v'):
                if key in item:
                    item[key + '_smooth'] = signal.savgol_filter(item[key], n, cfg['polynomial_order'], axis=0)
            for key in ('q', 'c'):
                item[key + '_dot'] = signal.savgol_filter(item[key], n, cfg['polynomial_order'],
                                                         deriv=1, delta=cfg['dt_s'], axis=0)
            if 'pred' in item:
                item['pred_dot'] = signal.savgol_filter(item['pred'], n, cfg['polynomial_order'],
                                                        deriv=1, delta=cfg['dt_s'], axis=0)
            out.append(item)
    return out


PAIRS = dict(velocity_vs_position=('q_dot', 'v_smooth', True),
             command_tracking=('c_dot', 'q_dot', True),
             prediction_vs_torque=('pred_smooth', 'tau_smooth', False))


def pair_samples(runs, pair, joint, lag_ms, cfg):
    ref_key, response_key, _ = PAIRS[pair]
    xs, ys, groups = [], [], []
    max_lag = cfg.get('tracking_max_lag_ms', cfg['max_lag_ms']) if pair == 'command_tracking' else cfg['max_lag_ms']
    guard = max_lag * .001 + cfg['smooth_samples'] // 2 * cfg['dt_s'] + .02
    for group, run in enumerate(runs):
        if ref_key not in run: continue
        t = run['t']
        mask = (t >= guard) & (t <= t[-1] - guard)
        if pair == 'prediction_vs_torque':
            mask &= abs(run['pred_dot'][:, joint]) > cfg['torque_active_slope_nm_s']
        else:
            mask &= abs(run[ref_key][:, joint]) > np.deg2rad(cfg['active_speed_deg_s'])
        # Same reference rows at every lag. No extrapolation or crossing gaps.
        idx = np.where(mask)[0]
        if not len(idx): continue
        x = run[ref_key][idx, joint]
        y = np.interp(t[idx] + lag_ms * .001, t, run[response_key][:, joint])
        good = np.isfinite(x) & np.isfinite(y)
        xs.extend(x[good]); ys.extend(y[good]); groups.extend([group] * int(good.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(groups)


def error_metrics(x, y, gain=1., offset=0.):
    if not len(x): return {'n': 0}
    err = y - (gain * x + offset)
    return dict(n=len(x), rmse=float(np.sqrt(np.mean(err**2))), bias=float(err.mean()),
                p95=float(np.percentile(abs(err), 95)),
                correlation=float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-12 else None)


def fit_lags(runs, cfg):
    fitted = {}
    for pair, (_, _, affine) in PAIRS.items():
        max_lag = cfg.get('tracking_max_lag_ms', cfg['max_lag_ms']) if pair == 'command_tracking' else cfg['max_lag_ms']
        lags = np.arange(-max_lag, max_lag + 1, cfg['lag_step_ms'])
        fitted[pair] = []
        for j in range(7):
            losses, params, group_costs = [], [], []
            x, y, g = pair_samples(runs, pair, j, 0, cfg)
            if len(x) < cfg['min_pair_rows'] or x.std() < 1e-5:
                fitted[pair].append(dict(joint=j + 1, status='insufficient_excitation', lag_ms=None)); continue
            groups = np.unique(g)
            for lag in lags:
                a, b, _ = pair_samples(runs, pair, j, lag, cfg)
                gain, offset = np.linalg.lstsq(np.column_stack((a, np.ones(len(a)))), b, rcond=None)[0] if affine else (1., 0.)
                err2 = (b - (gain * a + offset))**2
                losses.append(err2.mean()); params.append((float(gain), float(offset)))
                group_costs.append([err2[g == group].sum() for group in groups])
            best = int(np.argmin(losses)); lag = int(lags[best]); gain, offset = params[best]
            x1, y1, _ = pair_samples(runs, pair, j, lag, cfg)
            counts = np.array([sum(g == group) for group in groups])
            costs = np.array(group_costs)
            rng = np.random.default_rng(cfg['bootstrap_seed'] + j)
            boots = []
            for _ in range(cfg['bootstrap_repeats']):
                take = rng.integers(len(groups), size=len(groups))
                boots.append(int(lags[np.argmin(costs[:, take].sum(axis=1) / counts[take].sum())]))
            near = lags[np.array(losses) <= min(losses) * 1.01 + 1e-15]
            fitted[pair].append(dict(joint=j + 1, status='phase_diagnostic_only', lag_ms=lag,
                gain=gain, offset=offset, lag_grid_ms=lags, loss_curve=np.array(losses),
                near_minimum_1pct_ms=[int(near[0]), int(near[-1])],
                block_bootstrap_95pct_ms=np.percentile(boots, [2.5, 97.5]),
                bootstrap_method='event-cluster resampling of fixed train loss profiles; not a hardware latency CI',
                independent_events=len(groups), search_boundary=best in (0, len(lags) - 1),
                baseline=error_metrics(x, y), shifted=error_metrics(x1, y1),
                baseline_affine=error_metrics(x, y, gain, offset),
                shifted_affine=error_metrics(x1, y1, gain, offset)))
        print('fit train-only:', pair, [v['lag_ms'] for v in fitted[pair]], flush=True)
    return fitted


def validate_lags(runs, frozen):
    result = {}
    cfg = frozen['settings']
    for pair, params in frozen['fits'].items():
        result[pair] = []
        for j, p in enumerate(params):
            if p['lag_ms'] is None:
                result[pair].append(dict(joint=j + 1, status='train_unidentified')); continue
            x0, y0, _ = pair_samples(runs, pair, j, 0, cfg)
            x1, y1, _ = pair_samples(runs, pair, j, p['lag_ms'], cfg)
            base = error_metrics(x0, y0); shifted = error_metrics(x1, y1)
            result[pair].append(dict(joint=j + 1, frozen_lag_ms=p['lag_ms'],
                baseline=base, shifted=shifted,
                baseline_affine=error_metrics(x0, y0, p['gain'], p['offset']),
                shifted_affine=error_metrics(x1, y1, p['gain'], p['offset']),
                raw_rmse_change_percent=100 * (shifted['rmse'] / base['rmse'] - 1) if base.get('rmse', 0) else None,
                status='held_out_diagnostic_no_retuning'))
    return result


def noise(episodes, cfg):
    pieces, means = {k: [] for k in ('q', 'v', 'c', 'tau')}, {k: [] for k in ('q', 'v', 'tau')}
    reports = []
    for ep in episodes:
        if not (ep['kind'].startswith('static_') or ep['kind'] in ('baseline_before', 'baseline_after')): continue
        t = (ep['t'] - ep['t'][0]) * 1e-9
        mask = (t >= cfg['static_trim_s']) & (t <= t[-1] - cfg['static_trim_s'])
        if sum(mask) < 100: continue
        q, c = ep['q'][mask], ep['c'][mask]
        # No velocity-based stationary gate: a quantized velocity may be nonzero
        # even when the encoder position never changes.
        info = dict(event=ep['event'], rows=int(mask.sum()),
                    measured_position_span_deg=np.rad2deg(np.ptp(q, axis=0)),
                    command_span_deg=np.rad2deg(np.ptp(c, axis=0)))
        for k in pieces:
            a = ep[k][mask]
            pieces[k].append(signal.detrend(a, axis=0))
            if k in means: means[k].append(a.mean(axis=0))
        info['velocity_mean_deg_s'] = np.rad2deg(ep['v'][mask].mean(axis=0))
        info['torque_mean_nm'] = ep['tau'][mask].mean(axis=0)
        reports.append(info)
    summary = dict(segments=reports, definition='scheduled holds trimmed by 1 second at each edge; scatter may include real micro-motion, not pure sensor noise',
                   rows=sum(len(x) for x in pieces['q']))
    for k, values in pieces.items():
        if not values: continue
        scale = 180 / np.pi if k in ('q', 'v', 'c') else 1.
        a = np.vstack(values) * scale
        differences = np.vstack([np.diff(x, axis=0) for x in values]) * scale
        summary[k] = dict(units={'q': 'deg', 'v': 'deg/s', 'c': 'deg', 'tau': 'Nm'}[k],
            detrended_std=a.std(axis=0), robust_mad_sigma=1.4826 * np.median(abs(a - np.median(a, axis=0)), axis=0),
            adjacent_difference_sigma=differences.std(axis=0) / np.sqrt(2))
    return summary


def units_audit(episodes, robot):
    rows = []
    hashes = {}
    for j in range(7):
        p = robot / f'src/W3_ROBOT/w3_robot_bridge/config/motor_config/right_arm/can1_id{j+1}.yaml'
        cfg = yaml.safe_load(p.read_text()); hashes[str(p)] = digest(p)
        ranges = cfg['ranges']
        info = dict(joint=j + 1, model=cfg['motor_model'], nominal_units=dict(q='rad', v='rad/s', c='rad', tau='Nm'))
        for key, field, bits in [('q', 'position', 16), ('v', 'velocity', 12), ('tau', 'torque', 12)]:
            step = (ranges[field]['max'] - ranges[field]['min']) / (2**bits - 1)
            a = np.concatenate([ep[key][:, j] for ep in episodes])
            # Position offsets/signs can translate the grid; test grid spacing
            # with differences. float32 decode error is not a second sensor LSB.
            lattice = (a - a[0]) / step
            info[key] = dict(nominal_lsb=step,
                lsb_display_deg_or_deg_s=step * 180 / np.pi if key != 'tau' else None,
                lattice_error_p99_in_counts=np.percentile(abs(lattice - np.rint(lattice)), 99),
                minimum_observed_absolute_value=float(np.min(abs(a))),
                unique_values=len(np.unique(a)), range=[a.min(), a.max()])
            if key in ('v', 'tau'):
                info[key]['half_lsb_at_symmetric_zero'] = step / 2
        rows.append(info)
    return dict(per_joint=rows, configuration_hashes=hashes,
                torque_absolute_calibration='not identifiable: no raw current, independent load-cell, or firmware range readback in these logs',
                hardware_delay='not identifiable: CAN receive times and per-motor last_update are not exported into the recorded state',
                zero_code_note='symmetric uint12 ranges have no exactly zero code; half-LSB is encoding resolution, not proven mechanical motion')


def recorded_source_proof(robot, capture_metadata):
    commit = capture_metadata['software_commits']['w3']
    paths = ['src/W3_ROBOT/w3_robot_bridge/src/damiao_protocol.cpp',
             'src/W3_ROBOT/w3_robot_bridge/src/w3_robot_bridge_node.cpp',
             'src/ieir_controllers/src/dm_hardware_interface.cpp']
    paths += [f'src/W3_ROBOT/w3_robot_bridge/config/motor_config/right_arm/can1_id{j}.yaml' for j in range(1, 8)]
    result = {}
    for p in paths:
        data = subprocess.check_output(['git', 'show', commit + ':' + p], cwd=robot)
        same = data == (robot / p).read_bytes()
        if not same: raise ValueError('source/profile differs from recorded capture commit: ' + p)
        result[p] = dict(sha256=hashlib.sha256(data).hexdigest(), matches_recorded_commit=same)
    return dict(recorded_capture_commit=commit, files=result,
                limitation='source/profile consistency does not read back physical motor firmware configuration')


def checkpoint_predictions(episodes, run):
    import torch
    from factr2_next.inference.checkpoint import load_checkpoint
    from factr2_next.inference.offline_eval import predict_sequence
    torch.set_num_threads(4)
    loaded = load_checkpoint(run)
    for i, ep in enumerate(episodes):
        arrays = {k: ep[s].astype(np.float32) for k, s in zip(KEYS, ('q', 'v', 'c', 'tau'))}
        pred, _ = predict_sequence(loaded, arrays)
        ep['pred'] = np.full_like(ep['tau'], np.nan)
        ep['pred'][loaded.history - 1:] = pred
        if i % 50 == 0: print('fixed checkpoint replay:', i, '/', len(episodes), flush=True)


def cadence(episodes):
    intervals = np.concatenate([np.diff(ep['t']) / 1e6 for ep in episodes])
    return dict(rows=sum(len(ep['t']) for ep in episodes), interval_ms=stats(intervals),
                gaps_over_20ms=int(sum(intervals > 20)),
                full_q_v_tau_repeats=int(sum(np.all(np.diff(np.c_[ep['q'], ep['v'], ep['tau']], axis=0) == 0, axis=1).sum() for ep in episodes)))


def fit(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'frozen.json').exists(): raise ValueError('frozen artifact already exists; use a new output directory')
    audit = json.loads(Path(args.split_audit).read_text())
    capture_metadata_path = Path(audit['source']).with_suffix('.metadata.json')
    capture_metadata = json.loads(capture_metadata_path.read_text())
    source_proof = recorded_source_proof(Path(args.robot), capture_metadata)
    h5 = Path(args.train_h5).resolve()
    if h5.name != 'train.h5': raise ValueError('fit accepts explicit train.h5 only')
    split = json.loads((h5.parent / 'split.json').read_text())
    expected_hash = split['splits']['train'][0]['h5_sha256']
    if digest(h5) != expected_hash: raise ValueError('training hash mismatch')
    episodes = load_train(h5, audit)
    intervals = [(int(e['t'][0]), int(e['t'][-1])) for e in episodes]
    native, diag = read_bag(args.bag, intervals)
    native_match = add_native(episodes, native, diag)
    checkpoint_predictions(episodes, args.checkpoint)
    runs = regular_runs(episodes, SETTINGS)
    derivative_check = []
    for j in range(7):
        xs, ys = [], []
        for run in runs:
            mask = (run['t'] > .12) & (run['t'] < run['t'][-1] - .12)
            mask &= np.isfinite(run['native_v_smooth'][:, j])
            xs.extend(run['c_dot'][mask, j]); ys.extend(run['native_v_smooth'][mask, j])
        derivative_check.append(error_metrics(np.array(xs), np.array(ys)))
    filter_cfg = yaml.safe_load((ROOT / 'log/deploy_pose_trial_20261015/inference.yaml').read_text())
    artifact = dict(schema='w3_signal_audit_frozen_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='fit: train.h5 only; no val.h5, test.h5 or independent CSV opened',
        settings=SETTINGS, script_sha256=digest(__file__),
        training=dict(path=str(h5), sha256=digest(h5), split_audit_sha256=digest(args.split_audit),
                      intervals_ns=intervals, checkpoint_dir=str(Path(args.checkpoint).resolve()),
                      checkpoint_sha256=digest(Path(args.checkpoint) / 'model.pt'),
                      bag_files={str(p.resolve()): dict(sha256=digest(p), bytes=p.stat().st_size)
                                 for p in sorted(Path(args.bag).glob('*.db3'))}),
        lag_sign='positive L: response(t+L) compared with reference(t); grid resolution 10ms; not absolute sensor latency',
        timing=diagnostics_summary(diag), native_command_match=native_match,
        native_command_velocity_derivative_check=derivative_check,
        original_output_filter=filter_cfg.get('smoothing', {}),
        runtime_inference_config_sha256=digest(ROOT / 'log/deploy_pose_trial_20261015/inference.yaml'),
        recorded_source_proof=source_proof,
        native_header_to_bag_ms=stats((native['received'] - native['t']) / 1e6),
        units=units_audit(episodes, Path(args.robot)), cadence=cadence(episodes), noise=noise(episodes, SETTINGS),
        fits=fit_lags(runs, SETTINGS),
        torque_delay_identifiability='prediction_vs_torque is fixed-model phase error; cannot separate sensor delay from model/controller/friction dynamics',
        runtime_changes=False)
    # Verify the source files did not change during this computation.
    assert digest(h5) == expected_hash
    save(out / 'frozen.json', artifact)
    (out / 'frozen.sha256').write_text(digest(out / 'frozen.json') + '\n')
    print('FROZEN', out / 'frozen.json', digest(out / 'frozen.json'), flush=True)


def read_csv(path, filter_alpha):
    with Path(path).open() as f: rows = list(csv.DictReader(f))
    t = np.array([int(r['source_timestamp_ns']) for r in rows], np.int64)
    assert np.all(np.diff(t) > 0), 'independent timestamp nonmonotonic'
    arrays = {}
    for key, prefix, scale in [('q', 'q_deg', np.pi / 180), ('v', 'velocity_deg_s', np.pi / 180),
                               ('c', 'command_deg', np.pi / 180), ('tau', 'measured_tau_nm', 1),
                               ('pred', 'free_pred_tau_nm', 1)]:
        arrays[key] = np.array([[float(r[f'{prefix}_j{j+1}']) for j in range(7)] for r in rows]) * scale
    cuts = [0] + [i for i in range(1, len(rows)) if rows[i]['phase'] != rows[i-1]['phase']
                or rows[i]['step_index'] != rows[i-1]['step_index']] + [len(rows)]
    episodes = [dict(t=t[a:b], kind=rows[a]['phase'], event=rows[a]['phase'] + '_' + rows[a]['step_index'],
                     **{k: v[a:b] for k, v in arrays.items()}) for a, b in zip(cuts[:-1], cuts[1:])]
    age = (np.array([int(r['received_wall_ns']) for r in rows], np.int64) - t) / 1e6
    raw = np.array([[float(r[f'external_raw_nm_j{j+1}']) for j in range(7)] for r in rows])
    filtered = np.array([[float(r[f'external_filtered_nm_j{j+1}']) for j in range(7)] for r in rows])
    continuous = (np.diff(t) >= 5_000_000) & (np.diff(t) <= 15_000_000)
    predicted_filter = filter_alpha * raw[1:] + (1-filter_alpha) * filtered[:-1]
    fd = np.diff(arrays['c'], axis=0) / (np.diff(t) * 1e-9)[:, None]
    recorded_fd = np.array([[float(r[f'command_velocity_fd_deg_s_j{j+1}']) for j in range(7)] for r in rows[1:]]) * np.pi/180
    identities = dict(raw_equals_measured_minus_prediction_max_error_nm=np.max(abs(raw - (arrays['tau'] - arrays['pred'])), axis=0),
        ema_alpha_from_frozen_config=filter_alpha, continuous_ema_rows=int(continuous.sum()),
        ema_recurrence_max_error_nm=np.max(abs(filtered[1:][continuous] - predicted_filter[continuous]), axis=0),
        command_fd_column_check_max_error_rad_s=np.max(abs(fd - recorded_fd), axis=0),
        ema_low_frequency_group_delay_ms=1000 * (1-filter_alpha) / filter_alpha * .01,
        limitation='joined receive time includes inference, synchronizer and recorder callback; not pure torque sensor transport')
    return episodes, age, identities


def validate(args):
    path = Path(args.frozen)
    before = digest(path)
    if before != path.with_suffix('.sha256').read_text().strip(): raise ValueError('frozen hash mismatch')
    frozen = json.loads(path.read_text())
    if digest(__file__) != frozen['script_sha256']: raise ValueError('audit implementation changed after freeze')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation.json').exists(): raise ValueError('validation already exists; no overwrite')
    episodes, age, identities = read_csv(args.csv, frozen['original_output_filter']['ema_alpha'])
    all_t = np.concatenate([e['t'] for e in episodes])
    if contains(all_t, frozen['training']['intervals_ns']).any(): raise ValueError('validation overlaps training times')
    diag = []
    with Path(args.diagnostics).open() as f:
        for line in f:
            item = json.loads(line)
            if item['topic'] != '/factr2/right/adapter_status': continue
            for s in item['status']:
                if s['name'] == 'factr2/adapter/right':
                    diag.append((int(item['source_timestamp_ns']), int(item['received_wall_ns']), s['message'], s['values']))
    report = dict(schema='w3_signal_audit_validation_v1', frozen_sha256=before,
        source_csv=str(Path(args.csv).resolve()), source_csv_sha256=digest(args.csv),
        diagnostics_sha256=digest(args.diagnostics), independent_session=True, fit_or_parameter_search=False,
        timing=diagnostics_summary(diag), joined_receive_minus_source_ms=stats(age),
        algebraic_and_filter_checks=identities,
        cadence=cadence(episodes), noise=noise(episodes, frozen['settings']),
        fits=validate_lags(regular_runs(episodes, frozen['settings']), frozen))
    assert digest(path) == before, 'frozen parameters mutated'
    save(out / 'validation.json', report)
    make_plots(frozen, report, out)
    print('VALIDATED WITHOUT REFIT', out / 'validation.json', flush=True)


def make_plots(frozen, report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for row, pair in enumerate(PAIRS):
        for p in frozen['fits'][pair]:
            if p['lag_ms'] is None: continue
            loss = np.array(p['loss_curve'])
            axes[row, 0].plot(p['lag_grid_ms'], loss / max(min(loss), 1e-15), label='J' + str(p['joint']))
        axes[row, 0].set(title=pair + ': TRAIN loss only', xlabel='Response lag (ms)', ylabel='MSE / minimum')
        axes[row, 0].legend(ncol=4)
        values = report['fits'][pair]
        x = np.arange(1, 8)
        axes[row, 1].bar(x - .17, [p.get('baseline', {}).get('rmse', np.nan) for p in values], .34, label='zero lag')
        axes[row, 1].bar(x + .17, [p.get('shifted', {}).get('rmse', np.nan) for p in values], .34, label='frozen train lag')
        axes[row, 1].set(title='Independent log: raw error (no gain/bias correction)', xlabel='Joint', ylabel='rad/s' if row < 2 else 'Nm', xticks=x)
        axes[row, 1].legend()
    fig.suptitle('Phase diagnostics, NOT hardware sampling delays; no held-out lag search')
    fig.tight_layout(); fig.savefig(out / 'delay_validation.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for data, label in [(frozen, 'train scheduled holds'), (report, 'independent holds')]:
        axes[0].plot(range(1, 8), data['noise']['v']['detrended_std'], 'o-', label=label)
        axes[1].plot(range(1, 8), data['noise']['tau']['detrended_std'], 'o-', label=label)
    axes[0].set(xlabel='Joint', ylabel='Velocity scatter (deg/s)', title='Detrended within each hold')
    axes[1].set(xlabel='Joint', ylabel='Torque scatter (Nm)', title='Contains sensor noise + real micro-motion')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'static_scatter.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    f = sub.add_parser('fit')
    for key in ('train-h5', 'split-audit', 'bag', 'checkpoint', 'robot', 'output'):
        f.add_argument('--' + key, required=True)
    v = sub.add_parser('validate')
    for key in ('frozen', 'csv', 'diagnostics', 'output'):
        v.add_argument('--' + key, required=True)
    args = parser.parse_args()
    (fit if args.mode == 'fit' else validate)(args)


if __name__ == '__main__': main()
