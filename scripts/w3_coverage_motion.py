#!/usr/bin/python3.10
"""Plan, validate, execute and assess free-space coverage. Only run --send moves hardware."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

from w3_coverage_plan import ArmModel, build_plan, load_coverage, summary
from w3_motion_sequence import MotionClient, read_active, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def live_snapshot(side):
    """Read-only snapshot; does not require C, enable motors or publish trajectories."""
    from rcl_interfaces.srv import GetParameters
    client = MotionClient(side)
    try:
        client.check_mode()
        request = GetParameters.Request()
        request.names = ['robot_description']
        reply = client.call(GetParameters, '/robot_state_publisher/get_parameters', request)
        if len(reply.values) != 1 or reply.values[0].type != 4:
            raise RuntimeError('No live robot_description')
        client.wait_until(lambda: client.fresh('q') and client.fresh('cmd'), 'fresh q and command')
        initial = np.rad2deg(client.state['cmd']).tolist()
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            client.ros.spin_once(client.node, timeout_sec=.02)
            if not client.fresh('q') or not client.fresh('cmd'):
                raise RuntimeError('Stale state while planning')
            if np.max(np.abs(np.rad2deg(client.state['cmd']) - initial)) > .05:
                raise RuntimeError('Existing trajectory is moving; wait before planning')
        return reply.values[0].string_value, initial
    finally:
        client.close()


def window_indices(times, elapsed, horizon=.6):
    """Include preceding samples so controller interpolates the same curve at replacement."""
    start = max(0, int(np.searchsorted(times, elapsed, side='right')) - 2)
    end = min(len(times), max(start + 2, int(np.searchsorted(times, elapsed + horizon)) + 1))
    return start, end


class CoverageClient(MotionClient):
    def __init__(self, plan):
        super().__init__(plan['side'])
        self.plan = plan
        self.expected_deg = None
        self.hardware_bounds = ArmModel(plan['robot_description'], plan['side'], plan['config']['tool_frame']).limits_deg

    def guard(self):
        super().guard()
        config = self.plan['config']
        measured = np.rad2deg(self.state['q'])
        if np.any(measured < self.hardware_bounds[:, 0]) or np.any(measured > self.hardware_bounds[:, 1]):
            raise RuntimeError('Measured joint angle exceeds hardware/model limits')
        if np.max(np.abs(np.rad2deg(np.array(self.state['q']) - self.state['cmd']))) > config['tracking_limit_deg']:
            raise RuntimeError('Configured tracking error limit exceeded')
        # Detect another command source or a controller holding after a lost window.
        if self.expected_deg is not None and np.max(np.abs(np.rad2deg(self.state['cmd']) - self.expected_deg)) > .5:
            raise RuntimeError('Controller target differs from scheduled path by >0.5 degrees')

    def play(self, step, capture_guard):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        self.check_mode()
        self.guard()
        q = np.asarray(step['positions_deg'])
        times = np.asarray(step['times_seconds'])
        if np.max(np.abs(np.rad2deg(self.state['cmd']) - q[0])) > .05:
            raise RuntimeError('Controller start target differs from frozen trajectory')
        if self.publisher is None:
            self.publisher = self.node.create_publisher(JointTrajectory, '/joint_position_command', 10)
            self.wait_until(lambda: self.publisher.get_subscription_count() > 0, 'trajectory subscriber', 3)
        self.guard()
        if np.max(np.abs(np.rad2deg(self.state['cmd']) - q[0])) > .05:
            raise RuntimeError('Controller target changed during publisher discovery')
        base_ns = self.node.get_clock().now().nanoseconds + 150_000_000
        last_publish, final_published = -1e9, False
        # All windows share one absolute ROS start time. A stall >0.2s aborts;
        # previously published future motion is limited to about 0.6s.
        last_tick = time.monotonic()
        while True:
            self.ros.spin_once(self.node, timeout_sec=.01)
            now_tick = time.monotonic()
            if now_tick - last_tick > .2:
                raise RuntimeError('Motion scheduler stalled >200ms; stop sending windows')
            last_tick = now_tick
            elapsed = (self.node.get_clock().now().nanoseconds - base_ns) / 1e9
            if elapsed < -.2:
                raise RuntimeError('ROS clock moved backwards')
            command_elapsed = (self.state.get('cmd_stamp', base_ns) - base_ns) / 1e9
            self.expected_deg = np.array([np.interp(max(0, command_elapsed), times, q[:, i]) for i in range(7)])
            self.guard()
            capture_guard()
            if not final_published and now_tick - last_publish >= .15:
                start, end = window_indices(times, max(0, elapsed))
                message = JointTrajectory()
                message.joint_names = self.names
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(base_ns, 1_000_000_000)
                for index in range(start, end):
                    point = JointTrajectoryPoint()
                    point.positions = np.deg2rad(q[index]).tolist()
                    ns = round(times[index] * 1e9)
                    point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
                    message.points.append(point)
                self.publisher.publish(message)
                last_publish = now_tick
                final_published = end == len(times)
            if elapsed >= times[-1] + step['hold_seconds']:
                break
        return self.endpoint(np.deg2rad(q[-1]).tolist(), self.plan['config']['arrival_tolerance_deg'])


def check_start(plan, initial, description, command):
    if hashlib.sha256(description.encode()).hexdigest() != plan['model_sha256']:
        raise RuntimeError('Live robot_description changed; generate a new plan')
    if np.max(np.abs(np.rad2deg(initial) - plan['initial_deg'])) > plan['config']['start_tolerance_deg']:
        raise RuntimeError('Measured start pose changed; generate a new plan')
    if np.max(np.abs(np.rad2deg(command) - plan['initial_deg'])) > .05:
        raise RuntimeError('Controller start target changed; generate a new plan')


def execute(plan, client, capture_guard, report, save):
    for index, step in enumerate(plan['steps'], 1):
        capture_guard()
        client.guard()
        event = {k: step[k] for k in ('name', 'phase', 'kind', 'speed', 'repetition', 'joint')}
        event.update(index=index, utc=utc_now(), status='sending')
        report['events'].append(event)
        save()
        print(f'[{index}/{len(plan["steps"])}] {step["name"]}: {step["duration_seconds"]:.2f}s', flush=True)
        event['arrival_error_deg'] = client.play(step, capture_guard)
        event.update(status='completed', completed_utc=utc_now())
        save()
    report.update(status='completed', completed_utc=utc_now())
    save()


def send(plan, digest):
    active = read_active(plan['side'], digest)
    session = Path(active['session'])
    path = session / 'audit/motion_run.json'
    with (session / 'audit/motion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError('This capture already attempted motion; start a new C session')
        report = {'status': 'preflight', 'motion_sha256': digest, 'motion_name': plan['name'],
                  'side': plan['side'], 'started_utc': utc_now(), 'events': []}
        save = lambda: write_json(path, report)
        save()
        client = None
        try:
            def capture_guard():
                if read_active(plan['side'], digest) != active:
                    raise RuntimeError('Capture ended or changed')
            client = CoverageClient(plan)
            initial = client.preflight()
            check_start(plan, initial, client.description, client.state['cmd'])
            capture_guard()
            report.update(status='running', initial_feedback_deg=np.rad2deg(initial).tolist())
            save()
            execute(plan, client, capture_guard, report, save)
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=str(exc) or 'Ctrl-C', completed_utc=utc_now())
            save()
            raise
        finally:
            if client is not None:
                client.close()
    print(f'完成：{path}。回到 C 按 Enter 停录并检查实测覆盖。')


def coverage_report(session):
    """Measured coverage is descriptive; successful motion does not certify training validity."""
    import h5py
    from datetime import datetime
    plan, _, digest = load_coverage(session / 'audit/motion.yaml')
    model = ArmModel(plan['robot_description'], plan['side'], plan['config']['tool_frame'])
    files = sorted(session.glob('*.h5'))
    if len(files) != 1:
        raise ValueError('Expected one H5 in session')
    run = json.loads((session / 'audit/motion_run.json').read_text())
    episodes = []
    with h5py.File(files[0], 'r') as handle:
        for name in handle:
            group = handle[name]
            if len(group) == 0:
                continue  # A final empty episode can be created at a recorder boundary.
            q = group['joint_pos/data'][:]
            cmd = group['joint_cmd/data'][:]
            vel = group['joint_vel/data'][:]
            stamp = group['joint_pos/timestamps'][:].reshape(-1).astype(np.float64) * 1e-9
            # Writer timestamps are nanoseconds; keep episodes separate across dropped frames.
            if len(q):
                episodes.append((name, stamp, np.rad2deg(q), np.rad2deg(cmd), np.rad2deg(vel)))
    if not episodes:
        raise ValueError('No recorded rows')
    samples_by_speed = {'slow': [], 'fast': []}
    all_q = np.vstack([item[2] for item in episodes])
    bounds = np.asarray(plan['config']['working_bounds_deg'])
    report = {'motion_sha256': digest, 'motion_completed': run.get('status') == 'completed' and run.get('motion_sha256') == digest,
              'working_bounds_deg': bounds.tolist(), 'measured_min_deg': all_q.min(axis=0).tolist(),
              'measured_max_deg': all_q.max(axis=0).tolist(), 'events': [],
              'interpretation': 'Coverage only; also inspect quality.json, free-space assumption and train/val/test sessions.'}
    for event in run['events']:
        start = datetime.fromisoformat(event['utc']).timestamp()
        end = datetime.fromisoformat(event.get('completed_utc', run['completed_utc'])).timestamp()
        selected = [(q[(stamp >= start) & (stamp <= end)], cmd[(stamp >= start) & (stamp <= end)],
                     vel[(stamp >= start) & (stamp <= end)]) for _, stamp, q, cmd, vel in episodes]
        selected = [item for item in selected if len(item[0])]
        entry = {key: event[key] for key in ('name', 'phase', 'kind', 'speed', 'status')}
        entry['rows'] = sum(len(item[0]) for item in selected)
        if selected:
            q, cmd, vel = [np.vstack([item[i] for item in selected]) for i in range(3)]
            if event['phase'] == 'single_joint':
                samples_by_speed[event['speed']].append((q, vel))
            entry.update(actual_range_deg=np.ptp(q, axis=0).tolist(),
                         tracking_rmse_deg=np.sqrt(np.mean((q-cmd)**2, axis=0)).tolist(),
                         speed_p95_deg_s=np.percentile(np.abs(vel), 95, axis=0).tolist())
            if event['phase'] == 'cartesian':
                xyz = np.array([model.fk(np.deg2rad(row))[:3, 3] for row in q[::5]])
                entry['measured_ee_extent_m'] = np.ptp(xyz, axis=0).tolist()
        report['events'].append(entry)
    report['single_joint_coverage_by_speed'] = {}
    for speed, samples in samples_by_speed.items():
        if samples:
            q = np.vstack([item[0] for item in samples])
            vel = np.vstack([item[1] for item in samples])
            report['single_joint_coverage_by_speed'][speed] = {
                'actual_min_deg': q.min(axis=0).tolist(), 'actual_max_deg': q.max(axis=0).tolist(),
                'range_fraction_of_working_span': (np.ptp(q, axis=0)/(bounds[:, 1]-bounds[:, 0])).tolist(),
                'speed_p95_deg_s': np.percentile(np.abs(vel), 95, axis=0).tolist()}
    write_json(session / 'coverage.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    planner = sub.add_parser('plan', help='Generate a frozen plan, never publish motion')
    planner.add_argument('--config', type=Path, default=ROOT / 'config/w3/motions/left_coverage_conservative.yaml')
    source = planner.add_mutually_exclusive_group(required=True)
    source.add_argument('--live', action='store_true', help='Read current ROS model and stationary command')
    source.add_argument('--urdf', type=Path, help='Expanded URDF with ros2_control limits')
    planner.add_argument('--initial-deg', nargs=7, type=float)
    planner.add_argument('--output', required=True, type=Path)
    runner = sub.add_parser('run', help='Default is offline validation only')
    runner.add_argument('--file', required=True, type=Path)
    runner.add_argument('--send', action='store_true')
    reporter = sub.add_parser('report')
    reporter.add_argument('--session', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'plan':
            if args.output.exists():
                raise ValueError('Output exists; use a new filename to preserve frozen plans')
            config = yaml.safe_load(args.config.read_text())
            if args.live:
                if args.initial_deg is not None:
                    raise ValueError('--live reads controller target; omit --initial-deg')
                description, initial = live_snapshot(config['side'])
            else:
                if args.initial_deg is None:
                    raise ValueError('--urdf requires --initial-deg (seven absolute angles)')
                description, initial = args.urdf.read_text(), args.initial_deg
            plan = build_plan(config, description, initial)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            raw = yaml.dump(plan, Dumper=getattr(yaml, 'CSafeDumper', yaml.SafeDumper), sort_keys=False)
            with args.output.open('x') as handle:
                handle.write(raw)
            write_json(args.output.with_suffix('.summary.json'), summary(plan))
            print(json.dumps(summary(plan), indent=2, ensure_ascii=False))
            print(f'计划已保存：{args.output}；没有发送动作。')
        elif args.action == 'run':
            plan, _, digest = load_coverage(args.file)
            print(json.dumps(summary(plan), indent=2, ensure_ascii=False))
            if args.send:
                send(plan, digest)
            else:
                print('离线检查 PASS；未连接 ROS，未发送动作。')
        else:
            report = coverage_report(args.session)
            print(f'实测覆盖：{args.session / "coverage.json"}\n实测范围(deg)：{np.round(np.array(report["measured_max_deg"])-report["measured_min_deg"], 3)}')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'未完成：{exc or "Ctrl-C"}', flush=True)
        if args.action == 'run' and args.send:
            print('停止后续发送。已发送的小段可能继续约 0.8 秒；本工具不是硬件急停，请按现场停车流程处理并回 C 停录。')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
