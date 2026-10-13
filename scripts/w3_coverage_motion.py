#!/usr/bin/python3.10
"""Plan, validate, execute and assess free-space coverage. Only run --send moves hardware."""
import argparse
import fcntl
import hashlib
import json
import math
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

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


def trajectory_derivatives(times, positions_deg):
    """One C2 spline for the entire recording, independent of window boundaries."""
    from scipy.interpolate import CubicSpline
    times=np.asarray(times,dtype=float);q=np.asarray(positions_deg,dtype=float)
    curve=CubicSpline(times,q,axis=0,bc_type=((1,np.zeros(q.shape[1])),(1,np.zeros(q.shape[1]))))
    velocity=curve(times,1);acceleration=curve(times,2)
    velocity[[0,-1]]=0.;acceleration[[0,-1]]=0.
    return velocity,acceleration


def window_indices(times, elapsed, horizon=.6):
    """Include preceding samples so controller interpolates the same curve at replacement."""
    start = max(0, int(np.searchsorted(times, elapsed, side='right')) - 2)
    end = min(len(times), max(start + 2, int(np.searchsorted(times, elapsed + horizon)) + 1))
    return start, end


def scheduled_position(times, positions, elapsed):
    """One bracket lookup; avoid copying seven entire strided path columns."""
    index = int(np.searchsorted(times, elapsed, side='right'))
    if index == 0:
        return positions[0].copy()
    if index == len(times):
        return positions[-1].copy()
    fraction = (elapsed-times[index-1])/(times[index]-times[index-1])
    return positions[index-1] + fraction*(positions[index]-positions[index-1])


class CoverageClient(MotionClient):
    def __init__(self, plan, stationary_left_deg=None):
        super().__init__(plan['side'])
        self.plan = plan
        self.expected_deg = None
        self.hardware_bounds = ArmModel(plan['robot_description'], plan['side'], plan['config']['tool_frame']).limits_deg
        loaded = {j.get('name') for j in ET.fromstring(plan['robot_description']).findall('./ros2_control/joint')}
        self.stationary_left_deg = (np.asarray(stationary_left_deg, float)
            if plan['side'] == 'right' and {f'left_joint_{i}' for i in range(7)} <= loaded else None)
        if self.stationary_left_deg is not None and (self.stationary_left_deg.shape != (7,) or
                not np.isfinite(self.stationary_left_deg).all()):
            raise ValueError('Loaded left arm requires its collision-audited static pose')
        if self.stationary_left_deg is not None:
            from diagnostic_msgs.msg import DiagnosticArray
            from rclpy.qos import qos_profile_sensor_data
            self.node.create_subscription(DiagnosticArray, '/factr2/w3_health',
                                          self.left_health, qos_profile_sensor_data)

    def left_health(self, msg):
        self.state['left_healthy'] = False
        for status in msg.status:
            if status.name == 'factr2/w3/left':
                self.state['left_healthy'] = status.level in (0, b'\x00')
                self.state['left_health_received'] = time.monotonic()
                self.state['left_health_stamp'] = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec

    def joints(self, msg, key):
        super().joints(msg, key)
        if getattr(self, 'stationary_left_deg', None) is None:
            return
        names = [f'left_joint_{i}' for i in range(7)]
        values = dict(zip(msg.name, msg.position))
        mapped = 'left_q' if key == 'q' else 'left_cmd'
        if (len(msg.name) != len(msg.position) or len(msg.name) != len(set(msg.name))
                or not all(n in values and math.isfinite(values[n]) for n in names)):
            self.state.pop(mapped, None)
            return
        self.state[mapped] = [values[n] for n in names]
        self.state[mapped+'_received'] = time.monotonic()
        self.state[mapped+'_stamp'] = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec

    def health(self, msg):
        super().health(msg)
        for diagnostic in msg.status:
            if diagnostic.name == f'factr2/adapter/{self.side}':
                values = {v.key:v.value for v in diagnostic.values}
                self.state['adapter_hz'] = float(values.get('output_hz_last_second', '0'))
                self.state['adapter_reason'] = diagnostic.message

    def guard(self):
        super().guard()
        if getattr(self, 'stationary_left_deg', None) is not None:
            self.guard_stationary_left()
        config = self.plan['config']
        measured = np.rad2deg(self.state['q'])
        if np.any(measured < self.hardware_bounds[:, 0]) or np.any(measured > self.hardware_bounds[:, 1]):
            raise RuntimeError('Measured joint angle exceeds hardware/model limits')
        tracking = np.abs(np.rad2deg(np.array(self.state['q']) - self.state['cmd']))
        self.monitor_max_tracking = np.maximum(getattr(self, 'monitor_max_tracking', np.zeros(7)), tracking)
        if np.max(tracking) > config['tracking_limit_deg']:
            joint = int(np.argmax(tracking))
            raise RuntimeError(f'Configured tracking error limit exceeded: J{joint+1} '
                f'{tracking[joint]:.3f} deg > {config["tracking_limit_deg"]:.3f} deg; '
                f'measured={measured[joint]:.3f} deg, '
                f'target={np.rad2deg(self.state["cmd"])[joint]:.3f} deg')
        # Detect another command source or a controller holding after a lost window.
        if self.expected_deg is not None and np.max(np.abs(np.rad2deg(self.state['cmd']) - self.expected_deg)) > .5:
            raise RuntimeError('Controller target differs from scheduled path by >0.5 degrees')

    def guard_stationary_left(self):
        if not all(self.fresh(key) for key in ('left_q','left_cmd','left_health')):
            raise RuntimeError('Loaded left arm feedback/target stale; collision pose unknown')
        if not self.state.get('left_healthy'):
            raise RuntimeError('Loaded left arm motor feedback unhealthy; collision pose unknown')
        for key in ('left_q','left_cmd'):
            if np.max(np.abs(np.rad2deg(self.state[key])-self.stationary_left_deg)) > 3.:
                raise RuntimeError('Left arm moved beyond the audited static pose; stop sending windows')

    def play(self, step, capture_guard, on_tick=None):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        # Preparing a long campaign spline can take longer than the feedback
        # freshness window. Do it before the final ROS preflight, then consume
        # fresh inputs. No motion has been published during this preparation.
        q = np.asarray(step['positions_deg'])
        times = np.asarray(step['times_seconds'])
        velocity,acceleration=trajectory_derivatives(times,q)
        positions_rad, velocity_rad, acceleration_rad = map(np.deg2rad, (q, velocity, acceleration))
        self.check_mode()
        self.wait_until(self.ready, 'fresh feedback after trajectory preparation')
        self.guard()
        capture_guard()
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
        previous_phases = {}
        self.scheduler_monitor = dict(max_gap_ms=0., phase_max_ms={})
        while True:
            spin_start = time.monotonic()
            self.ros.spin_once(self.node, timeout_sec=.01)
            now_tick = time.monotonic()
            gap_ms = (now_tick-last_tick)*1000
            spin_ms = (now_tick-spin_start)*1000
            self.scheduler_monitor['max_gap_ms'] = max(self.scheduler_monitor['max_gap_ms'],gap_ms)
            if now_tick - last_tick > .2:
                raise RuntimeError(f'Motion scheduler stalled >200ms ({gap_ms:.1f}ms); '
                    f'spin={spin_ms:.1f}ms, previous phases={previous_phases}; stop sending windows')
            last_tick = now_tick
            phases = dict(spin=spin_ms)
            phase_start = now_tick
            elapsed = (self.node.get_clock().now().nanoseconds - base_ns) / 1e9
            if elapsed < -.2:
                raise RuntimeError('ROS clock moved backwards')
            command_elapsed = (self.state.get('cmd_stamp', base_ns) - base_ns) / 1e9
            self.expected_deg = scheduled_position(times, q, max(0, command_elapsed))
            self.guard()
            phases['state_guard'] = (time.monotonic()-phase_start)*1000
            phase_start = time.monotonic()
            capture_guard()
            phases['capture_guard'] = (time.monotonic()-phase_start)*1000
            phase_start = time.monotonic()
            if on_tick is not None:
                on_tick(max(0., elapsed))
            phases['audit_tick'] = (time.monotonic()-phase_start)*1000
            work_ms = (time.monotonic()-last_tick)*1000
            if work_ms > 200.:
                self.scheduler_monitor['max_gap_ms'] = max(self.scheduler_monitor['max_gap_ms'],work_ms)
                raise RuntimeError(f'Motion scheduler stalled >200ms ({work_ms:.1f}ms) '
                    f'before publish; phases={phases}; stop sending windows')
            phase_start = time.monotonic()
            if not final_published and now_tick - last_publish >= .15:
                start, end = window_indices(times, max(0, elapsed))
                message = JointTrajectory()
                message.joint_names = self.names
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(base_ns, 1_000_000_000)
                for index in range(start, end):
                    point = JointTrajectoryPoint()
                    point.positions = positions_rad[index].tolist()
                    point.velocities = velocity_rad[index].tolist()
                    point.accelerations = acceleration_rad[index].tolist()
                    ns = round(times[index] * 1e9)
                    point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
                    message.points.append(point)
                phases['build_window'] = (time.monotonic()-phase_start)*1000
                phase_start = time.monotonic()
                self.publisher.publish(message)
                phases['publish'] = (time.monotonic()-phase_start)*1000
                last_publish = now_tick
                final_published = end == len(times)
            previous_phases = phases
            for name, duration in phases.items():
                maximum = self.scheduler_monitor['phase_max_ms']
                maximum[name] = max(maximum.get(name,0.), duration)
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
    if plan['config'].get('trajectory_profile') == 'continuous_sine':
        return execute_continuous(plan, client, capture_guard, report, save)
    for index, step in enumerate(plan['steps'], 1):
        capture_guard()
        client.guard()
        event = {k: step[k] for k in ('name', 'phase', 'kind', 'speed', 'repetition', 'joint')}
        event.update(index=index, utc=utc_now(), status='sending')
        report['events'].append(event)
        save()
        event['arrival_error_deg'] = client.play(step, capture_guard)
        event.update(status='completed', completed_utc=utc_now())
        save()
    report.update(status='completed', completed_utc=utc_now())
    save()


def execute_continuous(plan, client, capture_guard, report, save):
    """One absolute clock and uninterrupted windows across all logical blocks."""
    steps=plan['steps']; times=[]; positions=[]; offset=0.; boundaries=[]
    for index,step in enumerate(steps):
        t=np.asarray(step['times_seconds'])+offset; q=step['positions_deg']
        times.extend(t[1:] if index else t); positions.extend(q[1:] if index else q)
        offset+=step['duration_seconds']; boundaries.append(offset)
    combined=dict(times_seconds=times,positions_deg=positions,hold_seconds=.8)
    index=-1; last_monitor=-1e9
    def tick(elapsed):
        nonlocal index, last_monitor
        # Inspect existing subscriptions; never spawn a new DDS participant or
        # open the live recording database to verify a moving robot.
        if elapsed-last_monitor >= 10.:
            report['live_monitor'] = dict(utc=utc_now(),elapsed_seconds=elapsed,
                adapter_hz_last_second=client.state.get('adapter_hz'),
                adapter_reason=client.state.get('adapter_reason'),health_ok=client.state.get('ok'),
                max_tracking_deg=getattr(client,'monitor_max_tracking',np.zeros(7)).tolist(),
                measured_deg=np.rad2deg(client.state['q']).tolist(),
                target_deg=np.rad2deg(client.state['cmd']).tolist() if 'cmd' in client.state else None)
            report['live_monitor']['scheduler'] = getattr(client,'scheduler_monitor',None)
            save();last_monitor=elapsed
        if index < 0:
            index=0; start(0)
        while index<len(steps) and elapsed>=boundaries[index]:
            waypoint_error=np.abs(np.rad2deg(client.state['q'])-steps[index]['positions_deg'][-1])
            # A logical repetition can cross this waypoint at full speed. It
            # is not a stop/arrival request. Preserve the motion tracking cap
            # against the actual controller target; play() separately checks
            # the final stopped position after its hold, using arrival tolerance.
            reference=np.rad2deg(client.state['cmd']) if 'cmd' in client.state else steps[index]['positions_deg'][-1]
            tracking_error=np.abs(np.rad2deg(client.state['q'])-reference)
            if np.max(tracking_error)>plan['config']['tracking_limit_deg']:
                raise RuntimeError(f'Continuous block tracking error {tracking_error.tolist()}')
            report['events'][-1].update(status='completed',completed_utc=utc_now(),
                boundary_type='continuous_schedule',waypoint_error_deg=waypoint_error.tolist(),
                tracking_error_deg=tracking_error.tolist())
            index+=1
            if index<len(steps): start(index)
            save()
    def start(i):
        event={k:steps[i][k] for k in ('name','phase','kind','speed','repetition','joint')}
        event.update(index=i+1,utc=utc_now(),status='sending');report['events'].append(event)
        save()
    client.play(combined,capture_guard,on_tick=tick)
    if index!=len(steps):raise RuntimeError('Incomplete continuous schedule')
    report.update(status='completed',completed_utc=utc_now());save()


def send(plan, digest):
    active = read_active(plan['side'], digest)
    session = Path(active['session'])
    collision = {}
    if plan['config'].get('baseline_protocol'):
        collision = json.loads((session/'audit/collision.json').read_text())
        if (collision.get('accepted') is not True or collision.get('plan_sha256') != digest
                or collision.get('robot_description_sha256') != plan['model_sha256']):
            raise RuntimeError('Frozen baseline collision report is missing or mismatched')
    path = session / 'audit/motion_run.json'
    with (session / 'audit/motion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError('This capture already attempted motion; start a new C session')
        report = {'status': 'preflight', 'motion_sha256': digest, 'motion_name': plan['name'],
                  'side': plan['side'], 'started_utc': utc_now(), 'events': []}
        from w3_motion_audit import MotionAuditWriter
        writer = MotionAuditWriter(path)
        save = lambda: writer.submit(report)
        save()
        writer.flush()  # Persist preflight before any robot trajectory publication.
        client = None
        try:
            def capture_guard():
                writer.check()
                if read_active(plan['side'], digest) != active:
                    raise RuntimeError('Capture ended or changed')
            client = CoverageClient(plan, stationary_left_deg=collision.get('left_pose_deg'))
            initial = client.preflight()
            check_start(plan, initial, client.description, client.state['cmd'])
            capture_guard()
            report.update(status='running', initial_feedback_deg=np.rad2deg(initial).tolist())
            save()
            execute(plan, client, capture_guard, report, save)
        except (Exception, KeyboardInterrupt) as exc:
            if client is not None:
                report['scheduler'] = getattr(client,'scheduler_monitor',None)
            report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=str(exc) or 'Ctrl-C', completed_utc=utc_now())
            save()
            raise
        finally:
            if client is not None:
                client.close()
            writer.close()  # Final completed/failed snapshot must reach disk before D exits.
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
