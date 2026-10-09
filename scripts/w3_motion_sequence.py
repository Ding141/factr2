#!/usr/bin/python3.10
"""Terminal D: validate a motion YAML offline; --send executes it during C's capture."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import yaml

from w3_joint_target import target_degrees, trajectory_points, validate_limits

ROOT = Path(__file__).resolve().parents[1]


def number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if not low <= value <= high:
        raise ValueError(f'{label} must be within {low}..{high}')
    return float(value)


def keys(value, required, optional, label):
    if not isinstance(value, dict) or required - value.keys() or value.keys() - required - optional:
        raise ValueError(f'{label}: missing/unknown keys; required={sorted(required)}, optional={sorted(optional)}')


def load_motion(path):
    """No ROS imports or robot queries. Reject the entire invalid plan before any send."""
    raw = Path(path).read_bytes()
    plan = yaml.safe_load(raw)
    keys(plan, {'version', 'name', 'side', 'reference', 'units', 'steps'},
         {'start_hold_seconds', 'finish_hold_seconds', 'arrival_tolerance_deg', 'max_step_deg'}, 'motion')
    if type(plan['version']) is not int or plan['version'] != 1:
        raise ValueError('version must be 1')
    if not isinstance(plan['name'], str) or not plan['name'].strip():
        raise ValueError('name must be a nonempty string')
    if plan['side'] not in ('left', 'right') or plan['reference'] != 'initial_feedback' or plan['units'] != 'deg':
        raise ValueError('Require side left/right, reference initial_feedback, units deg')
    plan['start_hold_seconds'] = number(plan.get('start_hold_seconds', 2), 'start hold', 0, 60)
    plan['finish_hold_seconds'] = number(plan.get('finish_hold_seconds', 3), 'finish hold', 0, 60)
    plan['arrival_tolerance_deg'] = number(plan.get('arrival_tolerance_deg', 1.5), 'arrival tolerance', .05, 2)
    plan['max_step_deg'] = number(plan.get('max_step_deg', 5), 'motion limit', .1, 10)
    if not isinstance(plan['steps'], list) or not 1 <= len(plan['steps']) <= 500:
        raise ValueError('steps must contain 1..500 steps')
    previous = [0.] * 7
    seen = set()
    for index, step in enumerate(plan['steps'], 1):
        keys(step, {'name', 'offset_deg', 'duration_seconds', 'hold_seconds'}, set(), f'step {index}')
        if not isinstance(step['name'], str) or not step['name'].strip() or step['name'] in seen:
            raise ValueError(f'step {index}: name must be nonempty and unique')
        seen.add(step['name'])
        if not isinstance(step['offset_deg'], list) or len(step['offset_deg']) != 7:
            raise ValueError(f'step {index}: offset_deg must contain seven angles')
        limit = plan['max_step_deg']
        step['offset_deg'] = [number(v, f'step {index} offset', -limit, limit) for v in step['offset_deg']]
        if max(abs(a - b) for a, b in zip(previous, step['offset_deg'])) > limit + 1e-9:
            raise ValueError(f'step {index}: segment exceeds {limit:g} degrees per joint')
        step['duration_seconds'] = number(step['duration_seconds'], f'step {index} duration', 3, 60)
        step['hold_seconds'] = number(step['hold_seconds'], f'step {index} hold', .5, 60)
        previous = step['offset_deg']
    return plan, raw, hashlib.sha256(raw).hexdigest()


def resolve_motion(plan, initial, description):
    """Every target uses the same initial feedback, never the latest feedback as an offset."""
    names = [f'{plan["side"]}_joint_{i}' for i in range(7)]
    target_degrees(initial)  # Validate dimension and finite feedback before resolving any step.
    validate_limits(description, names, initial)
    result = []
    for step in plan['steps']:
        goal = [q + math.radians(d) for q, d in zip(initial, step['offset_deg'])]
        validate_limits(description, names, goal)
        result.append({**step, 'target_rad': goal})
    return result


def active_path(side):
    return ROOT / 'data/quick_capture' / side / 'active.json'


def read_active(side, digest):
    try:
        info = json.loads(active_path(side).read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError('C 尚未用这份 --motion 文件开录；先等待 C 显示“已开始录制”') from exc
    if not isinstance(info, dict) or not isinstance(info.get('session'), str):
        raise RuntimeError('Invalid active capture information')
    if info.get('motion_sha256') != digest or info.get('side') != side:
        raise RuntimeError('D 的动作文件与 C 开录时冻结的动作文件不同；本次拒绝发送')
    if str(info.get('domain')) != os.environ.get('ROS_DOMAIN_ID', '0'):
        raise RuntimeError('C/D 的 ROS_DOMAIN_ID 不一致')
    pid = info.get('pid')
    if type(pid) is not int or pid <= 0:
        raise RuntimeError('Invalid C process identity')
    try:
        os.kill(pid, 0)  # Existence check only; sends no signal.
    except OSError as exc:
        raise RuntimeError('C 的采集进程已退出') from exc
    session = Path(info['session'])
    if session.parent != active_path(side).parent or not session.is_dir():
        raise RuntimeError('Invalid active capture directory')
    return info


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class MotionClient:
    """One ROS publisher for the whole plan. Construction is confined to --send."""
    def __init__(self, side, max_step_deg=5):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from diagnostic_msgs.msg import DiagnosticArray
        from sensor_msgs.msg import JointState
        self.ros = rclpy
        rclpy.init(args=[])
        self.node = Node(f'quick_motion_sequence_{side}')
        self.side = side
        self.max_step_deg = max_step_deg
        self.names = [f'{side}_joint_{i}' for i in range(7)]
        self.state = {}
        self.publisher = None
        for topic, key in [('/joint_states', 'q'), ('/joint_position_controller/command_state', 'cmd')]:
            self.node.create_subscription(JointState, topic,
                lambda msg, key=key: self.joints(msg, key), qos_profile_sensor_data)
        self.node.create_subscription(DiagnosticArray, f'/factr2/{side}/adapter_status',
                                      self.health, qos_profile_sensor_data)

    def joints(self, msg, key):
        if len(set(msg.name)) != len(msg.name) or len(msg.name) != len(msg.position):
            self.state.pop(key, None)
            return
        values = dict(zip(msg.name, msg.position))
        if not all(name in values and math.isfinite(values[name]) for name in self.names):
            self.state.pop(key, None)
            return
        self.state[key] = [values[name] for name in self.names]
        self.state[key + '_received'] = time.monotonic()
        self.state[key + '_stamp'] = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def health(self, msg):
        self.state['ok'] = False
        for diagnostic in msg.status:
            if diagnostic.name == f'factr2/adapter/{self.side}':
                self.state['ok'] = diagnostic.level in (0, b'\x00')
                self.state['health_received'] = time.monotonic()
                self.state['health_stamp'] = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def fresh(self, key):
        return (0 <= time.monotonic() - self.state.get(key + '_received', -1e9) <= .25 and
                0 < self.state.get(key + '_stamp', 0) and
                0 <= (self.node.get_clock().now().nanoseconds - self.state[key + '_stamp']) * 1e-9 <= .25)

    def ready(self):
        return (all(self.fresh(key) for key in ('q', 'cmd', 'health')) and self.state.get('ok') and
                all(key in self.state for key in ('q', 'cmd')))

    def wait_until(self, predicate, label, seconds=8):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.ros.spin_once(self.node, timeout_sec=.02)
            if predicate():
                return
        raise RuntimeError(f'Timeout: {label}; check mode, capture, ROS domain')

    def call(self, kind, service, request):
        client = self.node.create_client(kind, service)
        try:
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError(f'No service: {service}')
            future = client.call_async(request)
            self.wait_until(future.done, service)
            result = future.result()
            if result is None:
                raise RuntimeError(f'No result: {service}')
            return result
        finally:
            self.node.destroy_client(client)

    def check_mode(self):
        from controller_manager_msgs.srv import ListControllers
        reply = self.call(ListControllers, '/controller_manager/list_controllers', ListControllers.Request())
        active = {c.name for c in reply.controller if c.state == 'active'}
        if not {'joint_position_controller', 'gravity_compensation_controller', 'joint_state_broadcaster'} <= active:
            raise RuntimeError('请在网页手动切到关节位置；本工具不使能或切模式')
        if 'cartesian_position_controller' in active:
            raise RuntimeError('Cartesian controller must be inactive')
        other = [p.node_name for p in self.node.get_publishers_info_by_topic('/joint_position_command')
                 if not p.node_name.startswith('ieir_web_console_') and p.node_name != self.node.get_name()]
        if other:
            raise RuntimeError(f'Other motion publishers: {other}')

    def preflight(self):
        from rcl_interfaces.srv import GetParameters
        self.check_mode()
        request = GetParameters.Request()
        request.names = ['robot_description']
        reply = self.call(GetParameters, '/robot_state_publisher/get_parameters', request)
        if len(reply.values) != 1 or reply.values[0].type != 4:
            raise RuntimeError('Missing active robot_description')
        self.description = reply.values[0].string_value
        self.wait_until(self.ready, 'fresh real feedback, command state and adapter OK')
        # Do not interrupt a trajectory that was already moving before D started.
        command = list(self.state['cmd'])
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            self.ros.spin_once(self.node, timeout_sec=.02)
            self.guard()
            if max(abs(a - b) for a, b in zip(command, self.state['cmd'])) > math.radians(.05):
                raise RuntimeError('已有目标正在变化；先等当前运动结束')
        return list(self.state['q'])

    def guard(self):
        if not self.ready():
            raise RuntimeError('状态/command_state/real adapter 失效或超过 250ms；停止后续发送')
        if max(abs(q - c) for q, c in zip(self.state['q'], self.state['cmd'])) > math.radians(5):
            raise RuntimeError('实测与控制目标相差超过 5°；停止后续发送')

    def feedback(self):
        return list(self.state['q']), list(self.state['cmd'])

    def send(self, goal, duration, capture_guard):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        self.check_mode()
        self.wait_until(self.ready, 'fresh state before publish')
        self.guard()
        q, current_command = self.feedback()
        target_degrees(q, [math.degrees(v) for v in goal], max_delta_deg=self.max_step_deg)
        target_degrees(current_command, [math.degrees(v) for v in goal], max_delta_deg=self.max_step_deg)
        validate_limits(self.description, self.names, current_command)
        validate_limits(self.description, self.names, goal)
        if self.publisher is None:
            self.publisher = self.node.create_publisher(JointTrajectory, '/joint_position_command', 10)
            self.wait_until(lambda: self.publisher.get_subscription_count() > 0, 'trajectory subscriber', 3)
        # Discovery/services can take time; refresh all checks before the actual publish.
        self.guard()
        q, current_command = self.feedback()
        target_degrees(q, [math.degrees(v) for v in goal], max_delta_deg=self.max_step_deg)
        target_degrees(current_command, [math.degrees(v) for v in goal], max_delta_deg=self.max_step_deg)
        validate_limits(self.description, self.names, current_command)
        message = JointTrajectory()
        message.joint_names = self.names
        # Use the last controller target at t=0; tracking error must not reset it between steps.
        for positions, sec, ns in trajectory_points(current_command, goal, duration):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = ns
            message.points.append(point)
        capture_guard()
        self.publisher.publish(message)

    def monitor(self, seconds, capture_guard):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.ros.spin_once(self.node, timeout_sec=.02)
            self.guard()
            capture_guard()

    def endpoint(self, goal, tolerance):
        self.guard()
        q, cmd = self.feedback()
        if max(abs(a - b) for a, b in zip(goal, cmd)) > math.radians(.05):
            raise RuntimeError('控制器没有保持本步目标（可能未收到、被网页覆盖或切换了模式）')
        errors = [math.degrees(qi - gi) for qi, gi in zip(q, goal)]
        if max(abs(v) for v in errors) > tolerance:
            raise RuntimeError(f'到位误差超过 {tolerance:g}°：{[round(v, 3) for v in errors]}；停止后续发送')
        return errors

    def close(self):
        self.node.destroy_node()
        self.ros.try_shutdown()


def execute_motion(plan, resolved, client, capture_guard, report, save):
    """Scheduling is tested with a fake client, with no DDS or hardware initialization."""
    client.monitor(plan['start_hold_seconds'], capture_guard)
    for index, step in enumerate(resolved, 1):
        capture_guard()
        client.guard()
        event = {'index': index, 'name': step['name'], 'status': 'sending', 'utc': utc_now(),
                 'target_deg': [math.degrees(v) for v in step['target_rad']]}
        report['events'].append(event)
        save()
        print(f'[{index}/{len(resolved)}] {step["name"]}：运动 {step["duration_seconds"]:g}s，'
              f'保持 {step["hold_seconds"]:g}s', flush=True)
        client.send(step['target_rad'], step['duration_seconds'], capture_guard)
        event['status'] = 'published'
        save()
        # No automatic resend. Wait beyond the scheduled endpoint before the next command.
        client.monitor(step['duration_seconds'] + step['hold_seconds'], capture_guard)
        event['arrival_error_deg'] = client.endpoint(step['target_rad'], plan['arrival_tolerance_deg'])
        event.update(status='completed', completed_utc=utc_now())
        save()
    client.monitor(plan['finish_hold_seconds'], capture_guard)
    report.update(status='completed', completed_utc=utc_now())
    save()


def print_plan(plan):
    seconds = plan['start_hold_seconds'] + plan['finish_hold_seconds'] + sum(
        s['duration_seconds'] + s['hold_seconds'] for s in plan['steps'])
    print(f'{plan["name"]} / {plan["side"]}：{len(plan["steps"])} 步，计划约 {seconds:g} 秒（另有服务检查耗时）')
    print('单位：度；offset_deg 顺序 J1..J7；全部相对同一次开始时的实测姿态。')
    print(f'本文件动作上限：{plan["max_step_deg"]:g}°；运行中跟踪误差上限仍为 5°。')
    for index, step in enumerate(plan['steps'], 1):
        print(f'{index:02d} {step["name"]}: {step["offset_deg"]} '
              f'/ {step["duration_seconds"]:g}s + 保持 {step["hold_seconds"]:g}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--send', action='store_true', help='Execute on the robot; omitted = offline file check only')
    args = parser.parse_args()
    try:
        plan, _, digest = load_motion(args.file)
        print_plan(plan)
        if not args.send:
            print('文件检查 PASS；未连接 ROS，未发送动作。--send 才执行，现场路径仍须人工核对。')
            return 0
        active = read_active(plan['side'], digest)
        session = Path(active['session'])
        path = session / 'audit/motion_run.json'
        with (session / 'audit/motion.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('本次已有 D 运动进程') from exc
            if path.exists():
                raise RuntimeError('本次动作已经运行或尝试过；下一轮请重新在 C 开录，避免重复运动')
            report = {'status': 'preflight', 'motion_sha256': digest, 'motion_name': plan['name'],
                      'side': plan['side'], 'started_utc': utc_now(), 'events': []}
            save = lambda: write_json(path, report)
            save()
            client = None
            try:
                def capture_guard():
                    if read_active(plan['side'], digest) != active:
                        raise RuntimeError('C 的录制 session 已改变；停止后续发送')
                client = MotionClient(plan['side'], max_step_deg=plan['max_step_deg'])
                initial = client.preflight()
                resolved = resolve_motion(plan, initial, client.description)
                report.update(initial_feedback_deg=[math.degrees(v) for v in initial],
                              targets_deg=[[math.degrees(v) for v in s['target_rad']] for s in resolved],
                              status='running')
                save()
                print('起始角度 J1..J7(deg)：', report['initial_feedback_deg'], flush=True)
                execute_motion(plan, resolved, client, capture_guard, report, save)
                print(f'整套动作完成；执行报告：{path}\n请回到 C 按 Enter 停录并检查。', flush=True)
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              error=str(exc) or 'Ctrl-C', completed_utc=utc_now())
                save()
                raise
            finally:
                if client is not None:
                    client.close()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'未完成：{exc or "Ctrl-C"}', flush=True)
        if args.send:
            print('停止后续发送；已发布轨迹仍可能继续。按现场停车流程处理，再回 C 停录。', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
