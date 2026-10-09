#!/usr/bin/python3.10
"""Inspect joint angles; only --send publishes one explicit joint trajectory."""
import argparse
import math
import time
import xml.etree.ElementTree as ET


def target_degrees(current, degrees=None, step=None, max_delta_deg=5):
    if not math.isfinite(max_delta_deg) or not 0 < max_delta_deg <= 10:
        raise ValueError('Motion limit must be finite and within 0..10 degrees')
    if len(current) != 7 or not all(math.isfinite(v) for v in current):
        raise ValueError('Current state must contain seven finite angles')
    target = list(current)
    if degrees is not None:
        if len(degrees) != 7:
            raise ValueError('Supply seven angles in degrees')
        target = [math.radians(v) for v in degrees]
    elif step is not None:
        index, delta = step
        if index not in range(7):
            raise ValueError('Joint index must be 0..6')
        target[index] += math.radians(delta)
    if not all(math.isfinite(v) for v in target):
        raise ValueError('Target contains nonfinite values')
    if any(abs(t - q) > math.radians(max_delta_deg) + 1e-9 for q, t in zip(current, target)):
        raise ValueError(f'Pilot limit: each target must be within {max_delta_deg:g} degrees of current feedback')
    return target


def validate_limits(description, names, target):
    root = ET.fromstring(description)
    for name, value in zip(names, target):
        hardware = next((j for j in root.findall('./ros2_control/joint') if j.get('name') == name), None)
        if hardware is None:
            raise ValueError(f'{name} is not exposed by the active hardware')
        bounds = {p.get('name'): float(p.text) for p in hardware.findall('param')
                  if p.get('name') in ('urdf_lower', 'urdf_upper')}
        if set(bounds) != {'urdf_lower', 'urdf_upper'} or not all(math.isfinite(v) for v in bounds.values()):
            raise ValueError(f'Missing finite hardware limits for {name}')
        if not bounds['urdf_lower'] <= value <= bounds['urdf_upper']:
            raise ValueError(f'{name} target is outside the active hardware limits')


def trajectory_points(current, target, seconds):
    """Include current pose at t=0; a lone future point can jump on first update."""
    if not math.isfinite(seconds) or not 3 <= seconds <= 60:
        raise ValueError('Interpolation duration must be within 3..60 seconds')
    ns = round(seconds * 1_000_000_000)
    return [(list(current), 0, 0), (list(target), ns // 1_000_000_000, ns % 1_000_000_000)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--degrees', nargs=7, type=float, metavar='ANGLE')
    parser.add_argument('--step', nargs=2, type=float, metavar=('INDEX', 'DELTA_DEG'))
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--send', action='store_true', help='Publish the displayed target; otherwise read-only')
    args = parser.parse_args()
    if args.degrees is not None and args.step is not None:
        parser.error('Choose --degrees or --step')
    if args.send and args.degrees is None and args.step is None:
        parser.error('--send needs --degrees or --step')
    if not math.isfinite(args.seconds) or not 3 <= args.seconds <= 60:
        parser.error('--seconds must be finite and within 3..60')
    if args.step and (not math.isfinite(args.step[0]) or args.step[0] != int(args.step[0])):
        parser.error('Step joint index must be an integer 0..6')

    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from diagnostic_msgs.msg import DiagnosticArray
    from controller_manager_msgs.srv import ListControllers
    from rcl_interfaces.srv import GetParameters
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init(args=[])
    node = Node('quick_joint_target')
    names = [f'{args.side}_joint_{i}' for i in range(7)]
    state = {}

    def joints(msg):
        if len(set(msg.name)) != len(msg.name) or len(msg.name) != len(msg.position):
            return
        values = dict(zip(msg.name, msg.position))
        if all(name in values for name in names):
            state['q'] = [values[name] for name in names]
            state['received'] = time.monotonic()
            state['stamp'] = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def health(msg):
        for diagnostic in msg.status:
            if diagnostic.name == f'factr2/adapter/{args.side}':
                state['ok'] = diagnostic.level in (0, b'\x00')
                state['health_received'] = time.monotonic()
                state['health_stamp'] = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    node.create_subscription(JointState, '/joint_states', joints, qos_profile_sensor_data)
    node.create_subscription(DiagnosticArray, f'/factr2/{args.side}/adapter_status', health, qos_profile_sensor_data)

    def wait_until(predicate, label, seconds=8):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if predicate():
                return
        raise RuntimeError(f'Timeout: {label}; check ROS domain and W3/capture processes')

    def call(kind, service, request):
        client = node.create_client(kind, service)
        try:
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError(f'No service: {service}')
            future = client.call_async(request)
            wait_until(future.done, service)
            result = future.result()
            if result is None:
                raise RuntimeError(f'No result: {service}')
            return result
        finally:
            node.destroy_client(client)

    def fresh(key, stamp_key):
        return (key in state and 0 <= time.monotonic() - state[key] <= .25 and
                0 < state.get(stamp_key, 0) and
                0 <= (node.get_clock().now().nanoseconds - state[stamp_key]) * 1e-9 <= .25)

    try:
        wait_until(lambda: fresh('received', 'stamp'), 'fresh /joint_states')
        print(f'{args.side} joint_0..6 current(deg):', ' '.join(f'{math.degrees(q):.4f}' for q in state['q']))
        if args.degrees is None and args.step is None:
            return 0
        step = (int(args.step[0]), args.step[1]) if args.step else None
        if not args.send:
            target = target_degrees(state['q'], args.degrees, step)
            print('Preview target(deg):', ' '.join(f'{math.degrees(q):.4f}' for q in target))
            print('Read-only preview. Add --send only after checking the physical path.')
            return 0
        reply = call(ListControllers, '/controller_manager/list_controllers', ListControllers.Request())
        active = {c.name for c in reply.controller if c.state == 'active'}
        if not {'joint_position_controller', 'gravity_compensation_controller'} <= active:
            raise RuntimeError('Joint position and gravity controllers must already be active; no automatic switch')
        if 'cartesian_position_controller' in active:
            raise RuntimeError('Cartesian controller must be inactive')
        request = GetParameters.Request()
        request.names = ['robot_description']
        reply = call(GetParameters, '/robot_state_publisher/get_parameters', request)
        if len(reply.values) != 1 or reply.values[0].type != 4:
            raise RuntimeError('Missing active robot_description')
        description = reply.values[0].string_value
        # Idle UI publishers may remain after a prior web target; exclude only that known UI.
        other = [p.node_name for p in node.get_publishers_info_by_topic('/joint_position_command')
                 if not p.node_name.startswith('ieir_web_console_') and p.node_name != node.get_name()]
        if other:
            raise RuntimeError(f'Other trajectory publishers exist: {other}; stop replay/teleop first')
        wait_until(lambda: fresh('received', 'stamp') and state.get('ok') and
                   fresh('health_received', 'health_stamp'), 'fresh real adapter OK')
        current = list(state['q'])
        target = target_degrees(current, args.degrees, step)
        validate_limits(description, names, target)
        publisher = node.create_publisher(JointTrajectory, '/joint_position_command', 10)
        wait_until(lambda: publisher.get_subscription_count() > 0, 'trajectory subscriber', seconds=3)
        # Refresh the initial point after publisher discovery; the first point is current at t=0.
        wait_until(lambda: fresh('received', 'stamp') and state.get('ok') and
                   fresh('health_received', 'health_stamp'), 'fresh state before publish')
        current = list(state['q'])
        target_degrees(current, [math.degrees(q) for q in target])
        message = JointTrajectory()
        message.joint_names = names
        for positions, sec, ns in trajectory_points(current, target, args.seconds):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = ns
            message.points.append(point)
        print('Sending target(deg):', ' '.join(f'{math.degrees(q):.4f}' for q in target), flush=True)
        publisher.publish(message)
        # Allow reliable transport to acknowledge the one message before destroying the node.
        if not publisher.wait_for_all_acked(Duration(seconds=2)):
            raise RuntimeError('Trajectory acknowledgement timed out; result unknown, do not automatically retry')
        print(f'Published once, interpolation {args.seconds:g}s. Arrival/path safety is not confirmed.')
        return 0
    except (RuntimeError, ValueError, ET.ParseError) as exc:
        print(f'Refused: {exc}')
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
