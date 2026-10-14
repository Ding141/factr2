#!/usr/bin/env python3
"""Isolated ROS goal-to-native-controller test, double-backed interfaces, no CAN."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', type=int, default=227)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 200 <= args.domain <= 230:
        parser.error('Use a test-only domain in 200..230')
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY='1',
        ROS_LOG_DIR='/tmp/w3-pose-stream-ros')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
    from w3_pose_motion import PoseCoverageClient
    from w3_pose_protocol import compile_knots, pose_target
    from w3_coverage_plan import ArmModel
    from test_w3_coverage_motion import fixture_description
    root = Path(__file__).resolve().parents[1]
    mock_path = root.parent/'dual_arm_robot/build/ieir_controllers/command_state_mock'
    process = None; node = None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.with_suffix('.mock.log').open('w') as log:
        try:
            process = subprocess.Popen([str(mock_path), 'right', '10'], env=os.environ.copy(),
                                       stdout=log, stderr=subprocess.STDOUT)
            rclpy.init(args=[]); node = Node('pose_goal_stream_test')
            client = PoseCoverageClient.__new__(PoseCoverageClient)
            client.node = node; client.names = [f'right_joint_{i}' for i in range(7)]
            client.goal_topic = '/factr2/right/pose_goal'
            client.state = {}; client.pending_goal = None; client.bridge_error = None
            client.accepted_stamp = None
            native_seen, poses_seen, command_seen = [], [], []
            def command(msg):
                client.state['cmd'] = list(msg.position)
                command_seen.append(msg)
            def guard():
                if not command_seen or (node.get_clock().now().nanoseconds-
                        command_seen[-1].header.stamp.sec*10**9-command_seen[-1].header.stamp.nanosec) > 250_000_000:
                    raise RuntimeError('Mock native command state stale')
            client.guard = guard
            client.publisher = node.create_publisher(JointTrajectory, '/joint_position_command', 1)
            pose_pub = node.create_publisher(PoseStamped, '/factr2/right/pose_goal', 1)
            node.create_subscription(PoseStamped, '/factr2/right/pose_goal', client.accept_pose, 1)
            node.create_subscription(PoseStamped, '/factr2/right/pose_goal', poses_seen.append, 1)
            node.create_subscription(JointTrajectory, '/joint_position_command', native_seen.append, 1)
            node.create_subscription(JointState, '/joint_position_controller/command_state', command,
                                     qos_profile_sensor_data)
            deadline = time.monotonic()+5
            while (not command_seen or not client.goal_interfaces_ready()) and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.01)
            guard()
            assert client.goal_interfaces_ready(), 'Actual controller/stop/bridge interfaces required'
            start = np.rad2deg(client.state['cmd']); goal = start+np.array([.25, .1, -.1, .15, 0., .1, -.1])
            limits = dict(velocity_deg_s=[20.]*7, acceleration_deg_s2=[80.]*7)
            knots, t, q = compile_knots([start, goal], limits, .01)
            model = ArmModel(fixture_description(), 'right', 'right_attachment_point')
            step = dict(positions_deg=q.tolist(), command=dict(type='pose_goal', knots=knots,
                                                             target=pose_target(model, goal)))
            stamp = node.get_clock().now().nanoseconds+150_000_000
            native = client._native_message(step, stamp)
            client.pending_goal = (step, native, stamp, lambda: None)
            pose = PoseStamped(); pose.header.stamp = native.header.stamp
            target = step['command']['target']; pose.header.frame_id = target['frame_id']
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = target['position_m']
            (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
             pose.pose.orientation.w) = target['quaternion_xyzw']
            pose_pub.publish(pose)
            deadline = time.monotonic()+float(t[-1])+1.
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.005)
                if client.bridge_error:
                    raise client.bridge_error
            assert len(poses_seen) == 1 and len(native_seen) == 1, 'Goal must publish exactly once'
            # A future goal is installed at the first native update at/after
            # its stamp (300 Hz), so the effective handoff can be one tick late.
            starts = [m for m in command_seen if
                m.header.stamp.sec*10**9+m.header.stamp.nanosec >= stamp]
            assert starts
            handoff = starts[0].header.stamp.sec*10**9+starts[0].header.stamp.nanosec
            delay = (handoff-stamp)*1e-9
            assert 0 <= delay < .02
            errors, nominal_errors = [], []
            for msg in command_seen:
                elapsed = (msg.header.stamp.sec*10**9+msg.header.stamp.nanosec-handoff)*1e-9
                if 0 <= elapsed <= t[-1]-delay:
                    u = elapsed/(t[-1]-delay); s = u**3*(10-15*u+6*u*u)
                    errors.append(np.max(abs(np.rad2deg(msg.position)-(start+s*(goal-start)))))
                    nominal_u = (elapsed+delay)/t[-1]
                    nominal_s = nominal_u**3*(10-15*nominal_u+6*nominal_u*nominal_u)
                    nominal_errors.append(np.max(abs(np.rad2deg(msg.position)-(start+nominal_s*(goal-start)))))
            assert len(errors) >= 100 and max(errors) < 1e-6
            assert max(nominal_errors) < .01
            np.testing.assert_allclose(np.rad2deg(client.state['cmd']), goal, atol=1e-6)
            # Duplicate delivery cannot produce another native trajectory.
            pose_pub.publish(pose)
            deadline = time.monotonic()+.2
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.005)
            assert len(native_seen) == 1
            result = dict(passed=True, domain=args.domain, hardware=False, can=False,
                initial_pose_messages=1, native_trajectories=1, native_knots=len(native_seen[0].points),
                duplicate_replayed=False, command_samples_checked=len(errors), max_position_error_deg=max(errors),
                native_start_delay_seconds=delay, max_nominal_schedule_error_deg=max(nominal_errors))
            args.report.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result))
        finally:
            if node is not None: node.destroy_node()
            rclpy.try_shutdown()
            if process is not None and process.poll() is None:
                process.terminate(); process.wait(timeout=5)


if __name__ == '__main__':
    main()
