"""Direct PoseStamped plan execution through a local, frozen-goal ROS bridge.

Only the current collision-audited pose is accepted. Its offline inverse solution
fixes the 7-DoF null-space choice. The bridge sends one sparse native trajectory;
it never streams/interpolates joint commands. command_state remains authoritative.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from w3_coverage_motion import CoverageClient


def pose_matches(target, message):
    p, r = message.pose.position, message.pose.orientation
    actual = np.array([p.x, p.y, p.z, r.x, r.y, r.z, r.w], dtype=float)
    if (message.header.frame_id != target['frame_id'] or not np.isfinite(actual).all()
            or abs(np.linalg.norm(actual[3:])-1) > 1e-8):
        return False
    return (np.max(abs(actual[:3]-target['position_m'])) < 1e-8 and
        np.max(abs(Rotation.from_quat(actual[3:]).as_matrix()-
                   Rotation.from_quat(target['quaternion_xyzw']).as_matrix())) < 1e-8)


class PoseCoverageClient(CoverageClient):
    def __init__(self, plan, stationary_left_deg=None):
        super().__init__(plan, stationary_left_deg)
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Empty
        from trajectory_msgs.msg import JointTrajectory
        self.goal_topic = f'/factr2/{self.side}/pose_goal'
        self.pose_pub = self.node.create_publisher(PoseStamped, self.goal_topic, 1)
        self.node.create_subscription(PoseStamped, self.goal_topic, self.accept_pose, 1)
        self.publisher = self.node.create_publisher(JointTrajectory, '/joint_position_command', 1)
        self.stop_pub = self.node.create_publisher(Empty, '/joint_position_controller/stop', 1)
        self.pending_goal = None
        self.accepted_stamp = None
        self.bridge_error = None

    def preflight(self):
        from rcl_interfaces.srv import GetParameters
        initial = super().preflight()
        request = GetParameters.Request()
        request.names = ['joints', 'max_velocity', 'max_acceleration', 'max_jerk']
        reply = self.call(GetParameters, '/joint_position_controller/get_parameters', request)
        if len(reply.values) != 4 or reply.values[0].type != 9:
            raise RuntimeError('Missing native motion limit parameters')
        names = list(reply.values[0].string_array_value)
        if not set(self.names) <= set(names):
            raise RuntimeError('Native controller does not contain this arm')
        required = [np.deg2rad(np.max([s['velocity_deg_s'] for s in
                    self.plan['config']['speeds'].values()], axis=0)),
                    np.deg2rad(np.max([s['acceleration_deg_s2'] for s in
                    self.plan['config']['speeds'].values()], axis=0)), np.full(7, np.deg2rad(120.))]
        for param, need in zip(reply.values[1:], required):
            values = np.asarray(param.double_array_value)
            if param.type != 8 or len(values) != len(names) or not np.isfinite(values).all():
                raise RuntimeError('Require explicit native per-joint motion limits')
            if np.any(values[[names.index(n) for n in self.names]]+1e-8 < need):
                raise RuntimeError('Plan exceeds native motion limits; reduce recipe speeds')
        return initial

    def accept_pose(self, message):
        pending = self.pending_goal
        if pending is None:
            return  # Duplicate delivery cannot re-execute a goal.
        step, native, stamp, capture_guard = pending
        received = message.header.stamp.sec*10**9+message.header.stamp.nanosec
        if received != stamp or not pose_matches(step['command']['target'], message):
            self.bridge_error = RuntimeError('Pose differs from the frozen current goal')
            return
        try:
            self.guard(); capture_guard()
            if np.max(abs(np.rad2deg(self.state['cmd'])-step['positions_deg'][0])) > .05:
                raise RuntimeError('Command changed before pose acceptance')
            if not 0 < stamp-self.node.get_clock().now().nanoseconds <= 300_000_000:
                raise RuntimeError('Expired pose goal; never resend')
            self.pending_goal = None
            self.publisher.publish(native)
            self.accepted_stamp = stamp
        except Exception as exc:
            self.bridge_error = exc

    def _native_message(self, step, stamp):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        message = JointTrajectory()
        message.joint_names = self.names
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 10**9)
        for knot in step['command']['knots']:
            point = JointTrajectoryPoint()
            point.positions = np.deg2rad(knot['position_deg']).tolist()
            point.velocities = [0.]*7
            point.accelerations = [0.]*7
            ns = round(knot['time_seconds']*10**9)
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 10**9)
            message.points.append(point)
        return message

    def goal_interfaces_ready(self):
        # A rosbag subscriber counts as a subscriber too. Require the actual
        # controller/bridge identities, while allowing additional observers.
        required = {'/joint_position_command': 'joint_position_controller',
                    '/joint_position_controller/stop': 'joint_position_controller',
                    self.goal_topic: self.node.get_name()}
        return all(name in {endpoint.node_name for endpoint in
            self.node.get_subscriptions_info_by_topic(topic)} for topic, name in required.items())

    def play(self, step, capture_guard, on_tick=None):
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Empty
        self.expected_deg = None
        self.check_mode()
        self.wait_until(self.ready, 'fresh inputs before direct goal')
        self.guard(); capture_guard()
        if step['command']['type'] == 'hold':
            self.expected_deg = np.asarray(step['positions_deg'][0])
            self.monitor(step['duration_seconds'], capture_guard)
            return self.endpoint(np.deg2rad(step['positions_deg'][-1]),
                                 self.plan['config']['arrival_tolerance_deg'])
        self.wait_until(self.goal_interfaces_ready,
            'native controller, bounded stop and pose bridge subscribers')
        other = [p.node_name for p in self.node.get_publishers_info_by_topic(self.goal_topic)
                 if p.node_name != self.node.get_name()]
        if other:
            raise RuntimeError('Other pose goal publishers: '+str(other))
        self.guard(); capture_guard()
        if np.max(abs(np.rad2deg(self.state['cmd'])-step['positions_deg'][0])) > .05:
            raise RuntimeError('Controller start differs from audited direct goal')
        stamp = self.node.get_clock().now().nanoseconds + 150_000_000
        native = self._native_message(step, stamp)
        self.bridge_error = None
        try:
            if step['command']['type'] == 'pose_goal':
                target = step['command']['target']
                pose = PoseStamped(); pose.header = native.header
                pose.header.frame_id = target['frame_id']
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = target['position_m']
                (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                 pose.pose.orientation.w) = target['quaternion_xyzw']
                self.pending_goal = (step, native, stamp, capture_guard)
                self.pose_pub.publish(pose)
            else:
                self.publisher.publish(native)
                self.accepted_stamp = stamp
            self.wait_until(lambda: self.accepted_stamp == stamp or self.bridge_error is not None,
                            'single pose acceptance', seconds=.14)
            if self.bridge_error is not None:
                raise self.bridge_error
            knots = step['command']['knots']
            last_tick = time.monotonic()
            while True:
                self.ros.spin_once(self.node, timeout_sec=.005)
                elapsed = (self.node.get_clock().now().nanoseconds-stamp)*1e-9
                # Analytic reference, never published: same polynomial as the controller.
                index = min(int(np.searchsorted([k['time_seconds'] for k in knots],
                    max(0., elapsed), side='right'))-1, len(knots)-2)
                a, b = knots[index], knots[index+1]
                u = np.clip((elapsed-a['time_seconds'])/(b['time_seconds']-a['time_seconds']), 0, 1)
                s = u**3*(10-15*u+6*u*u)
                self.expected_deg = np.asarray(a['position_deg'])+s*(
                    np.asarray(b['position_deg'])-a['position_deg'])
                now = time.monotonic()
                if now-last_tick > .2:
                    raise RuntimeError('Direct goal monitoring stalled >200 ms')
                last_tick = now
                self.guard(); capture_guard()
                if on_tick:
                    on_tick(max(0., elapsed))
                if elapsed >= step['duration_seconds']+.05:
                    break
            return self.endpoint(np.deg2rad(step['positions_deg'][-1]),
                                 self.plan['config']['arrival_tolerance_deg'])
        except BaseException:
            self.pending_goal = None
            self.expected_deg = None
            self.stop_pub.publish(Empty())
            # Give the reliable publisher time to deliver the braking request.
            deadline = time.monotonic()+.2
            while time.monotonic() < deadline:
                self.ros.spin_once(self.node, timeout_sec=.01)
            raise
        finally:
            self.pending_goal = None
