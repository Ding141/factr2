"""Isolated ROS test: never run this synthetic publisher on robot domain 74."""
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import WrenchStamped
import yaml
from w3_endpoint_wrench import EndpointObserver
from w3_coverage_plan import ArmModel


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '172' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('Synthetic test requires dedicated localhost ROS_DOMAIN_ID=172')
    root = Path(__file__).resolve().parents[1]
    description = next(iter(yaml.safe_load((root / 'log/manual_start/right_observer_20261010_125336/live_position.yaml').read_text()).values()))['ros__parameters']['robot_description']
    cfg = yaml.safe_load((root / 'config/w3/right/endpoint_wrench.yaml').read_text())
    cfg['web']['port'] = 8083
    model = ArmModel(description, 'right', cfg['tool_frame'])
    q = np.deg2rad([10, 20, 10, -30, 5, 8, 5])
    pose, j = model.fk(q, True)
    tool = np.array([2., -1., .5, .2, -.3, .1])
    base = np.r_[pose[:3, :3] @ tool[:3], pose[:3, :3] @ tool[3:]]
    tau = j.T @ base
    rclpy.init(); fixture = Node('isolated_endpoint_fixture'); executor = SingleThreadedExecutor()
    with tempfile.TemporaryDirectory(dir=root / 'log/tmp') as temporary:
        path = Path(temporary) / 'endpoint.yaml'; path.write_text(yaml.safe_dump(cfg))
        observer = EndpointObserver(path)
        executor.add_node(fixture); executor.add_node(observer)
        pub_q = fixture.create_publisher(JointState, '/factr2/right/joint_pos', qos_profile_sensor_data)
        pub_tau = fixture.create_publisher(JointState, '/next/right/external_joint_torque', qos_profile_sensor_data)
        pub_status = fixture.create_publisher(DiagnosticArray, '/next/right/status', qos_profile_sensor_data)
        pub_description = fixture.create_publisher(String, '/robot_description', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        received = []
        sub = fixture.create_subscription(WrenchStamped, '/next/right/endpoint_wrench', received.append, qos_profile_sensor_data)
        pub_description.publish(String(data=description))
        def publish():
            stamp = fixture.get_clock().now().to_msg()
            for publisher, values in [(pub_q, q), (pub_tau, tau)]:
                msg = JointState(); msg.header.stamp = stamp; msg.name = observer.names; msg.position = values.tolist(); publisher.publish(msg)
            msg = DiagnosticArray(); msg.header.stamp = stamp
            status = DiagnosticStatus(); status.name = 'factr2/next/right'; status.level = DiagnosticStatus.OK
            status.values = [KeyValue(key='state', value='"valid"'), KeyValue(key='reason', value='"isolated_test"')]
            msg.status = [status]; pub_status.publish(msg)
        timer = fixture.create_timer(.01, publish)
        def spin(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end: executor.spin_once(timeout_sec=.01)
        try:
            spin(1.5)
            assert set(fixture.get_node_names()) <= {'isolated_endpoint_fixture', 'right_endpoint_wrench_observer'}, 'Unexpected node on test domain'
            spin(3.)
            assert len(received) > 200 and observer.state == 'valid'
            last = received[-1]; assert last.header.frame_id == 'right_attachment_point'
            values = np.array([last.wrench.force.x, last.wrench.force.y, last.wrench.force.z, last.wrench.torque.x, last.wrench.torque.y, last.wrench.torque.z])
            np.testing.assert_allclose(values, tool, atol=.003)
            snapshot = json.load(urllib.request.urlopen('http://127.0.0.1:8083/snapshot', timeout=2))
            assert snapshot['state'] == 'valid' and snapshot['wrench'] is not None
            assert len(snapshot['history']['mx']) > 20 and any(v is not None for v in snapshot['history']['mx'])
            before = observer.outputs; timer.cancel(); spin(.5)
            stale = json.load(urllib.request.urlopen('http://127.0.0.1:8083/snapshot', timeout=2))
            assert stale['state'] == 'unavailable' and stale['wrench'] is None
            assert observer.outputs - before <= 2 and stale['history']['mx'][-1] is None
            result = dict(test='isolated_ROS_172_no_CAN_no_motion', expected=tool.tolist(), actual=values.tolist(),
                          frame_id=last.header.frame_id, received=len(received), stale_suppression='PASS',
                          max_abs_error=float(np.max(np.abs(values - tool))))
            output = root / 'log/manual_start/right_observer_20261010_125336/endpoint_stream_test.json'
            output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
        finally:
            executor.remove_node(observer); executor.remove_node(fixture); executor.shutdown()
            observer.destroy_node(); fixture.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': main()
