"""Mock standard-message exchange, never creates command endpoints."""
import json
import os
import sys
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

rclpy.init()
node = rclpy.create_node('contract_probe_' + sys.argv[1])
print(json.dumps({k:os.environ.get(k) for k in ('ROS_DOMAIN_ID','RMW_IMPLEMENTATION','ROS_LOCALHOST_ONLY')}), flush=True)
received = []
topic = '/factr2_test/environment_joint_state'
expected = dict(name=['left_joint_0'], position=[0.125], velocity=[-0.25], effort=[1.5])
msg = JointState()
msg.header.stamp.sec = 123
msg.header.stamp.nanosec = 456789
for key, value in expected.items():
    setattr(msg, key, value)
if sys.argv[1] == 'receive':
    sub = node.create_subscription(JointState, topic, received.append, qos_profile_sensor_data)
    deadline = time.monotonic() + 12
    while not received and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    assert received, 'DDS message not received'
    actual = received[0]
    for key, value in expected.items():
        assert list(getattr(actual, key)) == value
    assert actual.header.stamp == msg.header.stamp
    print(json.dumps({'result':'PASS', 'stamp':[123,456789], 'fields':expected,
                      'qos':'BEST_EFFORT/VOLATILE/KEEP_LAST depth=5',
                      'executable':sys.executable}))
else:
    pub = node.create_publisher(JointState, topic, qos_profile_sensor_data)
    deadline = time.monotonic() + 15
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    assert pub.get_subscription_count() > 0, 'DDS peer discovery failed'
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.02)
node.destroy_node()
rclpy.shutdown()
