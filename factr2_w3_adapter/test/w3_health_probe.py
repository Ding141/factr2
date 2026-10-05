#!/usr/bin/env python3
"""Offline bridge-state publisher; run only in isolated W3 test domains."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from w3_robot_bridge.msg import MotorStateArray, MotorState

assert sys.executable == '/usr/bin/python3.10'
assert not any('/.venv/' in p for p in sys.path)
assert 'torch' not in sys.modules
control=Path(sys.argv[1])
rclpy.init(args=[])
node=rclpy.create_node('w3_raw_health_mock')
pub=node.create_publisher(MotorStateArray,'/w3_robot_bridge_node/state',qos_profile_sensor_data)
print(json.dumps({'python':sys.executable,'prefixes':__import__('os').environ.get('AMENT_PREFIX_PATH'),
                  'imports_torch':False,'mode':'mock bridge feedback, no hardware'}),flush=True)


def tick():
    mode=control.read_text().strip()
    if mode=='stop':return
    msg=MotorStateArray()
    msg.header.stamp=node.get_clock().now().to_msg()
    for channel in (0,1):
        if mode=='right_only' and channel==0:continue
        for i in range(7):
            m=MotorState()
            m.header.stamp=msg.header.stamp
            m.channel=channel; m.motor_index=i
            m.position=float(i)*0.01; m.velocity=0.0; m.torque=1.0
            m.online=True; m.enabled=True; m.error_flags=1
            if channel==0 and i==0:
                if mode=='missing_left':continue
                if mode=='offline_left':m.online=False
                if mode=='disabled_left':m.enabled=False
                if mode=='fault_left':m.error_flags=8
                if mode=='nan_left':m.torque=float('nan')
            msg.motors.append(m)
    if mode=='duplicate_left':msg.motors.append(deepcopy(msg.motors[0]))
    pub.publish(msg)


timer=node.create_timer(0.005,tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
try:rclpy.spin(node)
except KeyboardInterrupt:pass
finally:
    node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
