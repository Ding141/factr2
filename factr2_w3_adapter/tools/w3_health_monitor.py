#!/usr/bin/env python3
"""Run with system Python + W3 overlay; never source NEXT or import torch."""
from pathlib import Path
import sys
import time

# Direct source execution deliberately needs no adapter install in the W3 shell.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from diagnostic_msgs.msg import DiagnosticArray
from w3_robot_bridge.msg import MotorStateArray
from factr2_w3_adapter.health import HealthGate
from factr2_w3_adapter.diagnostics import diagnostic


class HealthMonitor(Node):
    def __init__(self, **kwargs):
        super().__init__('w3_health_monitor', **kwargs)
        readonly = ParameterDescriptor(read_only=True)
        hz = self.declare_parameter('publish_hz', 50.0, readonly).value
        timeout = self.declare_parameter('input_timeout_seconds', 0.25, readonly).value
        sides = self.declare_parameter('sides', ['left', 'right'], readonly).value
        self.gate = HealthGate(timeout, sides)
        import math
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('publish_hz must be finite and positive')
        self.subscription = self.create_subscription(MotorStateArray, '/w3_robot_bridge_node/state',
                                                       self._raw, qos_profile_sensor_data)
        self.publisher = self.create_publisher(DiagnosticArray, '/factr2/w3_health', qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / hz, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'Bridge-reported feedback only; Python={sys.executable}; no command endpoints')

    def _raw(self, msg):
        self.gate.ingest(msg, self.get_clock().now().nanoseconds, time.monotonic())

    def _tick(self):
        now = self.get_clock().now()
        msg = DiagnosticArray()
        msg.header.stamp = now.to_msg()
        for side, values in self.gate.assess(now.nanoseconds, time.monotonic()).items():
            msg.status.extend(diagnostic(now.to_msg(), f'factr2/w3/{side}', values,
                                         values['healthy'], values['reason']).status)
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HealthMonitor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
