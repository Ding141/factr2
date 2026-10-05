"""Single-arm, read-only W3 state adapter. Parameters are fixed at startup."""
from dataclasses import fields
import time
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray
from .diagnostics import diagnostic
from .validation import AdapterConfig, AdapterGate


class AdapterNode(Node):
    def __init__(self, **kwargs):
        super().__init__('w3_next_adapter', **kwargs)
        defaults = AdapterConfig(side='left')
        params = {}
        for field in fields(AdapterConfig):
            default = '' if field.name in ('side', 'output_root') else getattr(defaults, field.name)
            params[field.name] = self.declare_parameter(
                field.name, default, ParameterDescriptor(read_only=True)).value
        self.config = AdapterConfig(**params)
        self.gate = AdapterGate(self.config)
        self.state_sub = self.create_subscription(JointState, self.config.state_topic,
            lambda msg: self._input('state', msg), qos_profile_sensor_data)
        self.command_sub = self.create_subscription(JointState, self.config.command_state_topic,
            lambda msg: self._input('command', msg), qos_profile_sensor_data)
        self.health_sub = self.create_subscription(DiagnosticArray, self.config.health_topic,
            self._health, qos_profile_sensor_data)
        self.outputs = {key: self.create_publisher(JointState, f'{self.config.output_root}/{key}',
                                                  qos_profile_sensor_data)
                        for key in ('joint_pos', 'joint_vel', 'joint_cmd', 'joint_effort')}
        self.status_pub = self.create_publisher(DiagnosticArray,
            f'{self.config.output_root}/adapter_status', qos_profile_sensor_data)
        self.last_warnings = {}
        # Wall/steady timer continues watchdog checks while /clock is paused.
        self.timer = self.create_timer(1.0 / self.config.publish_hz, self._tick,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'{self.config.side}: {self.config.output_root}; '
                               f'hardware_health={self.config.require_hardware_health}; read-only')

    def _input(self, kind, msg):
        self.gate.ingest(kind, msg, self.get_clock().now().nanoseconds, time.monotonic())

    def _health(self, msg):
        self.gate.ingest_health(msg, self.get_clock().now().nanoseconds, time.monotonic())

    def _tick(self):
        now = self.get_clock().now()
        mono = time.monotonic()
        sample = self.gate.sample(now.nanoseconds, mono)
        if sample is not None:
            stamp, vectors = sample
            for key, (field, values) in vectors.items():
                msg = JointState()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                msg.name = list(self.config.names)
                setattr(msg, field, list(values))
                self.outputs[key].publish(msg)
        values = dict(self.gate.metrics, **self.gate.counts, side=self.config.side,
                      recent_rejection=self.gate.last_rejection,
                      state_source_stamp_ns=(self.gate.sources['state'].stamp if self.gate.sources['state'] else 0),
                      command_source_stamp_ns=(self.gate.sources['command'].stamp if self.gate.sources['command'] else 0))
        self.status_pub.publish(diagnostic(now.to_msg(), f'factr2/adapter/{self.config.side}',
                                          values, self.gate.reason == 'ok', self.gate.reason))
        reason = self.gate.reason
        if reason != 'ok' and mono - self.last_warnings.get(reason, float('-inf')) >= 1.0:
            self.get_logger().warning(reason)
            self.last_warnings[reason] = mono


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AdapterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
