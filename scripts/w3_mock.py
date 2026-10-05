#!/usr/bin/env python3
"""Seeded standard W3 state/command source. No hardware or command clients."""
import argparse
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def values(stamp, indices, seed=0):
    t = stamp * 1e-9
    q = [.1 * math.sin(t + i + seed * .01) for i in indices]
    vel = [.2 + i * .03 for i in indices]
    cmd = [.15 + i * .01 for i in indices]
    torque = [1. + i * .1 + .3 * x for i, x in zip(indices, q)]
    return q, vel, cmd, torque


class MockNode(Node):
    def __init__(self, side='left', hz=300, seed=0, faults=()):
        super().__init__('w3_standard_mock_' + side)
        self.side, self.seed, self.faults = side, seed, faults
        self.state = self.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.command = self.create_publisher(JointState, '/joint_position_controller/command_state', qos_profile_sensor_data)
        self.started = time.monotonic()
        self.last_stamp = None
        self.timer = self.create_timer(1 / hz, self.tick)

    def tick(self):
        elapsed = time.monotonic() - self.started
        kinds = [f['kind'] for f in self.faults if f['at'] <= elapsed < f['at'] + f['duration']]
        if 'pause' in kinds:
            return
        stamp = self.get_clock().now().nanoseconds
        if 'rollback' in kinds:
            stamp -= 300_000_000
        if 'duplicate' in kinds and self.last_stamp is not None:
            stamp = self.last_stamp
        else:
            self.last_stamp = stamp
        sides = ('left', 'right') if self.side == 'both' else (self.side,)
        names = [f'{s}_joint_{i}' for s in sides for i in range(7)]
        indices = list(range(7)) * len(sides)
        q, vel, cmd, torque = values(stamp, indices, self.seed)
        state, command = JointState(), JointState()
        for msg in (state, command):
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
            msg.name = names.copy()
        state.position, state.velocity, state.effort = q, vel, torque
        command.position, command.velocity = cmd, [0.] * len(cmd)
        if 'skew' in kinds:
            cs = stamp - 100_000_000
            command.header.stamp.sec, command.header.stamp.nanosec = divmod(cs, 10**9)
        if 'nan' in kinds:
            state.effort[0] = float('nan')
        if 'order' in kinds:
            state.name.reverse()  # names/values mismatch intentionally
        if 'missing' in kinds:
            state.name.pop(); state.position.pop(); state.velocity.pop(); state.effort.pop()
        self.state.publish(state)
        self.command.publish(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', choices=['left', 'right', 'both'], default='left')
    parser.add_argument('--hz', type=float, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--faults', default='[]', help='JSON: [{at:seconds,duration:seconds,kind:pause|...}]')
    args, ros = parser.parse_known_args()
    rclpy.init(args=ros)
    node = MockNode(args.side, args.hz, args.seed, json.loads(args.faults))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
