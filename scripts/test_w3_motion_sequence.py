"""Offline tests: fake publishers/ROS clients only; never initialize real ROS or hardware."""
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import yaml

import quick_capture as capture
import w3_motion_sequence as sequence
from w3_joint_target import target_degrees

EXAMPLE = sequence.ROOT / 'config/w3/motions/left_all_joints_pilot.yaml'
RIGHT_EXAMPLE = sequence.ROOT / 'config/w3/motions/right_all_joints_pilot.yaml'


def description(side='left'):
    return '<robot><ros2_control>' + ''.join(
        f'<joint name="{side}_joint_{i}"><param name="urdf_lower">-1</param>'
        '<param name="urdf_upper">1</param></joint>' for i in range(7)) + '</ros2_control></robot>'


class FakeClient:
    def __init__(self, side):
        self.side = side
        self.description = description(side)
        self.sent = []
        self.waited = []
        self.closed = False

    def preflight(self):
        return [.1] * 7

    def guard(self):
        pass

    def send(self, goal, duration, capture_guard):
        capture_guard()
        self.sent.append((list(goal), duration))

    def monitor(self, seconds, capture_guard):
        capture_guard()
        self.waited.append(seconds)

    def endpoint(self, goal, tolerance):
        return [.8] * 7  # Existing pilot tracking error does not accumulate into targets.

    def close(self):
        self.closed = True


class MotionSequenceTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.raw, self.digest = sequence.load_motion(EXAMPLE)

    def load_mutation(self, mutate):
        data = copy.deepcopy(self.plan)
        mutate(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'motion.yaml'
            path.write_text(yaml.safe_dump(data))
            return sequence.load_motion(path)

    def test_example_has_seven_independent_out_and_return_pairs(self):
        self.assertEqual(len(self.plan['steps']), 14)
        for index in range(7):
            expected = [0.] * 7
            expected[index] = 2
            self.assertEqual(self.plan['steps'][2 * index]['offset_deg'], expected)
            self.assertEqual(self.plan['steps'][2 * index + 1]['offset_deg'], [0.] * 7)
        self.assertEqual(2 + 3 + sum(s['duration_seconds'] + s['hold_seconds'] for s in self.plan['steps']), 103)

    def test_right_example_has_eight_degree_targets_and_explicit_ten_degree_limit(self):
        plan, _, _ = sequence.load_motion(RIGHT_EXAMPLE)
        self.assertEqual(plan['side'], 'right')
        self.assertEqual(plan['max_step_deg'], 10)
        self.assertEqual(len(plan['steps']), 14)
        resolved = sequence.resolve_motion(plan, [.1] * 7, description('right'))
        for index in range(7):
            expected = [0.] * 7
            expected[index] = 8
            self.assertEqual(plan['steps'][2 * index]['offset_deg'], expected)
            self.assertEqual(plan['steps'][2 * index + 1]['offset_deg'], [0.] * 7)
            self.assertAlmostEqual(resolved[2 * index]['target_rad'][index], .1 + math.radians(8))
            self.assertEqual(resolved[2 * index + 1]['target_rad'], [.1] * 7)
        self.assertTrue(all(s['duration_seconds'] == 8 and s['hold_seconds'] == 2 for s in plan['steps']))
        self.assertEqual(2 + 3 + sum(s['duration_seconds'] + s['hold_seconds'] for s in plan['steps']), 145)
        with self.assertRaises(ValueError):
            sequence.resolve_motion(plan, [.1] * 7, description('left'))

    def test_extended_limit_is_explicit_and_never_exceeds_ten_degrees(self):
        self.assertEqual(self.plan['max_step_deg'], 5)
        with self.assertRaises(ValueError):
            target_degrees([0.] * 7, step=(0, 8))
        self.assertAlmostEqual(target_degrees([0.] * 7, step=(0, 8), max_delta_deg=10)[0], math.radians(8))
        for degrees in (10.01, float('inf')):
            with self.assertRaises(ValueError):
                target_degrees([0.] * 7, step=(0, degrees), max_delta_deg=10)
        with self.assertRaises(ValueError):
            target_degrees([0.] * 7, step=(0, 8), max_delta_deg=11)
        for mutate in (
            lambda p: p.update(max_step_deg=11),
            lambda p: p.update(max_step_deg=10) or p['steps'][0].update(offset_deg=[11] + [0] * 6),
            lambda p: p.update(max_step_deg=10) or p['steps'][0].update(offset_deg=[8] + [0] * 6) or
                      p['steps'][1].update(offset_deg=[-8] + [0] * 6),
        ):
            with self.assertRaises(ValueError):
                self.load_mutation(mutate)

    def test_rejects_invalid_or_misspelled_file_values(self):
        mutations = [
            lambda p: p.update(side='dual'), lambda p: p.update(units='rad'),
            lambda p: p.update(version=True), lambda p: p.update(reference='latest_feedback'),
            lambda p: p.update(extra_option=1), lambda p: p.update(steps=[]),
            lambda p: p['steps'][0].update(offset_deg=[0] * 6),
            lambda p: p['steps'][0].update(offset_deg=[float('nan')] + [0] * 6),
            lambda p: p['steps'][0].update(offset_deg=[float('inf')] + [0] * 6),
            lambda p: p['steps'][0].update(offset_deg=[True] + [0] * 6),
            lambda p: p['steps'][0].update(offset_deg=[6] + [0] * 6),
            lambda p: p['steps'][0].update(duration_seconds=2),
            lambda p: p['steps'][0].update(hold_seconds=0),
            lambda p: p['steps'][1].update(name=p['steps'][0]['name']),
            lambda p: p['steps'][0].update(duration_second=5),
            lambda p: p.update(arrival_tolerance_deg=10),
            lambda p: p['steps'][0].update(offset_deg=[5] + [0] * 6) or
                      p['steps'][1].update(offset_deg=[-5] + [0] * 6),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.load_mutation(mutate)

    def test_targets_share_one_baseline_and_all_limits_checked_upfront(self):
        initial = [.1] * 7
        resolved = sequence.resolve_motion(self.plan, initial, description())
        for index in range(7):
            self.assertAlmostEqual(resolved[2 * index]['target_rad'][index], .1 + math.radians(2))
            self.assertEqual(resolved[2 * index + 1]['target_rad'], initial)
        initial[6] = .99  # Only the late J7 step violates the upper hardware limit.
        with self.assertRaises(ValueError):
            sequence.resolve_motion(self.plan, initial, description())
        self.assertEqual(initial[6], .99)

    def test_sequence_waits_full_duration_and_hold_before_next_send(self):
        client = FakeClient('left')
        report = {'events': [], 'status': 'running'}
        saved = []
        with patch('sys.stdout', new_callable=io.StringIO):
            sequence.execute_motion(self.plan, sequence.resolve_motion(self.plan, client.preflight(), description()),
                                    client, lambda: None, report, lambda: saved.append(copy.deepcopy(report)))
        self.assertEqual(len(client.sent), 14)
        self.assertEqual(client.waited, [2.] + [7.] * 14 + [3.])
        self.assertEqual(report['status'], 'completed')
        self.assertTrue(all(e['status'] == 'completed' for e in report['events']))
        self.assertEqual(client.sent[-1][0], [.1] * 7)
        self.assertEqual(len(saved), 14 * 3 + 1)

    def test_endpoint_or_capture_failure_never_sends_next_step(self):
        for fault in ('endpoint', 'capture'):
            client = FakeClient('left')
            if fault == 'endpoint':
                client.endpoint = Mock(side_effect=RuntimeError('arrival error'))
            def capture_guard():
                if fault == 'capture' and client.sent:
                    raise RuntimeError('C stopped')
            report = {'events': []}
            with patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(RuntimeError):
                sequence.execute_motion(self.plan, sequence.resolve_motion(self.plan, client.preflight(), description()),
                                        client, capture_guard, report, lambda: None)
            self.assertEqual(len(client.sent), 1)
            self.assertEqual(len(report['events']), 1)

    def test_freshness_and_tracking_gate_on_fake_state(self):
        client = sequence.MotionClient.__new__(sequence.MotionClient)
        client.max_step_deg = 10  # The larger motion limit must not relax the 5-degree tracking gate.
        client.node = Mock()
        client.node.get_clock.return_value.now.return_value.nanoseconds = 10_000_000_000
        client.state = {'q': [0] * 7, 'cmd': [0] * 7, 'ok': True}
        for key in ('q', 'cmd', 'health'):
            client.state[key + '_received'] = 20
            client.state[key + '_stamp'] = 10_000_000_000
        with patch.object(sequence.time, 'monotonic', return_value=20.1):
            client.guard()
            client.state['health_stamp'] = 9_700_000_000
            with self.assertRaises(RuntimeError):
                client.guard()
            client.state['health_stamp'] = 10_000_000_000
            client.state['cmd'][6] = math.radians(6)
            with self.assertRaises(RuntimeError):
                client.guard()
        with patch.object(sequence.time, 'monotonic', return_value=20.3), self.assertRaises(RuntimeError):
            client.guard()

    def test_endpoint_requires_controller_goal_and_bounded_actual_error(self):
        client = sequence.MotionClient.__new__(sequence.MotionClient)
        client.guard = Mock()
        client.state = {'q': [math.radians(.8)] * 7, 'cmd': [0.] * 7}
        self.assertEqual(len(client.endpoint([0.] * 7, 1.5)), 7)
        client.state['cmd'][0] = math.radians(.1)
        with self.assertRaises(RuntimeError):
            client.endpoint([0.] * 7, 1.5)
        client.state['cmd'][0] = 0.
        client.state['q'][0] = math.radians(1.6)
        with self.assertRaises(RuntimeError):
            client.endpoint([0.] * 7, 1.5)

    def test_send_uses_current_controller_target_at_t0_not_actual_feedback(self):
        class Trajectory:
            def __init__(self):
                self.points = []
        class Point:
            def __init__(self):
                self.time_from_start = types.SimpleNamespace(sec=0, nanosec=0)
        module = types.ModuleType('trajectory_msgs.msg')
        module.JointTrajectory, module.JointTrajectoryPoint = Trajectory, Point
        client = sequence.MotionClient.__new__(sequence.MotionClient)
        client.check_mode = Mock()
        client.wait_until = Mock()
        client.guard = Mock()
        client.names = [f'left_joint_{i}' for i in range(7)]
        client.max_step_deg = 5
        client.description = description()
        client.node = Mock()
        client.publisher = Mock()
        client.state = {'q': [.1] * 7, 'cmd': [.11] * 7}
        with patch.dict(sys.modules, {'trajectory_msgs.msg': module}):
            client.send([.12] * 7, 5, lambda: None)
        sent = client.publisher.publish.call_args.args[0]
        self.assertEqual(sent.points[0].positions, [.11] * 7)
        self.assertEqual(sent.points[0].time_from_start.sec, 0)
        self.assertEqual(sent.points[1].positions, [.12] * 7)
        self.assertEqual(sent.points[1].time_from_start.sec, 5)

    def test_default_cli_is_offline_and_never_constructs_ros_client(self):
        with patch('sys.argv', ['w3_motion_sequence.py', '--file', str(EXAMPLE)]), \
             patch.object(sequence, 'MotionClient', side_effect=AssertionError('No ROS allowed')), \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(sequence.main(), 0)
        self.assertIn('未连接 ROS', output.getvalue())
        result = subprocess.run(['/usr/bin/python3.10', str(sequence.ROOT / 'scripts/w3_motion_sequence.py'),
                                 '--file', str(EXAMPLE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_send_lifecycle_audit_and_duplicate_rejection_with_fake_ros(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / 'data/quick_capture/left/session'
            (session / 'audit').mkdir(parents=True)
            info = {'pid': 123456789, 'session': str(session), 'side': 'left', 'domain': 0,
                    'motion_sha256': self.digest}
            (session.parent / 'active.json').write_text(json.dumps(info))
            client = FakeClient('left')
            with patch.object(sequence, 'ROOT', root), patch.object(sequence.os, 'kill') as kill, \
                 patch.dict(sequence.os.environ, {'ROS_DOMAIN_ID': '0'}), \
                 patch.object(sequence, 'MotionClient', return_value=client), \
                 patch('sys.argv', ['sequence.py', '--file', str(EXAMPLE), '--send']), \
                 patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(sequence.main(), 0)
                self.assertEqual(sequence.main(), 1)  # Same capture cannot replay again.
                self.assertTrue(all(call.args == (123456789, 0) for call in kill.call_args_list))
            report = json.loads((session / 'audit/motion_run.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['motion_sha256'], self.digest)
            self.assertTrue(client.closed)
            self.assertEqual(len(client.sent), 14)

    def test_capture_hash_domain_and_exit_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / 'data/quick_capture/left/session'
            session.mkdir(parents=True)
            info = {'pid': 123456789, 'session': str(session), 'side': 'left', 'domain': 0,
                    'motion_sha256': self.digest}
            (session.parent / 'active.json').write_text(json.dumps(info))
            with patch.object(sequence, 'ROOT', root), patch.object(sequence.os, 'kill'), \
                 patch.dict(sequence.os.environ, {'ROS_DOMAIN_ID': '0'}):
                self.assertEqual(sequence.read_active('left', self.digest), info)
                with self.assertRaises(RuntimeError):
                    sequence.read_active('left', 'different')
                with patch.dict(sequence.os.environ, {'ROS_DOMAIN_ID': '73'}), self.assertRaises(RuntimeError):
                    sequence.read_active('left', self.digest)
                with patch.object(sequence.os, 'kill', side_effect=ProcessLookupError), self.assertRaises(RuntimeError):
                    sequence.read_active('left', self.digest)

    def test_capture_reports_incomplete_plan_as_failure(self):
        # The existing capture lifecycle test covers real offline H5 validation.
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            audit = session / 'audit'
            audit.mkdir()
            (audit / 'motion.yaml').write_bytes(self.raw)
            (session / 'fixture.h5').touch()
            quality = {'accepted': True, 'episodes': {}, 'errors': []}
            (session / 'quality.json').write_text(json.dumps(quality))
            with patch.object(capture.subprocess, 'run', return_value=types.SimpleNamespace(stdout='')), \
                 patch.object(capture, 'run', return_value='offline coverage'), \
                 patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(capture.summarize(session, {}), 1)
                self.assertIn('整套动作执行：FAIL', output.getvalue())
                sequence.write_json(audit / 'motion_run.json', {'status': 'completed',
                    'motion_sha256': self.digest, 'events': [{'status': 'completed'}] * 14})
                self.assertEqual(capture.summarize(session, {}), 0)


if __name__ == '__main__':
    unittest.main()
