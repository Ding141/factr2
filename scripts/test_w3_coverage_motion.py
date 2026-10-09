"""Offline meaningful motion tests; no ROS, motor commands or CAN initialization."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from w3_coverage_plan import ArmModel, build_plan, load_capture_motion, validate_plan
from w3_coverage_motion import CoverageClient, check_start, coverage_report, execute, window_indices

ROOT = Path(__file__).resolve().parents[1]


def fixture_description():
    # Actual project geometry; supply hardware limits offline, without loading ROS/xacro.
    path = ROOT.parent / 'dual_arm_robot/src/description/dual_arm_support/urdf/dual_arm_robot.urdf'
    root = ET.fromstring(path.read_text())
    for old in root.findall('ros2_control'):
        root.remove(old)
    control = ET.SubElement(root, 'ros2_control')
    for side in ('left', 'right'):
        for i in range(7):
            name = f'{side}_joint_{i}'
            geometric = root.find(f'joint[@name="{name}"]/limit')
            joint = ET.SubElement(control, 'joint', name=name)
            for key, attr in [('urdf_lower', 'lower'), ('urdf_upper', 'upper')]:
                ET.SubElement(joint, 'param', name=key).text = geometric.get(attr)
    return ET.tostring(root, encoding='unicode')


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.description = fixture_description()
        cls.config = yaml.safe_load((ROOT / 'config/w3/motions/left_coverage_conservative.yaml').read_text())
        cls.config['repeats'] = 1
        cls.plan = build_plan(cls.config, cls.description, [0]*7)

    def test_each_joint_covers_both_bounds_at_both_speeds(self):
        self.assertEqual(len(self.plan['steps']), 72)
        for speed in ('slow', 'fast'):
            for joint in range(7):
                rows = np.vstack([s['positions_deg'] for s in self.plan['steps']
                                  if s['phase'] == 'single_joint' and s['joint'] == joint and s['speed'] == speed])
                np.testing.assert_allclose([rows[:, joint].min(), rows[:, joint].max()], self.config['working_bounds_deg'][joint])
        phases = [s['phase'] for s in self.plan['steps']]
        self.assertLess(max(i for i, p in enumerate(phases) if p == 'single_joint'), min(i for i, p in enumerate(phases) if p == 'cartesian'))

    def test_fast_sweeps_are_shorter_and_return_start(self):
        slow = [s for s in self.plan['steps'] if s['phase'] == 'single_joint' and s['speed'] == 'slow']
        fast = [s for s in self.plan['steps'] if s['phase'] == 'single_joint' and s['speed'] == 'fast']
        self.assertTrue(all(f['duration_seconds'] < s['duration_seconds'] for s, f in zip(slow, fast)))
        np.testing.assert_allclose(self.plan['steps'][-1]['positions_deg'][-1], self.plan['initial_deg'])

    def test_jacobian_agrees_with_finite_difference(self):
        m = ArmModel(self.description, 'left', self.config['tool_frame'])
        q = np.deg2rad(self.config['home_deg'])
        pose, j = m.fk(q, True)
        for i in range(7):
            perturbed = q.copy(); perturbed[i] += 1e-6
            np.testing.assert_allclose((m.fk(perturbed)[:3, 3] - pose[:3, 3])/1e-6, j[:3, i], atol=3e-7)

    def test_cartesian_lines_and_circle_geometry(self):
        m = ArmModel(self.description, 'left', self.config['tool_frame'])
        center = m.fk(np.deg2rad(self.config['home_deg']))[:3, 3]
        for s in self.plan['steps']:
            if s['phase'] != 'cartesian': continue
            xyz = np.array([m.fk(np.deg2rad(row))[:3, 3] for row in s['positions_deg'][::5]])
            self.assertGreater(np.count_nonzero(np.ptp(s['positions_deg'], axis=0) > .001), 1)
            if s['kind'].startswith('line_'):
                axis = 'xyz'.index(s['kind'][5])
                self.assertLess(np.max(np.abs(np.delete(xyz-center, axis, axis=1))), .0001)
            else:
                circle_center = center.copy(); circle_center[0] -= self.config['cartesian']['circle_radius_m']
                radial = np.linalg.norm((xyz-circle_center)[:, :2], axis=1)
                self.assertLess(np.max(np.abs(radial-self.config['cartesian']['circle_radius_m'])), .0001)
                self.assertLess(np.max(np.abs(xyz[:, 2]-center[2])), .0001)

    def test_entire_plan_rejects_limits_speed_discontinuity_nonfinite_hash(self):
        mutations = [lambda p: p['steps'][2]['positions_deg'][4].__setitem__(0, 1000),
                     lambda p: p['steps'][2]['positions_deg'][4].__setitem__(0, float('nan')),
                     lambda p: p['steps'][2]['positions_deg'][4].__setitem__(0, 1),
                     lambda p: p['steps'][1]['positions_deg'][0].__setitem__(0, .01),
                     lambda p: p.__setitem__('model_sha256', 'bad'),
                     lambda p: p['config'].__setitem__('repeats', True)]
        for mutation in mutations:
            p = copy.deepcopy(self.plan); mutation(p)
            with self.assertRaises(ValueError): validate_plan(p)

    def test_unreachable_cartesian_path_is_rejected_before_any_send(self):
        c = copy.deepcopy(self.config); c['cartesian']['line_half_extent_m'] = .05
        with self.assertRaises(ValueError): build_plan(c, self.description, [0]*7)

    def test_capture_dispatch_preserves_legacy_pilot_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'motion.yaml'
            path.write_text(yaml.dump(self.plan, Dumper=yaml.CSafeDumper, sort_keys=False))
            plan, raw, digest = load_capture_motion(path)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(plan['side'], 'left')
        plan, _, _ = load_capture_motion(ROOT / 'config/w3/motions/left_all_joints_pilot.yaml')
        self.assertNotIn('schema', plan)

    def test_start_rejects_changed_model_or_pose_or_controller_target(self):
        p = self.plan
        check_start(p, [0]*7, self.description, [0]*7)
        for q, desc, cmd in [([.1]*7, self.description, [0]*7), ([0]*7, self.description+' ', [0]*7), ([0]*7, self.description, [.01]*7)]:
            with self.assertRaises(RuntimeError): check_start(p, q, desc, cmd)

    def test_windows_overlap_bracket_now_and_bound_future(self):
        times = np.arange(501)*.02
        for elapsed in np.arange(0, 9.9, .137):
            start, end = window_indices(times, elapsed)
            self.assertLessEqual(times[start], elapsed)
            self.assertGreater(times[end-1], elapsed)
            self.assertLessEqual(times[end-1], elapsed+.621)
            self.assertGreaterEqual(end-start, 2)

    def test_failure_prevents_all_following_steps(self):
        class Fake:
            def __init__(self): self.calls = 0
            def guard(self): pass
            def play(self, step, guard):
                self.calls += 1
                if self.calls == 2: raise RuntimeError('tracking fault')
                return [0]*7
        client = Fake(); report = {'events': []}
        with self.assertRaisesRegex(RuntimeError, 'tracking fault'):
            execute(self.plan, client, lambda: None, report, lambda: None)
        self.assertEqual(client.calls, 2)
        self.assertEqual(report['events'][0]['status'], 'completed')
        self.assertEqual(report['events'][1]['status'], 'sending')
        self.assertNotIn('status', report)

    def test_streaming_playback_uses_common_clock_and_stops_on_capture_fault(self):
        step = self.plan['steps'][1]
        class Message:
            def __init__(self):
                self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
                self.points = []
        class Point:
            def __init__(self):
                self.time_from_start = SimpleNamespace(sec=0, nanosec=0)
        class Publisher:
            def __init__(self): self.messages = []
            def get_subscription_count(self): return 1
            def publish(self, message): self.messages.append(message)
        class Node:
            def __init__(self): self.ns = 1_000_000_000; self.pub = Publisher()
            def get_clock(self): return self
            def now(self): return SimpleNamespace(nanoseconds=self.ns)
            def create_publisher(self, *args): return self.pub
        def make_client():
            client = CoverageClient.__new__(CoverageClient)
            client.node = Node(); client.plan = self.plan; client.names = self.plan['joint_names']
            client.publisher = None; client.expected_deg = None
            client.state = {'cmd': np.deg2rad(step['positions_deg'][0]).tolist(), 'cmd_stamp': client.node.ns}
            def spin(*args, **kwargs):
                client.node.ns += 10_000_000
                if client.node.pub.messages:
                    msg = client.node.pub.messages[-1]
                    base = msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec
                    elapsed = (client.node.ns-base)*1e-9
                    times = [p.time_from_start.sec+p.time_from_start.nanosec*1e-9 for p in msg.points]
                    client.state['cmd'] = [np.interp(elapsed, times, [p.positions[i] for p in msg.points]) for i in range(7)]
                client.state['cmd_stamp'] = client.node.ns
            client.ros = SimpleNamespace(spin_once=spin)
            client.check_mode = lambda: None
            def guard():
                if client.expected_deg is not None:
                    np.testing.assert_allclose(np.rad2deg(client.state['cmd']), client.expected_deg, atol=.00001)
            client.guard = guard
            client.endpoint = lambda goal, tolerance: [0]*7
            return client
        modules = {'trajectory_msgs': SimpleNamespace(), 'trajectory_msgs.msg': SimpleNamespace(JointTrajectory=Message, JointTrajectoryPoint=Point)}
        with patch.dict(sys.modules, modules):
            client = make_client()
            def fault():
                if len(client.node.pub.messages) >= 2:
                    raise RuntimeError('capture lost')
            with patch('w3_coverage_motion.time.monotonic', side_effect=lambda: client.node.ns*1e-9):
                with self.assertRaisesRegex(RuntimeError, 'capture lost'):
                    client.play(step, fault)
            self.assertEqual(len(client.node.pub.messages), 2)
            self.assertEqual(len({(m.header.stamp.sec,m.header.stamp.nanosec) for m in client.node.pub.messages}), 1)
            # Simulated monotonic time advances with ROS time, as a real scheduler would.
            client = make_client()
            with patch('w3_coverage_motion.time.monotonic', side_effect=lambda: client.node.ns*1e-9):
                self.assertEqual(client.play(step, lambda: None), [0]*7)
            messages = client.node.pub.messages
            self.assertGreater(len(messages), 3)
            self.assertEqual(len({(m.header.stamp.sec,m.header.stamp.nanosec) for m in messages}), 1)
            np.testing.assert_allclose(messages[-1].points[-1].positions, np.deg2rad(step['positions_deg'][-1]))

    def test_report_uses_nanosecond_h5_stamps_and_keeps_real_angles(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp); (session/'audit').mkdir()
            (session/'audit/motion.yaml').write_text(yaml.dump(self.plan, Dumper=yaml.CSafeDumper))
            now = datetime.now(timezone.utc).timestamp()
            utc = datetime.fromtimestamp(now, timezone.utc).isoformat()
            end = datetime.fromtimestamp(now+2, timezone.utc).isoformat()
            event = {k: self.plan['steps'][1][k] for k in ('name','phase','kind','speed')}
            event.update(status='completed',utc=utc,completed_utc=end)
            (session/'audit/motion_run.json').write_text(json.dumps({'status':'completed','motion_sha256':hashlib.sha256((session/'audit/motion.yaml').read_bytes()).hexdigest(),'events':[event],'completed_utc':end}))
            with h5py.File(session/'test.h5','w') as h:
                h.create_group('ep_0001')  # Empty final episode must not hide valid rows.
                for stream in ('joint_pos','joint_cmd','joint_vel'):
                    g=h.create_group('ep_0000/'+stream)
                    g['timestamps']=np.array((now+np.arange(60)*.02)*1e9,dtype=np.int64)
                    g['data']=np.ones((60,7))*.01
            report=coverage_report(session)
            self.assertEqual(report['events'][0]['rows'],60)
            self.assertTrue(report['motion_completed'])
            np.testing.assert_allclose(report['measured_min_deg'],np.rad2deg([.01]*7))


if __name__ == '__main__':
    unittest.main()
