"""Offline checks only: no ROS initialization, CAN, controllers or targets."""
import json
import io
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

import quick_capture as capture
import w3_joint_target as target
import w3_motion_sequence as sequence


def description(path):
    joints = ''.join(f'<joint name="left_joint_{i}"><param name="urdf_lower">-1</param>'
                     '<param name="urdf_upper">1</param></joint>' for i in range(7))
    return f'<robot><ros2_control><hardware><param name="offsets_yaml">{path}</param>' \
           f'</hardware>{joints}</ros2_control></robot>'


class QuickFlowTests(unittest.TestCase):
    def test_domain_zero_overrides_next_default_73(self):
        with patch.dict(capture.os.environ, {}, clear=True):
            env = capture.runtime_environment(0)
        self.assertEqual(env['ROS_DOMAIN_ID'], '0')
        self.assertEqual(env['ROS_LOCALHOST_ONLY'], '0')
        command = capture.w3_command(['ros2', 'param', 'dump', '/robot_state_publisher'], env)
        self.assertIn('ROS_DOMAIN_ID=0', command)
        self.assertIn('ROS_LOCALHOST_ONLY=0', command)
        self.assertNotIn('real_robot.launch.py', command)

    def test_inactive_not_mistaken_for_active(self):
        name = 'joint_position_controller'
        self.assertFalse(capture.controller_is_active(f'{name} plugin inactive', name))
        self.assertFalse(capture.controller_is_active(f'other_{name} plugin active', name))
        self.assertTrue(capture.controller_is_active(f'\x1b[92m{name}\x1b[0m plugin active', name))

    def test_configuration_uses_loaded_calibration_and_freezes_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            templates = root / 'config/w3/left'
            templates.mkdir(parents=True)
            shutil.copy2(capture.ROOT / 'config/w3/left/record.yaml', templates / 'record.yaml')
            calibration = root / 'actual.yaml'
            calibration.write_text('transport: w3\noffsets: []\n')
            friction = root / 'friction.yaml'
            friction.write_text('model: example\n')
            snapshots = {
                'controllers.txt': 'joint_position_controller plugin active',
                'position.yaml': yaml.safe_dump({'p': {'ros__parameters': {'kp_gains': [1]*7}}}),
                'gravity.yaml': yaml.safe_dump({'g': {'ros__parameters': {'friction_model_yaml': str(friction)}}}),
                'robot_description.yaml': yaml.safe_dump({'r': {'ros__parameters': {'robot_description': description(calibration)}}})}
            session = root / 'session'
            with patch.object(capture, 'ROOT', root):
                capture.prepare(session, 'left', snapshots, 'empty_v1', 'bare_attachment_v1')
            cfg = yaml.safe_load((session / 'record.yaml').read_text())
            self.assertEqual(cfg['side'], 'left')
            self.assertEqual(cfg['metadata']['source'], 'real')
            self.assertTrue(cfg['metadata']['health_gate_enabled'])
            self.assertFalse(cfg['metadata']['contact']['present'])
            self.assertEqual(Path(cfg['metadata']['calibration']['path']).read_bytes(), calibration.read_bytes())
            self.assertEqual((session / 'audit/friction_model.yaml').read_bytes(), friction.read_bytes())

    def test_refuses_ambiguous_or_gripper_description(self):
        with self.assertRaises(ValueError):
            capture.calibration_from_description('<robot/>')
        with self.assertRaises(ValueError):
            capture.calibration_from_description(description('/tmp/x').replace(
                '<robot>', '<robot><link name="left_gripper_gripper_base_link"/>'))

    def test_step_changes_only_requested_joint_and_checks_index(self):
        current = [.1]*7
        result = target.target_degrees(current, step=(6, 1))
        self.assertEqual(result[:6], current[:6])
        self.assertAlmostEqual(result[6], .1 + math.radians(1))
        with self.assertRaises(ValueError):
            target.target_degrees(current, step=(7, 1))

    def test_rejects_nonfinite_and_large_motion(self):
        for value in (float('nan'), float('inf'), 6):
            with self.assertRaises(ValueError):
                target.target_degrees([0]*7, step=(0, value))

    def test_active_hardware_limits_and_side_checked(self):
        names = [f'left_joint_{i}' for i in range(7)]
        target.validate_limits(description('/tmp/x'), names, [0]*7)
        with self.assertRaises(ValueError):
            target.validate_limits(description('/tmp/x'), names, [1.1]*7)
        with self.assertRaises(ValueError):
            target.validate_limits(description('/tmp/x'), ['right_joint_0']*7, [0]*7)

    def test_first_point_current_at_zero_avoids_future_point_jump(self):
        current, goal = [.2]*7, [.21]*7
        points = target.trajectory_points(current, goal, 5.25)
        self.assertEqual(points, [(current, 0, 0), (goal, 5, 250_000_000)])
        with self.assertRaises(ValueError):
            target.trajectory_points(current, goal, 0)

    def test_cli_help_and_missing_send_target_never_initialize_ros(self):
        for filename in ('quick_capture.py', 'w3_joint_target.py'):
            result = subprocess.run(['/usr/bin/python3.10', str(capture.ROOT / 'scripts' / filename), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(['/usr/bin/python3.10', str(capture.ROOT / 'scripts/w3_joint_target.py'), '--send'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--send needs', result.stderr)

    def test_next_environment_omits_empty_dds_path_and_preserves_explicit_path(self):
        for value in ('', str(capture.ROOT / 'config/w3/dds_loopback.xml')):
            env = capture.runtime_environment(0)
            env['FASTRTPS_DEFAULT_PROFILES_FILE'] = value
            result = subprocess.run(capture.next_command(['python', '-c',
                'import os,json; print(json.dumps(os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")))']),
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), value if value else None)

    def test_capture_lifecycle_with_fake_processes_and_real_offline_h5_checker(self):
        # Every ROS command/process is replaced; only the offline H5 checker is executed.
        from factr2_next.data_collection.h5_writer import H5Writer
        from factr2_next.data_collection.quality import session_metadata
        import numpy as np
        real_root = capture.ROOT
        original_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / 'config/w3/left'
            template.mkdir(parents=True)
            shutil.copy2(real_root / 'config/w3/left/record.yaml', template / 'record.yaml')
            calibration = root / 'actual.yaml'
            calibration.write_text('transport: w3\noffsets: []\n')
            observed_commands, stopped = [], []
            motion_file = real_root / 'config/w3/motions/left_all_joints_pilot.yaml'
            plan, motion_raw, motion_hash = sequence.load_motion(motion_file)

            def fake_run(command, env, timeout=15):
                observed_commands.append(command)
                if 'list_controllers' in command:
                    return '\n'.join(f'{name} plugin active' for name in
                        ('joint_state_broadcaster', 'gravity_compensation_controller', 'joint_position_controller'))
                if 'dump' in command:
                    params = {'robot_description': description(calibration)} if command[-1] == '/robot_state_publisher' else {}
                    return yaml.safe_dump({'node': {'ros__parameters': params}})
                if 'call' in command:
                    return 'response: SetBool_Response(success=True, message="ok")'
                return ''

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.pid = 99999999  # killpg is also mocked; never signals any real process.
                    self.code = None
                    self.label = Path(kwargs['stdout'].name).stem
                    if 'next_record' in command:
                        value = next(c for c in command if c.startswith('config_file:='))
                        config_path = Path(value.split(':=', 1)[1])
                        cfg = yaml.safe_load(config_path.read_text())
                        path = Path(cfg['output_dir']) / 'offline_fixture.h5'
                        writer = H5Writer(path, cfg['session_name'], cfg['topics'], session_metadata(cfg, config_path))
                        writer.start_episode()
                        for i in range(100):
                            sample = {key: np.full(7, i * .001, dtype=np.float32) for key in cfg['topics']}
                            writer.append(1_000_000_000 + i * 20_000_000, sample)
                        writer.close()

                def poll(self):
                    return self.code

                def wait(self, timeout):
                    stopped.append(self.label)
                    self.code = 0
                    return 0

                def send_signal(self, sig):
                    if self.label != 'adapter':
                        raise AssertionError('Only the launch parent should receive send_signal')

            def fake_popen(command, **kwargs):
                if hasattr(kwargs.get('stdout'), 'name'):
                    return FakeProcess(command, **kwargs)
                # Permit only offline git/checker work; never execute a real ROS command here.
                if 'ros2' in command and 'next_check_h5' not in command:
                    raise AssertionError(f'Unexpected real ROS subprocess: {command}')
                return original_popen(command, **kwargs)

            def fake_select(*_):
                # C has just opened recording. Verify its frozen plan and simulate D's saved result.
                active_file = root / 'data/quick_capture/left/active.json'
                active = json.loads(active_file.read_text())
                session = Path(active['session'])
                self.assertEqual(active['motion_sha256'], motion_hash)
                self.assertEqual((session / 'audit/motion.yaml').read_bytes(), motion_raw)
                cfg = yaml.safe_load((session / 'record.yaml').read_text())
                self.assertEqual(cfg['metadata']['trajectory_id'], f'{plan["name"]}:sha256:{motion_hash}')
                sequence.write_json(session / 'audit/motion_run.json', {'status': 'completed',
                    'motion_sha256': motion_hash, 'events': [{'status': 'completed'}] * len(plan['steps'])})
                return [stdin], [], []

            with patch.object(capture, 'ROOT', root), \
                 patch.object(capture, 'run', side_effect=fake_run), \
                 patch.object(capture, 'next_command', side_effect=lambda args:
                     ['/bin/bash', str(real_root / 'scripts/next.sh'), *map(str, args)]), \
                 patch.object(capture.subprocess, 'Popen', side_effect=fake_popen), \
                 patch.object(capture.os, 'killpg'), \
                 patch('sys.argv', ['quick_capture.py', '--side', 'left', '--domain', '0', '--motion', str(motion_file)]), \
                 patch('sys.stdin') as stdin, \
                 patch.object(capture.select, 'select', side_effect=fake_select), \
                 patch('sys.stdout', new_callable=io.StringIO) as output:
                stdin.isatty.return_value = True
                stdin.readline.return_value = '\n'
                self.assertEqual(capture.main(), 0)
                self.assertIn('质量检查：PASS', output.getvalue())
                self.assertIn('整套动作执行：PASS', output.getvalue())
                self.assertFalse((root / 'data/quick_capture/left/active.json').exists())
            self.assertEqual(stopped, ['recorder', 'bag', 'adapter', 'health'])
            calls = [c[-1] for c in observed_commands if 'call' in c]
            self.assertEqual(calls, ['{data: true}', '{data: false}'])
            self.assertFalse(any('real_robot.launch.py' in c or 'switch_controller' in c or
                                 '/joint_position_command' in c for c in observed_commands))


if __name__ == '__main__':
    unittest.main()
