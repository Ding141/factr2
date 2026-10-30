"""MD online/offline parity and isolated ROS observation checks; no robot control."""
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import yaml

from factr2_next.inference.md_observer import MDHistoryBuffer, MDInferenceNode, load_md_checkpoint
from factr2_next.training.motion_discriminator import MotionDiscriminator
from factr2_next.w3_samples import features

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/right/md_ablation_20261016_v2'


def test_online_windows_preserve_md_from_before_window_and_reset_on_gap():
    recipe = dict(thresholds_rad_s=[[.1] * 7] * 15, max_gap_ns=40_000_000)
    buffer = MDHistoryBuffer(50, recipe)
    v = np.zeros((180, 7), np.float32); v[5] = .3; v[120] = -.2
    stamps = np.arange(len(v), dtype=np.int64) * 10_000_000 + 1
    q = np.full_like(v, .5); cmd = q + .01
    expected = np.c_[features(q, v, cmd), MotionDiscriminator(recipe['thresholds_rad_s']).transform(v, stamps)]
    for i, stamp in enumerate(stamps):
        buffer.append(q[i], v[i], cmd[i], stamp)
        if buffer.ready:
            np.testing.assert_array_equal(buffer.array(), expected[i-49:i+1])
    assert buffer.md.known.all()
    buffer.append(q[-1], np.zeros(7), cmd[-1], int(stamps[-1]) + 50_000_000)
    assert len(buffer.rows) == 1 and not buffer.ready and not buffer.md.known.any()
    np.testing.assert_array_equal(buffer.rows[-1][21:], 0)
    buffer.clear()
    assert not buffer.rows and buffer.md.last_stamp is None


def test_frozen_md_checkpoint_prediction_matches_ablation_loader_and_rejects_recipe(tmp_path):
    from evaluate_w3_md_ablation import load_md
    torch.set_num_threads(1)
    loaded, _ = load_md_checkpoint(RUN / 'next_md_seed0', RUN / 'md_recipe.json')
    original, norm = load_md(RUN / 'next_md_seed0')
    x = torch.from_numpy(np.random.default_rng(3).normal(size=(2, 50, 126)).astype(np.float32))
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(x), original(x), rtol=0, atol=0)
    for key in norm:
        np.testing.assert_array_equal(loaded.normalization[key], norm[key])
    changed = tmp_path / 'recipe.json'
    changed.write_text((RUN / 'md_recipe.json').read_text() + '\n')
    with pytest.raises(ValueError, match='recipe_hash'):
        load_md_checkpoint(RUN / 'next_md_seed0', changed)


@pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID') != '173', reason='Requires isolated localhost ROS domain 173')
def test_isolated_ros_md_inference_endpoint_and_stale_reset(tmp_path):
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from sensor_msgs.msg import JointState
    from w3_endpoint_wrench import EndpointObserver
    rclpy.init()
    fixture = rclpy.create_node('isolated_md_fixture')
    node = MDInferenceNode(parameter_overrides=[rclpy.parameter.Parameter(
        'config_file', value=str(ROOT / 'config/w3/right/inference_md_observation_20261016.yaml'))],
        cli_args=['--ros-args', '-r', '__node:=isolated_md_inference'], use_global_arguments=False)
    endpoint_cfg = yaml.safe_load((ROOT / 'config/w3/right/endpoint_wrench_md_observation_20261016.yaml').read_text())
    endpoint_cfg['web']['port'] = 0
    endpoint_path = tmp_path / 'endpoint.yaml'; endpoint_path.write_text(yaml.safe_dump(endpoint_cfg))
    endpoint = EndpointObserver(endpoint_path)
    executor = SingleThreadedExecutor()
    for item in (fixture, node, endpoint): executor.add_node(item)
    publishers = [fixture.create_publisher(JointState, '/factr2/right/' + suffix, qos_profile_sensor_data)
                  for suffix in ('joint_pos', 'joint_vel', 'joint_cmd', 'joint_effort')]
    adapter = fixture.create_publisher(DiagnosticArray, '/factr2/right/adapter_status', qos_profile_sensor_data)
    emitted = {}; received = {}; diagnostic_states = []
    subscriptions = [fixture.create_subscription(JointState, '/next/right/md/free_joint_torque_pred',
        lambda msg: received.update({msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec: np.array(msg.position)}),
        qos_profile_sensor_data), fixture.create_subscription(DiagnosticArray, '/next/right/md/status',
        lambda msg: diagnostic_states.extend(json.loads(v.value) for s in msg.status for v in s.values if v.key == 'state'),
        qos_profile_sensor_data)]
    q = np.deg2rad([0, 50, 0, -55, 0, 0, 0]).astype(np.float32)
    def publish():
        stamp = fixture.get_clock().now().to_msg()
        ns = stamp.sec * 10**9 + stamp.nanosec
        v = np.full(7, .12 if len(emitted) < 80 else -.08, np.float32)
        emitted[ns] = (q.copy(), v, q.copy())
        status = DiagnosticArray(); status.header.stamp = stamp
        status.status = [DiagnosticStatus(name='factr2/adapter/right', level=DiagnosticStatus.OK,
                        message='isolated_fixture', values=[KeyValue(key='test', value='true')])]
        adapter.publish(status)
        for publisher, values, field in zip(publishers, (q, v, q, np.zeros(7)), ('position', 'velocity', 'position', 'effort')):
            msg = JointState(); msg.header.stamp = stamp; msg.name = node.names
            setattr(msg, field, values.astype(float).tolist()); publisher.publish(msg)
    def spin(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline: executor.spin_once(timeout_sec=.002)
    timer = None
    try:
        spin(.4)
        assert set(fixture.get_node_names()) <= {'isolated_md_fixture', 'isolated_md_inference', 'right_endpoint_wrench_md_observer'}
        startup_resets = node.counts['resets']
        timer = fixture.create_timer(.01, publish)
        spin(2.6); timer.cancel(); spin(.08)
        assert len(received) > 150 and 'valid' in diagnostic_states
        assert node.counts['dropped'] == 0 and node.counts['resets'] == startup_resets
        stamps = np.array(sorted(emitted), np.int64)
        memory = MotionDiscriminator(node.md_recipe['thresholds_rad_s']).transform(
            np.stack([emitted[t][1] for t in stamps]), stamps)
        x = np.c_[features(*[np.stack([emitted[t][i] for t in stamps]) for i in range(3)]), memory]
        last_stamp = max(received); index = int(np.flatnonzero(stamps == last_stamp)[0])
        norm = node.loaded.normalization
        batch = torch.from_numpy((x[index-49:index+1] - norm['x_mean']) / norm['x_std']).unsqueeze(0)
        with torch.inference_mode(): expected = node.loaded.model(batch).numpy()[0] * norm['y_std'] + norm['y_mean']
        np.testing.assert_allclose(received[last_stamp], expected, atol=1e-6, rtol=1e-6)
        assert endpoint.torque_sub.sub.topic_name == '/next/right/md/external_joint_torque'
        assert endpoint.status_sub.topic_name == '/next/right/md/status'
        assert 'NEXT+MD' in endpoint.page_html and '8083' in endpoint.page_html
        for name in ('isolated_md_inference', 'right_endpoint_wrench_md_observer'):
            assert not fixture.get_client_names_and_types_by_node(name, '/')
            topics = fixture.get_publisher_names_and_types_by_node(name, '/')
            assert all(t.startswith('/next/right/md/') or t in ('/rosout', '/parameter_events') for t, _ in topics)
        spin(.35)
        assert node.state == 'stale' and not node.buffer.rows and not node.buffer.md.known.any()
        assert node.torque_filter.y is None
        assert endpoint._snapshot()['wrench'] is None
    finally:
        if timer is not None: timer.cancel()
        for item in (endpoint, node, fixture): executor.remove_node(item)
        executor.shutdown()
        endpoint.destroy_node(); node.destroy_node(); fixture.destroy_node(); rclpy.shutdown()
