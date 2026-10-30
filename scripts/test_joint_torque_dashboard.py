"""Paper torque display and isolated ROS transport; no hardware interfaces."""
import json
import os
from pathlib import Path
import re
import time
from urllib.request import urlopen

import numpy as np
import pytest
import yaml

from factr2_next.visualization.web_node import WebNode, page
from factr2_next.w3_config import validate_w3_config
from w3_endpoint_wrench import joint_torque_dashboard_config, parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_existing_final_window_defaults_to_paper_quantity():
    assert parse_args([]).view == 'joint_torque'
    assert parse_args(['--view', 'endpoint_wrench']).view == 'endpoint_wrench'
    assert parse_args(['--mode', 'force_only']).view == 'endpoint_wrench'
    with pytest.raises(SystemExit):
        parse_args(['--view', 'joint_torque', '--mode', 'full_wrench'])


@pytest.mark.parametrize('filename', [
    'endpoint_wrench.yaml', 'endpoint_wrench_targeted_20261016.yaml',
    'endpoint_wrench_md_observation_20261016.yaml'])
def test_window_keeps_its_port_and_actual_inference_topics(filename):
    path = ROOT / 'config/w3/right' / filename
    original = yaml.safe_load(path.read_text())
    inference = yaml.safe_load(Path(original['inference_config']).read_text())
    cfg = joint_torque_dashboard_config(path)
    validate_w3_config(cfg, 'visualize')
    assert cfg['outputs'] == inference['outputs']
    assert cfg['joint_names'] == inference['joint_names']
    assert cfg['web'] == original['web']
    assert cfg['model_notice'] == original.get('model_notice')
    assert cfg['plot']['view'] == 'joint_torque'


def test_paper_page_shows_signed_joint_residuals_by_default():
    names = [f'right_joint_{i}' for i in range(7)]
    markup = page(names, view='joint_torque', model_notice='<pending>')
    groups = json.loads(re.search(r'const groups=(.*);', markup).group(1))
    assert len(groups) == 2
    raw, filtered = [group[1] for group in groups]
    assert [row[0] for row in raw] == [f'raw_j{i}' for i in range(1, 8)]
    assert all(row[3] for row in raw) and not any(row[3] for row in filtered)
    assert 'J7 · right_joint_6' in raw[-1][1]
    assert 'τ_ext = τ_measured − τ_free_pred' in markup
    assert '关节外部力矩（N·m）' in markup and '时间（s）' in markup
    assert '&lt;pending&gt;' in markup and '<pending>' not in markup
    assert 'next.joint_torque.' in markup  # Old diagnostic checkbox settings do not hide joints.
    assert 'data.state==="valid"' in markup


@pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID') != '199',
                    reason='Synthetic publishers require isolated localhost domain 199')
def test_joint_page_works_without_jacobian_or_robot_state_and_marks_stale():
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    cfg = joint_torque_dashboard_config(ROOT / 'config/w3/right/endpoint_wrench_targeted_20261016.yaml')
    cfg['web']['port'] = 0
    rclpy.init()
    fixture = rclpy.create_node('isolated_joint_torque_fixture')
    observer = None
    executor = SingleThreadedExecutor()
    try:
        observer = WebNode(config=cfg, node_name='isolated_joint_torque_dashboard')
        executor.add_node(fixture)
        executor.add_node(observer)
        names = cfg['joint_names']
        raw = np.array([.1, -.2, .3, -.4, .5, -.6, .7])
        pubs = {key: fixture.create_publisher(JointState, cfg['outputs'][key].format(
            next_topic_root=cfg['next_topic_root']), qos_profile_sensor_data)
            for key in ('external_joint_torque_raw', 'external_joint_torque', 'free_joint_torque_pred')}
        status_pub = fixture.create_publisher(DiagnosticArray, '/next/right/status', qos_profile_sensor_data)

        def publish():
            stamp = fixture.get_clock().now().to_msg()
            for key, value in [('external_joint_torque_raw', raw),
                               ('external_joint_torque', raw / 2),
                               ('free_joint_torque_pred', np.full(7, 90.))]:
                msg = JointState()
                msg.name, msg.position, msg.header.stamp = names, value.tolist(), stamp
                pubs[key].publish(msg)
            message = DiagnosticArray()
            message.header.stamp = stamp
            status = DiagnosticStatus()
            status.name = 'factr2/next/right'
            status.values = [KeyValue(key='state', value='"valid"'),
                             KeyValue(key='reason', value='"isolated_fixture"')]
            message.status = [status]
            status_pub.publish(message)

        def spin(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                executor.spin_once(timeout_sec=.005)

        timer = fixture.create_timer(.02, publish)
        spin(1.5)
        url = f'http://127.0.0.1:{observer.httpd.server_port}'
        with urlopen(url + '/snapshot', timeout=2) as response:
            snapshot = json.load(response)
        assert snapshot['state'] == 'valid' and snapshot['view'] == 'joint_torque'
        assert snapshot['quantity'] == 'external_joint_torque'
        np.testing.assert_allclose([snapshot[f'raw_j{i}'][-1] for i in range(1, 8)], raw)
        np.testing.assert_allclose([snapshot[f'ext_j{i}'][-1] for i in range(1, 8)], raw / 2)
        assert snapshot['free_j7'][-1] == 90.
        subscriptions = fixture.get_subscriber_names_and_types_by_node(observer.get_name(), '/')
        assert not any(topic in ('/robot_description', '/factr2/right/joint_pos')
                       or 'endpoint_wrench' in topic for topic, _ in subscriptions)
        assert set(fixture.get_node_names()) <= {'isolated_joint_torque_fixture', 'isolated_joint_torque_dashboard'}
        timer.cancel()
        spin(.4)
        assert json.loads(observer._snapshot())['state'] == 'stale'
    finally:
        executor.shutdown()
        if observer is not None:
            observer.destroy_node()
        fixture.destroy_node()
        rclpy.try_shutdown()
