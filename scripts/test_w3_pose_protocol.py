"""Goal execution, measured speed and paper-contract regressions; no hardware."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

from geometry_msgs.msg import PoseStamped
from w3_pose_protocol import build_pose_protocol
from w3_pose_motion import PoseCoverageClient, pose_matches
from w3_coverage_plan import ArmModel, validate_plan
from w3_baseline_protocol import conditions
from test_w3_coverage_motion import fixture_description
from test_w3_next_revision import coverage_dataset, requirements
from factr2_next.training.coverage_audit import audit_coverage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plans():
    recipe = yaml.safe_load((ROOT/'config/w3/motions/right_next_pose_protocol_v3.yaml').read_text())
    recipe['protocol']['task_pose_count'] = 2
    return [build_pose_protocol(recipe, fixture_description(), recipe['coverage']['home_deg'], role)
            for role in ('train', 'val')]


def test_direct_goals_are_sparse_exactly_audited_and_well_conditioned(plans):
    for plan in plans:
        model = validate_plan(plan)
        assert plan['config']['execution_mode'] == 'direct_pose_goals'
        for step in plan['steps']:
            command = step['command']
            if command['type'] == 'pose_goal':
                assert len(command['knots']) == 2
            elif command['type'] == 'joint_trajectory':
                assert len(command['knots']) == 4
                joint = step['joint']
                q = np.asarray(step['positions_deg'])
                np.testing.assert_allclose([q[:, joint].min(), q[:, joint].max()],
                                           plan['config']['working_bounds_deg'][joint])
            if (step['phase'] == 'cartesian' or
                    (step['phase'] == 'transition' and step['kind'] != 'approach')):
                assert max(conditions(model, step['positions_deg'])) <= 20
    a, b = plans
    assert a['config']['baseline_protocol']['task_poses_deg'] == b['config']['baseline_protocol']['task_poses_deg']
    assert a['steps'] != b['steps']
    for joint in range(7):
        sweeps = [s for s in a['steps'] if s['joint'] == joint]
        assert len(sweeps) == 2 and sweeps[1]['duration_seconds'] < sweeps[0]['duration_seconds']


@pytest.mark.parametrize('corruption', ['pose', 'knots', 'sample', 'mode'])
def test_goal_path_and_pose_cannot_bypass_frozen_collision_samples(plans, corruption):
    plan = copy.deepcopy(plans[0])
    step = next(s for s in plan['steps'] if s['command']['type'] == 'pose_goal')
    if corruption == 'pose':
        step['command']['target']['position_m'][0] += .005
    elif corruption == 'knots':
        step['command']['knots'][-1]['position_deg'][0] += 1
    elif corruption == 'sample':
        step['positions_deg'][10][0] += .005
    else:
        plan['config']['execution_mode'] = 'unvalidated_stream'
    with pytest.raises(ValueError):
        validate_plan(plan)


def bridge_fixture(step):
    client = PoseCoverageClient.__new__(PoseCoverageClient)
    publications = []
    client.publisher = SimpleNamespace(publish=publications.append)
    client.guard = lambda: None
    client.node = SimpleNamespace(get_clock=lambda:SimpleNamespace(
        now=lambda:SimpleNamespace(nanoseconds=1_000_000_000)))
    client.state = {'cmd': np.deg2rad(step['positions_deg'][0])}
    client.bridge_error = None
    client.accepted_stamp = None
    native = object()
    client.pending_goal = (step, native, 1_150_000_000, lambda: None)
    msg = PoseStamped()
    msg.header.frame_id = step['command']['target']['frame_id']
    msg.header.stamp.sec = 1; msg.header.stamp.nanosec = 150_000_000
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = step['command']['target']['position_m']
    (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
     msg.pose.orientation.w) = step['command']['target']['quaternion_xyzw']
    return client, publications, msg


def test_one_pose_causes_one_complete_goal_and_duplicates_do_not_replay(plans):
    step = next(s for s in plans[0]['steps'] if s['command']['type'] == 'pose_goal')
    client, publications, msg = bridge_fixture(step)
    client.accept_pose(msg)
    assert len(publications) == 1 and client.accepted_stamp == 1_150_000_000
    client.accept_pose(msg)
    assert len(publications) == 1
    client.names = plans[0]['joint_names']
    native = client._native_message(step, 1_150_000_000)
    assert len(native.points) == 2 < len(step['positions_deg'])
    np.testing.assert_allclose(native.points[-1].positions, np.deg2rad(step['positions_deg'][-1]))
    assert list(native.points[-1].velocities) == [0.]*7 and list(native.points[-1].accelerations) == [0.]*7
    for name in ('x','y','z','w'):
        setattr(msg.pose.orientation, name, -getattr(msg.pose.orientation, name))
    assert pose_matches(step['command']['target'], msg)  # Quaternion sign equivalence.


@pytest.mark.parametrize('corruption', ['pose', 'expired', 'seed', 'capture'])
def test_unapproved_stale_or_changed_pose_is_rejected_without_native_publish(plans, corruption):
    step = next(s for s in plans[0]['steps'] if s['command']['type'] == 'pose_goal')
    client, publications, msg = bridge_fixture(step)
    if corruption == 'pose':
        msg.pose.position.x += .01
    elif corruption == 'expired':
        client.node.get_clock = lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=2_000_000_000))
    elif corruption == 'seed':
        client.state['cmd'][0] += np.deg2rad(1)
    else:
        def capture_failed(): raise RuntimeError('Capture stopped')
        client.pending_goal = (*client.pending_goal[:3], capture_failed)
    client.accept_pose(msg)
    assert publications == [] and client.bridge_error is not None


def test_fast_label_does_not_substitute_for_measured_bidirectional_motion():
    ds = coverage_dataset()
    for info in ds.episode_info:
        info['speed'] = 'fast'
    req = dict(requirements(), minimum_fast_fraction=.15, minimum_fast_speed_p95_deg_s=14.,
               minimum_fast_direction_samples=100)
    report = audit_coverage(ds, ds, req)
    assert 'train:insufficient_measured_bidirectional_fast_speed' in report['reasons']
    for (x, _), info in zip(ds.episodes, ds.episode_info):
        if info['phase'] in ('cartesian', 'single_joint') and not info['kind'].startswith('static'):
            x[:, 7:14] = np.deg2rad(np.where(np.arange(len(x))[:, None]%2, 18., -18.))
    assert audit_coverage(ds, ds, req)['passed']


def test_direct_capture_preparation_chooses_speed_audited_training_template(tmp_path, plans):
    import test_w3_next_revision as legacy
    # Exercise complete raw-H5/metadata/event derivation and independent split
    # validation with the new frozen command schema, then inspect the template.
    legacy.test_v2_preparation_freezes_shared_envelope_and_rejects_mismatched_anchors(tmp_path, plans)
    cfg = yaml.safe_load((tmp_path/'campaign/train.yaml').read_text())
    assert cfg['data']['coverage_requirements']['minimum_fast_speed_p95_deg_s'] == 14.
    assert cfg['data']['coverage_requirements']['minimum_fast_direction_samples'] == 100


def test_training_reports_return_baseline_gap_and_speed_groups_without_changing_mse():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from factr2_next.training.validation_metrics import validation_epoch
    class OrderedDataset(Dataset):
        ends = [5, 10]
        episode_info = [dict(kind='static_pose0_from_low', path='session', episode='low',
                             phase='cartesian', speed='slow'),
                        dict(kind='static_pose0_from_high', path='session', episode='high',
                             phase='cartesian', speed='slow')]
        def __len__(self): return 10
        def __getitem__(self, i): return torch.zeros(1), torch.full((7,), 1. if i < 5 else 2.)
    class Zero(torch.nn.Module):
        def forward(self, x): return torch.zeros((len(x), 7))
    loss, diag = validation_epoch(Zero(), DataLoader(OrderedDataset(), batch_size=3),
                                 torch.device('cpu'), dict(y_std=np.full(7, 2.)), 'physical_nm')
    assert loss == pytest.approx(10.)
    assert diag['matched_return_baselines'][0]['baseline_gap_nm'] == [2.]*7
    assert diag['groups_by_speed']['static:slow']['samples'] == 10


@pytest.mark.parametrize('command_type', ['pose_goal', 'joint_trajectory'])
def test_publication_failure_requests_braking_and_never_retries(plans, command_type):
    import time
    step = next(s for s in plans[0]['steps'] if s['command']['type'] == command_type)
    client = PoseCoverageClient.__new__(PoseCoverageClient)
    client.names = plans[0]['joint_names']; client.plan = plans[0]
    client.state = {'cmd':np.deg2rad(step['positions_deg'][0])}
    client.guard = lambda: None; client.check_mode = lambda: None
    client.ready = lambda: True
    client.wait_until = lambda predicate, *args, **kwargs: None
    client.node = SimpleNamespace(get_clock=lambda:SimpleNamespace(
        now=lambda:SimpleNamespace(nanoseconds=10**9)), get_name=lambda:'fixture',
        get_publishers_info_by_topic=lambda topic:[])
    calls, stops = [], []
    def failed_publish(msg):
        calls.append(msg)
        raise RuntimeError('Injected transport publication failure')
    client.pose_pub = SimpleNamespace(publish=failed_publish, get_subscription_count=lambda:1)
    client.publisher = SimpleNamespace(publish=failed_publish, get_subscription_count=lambda:1)
    client.stop_pub = SimpleNamespace(publish=stops.append, get_subscription_count=lambda:1)
    client.ros = SimpleNamespace(spin_once=lambda *args, **kwargs:time.sleep(.01))
    client.goal_topic = '/factr2/right/pose_goal'
    with pytest.raises(RuntimeError, match='publication failure'):
        client.play(step, lambda: None)
    assert len(calls) == 1 and len(stops) == 1 and client.pending_goal is None


def test_rosbag_observers_are_allowed_but_cannot_substitute_for_stop_controller():
    client = PoseCoverageClient.__new__(PoseCoverageClient)
    client.goal_topic = '/factr2/right/pose_goal'
    subscriptions = {topic:[SimpleNamespace(node_name='rosbag2_recorder')] for topic in
        ('/joint_position_command', '/joint_position_controller/stop', client.goal_topic)}
    client.node = SimpleNamespace(get_name=lambda:'pose_sender',
        get_subscriptions_info_by_topic=lambda topic:subscriptions[topic])
    assert not client.goal_interfaces_ready()
    for topic in ('/joint_position_command', '/joint_position_controller/stop'):
        subscriptions[topic].append(SimpleNamespace(node_name='joint_position_controller'))
    subscriptions[client.goal_topic].append(SimpleNamespace(node_name='pose_sender'))
    assert client.goal_interfaces_ready()
    subscriptions['/joint_position_controller/stop'].pop()
    assert not client.goal_interfaces_ready()
