"""NEXT revision regressions: coverage, stateless release and observable force."""
import copy
from datetime import datetime, timezone
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from factr2_next.training.coverage_audit import audit_coverage
from factr2_next.training.validation_metrics import validation_epoch
from factr2_next.training.models import build_model
from factr2_next.inference.endpoint_wrench import estimate_wrench
from w3_baseline_protocol import build_protocol, conditions
from w3_coverage_plan import ArmModel, validate_plan
from test_w3_coverage_motion import fixture_description

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plans():
    recipe = yaml.safe_load((ROOT/'config/w3/motions/right_next_protocol_v2.yaml').read_text())
    home = np.asarray(recipe['coverage']['home_deg'])
    recipe['coverage']['working_bounds_deg'] = np.c_[home-12, home+12].tolist()
    recipe['coverage']['cartesian'].update(line_half_extent_m=.002, circle_radius_m=.002)
    recipe['protocol'].update(task_pose_count=2, dwell_seconds=3.,
                              task_pose_bounds_deg=np.c_[home-5, home+5].tolist())
    recipe['task_coverage'].update(coupled_amplitude_deg=[2]*7, coupled_cycles=1, cartesian_repeats=1)
    return [build_protocol(recipe, fixture_description(), home.tolist(), role)[0]
            for role in ('train', 'val')]


def test_shared_anchors_fresh_trajectories_and_exhaustive_conditioning(plans):
    train, val = plans
    assert train['config']['baseline_protocol']['version'] == 2
    assert train['config']['baseline_protocol']['task_poses_deg'] == val['config']['baseline_protocol']['task_poses_deg']
    assert train['config']['baseline_protocol']['recipe_sha256'] == val['config']['baseline_protocol']['recipe_sha256']
    a = next(s for s in train['steps'] if 'coupled_sine' in s['kind'])
    b = next(s for s in val['steps'] if s['kind'] == a['kind'] and s['speed'] == a['speed'])
    assert a['positions_deg'] != b['positions_deg']
    for plan in plans:
        validate_plan(plan)  # Includes all joint/speed/acceleration/jerk/boundary checks.
        model = ArmModel(plan['robot_description'], 'right', 'right_attachment_point')
        for step in plan['steps']:
            if step['phase'] == 'cartesian' or (step['phase'] == 'transition' and step['kind'] != 'approach'):
                assert max(conditions(model, step['positions_deg'])) <= 20
        assert all(np.count_nonzero(np.ptp(s['positions_deg'], axis=0) > 1.) >= 6
                   for s in plan['steps'] if 'coupled_sine' in s['kind'])


def test_batched_geometry_matches_scalar_and_detects_interior_singularity(plans):
    model = ArmModel(plans[0]['robot_description'], 'right', 'right_attachment_point')
    q = np.random.default_rng(17).uniform(-.6, .6, (77, 7))
    q[38] = 0
    np.testing.assert_allclose(model.jacobians(q), np.stack([model.fk(row, True)[1] for row in q]), atol=1e-14)
    quality = conditions(model, np.rad2deg(q))
    assert quality[38] is None or quality[38] > 100


def coverage_dataset(static_pose=None, static_torque=0., counts=(300, 800, 400)):
    episodes, info, ends = [], [], []
    for (kind, phase), n in zip((('static_pose0_from_low', 'cartesian'),
                                ('pose0_coupled_sine0', 'cartesian'), ('sine_j1', 'single_joint')), counts):
        x = np.zeros((n, 21), np.float32)
        y = np.zeros((n, 7), np.float32)
        if kind.startswith('static_'):
            x[:, :7] = np.deg2rad(static_pose if static_pose is not None else [0]*7)
            y[:, 0] = static_torque
        episodes.append((x, y))
        info.append(dict(kind=kind, phase=phase, episode=str(len(info)), path='synthetic'))
        ends.append((ends[-1] if ends else 0)+n-49)
    return SimpleNamespace(episodes=episodes, episode_info=info, history=50, ends=ends)


def requirements():
    return dict(minimum_static_fraction=.10, minimum_multi_joint_fraction=.25,
                maximum_single_joint_fraction=.60, max_validation_pose_distance_deg=5.)


def test_measured_coverage_rejects_axis_dominated_data_and_uncovered_static_poses():
    train = coverage_dataset()
    assert audit_coverage(train, coverage_dataset(), requirements())['passed']
    dominated = coverage_dataset(counts=(80, 80, 2000))
    report = audit_coverage(dominated, dominated, requirements())
    assert not report['passed'] and 'train:single_joint_dominates' in report['reasons']
    report = audit_coverage(train, coverage_dataset(static_pose=[15]*7), requirements())
    assert not report['passed'] and 'val:uncovered_static_pose:0' in report['reasons']
    assert report['test_used'] is False


def test_static_ambiguity_warns_without_falsely_relabelling_free_torque():
    train, val = coverage_dataset(), coverage_dataset(static_torque=.6)
    report = audit_coverage(train, val, requirements())
    assert report['passed'] and report['potential_static_ambiguity']
    assert train.episodes[0][1][0, 0] == 0
    assert val.episodes[0][1][0, 0] == pytest.approx(.6)


def test_validation_group_metrics_reveal_static_bias_without_changing_mse():
    class OrderedDataset(Dataset):
        ends = [3, 10]
        episode_info = [dict(kind='static_pose0_from_low', path='a', episode='0', phase='cartesian'),
                        dict(kind='pose0_coupled_sine0', path='a', episode='1', phase='cartesian')]
        def __len__(self): return 10
        def __getitem__(self, i): return torch.zeros(1), torch.full((7,), 1. if i < 3 else 2.)
    class Zero(torch.nn.Module):
        def forward(self, x): return torch.zeros((len(x), 7))
    loss, diag = validation_epoch(Zero(), DataLoader(OrderedDataset(), batch_size=4),
                                  torch.device('cpu'), {'y_std':np.full(7, 3.)}, 'physical_nm')
    assert loss == pytest.approx((3*9+7*36)/10)
    assert diag['groups']['static']['signed_bias_nm'] == [3.]*7
    assert diag['groups']['multi_joint']['per_joint_rmse_nm'] == [6.]*7
    assert diag['static_episodes'][0]['samples'] == 3


def test_paper_lstm_has_no_hidden_memory_after_history_replacement():
    torch.set_num_threads(1)
    cfg = yaml.safe_load((ROOT/'config/w3/right/train_next_v2.yaml').read_text())
    model = build_model(cfg['model'], 21, 7, 50).eval()
    before = torch.zeros(1, 50, 21)
    contact_history = torch.ones_like(before)
    with torch.inference_mode():
        pre = model(before)
        model(contact_history)
        post = model(before)
    torch.testing.assert_close(post, pre, atol=0, rtol=0)


def test_force_only_does_not_require_angular_rank_but_rejects_moments_and_linear_singularity():
    j = np.c_[np.diag([.3, .4, .5, 1., 1., 1.]), np.zeros(6)]
    j[5] = 0  # 6-D wrench unobservable; point-force subproblem remains full rank.
    expected = np.r_[2., -1., 3., 0., 0., 0.]
    tau = j.T @ expected
    assert estimate_wrench(j, tau)[0] is None
    result, diag = estimate_wrench(j, tau, mode='force_only', damping=0)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert diag['valid'] and not diag['moments_observed'] and diag['observable_rank'] == 3
    tau[3] += 2.  # An external moment violates the point-force hypothesis.
    assert estimate_wrench(j, tau, mode='force_only')[1]['reason'] == 'inconsistent_with_tool_contact'
    j[2] = 0
    assert estimate_wrench(j, np.zeros(7), mode='force_only')[1]['reason'] == 'near_singular'


def test_v2_preparation_freezes_shared_envelope_and_rejects_mismatched_anchors(tmp_path, plans):
    import h5py
    from test_w3_quality import fixture
    from factr2_next.data_collection.quality import sha256, sidecar_path
    from prepare_w3_baseline_campaign import prepare
    sessions = {}
    calibration = tmp_path/'calibration.yaml'
    calibration.write_text('test: frozen\n')
    for i, role in enumerate(('train', 'val', 'test')):
        folder = tmp_path/role
        audit = folder/'audit'; audit.mkdir(parents=True)
        plan = copy.deepcopy(plans[0]); plan['steps'] = plan['steps'][:2]
        plan['config']['baseline_protocol'].update(role=role, seed=i+1)
        validate_plan(plan)
        (audit/'motion.yaml').write_text(yaml.safe_dump(plan, sort_keys=False))
        digest = sha256(audit/'motion.yaml')
        events, elapsed, start_ns = [], 0., 10**15
        for step in plan['steps']:
            event = {k:step[k] for k in ('name', 'kind', 'phase', 'speed')}
            event.update(status='completed', utc=datetime.fromtimestamp(start_ns/1e9+elapsed, timezone.utc).isoformat())
            elapsed += step['duration_seconds']
            event['completed_utc'] = datetime.fromtimestamp(start_ns/1e9+elapsed, timezone.utc).isoformat()
            events.append(event)
        (audit/'motion_run.json').write_text(json.dumps(dict(status='completed', motion_sha256=digest, events=events)))
        (audit/'collision.json').write_text(json.dumps(dict(accepted=True, plan_sha256=digest,
                                                          robot_description_sha256=plan['model_sha256'])))
        n = int(round(elapsed*100))+1
        path = fixture(folder/'raw.h5', phase=i, hz=100, n=n)
        q = np.vstack([s['positions_deg'] if j == 0 else s['positions_deg'][1:]
                       for j,s in enumerate(plan['steps'])])
        with h5py.File(path, 'r+') as h:
            h.attrs['session_name'] = role
            for key in ('joint_pos', 'joint_cmd'):
                h[f'ep_0000/{key}/data'][:] = np.deg2rad(q)
            for key in h['ep_0000']:
                h[f'ep_0000/{key}/timestamps'][:] = start_ns+np.arange(n, dtype=np.int64)*10_000_000
        for name in ('position', 'gravity', 'friction'):
            (audit/(name+'.yaml')).write_text('parameters: {gain: 1}\n')
        meta = json.loads(sidecar_path(path).read_text())
        meta.update(session_id=role, side='right', joint_order=[f'right_joint_{j}' for j in range(7)],
                    source='real', health_gate_enabled=True,
                    calibration=dict(path=str(calibration), sha256=sha256(calibration)),
                    control=dict(gains=dict(position_params=str(audit/'position.yaml')),
                        feedforward=dict(gravity_params=str(audit/'gravity.yaml'), friction_model=str(audit/'friction.yaml'))))
        for value in meta['topics'].values():
            value['topic'] = value['topic'].replace('/left/', '/right/')
        meta['h5_sha256'] = sha256(path)
        sidecar_path(path).write_text(json.dumps(meta))
        sessions[role] = [str(folder)]
    output = prepare({'sessions':sessions}, tmp_path/'campaign')
    cfg = yaml.safe_load((output/'train.yaml').read_text())
    assert cfg['train']['batch_size'] == 2048 and cfg['data']['coverage_requirements']
    manifest = json.loads((output/'split.json').read_text())
    for role in sessions:
        meta = json.loads(sidecar_path(manifest['splits'][role][0]['path']).read_text())
        assert meta['operating_envelope']['position_bounds_deg'] == plans[0]['config']['working_bounds_deg']
        assert meta['task_coverage']['task_poses_deg'] == plans[0]['config']['baseline_protocol']['task_poses_deg']
    path = tmp_path/'val/audit/motion.yaml'
    plan = yaml.safe_load(path.read_text())
    plan['config']['baseline_protocol']['task_poses_deg'][0][0] += .01
    path.write_text(yaml.safe_dump(plan, sort_keys=False)); digest = sha256(path)
    for name, key in (('motion_run.json', 'motion_sha256'), ('collision.json', 'plan_sha256')):
        item = json.loads((path.parent/name).read_text()); item[key] = digest
        (path.parent/name).write_text(json.dumps(item))
    with pytest.raises(ValueError, match='same task coverage'):
        prepare({'sessions':sessions}, tmp_path/'bad_campaign')
    assert not (tmp_path/'bad_campaign').exists()


def test_training_saves_physical_group_diagnostics_and_unchanged_checkpoint(tmp_path):
    from test_training_contract import config
    from factr2_next.training.w3_train import train
    from factr2_next.inference.checkpoint import load_checkpoint
    cfg = config(tmp_path)
    cfg['train'].update(loss_units='physical_nm', validation_diagnostics=True)
    run = train(cfg)
    loaded = load_checkpoint(run)
    metrics = json.loads((run/'metrics.json').read_text())
    assert metrics['validation_diagnostics'][0]['groups']['unlabelled']['samples'] == metrics['window_counts']['val']
    assert loaded.history == 50 and loaded.model.state_mode == 'stateless'
    assert set(loaded.metadata['training_source_files_sha256']) >= {'coverage_audit.py', 'validation_metrics.py'}


def test_point_force_observer_shows_unknown_moments_and_preserves_joint_residual(plans):
    # Exercise the actual ROS-message conversion without a ROS graph or server.
    from sensor_msgs.msg import JointState
    from w3_endpoint_wrench import EndpointObserver
    from factr2_next.inference.operating_envelope import OperatingEnvelope
    model = ArmModel(plans[0]['robot_description'], 'right', 'right_attachment_point')
    q = np.deg2rad([10, 20, 10, -30, 5, 8, 5])
    pose, j = model.fk(q, True)
    force = np.array([2., -1., .5])
    tau = j[:3].T @ (pose[:3, :3] @ force)
    published = []
    ns = 10**9
    observer = SimpleNamespace(names=model.names, cfg={'solver':dict(mode='force_only', damping=0)},
        mode='force_only', frame_id='right_attachment_point', model=model, lock=threading.Lock(),
        next_status={'state':'valid'}, next_status_at=time.monotonic(), last_stamp=None,
        envelope=OperatingEnvelope(None, 7), publisher=SimpleNamespace(publish=published.append),
        outputs=0, history={'t':[], 'mx':[]},
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=ns)))
    def fail(reasons):
        raise AssertionError(reasons)
    observer._unavailable = fail
    messages = []
    for values in (q, tau):
        msg = JointState(); msg.header.stamp.sec = 1
        msg.name = model.names; msg.position = values.tolist(); messages.append(msg)
    before = list(messages[1].position)
    EndpointObserver._pair(observer, *messages)
    np.testing.assert_allclose([published[0].wrench.force.x, published[0].wrench.force.y,
                               published[0].wrench.force.z], force, atol=1e-12)
    snapshot = EndpointObserver._snapshot(observer)
    assert snapshot['state'] == 'valid' and snapshot['wrench']['mx'] is None
    assert not snapshot['moments_observed'] and snapshot['estimation_mode'] == 'force_only'
    assert published[0].wrench.torque.x == 0  # Explicit hypothesis, not a measured moment.
    assert list(messages[1].position) == before
