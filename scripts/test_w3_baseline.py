"""Offline regressions for baseline drift, release sensitivity and split leakage."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'factr2_next/src/factr2_next/test'))
from test_w3_quality import fixture
from test_training_contract import config
from test_w3_coverage_motion import fixture_description
from factr2_next.data_collection.quality import sidecar_path, sha256, validate_manifest
from factr2_next.data_collection.campaign import control_profile, validate_independent_manifest
from factr2_next.training.train import run_epoch
from factr2_next.training.w3_train import train
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.endpoint_wrench import geometry_quality, estimate_wrench
from w3_baseline_protocol import build_protocol, conditions
from w3_coverage_plan import ArmModel, validate_plan
from prepare_w3_baseline_campaign import prepare, collect_source, event_blocks
from evaluate_w3_baseline import release_case, validate_thresholds, endpoint_free_metrics
from quick_capture import check_capture_space


def test_parameter_snapshot_uses_live_discovery_and_only_retries_node_not_found(monkeypatch):
    import quick_capture
    calls = []
    def read(command, env, timeout):
        calls.append(command)
        if len(calls) == 1:
            raise RuntimeError('Node not found')
        return '/joint_position_controller:\n  ros__parameters:\n    kp: [1.0]\n'
    monkeypatch.setattr(quick_capture, 'run', read)
    output = quick_capture.runtime_parameters('/joint_position_controller', {}, ROOT.parent/'dual_arm_robot')
    assert quick_capture.parameters(output)['kp'] == [1.0]
    assert len(calls) == 2 and all('--no-daemon' in c and '--spin-time' in c for c in calls)
    def failed(*args, **kwargs):
        raise RuntimeError('Parameter service failure')
    monkeypatch.setattr(quick_capture, 'run', failed)
    with pytest.raises(RuntimeError, match='Parameter service failure'):
        quick_capture.runtime_parameters('/joint_position_controller', {}, ROOT.parent/'dual_arm_robot')


@pytest.fixture(scope='module')
def protocol():
    recipe = yaml.safe_load((ROOT/'config/w3/motions/right_baseline_protocol.yaml').read_text())
    home = np.asarray(recipe['coverage']['home_deg'])
    recipe['coverage']['working_bounds_deg'] = np.c_[home-6, home+6].tolist()
    recipe['coverage']['cartesian'].update(line_half_extent_m=.002, circle_radius_m=.002)
    recipe['protocol'].update(dwell_seconds=3., task_pose_count=2,
                              task_pose_bounds_deg=np.c_[home-1, home+1].tolist())
    plan, _ = build_protocol(recipe, fixture_description(), home.tolist(), 'train')
    return recipe, plan


def acceptance():
    return yaml.safe_load((ROOT/'config/w3/right/baseline_acceptance.yaml').read_text())


def test_smooth_protocol_ranges_static_returns_and_geometry(protocol):
    recipe, plan = protocol; validate_plan(plan)
    for speed in ('slow', 'fast'):
        sweeps = [s for s in plan['steps'] if s['phase'] == 'single_joint' and s['speed'] == speed]
        assert {s['joint'] for s in sweeps} == set(range(7))
        for s in sweeps:
            a = np.asarray(s['positions_deg'])[:, s['joint']]
            np.testing.assert_allclose([a.min(), a.max()], recipe['coverage']['working_bounds_deg'][s['joint']], atol=.005)
    m = ArmModel(plan['robot_description'], 'right', 'right_attachment_point')
    for s in plan['steps']:
        if s['phase'] == 'cartesian':
            assert max(conditions(m, np.asarray(s['positions_deg'])[::10])) <= 20
        if s['kind'].startswith('static_'):
            assert np.ptp(np.asarray(s['positions_deg']), axis=0).max() == 0
    for i in range(2):
        a = next(s for s in plan['steps'] if s['kind'] == f'static_pose{i}_from_low')
        b = next(s for s in plan['steps'] if s['kind'] == f'static_pose{i}_from_high')
        np.testing.assert_allclose(a['positions_deg'], b['positions_deg'], atol=1e-9)
    modified = copy.deepcopy(plan); modified['steps'][0]['positions_deg'][100][0] += 1
    with pytest.raises(ValueError): validate_plan(modified)


def test_singular_home_and_duplicate_seed_refused(protocol):
    recipe, _ = protocol; bad = copy.deepcopy(recipe)
    bad['protocol']['seeds']['val'] = bad['protocol']['seeds']['train']
    with pytest.raises(ValueError, match='different fixed seeds'):
        build_protocol(bad, fixture_description(), bad['coverage']['home_deg'], 'train')
    m = ArmModel(fixture_description(), 'right', 'right_attachment_point')
    singular = conditions(m, [[0]*7])[0]
    assert singular is None or singular > 100
    j = np.c_[np.eye(6), np.zeros(6)]; j[5] = 0
    assert geometry_quality(j)['observable_rank'] == 5
    assert estimate_wrench(j, np.zeros(7))[0] is None


def test_physical_mse_and_checkpoint_compatibility(tmp_path):
    x = torch.ones(2, 1); y = torch.zeros(2, 2)
    class Constant(torch.nn.Module):
        def forward(self, x): return torch.ones(len(x), 2)
    assert run_epoch(Constant(), [(x,y)], torch.device('cpu')) == 1
    assert run_epoch(Constant(), [(x,y)], torch.device('cpu'), output_scale=[2.,3.]) == 6.5
    cfg = config(tmp_path); cfg['train']['loss_units'] = 'physical_nm'
    run = train(cfg); loaded = load_checkpoint(run)
    assert loaded.config['train']['loss_units'] == 'physical_nm'
    assert json.loads((run/'metrics.json').read_text())['loss_units'] == 'physical_nm'


def test_release_recovers_and_zero_estimator_bias_drift_fail():
    cfg = acceptance(); validate_thresholds(cfg)
    t = np.arange(2000)/100; q = np.zeros((len(t),7)); vel = q.copy()
    case = dict(id='test', operator_verified=True,
                spans=dict(before_free=[.5,6.], contact=[6.,8.], after_free=[8.,16.]))
    r = q.copy(); r[(t >= 6) & (t < 8),0] = 1.
    assert release_case(r,t,q,vel,case,cfg)['passed']
    assert not release_case(q,t,q,vel,case,cfg)['passed']
    biased = r.copy(); biased[t >= 8,0] = .3
    assert not release_case(biased,t,q,vel,case,cfg)['passed']
    drift = r.copy(); drift[(t >= 10)&(t < 13),0] = -.08; drift[t >= 13,0] = .08
    assert not release_case(drift,t,q,vel,case,cfg)['passed']
    case['operator_verified'] = False
    with pytest.raises(ValueError, match='Manual contact'): release_case(r,t,q,vel,case,cfg)


def test_endpoint_does_not_hide_invalid_or_biased_rows(protocol):
    _, plan = protocol
    m = ArmModel(plan['robot_description'], 'right', 'right_attachment_point'); cfg = acceptance()
    q = np.repeat(np.deg2rad(plan['config']['home_deg'])[None,:], 100, axis=0)
    r = np.zeros((100,7))
    assert endpoint_free_metrics(r,q,m,cfg)['passed']
    assert not endpoint_free_metrics(r,np.zeros_like(q),m,cfg)['passed']
    wrench = np.array([10.,0,0,0,0,0])
    r[:] = m.fk(q[0],True)[1].T @ wrench
    assert not endpoint_free_metrics(r,q,m,cfg)['passed']


def test_campaign_whole_event_guards_leakage_and_gain_change(tmp_path, protocol):
    recipe, prototype = protocol
    calibration = tmp_path/'calibration.yaml'; calibration.write_text('test: frozen\n')
    sessions = {}
    for i, role in enumerate(('train','val','test')):
        folder = tmp_path/role; audit = folder/'audit'; audit.mkdir(parents=True)
        plan = copy.deepcopy(prototype); plan['config']['baseline_protocol']['role'] = role
        plan['config']['baseline_protocol']['seed'] = recipe['protocol']['seeds'][role]
        (audit/'motion.yaml').write_text(yaml.safe_dump(plan, sort_keys=False))
        digest = sha256(audit/'motion.yaml')
        (audit/'collision.json').write_text(json.dumps(dict(accepted=True,plan_sha256=digest,
                                                   robot_description_sha256=plan['model_sha256'])))
        elapsed = 0.; events = []; start_ns = 10**15
        for step in plan['steps']:
            e = {k:step[k] for k in ('name','kind','phase','speed')}
            e.update(status='completed',utc=datetime.fromtimestamp(start_ns/1e9+elapsed,timezone.utc).isoformat())
            elapsed += step['duration_seconds']
            e['completed_utc'] = datetime.fromtimestamp(start_ns/1e9+elapsed,timezone.utc).isoformat(); events.append(e)
        (audit/'motion_run.json').write_text(json.dumps(dict(status='completed',motion_sha256=digest,events=events)))
        p = fixture(folder/'raw.h5',phase=i,hz=100)
        n = int(round(elapsed*100))+1
        planned = np.vstack([np.asarray(s['positions_deg']) if j == 0 else np.asarray(s['positions_deg'])[1:]
                             for j,s in enumerate(plan['steps'])])
        assert len(planned) == n
        with h5py.File(p,'r+') as h:
            h.attrs['session_name'] = role
            for key in h['ep_0000']:
                g = h['ep_0000'][key]; g['data'].resize((n,7)); g['data'][:] = i*.001
                if key in ('joint_pos', 'joint_cmd'): g['data'][:] = np.deg2rad(planned)
                g['timestamps'].resize(n,axis=0); g['timestamps'][:] = start_ns+np.arange(n,dtype=np.int64)*10_000_000
        for name in ('position','gravity','friction'):
            (audit/(name+'.yaml')).write_text('parameters: {gain: 1}\n')
        meta = json.loads(sidecar_path(p).read_text())
        meta.update(session_id=role,side='right',joint_order=[f'right_joint_{j}' for j in range(7)],source='real',
                    health_gate_enabled=True,calibration=dict(path=str(calibration),sha256=sha256(calibration)),
                    control=dict(gains=dict(position_params=str(audit/'position.yaml')),
                        feedforward=dict(gravity_params=str(audit/'gravity.yaml'),friction_model=str(audit/'friction.yaml'))))
        for v in meta['topics'].values(): v['topic'] = v['topic'].replace('/left/','/right/')
        meta['h5_sha256'] = sha256(p); meta['episodes']['ep_0000']['rows'] = n
        sidecar_path(p).write_text(json.dumps(meta)); sessions[role] = [str(folder)]
    output = prepare({'sessions':sessions}, tmp_path/'campaign')
    manifest, reports = validate_manifest(output/'split.json')
    validate_independent_manifest(manifest,reports)
    for role in ('train','val','test'):
        meta = reports[role][0]['metadata']
        assert all(e['rows'] >= 50 for e in meta['episodes'].values())
        assert meta['derivation']['guard_frames_each_end'] == {'static':100, 'dynamic':0}
    # A stopped session can contribute explicitly selected completed training
    # blocks, with its failed block excluded and the original audit unchanged.
    run_path = tmp_path/'train/audit/motion_run.json'
    original_run = json.loads(run_path.read_text()); stopped = copy.deepcopy(original_run)
    stopped['status'] = 'failed'; stopped['events'][-1]['status'] = 'sending'
    run_path.write_text(json.dumps(stopped))
    event_name = stopped['events'][2]['name']
    chosen = collect_source(tmp_path/'train', 'train', [event_name])
    assert [e['name'] for e in chosen[3]['events']] == [event_name]
    with pytest.raises(ValueError, match='Selected event did not complete'):
        collect_source(tmp_path/'train', 'train', [stopped['events'][-1]['name']])
    with pytest.raises(ValueError, match='complete matching'):
        collect_source(tmp_path/'train', 'train')
    with pytest.raises(ValueError, match='only for training'):
        collect_source(tmp_path/'val', 'val', [event_name])
    from w3_baseline_resume import resume_training_plan
    resumed, reference, receipt = resume_training_plan(recipe, prototype['robot_description'],
                                                     recipe['coverage']['home_deg'], tmp_path/'train')
    validate_plan(resumed)
    assert len(receipt['selected_completed_events']) == 7
    assert not any(s['phase'] == 'single_joint' and s['speed'] == 'slow' for s in resumed['steps'])
    assert {s['joint'] for s in resumed['steps'] if s['phase'] == 'single_joint' and s['speed'] == 'fast'} == set(range(7))
    assert len(reference['steps']) > len(resumed['steps'])
    all_sweeps, _, all_receipt = resume_training_plan(recipe, prototype['robot_description'],
        recipe['coverage']['home_deg'], tmp_path/'train', completed_speeds=('slow','fast'))
    assert len(all_receipt['selected_completed_events']) == 14
    assert not any(s['phase'] == 'single_joint' for s in all_sweeps['steps'])
    bad_recipe = copy.deepcopy(recipe)
    bad_recipe['coverage']['speeds']['fast']['velocity_deg_s'] = [
        v*1.01 for v in recipe['coverage']['speeds']['fast']['velocity_deg_s']]
    with pytest.raises(ValueError,match='selected speed profile mismatch'):
        resume_training_plan(bad_recipe, prototype['robot_description'],
            recipe['coverage']['home_deg'], tmp_path/'train', completed_speeds=('slow','fast'))
    run_path.write_text(json.dumps(original_run))
    refs = {role:str(tmp_path/role/'audit/motion.yaml') for role in sessions}
    missing = copy.deepcopy(sessions)
    missing['train'] = [dict(session=str(tmp_path/'train'), events=[event_name])]
    with pytest.raises(ValueError, match='No static baseline|Incomplete resumed protocol'):
        prepare(dict(sessions=missing, protocol_reference=refs), tmp_path/'missing_campaign')
    selected_names = [e['name'] for e in original_run['events'] if e['phase'] != 'transition']
    missing_fast = next(e['name'] for e in original_run['events'] if e['phase'] == 'single_joint' and e['speed'] == 'fast')
    missing['train'][0]['events'] = [n for n in selected_names if n != missing_fast]
    with pytest.raises(ValueError, match='Incomplete resumed protocol'):
        prepare(dict(sessions=missing, protocol_reference=refs), tmp_path/'missing_fast_campaign')
    missing['train'][0]['events'] = selected_names
    complete = prepare(dict(sessions=missing, protocol_reference=refs), tmp_path/'selected_campaign')
    receipt = json.loads((complete/'protocol_completeness.json').read_text())
    assert all(r['complete'] for r in receipt.values())
    # Evaluation recovery is explicit, keeps the original role/reference, and
    # still requires every frozen protocol block before producing a split.
    val_names = [e['name'] for e in json.loads((tmp_path/'val/audit/motion_run.json').read_text())['events']
                 if e['phase'] != 'transition']
    eval_sessions = copy.deepcopy(sessions)
    eval_sessions['val'] = [dict(session=str(tmp_path/'val'),events=val_names)]
    with pytest.raises(ValueError,match='only for training'):
        prepare(dict(sessions=eval_sessions,protocol_reference=refs),tmp_path/'implicit_val')
    eval_out = prepare(dict(sessions=eval_sessions,protocol_reference=refs,
        resumed_evaluation_roles=['val']),tmp_path/'explicit_val')
    assert json.loads((eval_out/'protocol_completeness.json').read_text())['val']['complete']
    val_path=tmp_path/'val/audit/motion_run.json'
    val_original=json.loads(val_path.read_text());val_stopped=copy.deepcopy(val_original)
    val_stopped['status']='failed';val_stopped['events'][-1]['status']='sending'
    val_path.write_text(json.dumps(val_stopped))
    from w3_baseline_resume import resume_evaluation_plan
    val_plan,val_receipt=resume_evaluation_plan(refs['val'],tmp_path/'val',
        prototype['robot_description'],recipe['coverage']['home_deg'])
    assert val_receipt['recaptured_groups']==['home_finish']
    assert val_plan['config']['baseline_protocol']['role']=='val'
    assert val_stopped['events'][-1]['name'] not in val_receipt['events']
    assert not val_receipt['evaluation_scores_inspected']
    with pytest.raises(ValueError,match='exact frozen reference'):
        resume_evaluation_plan(refs['test'],tmp_path/'val',prototype['robot_description'],recipe['coverage']['home_deg'])
    val_path.write_text(json.dumps(val_original))
    reports['val'][0]['metadata']['derivation']['parent_session_id'] = reports['train'][0]['metadata']['derivation']['parent_session_id']
    with pytest.raises(ValueError,match='independent_session_overlap'): validate_independent_manifest(manifest,reports)
    raw = tmp_path/'train/raw.h5'
    with h5py.File(raw,'r+') as h: h['ep_0000/joint_pos/data'][:] = 0.
    meta = json.loads(sidecar_path(raw).read_text()); meta['h5_sha256'] = sha256(raw)
    sidecar_path(raw).write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Insufficient measured single-joint coverage'):
        prepare({'sessions':sessions}, tmp_path/'bad_campaign')
    _, fresh = validate_manifest(output/'split.json')
    (tmp_path/'val/audit/position.yaml').write_text('parameters: {gain: 2}\n')
    with pytest.raises(ValueError,match='control_profile_hash'): validate_independent_manifest(manifest,fresh)


def test_event_extraction_preserves_raw_episode_boundaries_and_rejects_missing_time(tmp_path):
    start = 10**15
    raw = tmp_path/'split.h5'
    with h5py.File(raw, 'w') as h:
        h.create_group('ep_0002')  # An excluded final episode has no datasets.
        for i in range(2):
            group = h.create_group(f'ep_{i:04d}/joint_pos')
            t = start+np.arange(i*100, (i+1)*100, dtype=np.int64)*10_000_000
            group.create_dataset('timestamps', data=t)
            q = np.zeros((100,7)); q[:,0] = np.deg2rad(np.linspace(-1+2*i/2, i, 100))
            group.create_dataset('data', data=q)
    event = dict(name='sweep',phase='single_joint',kind='sine_j1',speed='slow',
                 utc=datetime.fromtimestamp(start/1e9,timezone.utc).isoformat(),
                 completed_utc=datetime.fromtimestamp(start/1e9+2,timezone.utc).isoformat())
    plan = dict(steps=[dict(name='sweep',joint=0)],config=dict(working_bounds_deg=[[-1,1]]*7))
    blocks = event_blocks(raw,dict(sample_hz=100),dict(events=[event]),plan)
    assert [b['source_episode'] for b in blocks] == ['ep_0000','ep_0001']
    assert all(b['source_row_stop_exclusive']-b['source_row_start'] == 100 for b in blocks)
    with h5py.File(raw, 'r+') as h:
        h['ep_0001/joint_pos/timestamps'][:] += 510_000_000
    with pytest.raises(ValueError,match='Incomplete continuous event coverage'):
        event_blocks(raw,dict(sample_hz=100),dict(events=[event]),plan)


def test_disk_budget_stops_before_capture(monkeypatch):
    class Space: free = 600*1024**2
    monkeypatch.setattr('quick_capture.shutil.disk_usage', lambda p:Space())
    check_capture_space(ROOT)
    with pytest.raises(RuntimeError,match='磁盘空间不足'): check_capture_space(ROOT,1000,'minimal')


def test_long_spline_preparation_refreshes_feedback_before_any_publish(monkeypatch):
    from types import SimpleNamespace
    from w3_coverage_motion import CoverageClient
    client = CoverageClient.__new__(CoverageClient)
    calls = []; fresh = [False]
    def prepare(t,q):
        calls.append('prepare');fresh[0]=False
        return np.zeros_like(q),np.zeros_like(q)
    monkeypatch.setattr('w3_coverage_motion.trajectory_derivatives',prepare)
    def wait(predicate,label,seconds=8):
        calls.append('refresh');fresh[0]=True;assert predicate()
    def guard():
        assert fresh[0], 'Preparatory CPU work must not reuse stale feedback'
        calls.append('guard')
    class Publisher:
        def get_subscription_count(self):return 1
        def publish(self,msg):
            assert fresh[0];calls.append('publish')
    class Clock:
        ns=1_000_000_000
        def now(self):
            value=SimpleNamespace(nanoseconds=self.ns);self.ns+=1_000_000_000;return value
    clock=Clock()
    client.node=SimpleNamespace(get_clock=lambda:clock,create_publisher=lambda *a:Publisher())
    client.ros=SimpleNamespace(spin_once=lambda *a,**kw:None)
    client.names=[f'right_joint_{j}' for j in range(7)];client.publisher=None
    client.state={'cmd':np.zeros(7),'q':np.zeros(7)}
    client.plan={'config':{'arrival_tolerance_deg':1.5}}
    client.check_mode=lambda:calls.append('mode');client.ready=lambda:fresh[0]
    client.wait_until=wait;client.guard=guard;client.endpoint=lambda *a:None
    client.play({'positions_deg':np.zeros((101,7)), 'times_seconds':np.linspace(0,1,101),
                 'hold_seconds':0},lambda:None)
    assert calls[:4]==['prepare','mode','refresh','guard']
    assert 'publish' in calls
