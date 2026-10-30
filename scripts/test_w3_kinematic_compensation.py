"""Causality, external-torque transfer, resets and real-record replay parity."""
import copy
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from factr2_next.inference.kinematic_compensation import (
    SCHEMA, DEFAULT_SIGNAL, Compensator, ResidualModel, SignalProcessor, feature_names, digest,
)


def test_predeclared_physical_blocks_override_legacy_cycle_split(tmp_path):
    from fit_w3_kinematic_compensation import recent
    path = tmp_path/'labeled_samples.csv'
    blocks = {'cycle1_reserved_test':'test', 'cycle2_reserved_train':'train',
              'cycle3_reserved_val':'val'}
    fields = ['source_timestamp_ns', 'next_state', 'exact_event'] + [
        f'{key}_j{j}' for key in ('q_deg', 'velocity_deg_s', 'command_deg',
                                 'measured_tau_nm', 'free_pred_tau_nm') for j in range(1,8)]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for bi, name in enumerate(blocks):
            for k in range(400):
                row = {field:0. for field in fields}
                row.update(source_timestamp_ns=1_000_000_000+(bi*400+k)*10_000_000,
                           next_state='valid', exact_event=name)
                writer.writerow(row)
    item = recent(path, blocks)
    for name, role in blocks.items():
        mask = item['labels']==name
        assert set(item['role'][mask]) == {role, 'unused'}
        used = mask & (item['role']==role)
        assert used.sum()==200
        assert item['t'][used].min() >= item['t'][mask].min()+1_000_000_000
        assert item['t'][used].max() <= item['t'][mask].max()-1_000_000_000
    with pytest.raises(ValueError, match='missing from frozen split'):
        recent(path, {'cycle1_reserved_test':'test'})


def test_stationary_training_noise_prevents_acceleration_overfit():
    from fit_w3_kinematic_compensation import train_artifact
    count=400
    x=np.zeros((count,70));x[:,0]=.02*np.sin(np.arange(count)*.04)
    x[:200,56]=np.tile([-.02,.02],100)
    x[200:,56]=.03*np.sin(np.arange(200)*.04)
    labels=np.array(['baseline_before']*200+['movement']*200)
    target=np.zeros((count,7));target[:,0]=2*x[:,0]+x[:,56]
    item=dict(features=x,target=target,x=np.zeros((count,21)),groups=np.full(count,'one_recording'),
              labels=labels,eligible=np.ones(count,bool))
    fitted=train_artifact([(item,np.ones(count,bool))],'velocity_acceleration',.01,'test','local',.005)
    assert fitted['excitation_floor'][56]>.08
    assert not fitted['active_features'][56]
    np.testing.assert_array_equal(np.array(fitted['weights'])[56],0.)
    assert fitted['active_features'][0]


def artifact(base_hash='test'):
    n=len(feature_names('velocity_acceleration'))
    w=np.zeros((n,7));w[7,0]=3.
    return dict(schema=SCHEMA,feature_mode='velocity_acceleration',feature_names=feature_names('velocity_acceleration'),
        feature_scale=[1.]*n,feature_min=[-100.]*n,feature_max=[100.]*n,weights=w.tolist(),
        active_features=[True]*n,correction_cap_nm=[.5]*7,enabled_joints=[True]*7,
        position_bounds_rad=[[-3.,3.]]*7,signal=DEFAULT_SIGNAL,uses_torque_as_feature=False,
        online_learning=False,base_metadata_sha256=base_hash)


def frame(p,k,torque=0.,offset=0.):
    q=np.zeros(7);q[0]=.03*np.sin(k*.01)
    cmd=q.copy();cmd[0]+=offset
    return p.update(1_000_000_000+k*10_000_000,q,np.full(7,.01),cmd,np.full(7,torque),np.zeros(7))


def test_future_samples_cannot_change_prior_outputs():
    a,b=[Compensator(ResidualModel(artifact())) for _ in range(2)]
    for k in range(200):
        ra=frame(a,k);rb=frame(b,k,torque=.7 if k>=150 else 0.)
        if k<150:np.testing.assert_array_equal(ra['final'],rb['final'])
        np.testing.assert_array_equal(ra['correction'],rb['correction'])


def test_sustained_torque_not_learned_away_and_delay_is_explicit():
    a,b=[Compensator(ResidualModel(artifact())) for _ in range(2)]
    delta=[]
    for k in range(400):
        ra=frame(a,k);rb=frame(b,k,torque=.3 if k>=100 else 0.)
        np.testing.assert_array_equal(ra['correction'],rb['correction'])
        delta.append(rb['final']-ra['final'])
    delta=np.array(delta)
    np.testing.assert_allclose(delta[-1],.3,atol=1e-10)
    assert 0<delta[100,0]<.15
    assert np.flatnonzero(delta[100:,0]>=.27)[0]*10==80


def test_position_kf_does_not_treat_half_velocity_bin_as_motion():
    p=SignalProcessor()
    q=np.zeros(7)
    for k in range(200):
        f=p.update(1_000_000_000+k*10_000_000,q,np.full(7,np.deg2rad(.6296888)),q,q,q)
    np.testing.assert_array_equal(f['actual'][1:],0.)
    assert np.all(f['motor_velocity']>0)


def test_velocity_and_acceleration_from_position_ramp():
    p=SignalProcessor()
    for k in range(300):
        q=np.full(7,k*.01*.02)
        f=p.update(1_000_000_000+k*10_000_000,q,np.zeros(7),q,q,q)
    np.testing.assert_allclose(f['actual'][1],.02,atol=1e-6)
    np.testing.assert_allclose(f['actual'][2],0.,atol=1e-6)


def test_matched_input_filters_and_residual_identity():
    p=Compensator(ResidualModel(artifact()))
    for k in range(200):
        r=frame(p,k,torque=np.sin(k*.3))
        np.testing.assert_allclose(r['measured']-r['free_total'],r['raw'],atol=1e-15)


def test_gap_resets_history_and_does_not_zero_true_residual():
    p=Compensator(ResidualModel(artifact()))
    for k in range(100):frame(p,k,torque=.4)
    q=np.zeros(7)
    r=p.update(3_000_000_000,q,q,q,np.full(7,.4),q)
    assert r['reset'] and not r['ready']
    np.testing.assert_allclose(r['correction'],0.)
    np.testing.assert_allclose(r['final'],.4)
    with pytest.raises(ValueError,match='stamp not advancing'):
        p.update(3_000_000_000,q,q,q,q,q)
    assert p.signal.last is None


def test_out_of_support_bypasses_correction_without_clipping_external_torque():
    a=artifact();a['position_bounds_rad']=[[-.001,.001]]*7
    p=Compensator(ResidualModel(a))
    for k in range(200):r=frame(p,k,torque=2.)
    assert not r['supported']
    np.testing.assert_array_equal(r['correction'],0.)
    np.testing.assert_allclose(r['final'],2.,atol=1e-10)


@pytest.mark.parametrize('bad',['torque_feature','online','order','nan','inactive_weight'])
def test_rejects_malformed_or_adaptive_models(bad):
    a=artifact()
    if bad=='torque_feature':a['uses_torque_as_feature']=True
    elif bad=='online':a['online_learning']=True
    elif bad=='order':a['feature_names'].reverse()
    elif bad=='nan':a['weights'][0][0]=float('nan')
    else:a['active_features'][7]=False
    with pytest.raises(ValueError):ResidualModel(a)


def test_compensation_page_defaults_to_final_and_keeps_reference():
    from observe_w3_kinematic_compensation import web_config
    from factr2_next.visualization.web_node import page
    from factr2_next.w3_config import validate_w3_config
    c=web_config(8083);validate_w3_config(c,'visualize')
    text=page(c['joint_names'],feedback=True,view='joint_torque',torque_display=c['plot']['torque_display'])
    groups=json.loads(text.split('const groups=',1)[1].split(';',1)[0])
    assert '最终平滑输出' in str(groups) and 'NEXT 原始残差对照' in str(groups)
    assert groups[1][1][0][-1] is True and groups[0][1][0][-1] is False
    assert 'next.kinematic_compensated.' in text
    assert '"ext_j1", "J1' in text
    c.pop('observation_mode')
    with pytest.raises(ValueError):validate_w3_config(c,'visualize')


@pytest.mark.skipif(not Path('data/right/kinematic_compensation_20261023_v2/model.json').exists(),reason='local completed pilot only')
def test_completed_record_streaming_and_fitting_replay_are_identical():
    root=Path('data/right/kinematic_compensation_20261023_v2')
    p=Compensator(ResidualModel(json.loads((root/'model.json').read_text())))
    with Path('log/reciprocal_frozen_sigma_20261023_v3/labeled_samples.csv').open() as f:rows=list(csv.DictReader(f))
    with np.load(root/'replay.npz') as h:
        expected=h['final'];correction=h['correction_nm'];scope=h['supported']
    for i,r in enumerate(rows):
        def values(key):return np.array([float(r[f'{key}_j{j}']) for j in range(1,8)])
        q,v,cmd=[np.deg2rad(values(k)) for k in ('q_deg','velocity_deg_s','command_deg')]
        out=p.update(int(r['source_timestamp_ns']),q,v,cmd,values('measured_tau_nm'),values('free_pred_tau_nm'))
        np.testing.assert_allclose(out['final'],expected[i],atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(out['correction'],correction[i],atol=1e-8,rtol=1e-8)
        assert bool(out['supported'] and out['ready']) == bool(scope[i])


@pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID')!='199',reason='requires isolated domain 199')
@pytest.mark.parametrize('model_kind',['legacy','nonlinear_fallback'])
def test_isolated_transport_stale_and_read_only(tmp_path,model_kind):
    assert os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
    from sensor_msgs.msg import JointState
    from factr2_next.inference.kinematic_observer import KinematicObserver,BASE_ROOT,NAMES
    from factr2_next.inference.kinematic_compensation import ROOT
    base=tmp_path/'base';base.mkdir();(base/'metadata.json').write_text(json.dumps(dict(operating_envelope=dict(position_bounds_deg=[[-180,180]]*7))))
    model=tmp_path/'model.json';model_artifact=artifact(digest(base/'metadata.json'))
    if model_kind=='nonlinear_fallback':
        from test_w3_pose_dynamic_compensation import nonlinear_artifact,pose_artifact
        model_artifact=nonlinear_artifact();model_artifact['support_fallback']=pose_artifact()
        model_artifact['base_metadata_sha256']=digest(base/'metadata.json')
        model_artifact['support_fallback']['base_metadata_sha256']=digest(base/'metadata.json')
    model.write_text(json.dumps(model_artifact))
    rclpy.init();source=rclpy.create_node('isolated_kinematic_source');node=KinematicObserver(model,base)
    ex=SingleThreadedExecutor();ex.add_node(source);ex.add_node(node)
    topics=['/factr2/right/joint_pos','/factr2/right/joint_vel','/factr2/right/joint_cmd','/factr2/right/joint_effort',BASE_ROOT+'/free_joint_torque_pred']
    pubs=[source.create_publisher(JointState,t,10) for t in topics]
    bp=source.create_publisher(DiagnosticArray,BASE_ROOT+'/status',qos_profile_sensor_data)
    ap=source.create_publisher(DiagnosticArray,'/factr2/right/adapter_status',qos_profile_sensor_data)
    got=[];source.create_subscription(JointState,ROOT+'/external_joint_torque',lambda m:got.append(m),10)
    def tick():
        ts=source.get_clock().now().to_msg()
        for pub,name in [(bp,'factr2/next/right'),(ap,'factr2/adapter/right')]:
            msg=DiagnosticArray();msg.header.stamp=ts;s=DiagnosticStatus();s.name=name;s.level=DiagnosticStatus.OK;s.values=[KeyValue(key='state',value='"valid"')];msg.status=[s];pub.publish(msg)
        for i,pub in enumerate(pubs):
            m=JointState();m.header.stamp=ts;m.name=NAMES
            m.position=[.1 if i==4 else 0.]*7;m.velocity=[0.]*7;m.effort=[.4]*7;pub.publish(m)
    timer=source.create_timer(.01,tick)
    try:
        end=time.monotonic()+8
        while time.monotonic()<end and not (node.state=='valid' and len(got)>60):ex.spin_once(timeout_sec=.005)
        assert node.state=='valid' and len(got)>60
        np.testing.assert_allclose(got[-1].position,.3,atol=1e-5)
        published=[t for t,_ in node.get_publisher_names_and_types_by_node(node.get_name(),'/')]
        assert published and all(t.startswith(ROOT+'/') or t in ('/rosout','/parameter_events') for t in published)
        timer.cancel();end=time.monotonic()+.4
        while time.monotonic()<end:ex.spin_once(timeout_sec=.005)
        assert node.state=='stale' and node.latest is None
    finally:ex.shutdown();node.destroy_node();source.destroy_node();rclpy.try_shutdown()
