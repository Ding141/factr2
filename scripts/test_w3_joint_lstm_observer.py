"""Read-only deployment contracts and causal filter transfer for new means."""
import json
import shutil
import os
import time

import numpy as np
import pytest
import torch
import yaml

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.inference_node import TorqueFilter
from factr2_next.inference.joint_lstm_observer import load_bundle, ResidualCascade, ROOT, SCHEMA
from factr2_next.training.w3_train import train
from factr2_next.w3_config import validate_w3_config
from test_w3_next_u_ablation import fixture_campaign
from export_w3_joint_lstm import configs, REPO


def test_preserved_shared_lstm_mean_is_identical_without_sigma_or_residual():
    from observe_w3_shared_baseline import DEFAULT
    from factr2_next.inference.frozen_sigma_observer import load_observer
    before=sha256(DEFAULT/'mean_checkpoint/model.pt')
    loaded=load_observer(DEFAULT)
    x=torch.randn(3,50,21)
    with torch.inference_mode():
        frozen_mean=loaded.model(x)
        shared_mean=loaded.model.base(x)
    torch.testing.assert_close(shared_mean,frozen_mean,rtol=0,atol=0)
    assert sha256(DEFAULT/'mean_checkpoint/model.pt')==before


def test_cascade_equals_paired_filters_and_preserves_sustained_torque():
    rng=np.random.default_rng(0)
    tau,mu=rng.normal(size=(400,7)),rng.normal(size=(400,7))
    a,b=[TorqueFilter(dict(mode='lowpass',sample_hz=100,cutoff_hz=12)) for _ in range(2)]
    final=TorqueFilter(dict(mode='lowpass',sample_hz=100,cutoff_hz=5))
    cascade=ResidualCascade()
    for x,y in zip(tau,mu):
        np.testing.assert_allclose(cascade.update(x-y),final.update(a.update(x)-b.update(y)),atol=5e-7)
    cascade.reset()
    for _ in range(200):out=cascade.update(np.full(7,.3))
    np.testing.assert_allclose(out,.3,atol=1e-6)
    assert not hasattr(cascade,'model')


@pytest.fixture
def bundle(fixture_campaign):
    tmp,_,manifest,_=fixture_campaign
    cfg=yaml.safe_load((REPO/'config/w3/right/train_joint_lstm.yaml').read_text())
    cfg['data']['manifest']=str(manifest)
    cfg['train'].update(epochs=1,batch_size=8,cpu_threads=1,stop_file=str(tmp/'STOP'))
    cfg['save']['output_dir']=str(tmp/'runs')
    checkpoint=train(cfg);out=tmp/'bundle';out.mkdir()
    shutil.copytree(checkpoint,out/'mean_checkpoint')
    envelope=dict(position_bounds_deg=[[-20.,20.]]*7,position_margin_deg=0.)
    inference,web=configs(out,envelope)
    for name,value in (('inference.yaml',inference),('web.yaml',web)):
        (out/name).write_text(yaml.safe_dump(value))
    files=['inference.yaml','web.yaml']+['mean_checkpoint/'+f for f in ('model.pt','config.yaml','metrics.json','metadata.json','normalization.npz')]
    write_json(out/'bundle.json',dict(schema=SCHEMA,architecture='independent_joint_lstm_v1',
        controller_changes=False,compensation_enabled=False,operating_envelope=envelope,files={f:sha256(out/f) for f in files}))
    return out


def test_bundle_preserves_new_mean_and_namespace_contract(bundle):
    loaded,meta=load_bundle(bundle)
    assert not any(p.requires_grad for p in loaded.model.parameters())
    assert loaded.metadata['architecture']=='independent_joint_lstm_v1'
    assert len(loaded.model.joints)==7
    for name,purpose in (('inference.yaml','inference'),('web.yaml','visualize')):
        cfg=yaml.safe_load((bundle/name).read_text())
        validate_w3_config(cfg,purpose)
        assert cfg['next_topic_root']==ROOT
        del cfg['observation_mode']
        with pytest.raises(ValueError,match='Wrong NEXT output root'):
            validate_w3_config(cfg,purpose)
    assert yaml.safe_load((bundle/'inference.yaml').read_text())['contact']['enabled'] is False


@pytest.mark.parametrize('file',['mean_checkpoint/model.pt','inference.yaml','web.yaml'])
def test_bundle_tamper_rejected(bundle,file):
    with (bundle/file).open('ab') as h:h.write(b'changed')
    with pytest.raises(ValueError,match='bundle_hash'):load_bundle(bundle)


def test_activation_is_pinned_and_preserves_previous_profile(bundle,tmp_path,monkeypatch):
    import activate_w3_joint_lstm as activation
    fake_repo=tmp_path/'repo';config=fake_repo/'config/w3/right';config.mkdir(parents=True)
    previous={'schema':'previous','kind':'old'}
    write_json(config/'observer_active.json',previous)
    monkeypatch.setattr(activation,'REPO',fake_repo)
    meta=json.loads((bundle/'bundle.json').read_text())
    report=tmp_path/'evaluation';report.mkdir()
    mean_sha=sha256(bundle/'mean_checkpoint/model.pt')
    write_json(report/'comparison.json',dict(mean_sha256=mean_sha,splits={'val':{},'test':{}}))
    meta.update(mean_model_sha256=mean_sha,evaluation=str(report),evaluation_sha256=sha256(report/'comparison.json'))
    write_json(bundle/'bundle.json',meta)
    active=activation.activate(bundle,tmp_path/'activation')
    assert active['kind']=='joint_lstm' and active['controller_changes'] is False
    assert json.loads((tmp_path/'activation/observer_active.json').read_text())==previous
    assert active['bundle_sha256']==sha256(bundle/'bundle.json')
    # Missing test evaluation cannot silently change the existing profile.
    write_json(report/'comparison.json',dict(mean_sha256=mean_sha,splits={'val':{}}))
    meta['evaluation_sha256']=sha256(report/'comparison.json');write_json(bundle/'bundle.json',meta)
    with pytest.raises(ValueError,match='complete_evaluation'):
        activation.activate(bundle,tmp_path/'bad_activation')
    assert json.loads((config/'observer_active.json').read_text())==active


@pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID')!='199',reason='Requires isolated ROS domain 199')
@pytest.mark.parametrize('variant',['joint','shared'])
def test_isolated_joint_lstm_observer_fresh_stale_scope_and_readonly(bundle,variant):
    assert os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy,qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus
    from factr2_next.inference.joint_lstm_observer import JointLSTMInferenceNode
    from factr2_next.visualization.web_node import WebNode
    if variant=='shared':
        from observe_w3_shared_baseline import SharedBaselineInferenceNode,DEFAULT
        from factr2_next.inference.frozen_sigma_observer import load_observer
        bundle=DEFAULT;loaded=load_observer(bundle)
    else:loaded,_=load_bundle(bundle)
    norm=loaded.normalization
    inference_cfg=yaml.safe_load((bundle/'inference.yaml').read_text())
    bounds=np.deg2rad(np.asarray(inference_cfg['operating_envelope']['position_bounds_deg']))
    position=bounds.mean(1).astype(np.float32)
    rows=np.tile(np.r_[position,np.zeros(14)].astype(np.float32),(50,1))
    torch.set_num_threads(1)
    with torch.inference_mode():free=loaded.model(torch.from_numpy((rows-norm['x_mean'])/norm['x_std'])[None]).numpy()[0]*norm['y_std']+norm['y_mean']
    delta=np.arange(1,8,dtype=np.float32)*.03
    rclpy.init();fixture=rclpy.create_node('isolated_joint_lstm_fixture');executor=SingleThreadedExecutor()
    observer=web=None
    try:
        overrides=[Parameter('config_file',value=str(bundle/'inference.yaml'))]
        observer=(SharedBaselineInferenceNode(parameter_overrides=overrides) if variant=='shared'
                  else JointLSTMInferenceNode(bundle,parameter_overrides=overrides))
        cfg=yaml.safe_load((bundle/'web.yaml').read_text());cfg['web']['port']=0
        web=WebNode(cfg,node_name='isolated_joint_lstm_web')
        for node in (fixture,observer,web):executor.add_node(node)
        qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE)
        pubs=[fixture.create_publisher(JointState,'/factr2/right/'+suffix,qos) for suffix in ('joint_pos','joint_vel','joint_cmd','joint_effort')]
        health=fixture.create_publisher(DiagnosticArray,'/factr2/right/adapter_status',qos_profile_sensor_data)
        def publish():
            stamp=fixture.get_clock().now().to_msg()
            msg=DiagnosticArray();msg.header.stamp=stamp;msg.status=[DiagnosticStatus(name='factr2/adapter/right',level=DiagnosticStatus.OK,message='isolated')];health.publish(msg)
            for i,pub in enumerate(pubs):
                m=JointState();m.header.stamp=stamp;m.name=cfg['joint_names']
                if i==1:m.velocity=[0.]*7
                elif i==3:m.effort=(free+delta).astype(float).tolist()
                else:m.position=position.tolist()
                pub.publish(m)
        timer=fixture.create_timer(.01,publish)
        def spin(duration):
            end=time.monotonic()+duration
            while time.monotonic()<end:executor.spin_once(timeout_sec=.002)
        spin(2.)
        assert observer.state=='valid',observer.reason
        np.testing.assert_allclose(observer.last_external_output,delta,atol=3e-6)
        for node in (observer,web):
            published=fixture.get_publisher_names_and_types_by_node(node.get_name(),'/')
            assert not any('/command' in topic or '/controller' in topic for topic,_ in published)
        position[:]=bounds[:,1]+.1;spin(.7)
        assert observer.state=='out_of_scope'
        timer.cancel();spin(.4)
        assert observer.state=='stale'
    finally:
        executor.shutdown()
        if web is not None:web.destroy_node()
        if observer is not None:observer.destroy_node()
        fixture.destroy_node();rclpy.try_shutdown()
