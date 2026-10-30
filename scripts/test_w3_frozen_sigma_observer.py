"""Observer artifacts, exact mean/sigma parity and isolated transport tests."""
import copy
import json
import os
import time
from urllib.request import urlopen

import numpy as np
import pytest
import torch
import yaml

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.frozen_sigma_observer import ROOT, SCHEMA, load_observer, FrozenSigmaInferenceNode
from factr2_next.visualization.frozen_sigma_web import valid_packet, FrozenSigmaWeb, PANEL, SCRIPT
from factr2_next.visualization.web_node import page
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.w3_config import validate_w3_config
from factr2_next.training.frozen_variance import head_variance
from test_w3_frozen_sigma import completed_parent
from test_w3_next_u_ablation import fixture_campaign
import run_w3_frozen_sigma as trial
from export_w3_frozen_sigma import export


@pytest.fixture
def bundle(completed_parent):
    tmp, parent = completed_parent
    source = trial.prepare(parent, tmp/"sigma", dict(trial.DEFAULT_RECIPE, epochs=2, cpu_threads=1, batch_size=8))
    trial.run(source)
    output = export(source, tmp/"observer")
    return source, output


def test_exported_observer_preserves_single_window_mean_and_sigma(bundle):
    source, output = bundle
    loaded = load_observer(output)
    _, cfg, manifest, original, norm = trial.verify(source)
    head, hidden_norm, _ = trial.load_head(source)
    ds = trial.core.dataset(manifest, cfg, "val")
    ds.norm = norm
    for i in (0, len(ds)-1):
        x = ds[i][0].unsqueeze(0)
        with torch.inference_mode():
            mu, variance = loaded.model.distribution(x)
            target, _ = original(x)
            sequence, _ = original.base.lstm(x)
            hidden = original.base.head[:-1](sequence[:,-1]).numpy()
        torch.testing.assert_close(mu, target, atol=0, rtol=0)
        np.testing.assert_allclose(variance.numpy(), head_variance(head, dict(hidden=hidden), hidden_norm), atol=1e-8)
    assert loaded.metadata["deployment_ready"] is False
    for name,purpose in (("inference.yaml","inference"),("web.yaml","visualize")):
        config=yaml.safe_load((output/name).read_text())
        validate_w3_config(config,purpose)
        config.pop("observation_mode")
        with pytest.raises(ValueError,match="Wrong NEXT output root"):
            validate_w3_config(config,purpose)
    assert all(not p.requires_grad for p in loaded.model.parameters())
    assert all(not m.training for m in loaded.model.modules())
    with pytest.raises(ValueError):
        load_checkpoint(output)


@pytest.mark.parametrize("file", ["mean_checkpoint/model.pt", "variance_head/head.pt", "variance_head/normalization.npz"])
def test_deployment_hash_rejection(bundle, file):
    _, output = bundle
    with (output/file).open("ab") as h:
        h.write(b"tampered")
    with pytest.raises(ValueError, match="observer_hash"):
        load_observer(output)


def packet():
    sigma = np.arange(1,8)*.01
    return dict(schema=SCHEMA,joint_names=[f"right_joint_{j}" for j in range(7)],
        stamp_ns=1_000_000_000,state="valid",calibration="uncalibrated",units="Nm",
        raw_external_nm=[.2]*7,free_mean_nm=[.5]*7,sigma_nm=sigma.tolist(),fixed_sigma_nm=[.03]*7,
        nominal_free_residual_lower_nm=(-1.959963984540054*sigma).tolist(),
        nominal_free_residual_upper_nm=(1.959963984540054*sigma).tolist())


@pytest.mark.parametrize("change", ["stale","future","names","nan","band","missing_state"])
def test_sigma_packet_rejects_invalid_or_unpaired_data(change):
    p=packet();names=p["joint_names"].copy()
    assert valid_packet(p,names,1_050_000_000) is p
    if change=="stale":p["stamp_ns"]=1
    elif change=="future":p["stamp_ns"]=2_000_000_000
    elif change=="names":p["joint_names"].reverse()
    elif change=="nan":p["sigma_nm"][0]=float('nan')
    elif change=="band":p["nominal_free_residual_lower_nm"][0]=-.5
    else:p.pop("state")
    with pytest.raises((ValueError,AssertionError)):
        valid_packet(p,names,1_050_000_000)


def test_page_labels_are_uncalibrated_and_clear_on_connection_loss():
    markup=page([f"right_joint_{j}" for j in range(7)],view="joint_torque",extra_panel=PANEL,extra_script=SCRIPT)
    assert "未校准" in markup and "不是外力真值的 95% 置信区间" in markup
    assert 'snapshot.state==="valid"' in markup
    assert 'events.onerror' in markup and 'updateUncertainty(data)' in markup
    assert '__EXTRA_' not in markup


@pytest.mark.skipif(os.environ.get("ROS_DOMAIN_ID")!="199",reason="Requires isolated ROS domain 199")
def test_isolated_observer_and_web_stale_scope_and_no_control_publishers(bundle):
    assert os.environ.get("ROS_LOCALHOST_ONLY")=="1"
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
    _, output=bundle
    cfg=yaml.safe_load((output/"inference.yaml").read_text())
    web_cfg=yaml.safe_load((output/"web.yaml").read_text());web_cfg["web"]["port"]=0
    loaded=load_observer(output)
    bounds=np.array(cfg["operating_envelope"]["position_bounds_deg"])
    q=np.deg2rad(bounds.mean(1)).astype(np.float32)
    rows=np.tile(np.r_[q,np.zeros(7),np.zeros(7)].astype(np.float32),(50,1))
    norm=loaded.normalization
    with torch.inference_mode():
        mean,v=loaded.model.distribution(torch.from_numpy((rows-norm['x_mean'])/norm['x_std'])[None])
    free=mean.numpy()[0]*norm['y_std']+norm['y_mean'];sigma=np.sqrt(v.numpy()[0])*norm['y_std']
    delta=np.array([.1,-.2,.3,-.4,.5,-.6,.7],np.float32)
    rclpy.init();fixture=rclpy.create_node('isolated_sigma_fixture')
    observer=web=None;executor=SingleThreadedExecutor()
    try:
        observer=FrozenSigmaInferenceNode(parameter_overrides=[Parameter('config_file',value=str(output/'inference.yaml'))])
        web=FrozenSigmaWeb(web_cfg,node_name='isolated_sigma_web')
        for node in (fixture,observer,web):executor.add_node(node)
        topics={'joint_pos':'joint_pos','joint_vel':'joint_vel','joint_cmd':'joint_cmd','measured_joint_torque':'joint_effort'}
        pubs={key:fixture.create_publisher(JointState,'/factr2/right/'+suffix,qos_profile_sensor_data) for key,suffix in topics.items()}
        health=fixture.create_publisher(DiagnosticArray,'/factr2/right/adapter_status',qos_profile_sensor_data)
        position=q.copy()
        def publish():
            stamp=fixture.get_clock().now().to_msg()
            status=DiagnosticStatus(name='factr2/adapter/right',level=DiagnosticStatus.OK,message='isolated')
            msg=DiagnosticArray();msg.header.stamp=stamp;msg.status=[status];health.publish(msg)
            for key,pub in pubs.items():
                m=JointState();m.name=cfg['joint_names'];m.header.stamp=stamp
                if key=='joint_vel':m.velocity=[0.]*7
                elif key=='measured_joint_torque':m.effort=(free+delta).astype(float).tolist()
                else:m.position=position.astype(float).tolist()
                pub.publish(m)
        timer=fixture.create_timer(.01,publish)
        def spin(seconds):
            until=time.monotonic()+seconds
            while time.monotonic()<until:executor.spin_once(timeout_sec=.002)
        spin(2.5)
        url=f'http://127.0.0.1:{web.httpd.server_port}/snapshot'
        with urlopen(url,timeout=2) as response:s=json.load(response)
        assert s['state']=='valid' and s['uncertainty']['available'],s
        p=s['uncertainty']['packet']
        from record_w3_atomic_waveforms import validate_base_frame
        frame=validate_base_frame(p,sha256(output/'metadata.json'))
        np.testing.assert_allclose(frame['base_external_raw_nm'],delta,atol=2e-6)
        assert p['compensation_applied'] is False
        np.testing.assert_allclose(p['raw_external_nm'],delta,atol=2e-6)
        np.testing.assert_allclose(p['sigma_nm'],sigma,atol=2e-6)
        for node in (observer,web):
            topics=fixture.get_publisher_names_and_types_by_node(node.get_name(),'/')
            assert not any('command' in topic or '/controller' in topic for topic,_ in topics)
        position[:]=np.deg2rad(bounds[:,0])-0.1
        spin(.7)
        s=json.loads(web._snapshot())
        assert s['state']=='out_of_scope' and not s['uncertainty']['available']
        assert s['uncertainty']['packet'] is None
        timer.cancel();spin(.4)
        s=json.loads(web._snapshot())
        assert s['state']=='stale' and s['uncertainty']['packet'] is None
    finally:
        executor.shutdown()
        if web is not None:web.destroy_node()
        if observer is not None:observer.destroy_node()
        fixture.destroy_node();rclpy.try_shutdown()
