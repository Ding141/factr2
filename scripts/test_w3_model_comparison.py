"""Isolated ROS verification of coherent old/new inference and filter resets."""
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
import json
from std_msgs.msg import String
from rclpy.qos import QoSProfile,DurabilityPolicy
import yaml
from test_w3_endpoint_wrench import model as geometry_model
from w3_endpoint_force_panel import PANEL_HTML,PANEL_SCRIPT

from observe_w3_model_comparison import ComparisonNode,predict_mean,dashboard_config
from factr2_next.visualization.web_node import WebNode
from factr2_next.inference.joint_lstm_observer import ResidualCascade


@pytest.mark.parametrize('residual_enabled',[False,True])
def test_same_history_predictions_causal_filters_and_stale_reset(tmp_path,residual_enabled):
    assert os.environ.get('ROS_DOMAIN_ID')=='199','Never publish synthetic samples on the hardware domain'
    root=Path(__file__).resolve().parents[1]
    bundle=root/'models/w3/right/joint_lstm_observer_20261028_v1'
    if not bundle.exists():pytest.skip('Real trained artifacts unavailable')
    residual_path=None
    if residual_enabled:
        from test_w3_kinematic_compensation import artifact
        from factr2_next.inference.kinematic_compensation import digest
        a=artifact(digest(bundle/'mean_checkpoint/metadata.json'))
        a.update(mean_model_sha256=digest(bundle/'mean_checkpoint/model.pt'),application='subtract_before_identical_RC12_RC5')
        residual_path=tmp_path/'model.json';residual_path.write_text(json.dumps(a))
    rclpy.init();node=ComparisonNode(bundle,residual_model=residual_path,parameter_overrides=[Parameter('config_file',value=str(bundle/'inference.yaml'))])
    import torch
    assert torch.get_num_threads()==1
    geometry=geometry_model();force_cfg=yaml.safe_load((root/'config/w3/right/endpoint_force_comparison.yaml').read_text())
    force_cfg.update(robot_description_sha256=geometry.sha256,root_frame=geometry.root_frame)
    node.enable_endpoint_force(force_cfg)
    description=node.create_publisher(String,'/robot_description',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    description.publish(String(data=geometry.description))
    webs=[WebNode(dashboard_config(bundle,node.old_bundle,old,0,residual=residual_enabled),node_name='test_web_'+str(old),
        extra_panel=PANEL_HTML,extra_script=PANEL_SCRIPT,
        extra_snapshot=lambda k=('old' if old else 'new'):dict(endpoint_force=node.endpoint_force.snapshot(k))) for old in (True,False)]
    names=[f'right_joint_{j}' for j in range(7)]
    pubs=[node.create_publisher(JointState,'/factr2/right/'+key,20)
          for key in ('joint_pos','joint_vel','joint_cmd','joint_effort')]
    health=node.create_publisher(DiagnosticArray,'/factr2/right/adapter_status',qos_profile_sensor_data)
    packets=[];last=None
    def run(seconds,publish=True):
        nonlocal last
        start=time.monotonic();next_send=start
        while time.monotonic()-start<seconds:
            now=time.monotonic()
            if publish and now>=next_send:
                stamp=node.get_clock().now().to_msg()
                diag=DiagnosticArray();diag.header.stamp=stamp
                st=DiagnosticStatus();st.name='factr2/adapter/right';st.level=DiagnosticStatus.OK
                st.values=[KeyValue(key='output_hz_last_second',value='100.0')];diag.status=[st];health.publish(diag)
                q=np.deg2rad([1.,50.,-4.,-62.,1.,-2.,0.]);q[0]+=.001*np.sin(now)
                for pub,field,values in zip(pubs,('position','velocity','position','effort'),(q,np.zeros(7),q+.001,np.arange(7)*.1)):
                    m=JointState();m.header.stamp=stamp;m.name=names;setattr(m,field,values.tolist());pub.publish(m)
                next_send=now+.01
            rclpy.spin_once(node,timeout_sec=.001)
            if node.last_packet and node.last_packet['stamp_ns']!=last:
                last=node.last_packet['stamp_ns'];packets.append(node.last_packet)
    try:
        run(3.)
        assert len(packets)>100
        assert len(packets)==node.counts['outputs']
        for web in webs:
            snapshot=json.loads(web._snapshot())
            assert 'ext_j7' in snapshot and 'raw_j7' in snapshot
            assert len(snapshot['endpoint_force']['history']['t'])>5
            assert snapshot['endpoint_force']['frame_id']=='right_attachment_point'
            assert 'endpoint-force-plot' in web.html and 'updateEndpointForce(data)' in web.html
        history,_,_=node.pair;p=packets[-1];a=p['recording_frame']
        assert p['history_sha256']==hashlib.sha256(history.tobytes()).hexdigest()
        assert p['same_input_history'] and p['compensation_enabled']==residual_enabled and not p['sigma_used']
        np.testing.assert_allclose(predict_mean(node.old_loaded,history),a['old_free_nm'],atol=0,rtol=0)
        base=predict_mean(node.loaded,history)
        if residual_enabled:
            from record_w3_model_comparison import validate_packet
            validate_packet(p,node.model_hashes,node.residual.hash)
            base=(base+np.asarray(a['correction_nm'],np.float32)).astype(np.float32)
        np.testing.assert_allclose(base,a['new_free_nm'],atol=0,rtol=0)
        filters={'old':ResidualCascade(),'new':ResidualCascade()}
        for packet in packets:
            a=packet['recording_frame']
            for kind in filters:
                expected=np.asarray(a['measured_tau_nm'],np.float32)-np.asarray(a[kind+'_free_nm'],np.float32)
                np.testing.assert_array_equal(expected,a[kind+'_raw_nm'])
                np.testing.assert_allclose(filters[kind].update(expected),a[kind+'_filtered_nm'],atol=1e-7,rtol=0)
        run(.35,False);assert node.state=='stale'
        assert node.endpoint_force.snapshot('new')['force'] is None
        count=len(packets);run(1.)
        assert len(packets)>count
        a=packets[count]['recording_frame']
        for kind in ('old','new'):
            np.testing.assert_allclose(a[kind+'_raw_nm'],a[kind+'_filtered_nm'],atol=1e-7,rtol=0)
        assert not node.get_publishers_info_by_topic('/joint_position_command')
        assert not node.get_publishers_info_by_topic('/joint_position_controller/stop')
    finally:
        for web in webs:web.destroy_node()
        node.destroy_node();rclpy.try_shutdown()
