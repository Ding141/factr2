"""Same-frame, same-history old/new torque comparison; no actuator publishers."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import hashlib

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from rclpy.clock import Clock,ClockType
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import torch
import yaml

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.frozen_sigma_observer import load_observer
from factr2_next.inference.joint_lstm_observer import JointLSTMInferenceNode, ResidualCascade
from factr2_next.inference.operating_envelope import OperatingEnvelope
from factr2_next.visualization.web_node import WebNode
from observe_w3_joint_lstm import resolve_bundle
from observe_w3_shared_baseline import DEFAULT as OLD_BUNDLE

OLD_ROOT='/next/right/frozen_sigma'
PACKET_TOPIC='/next/right/model_comparison/frame'


def predict_mean(loaded, history):
    norm=loaded.normalization
    x=torch.from_numpy((history-norm['x_mean'])/norm['x_std']).unsqueeze(0).to(loaded.device)
    with torch.inference_mode():y=loaded.model(x).cpu().numpy()[0]
    return (y*norm['y_std']+norm['y_mean']).astype(np.float32)


class ComparisonNode(JointLSTMInferenceNode):
    def __init__(self,bundle,old_bundle=OLD_BUNDLE,cpu_threads=1,residual_model=None,**kwargs):
        if cpu_threads not in (1,2,4):raise ValueError('comparison_cpu_threads')
        self.residual=None
        if residual_model is not None:
            from w3_joint_residual import JointResidual
            self.residual=JointResidual(residual_model,bundle)
        self.old_bundle=Path(old_bundle).resolve()
        if self.residual is None:
            self.old_loaded=load_observer(self.old_bundle)
            self.old_loaded.model=self.old_loaded.model.base
            self.old_loaded.model.eval().requires_grad_(False)
            self.old_envelope=OperatingEnvelope(self.old_loaded.metadata['operating_envelope'],7)
        self.old_filter=ResidualCascade();self.pair=None;self.last_packet=None
        self.model_hashes=dict(old=sha256(self.old_bundle/'mean_checkpoint/model.pt'),
                               new=sha256(Path(bundle)/'mean_checkpoint/model.pt'))
        self.old_metadata_hash=sha256(self.old_bundle/'metadata.json')
        self.old_state,self.old_reason='warming','waiting_50_fresh_frames'
        super().__init__(bundle,**kwargs)
        if self.residual is not None:
            self.old_loaded=self.loaded;self.old_envelope=self.operating_envelope
            self.model_hashes['old']=self.model_hashes['new']
        self.last_correction=np.zeros(7,dtype=np.float32);self.correction_supported=False
        # Small batch-one branches spend more time coordinating CPU workers
        # than computing. Override only runtime scheduling, preserving the
        # signed checkpoint configuration and every learned parameter.
        torch.set_num_threads(cpu_threads)
        self.packet_pub=self.create_publisher(String,PACKET_TOPIC,100)
        self.old_pubs={k:self.create_publisher(JointState,OLD_ROOT+'/'+topic,10) for k,topic in
            (('free','free_joint_torque_pred'),('raw','external_joint_torque/raw'),('filtered','external_joint_torque'))}
        self.old_status_pub=self.create_publisher(DiagnosticArray,OLD_ROOT+'/status',qos_profile_sensor_data)

    def _predict_free_torque(self):
        history=self.buffer.array().copy()
        if self.residual is None:
            old=predict_mean(self.old_loaded,history)
            new=super()._predict_free_torque()
        else:
            old=super()._predict_free_torque()
            self.last_correction,self.correction_supported=self.residual.correction(
                allow=not self.operating_envelope.outside_joints(history))
            new=(old+self.last_correction).astype(np.float32)
        if not np.isfinite(old).all():raise FloatingPointError('old_prediction_nonfinite')
        self.pair=(history,old,new)
        return new

    def _invalidate(self,*args,**kwargs):
        self.old_filter.reset();self.pair=None
        if self.residual is not None:self.residual.reset()
        return super()._invalidate(*args,**kwargs)

    def _append_history(self,joint_pos,joint_vel,joint_cmd):
        super()._append_history(joint_pos,joint_vel,joint_cmd)
        if self.residual is not None:
            self.residual.update(self.last_stamp,joint_pos,joint_vel,joint_cmd)

    def _callback(self,*msgs):
        before=self.counts['outputs']
        try:super()._callback(*msgs)
        except FloatingPointError:
            self._invalidate('old_prediction_nonfinite');return
        if self.counts['outputs']==before:return
        history,old,new=self.pair
        measured=self._extract(msgs[-1],'measured_joint_torque')
        old_raw=measured-old;new_raw=measured-new
        old_filtered=self.old_filter.update(old_raw)
        outside=self.old_envelope.outside_joints(history)
        self.old_state='out_of_scope' if outside else 'valid'
        self.old_reason='outside_collection_range:'+','.join(self.names[j] for j in outside) if outside else 'ok'
        stamp=msgs[-1].header.stamp
        for key,value in (('free',old),('raw',old_raw),('filtered',old_filtered)):
            self.old_pubs[key].publish(self._joint_state(value,stamp))
        values=dict(q_deg=np.rad2deg(history[-1,:7]),velocity_deg_s=np.rad2deg(history[-1,7:14]),
            command_deg=np.rad2deg(history[-1,:7]+history[-1,14:]),measured_tau_nm=measured,
            old_free_nm=old,new_free_nm=new,old_raw_nm=old_raw,new_raw_nm=new_raw,
            old_filtered_nm=old_filtered,new_filtered_nm=self.last_external_output)
        packet=dict(schema='w3_joint_lstm_comparison_v1',stamp_ns=self.last_output_stamp,
            joint_names=self.names,units='Nm',model_sha256=self.model_hashes,
            compensation_enabled=self.residual is not None,sigma_used=False,history_frames=len(history),
            same_input_history=True,history_sha256=hashlib.sha256(history.tobytes()).hexdigest(),
            state=self.state,reason=self.reason,old_state=self.old_state,old_reason=self.old_reason,
            new_state=self.state,new_reason=self.reason,filter='LP12_then_LP5_100Hz',
            recording_frame={k:np.asarray(v,dtype=float).tolist() for k,v in values.items()})
        if self.residual is not None:
            packet.update(comparison_kind='same_joint_lstm_residual_ablation_v1',
                compensation_sha256=self.residual.hash,compensation_applied=self.correction_supported,
                enabled_joints=self.residual.model.enabled.tolist())
            packet['recording_frame'].update(correction_nm=self.last_correction.astype(float).tolist(),
                estimated_velocity_deg_s=np.rad2deg(self.residual.frame['actual'][1]).tolist(),
                estimated_acceleration_deg_s2=np.rad2deg(self.residual.frame['actual'][2]).tolist())
        self.last_packet=packet
        self.packet_pub.publish(String(data=json.dumps(packet,allow_nan=False,separators=(',',':'))))
        if hasattr(self,'endpoint_force'):self.endpoint_force.ingest(packet)

    def enable_endpoint_force(self,cfg):
        from w3_endpoint_force_panel import EndpointForcePanel
        self.endpoint_force=EndpointForcePanel(cfg,self.model_hashes)
        self.endpoint_pub=self.create_publisher(String,'/next/right/model_comparison/endpoint_force',10)
        def description(msg):
            try:self.endpoint_force.description(msg.data)
            except ValueError as exc:
                self.endpoint_force.model=None;self.endpoint_force.description_reason=str(exc)
                self.get_logger().error('Endpoint model rejected: '+str(exc))
        self.endpoint_description_sub=self.create_subscription(String,'/robot_description',description,
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        def tick():
            self.endpoint_force.tick(self.get_clock().now().nanoseconds)
            snapshots={k:self.endpoint_force.snapshot(k) for k in ('old','new')}
            for s in snapshots.values():s.pop('history')
            value=dict(schema='w3_comparison_endpoint_force_v1',model_sha256=self.model_hashes,
                compensation_enabled=self.residual is not None,**snapshots)
            self.endpoint_pub.publish(String(data=json.dumps(value,allow_nan=False,separators=(',',':'))))
        self.endpoint_timer=self.create_timer(.1,tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _publish_status(self):
        super()._publish_status()
        if not hasattr(self,'old_status_pub'):return
        good=self.state in ('valid','out_of_scope')
        state=self.old_state if good else self.state;reason=self.old_reason if good else self.reason
        st=DiagnosticStatus();st.name='factr2/next/right';st.level=DiagnosticStatus.OK if state=='valid' else DiagnosticStatus.WARN
        st.message=reason;st.values=[KeyValue(key=k,value=json.dumps(v)) for k,v in
            dict(state=state,reason=reason,outputs=self.counts['outputs'],compensation_enabled=False).items()]
        msg=DiagnosticArray();msg.header.stamp=self.get_clock().now().to_msg();msg.status=[st]
        self.old_status_pub.publish(msg)

    def _watchdog(self):
        if time.monotonic()-self.last_identity_check>=1.:
            try:
                if self.residual is None:
                    self.identity_ok &= (sha256(self.old_bundle/'metadata.json')==self.old_metadata_hash
                        and sha256(self.old_bundle/'mean_checkpoint/model.pt')==self.model_hashes['old'])
                else:
                    self.identity_ok &= (sha256(self.residual.path)==self.residual.hash
                        and sha256(self.bundle/'mean_checkpoint/model.pt')==self.model_hashes['new'])
            except OSError:self.identity_ok=False
        super()._watchdog()


def dashboard_config(bundle,old_bundle,old,port,residual=False):
    cfg=yaml.safe_load(((Path(old_bundle) if old and not residual else Path(bundle))/'web.yaml').read_text())
    cfg['web']['port']=port
    if old:
        cfg['next_topic_root']=OLD_ROOT;cfg['model_label']='旧模型 · 共享 LSTM · 同帧对比'
        cfg['plot']['torque_display']=dict(primary='ext',storage_prefix='next.shared.comparison.',
            raw_label='旧模型 · 原始外力矩（N·m）',filtered_label='旧模型 · 平滑外力矩（N·m）',
            description='保留的共享LSTM同时预测七个关节；与新模型读取相同50帧输入，附加补偿关闭，滤波相同。')
    else:cfg['model_label']='新模型 · 七关节独立 LSTM · 同帧对比'
    cfg['model_notice']='两个模型读取完全相同的50帧状态，附加残差补偿关闭，不使用sigma，均使用12Hz和5Hz低通。'
    if residual:
        if old:cfg['observation_mode']='uncalibrated_research'  # legacy baseline topic, no sigma
        cfg['model_label']='当前独立 LSTM · '+('未补偿基线' if old else '已启用残差补偿')
        cfg['plot']['torque_display']=dict(primary='ext',storage_prefix='next.joint.residual.'+str(old)+'.',
            raw_label=('未补偿' if old else '补偿后')+' · 输出滤波前（N·m）',
            filtered_label=('未补偿' if old else '补偿后')+' · 同滤波输出（N·m）',
            description='两路来自同一个独立关节LSTM及同一帧输入；仅补偿分支增加冻结的运动学残差修正，两路滤波相同。')
        cfg['model_notice']='8082：当前独立LSTM未补偿；8083：同一模型残差补偿。不是旧共享LSTM。补偿仅用运动学输入，无在线归零，不使用sigma；真实接触尚未标定。'
    return cfg


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-file',type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,choices=(1,2,4),default=1)
    p.add_argument('--residual-model',type=Path)
    p.add_argument('--endpoint-config',type=Path,default=Path(__file__).resolve().parents[1]/'config/w3/right/endpoint_force_comparison.yaml')
    a=p.parse_args(argv);bundle=resolve_bundle()
    if os.environ.get('ROS_DOMAIN_ID')!='74':p.error('Real comparison requires domain 74')
    rclpy.init();executor=SingleThreadedExecutor();node=None;webs=[]
    try:
        node=ComparisonNode(bundle,cpu_threads=a.cpu_threads,residual_model=a.residual_model,parameter_overrides=[Parameter('config_file',value=str(bundle/'inference.yaml'))],
            cli_args=['--ros-args','-r','__node:=next_model_comparison_right'],use_global_arguments=False)
        executor.add_node(node)
        if a.endpoint_config:node.enable_endpoint_force(yaml.safe_load(a.endpoint_config.read_text()))
        for old,port in ((True,8082),(False,8083)):
            cfg=dashboard_config(bundle,node.old_bundle,old,port,residual=node.residual is not None)
            extra={}
            if a.endpoint_config:
                from w3_endpoint_force_panel import PANEL_HTML,PANEL_SCRIPT
                kind='old' if old else 'new'
                extra=dict(extra_panel=PANEL_HTML,extra_script=PANEL_SCRIPT,
                    extra_snapshot=lambda k=kind:dict(endpoint_force=node.endpoint_force.snapshot(k)))
            web=WebNode(cfg,node_name='model_comparison_dashboard_'+str(port),use_global_arguments=False,**extra)
            webs.append(web);executor.add_node(web)
        def status():
            packet=node.last_packet
            write_json(a.state_file,dict(service='running',pid=os.getpid(),state=node.state,reason=node.reason,
                outputs=node.counts['outputs'],counts=node.counts,model_sha256=node.model_hashes,
                compensation_enabled=node.residual is not None,same_input_history=True,packet_topic=PACKET_TOPIC,
                compensation_sha256=node.residual.hash if node.residual else None,
                compensation_applied=node.correction_supported,
                enabled_joints=node.residual.model.enabled.tolist() if node.residual else [False]*7,
                endpoint_force_enabled=hasattr(node,'endpoint_force'),
                last_stamp_ns=node.last_output_stamp,last_packet_mono_age_seconds=None if node.last_output_stamp is None else
                    (node.get_clock().now().nanoseconds-node.last_output_stamp)*1e-9,
                old_state=packet['old_state'] if packet else 'warming',
                infer_ms_p99=float(np.percentile(node.infer_times,99)) if node.infer_times else None,
                cpu_threads=torch.get_num_threads(),
                old_url='http://127.0.0.1:8082/',new_url='http://127.0.0.1:8083/'))
        node.create_timer(1.,status);status();executor.spin()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        executor.shutdown()
        for web in webs:web.destroy_node()
        if node is not None:node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
