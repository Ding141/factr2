"""Read-only downstream observer. No actuation or controller interfaces."""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String

from .kinematic_compensation import Compensator, ResidualModel, ROOT, SCHEMA, digest

BASE_ROOT = '/next/right/frozen_sigma'
NAMES = [f'right_joint_{i}' for i in range(7)]


def stamp(msg):
    return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec


class KinematicObserver(Node):
    def __init__(self, model_path, base_bundle, **kwargs):
        super().__init__('right_kinematic_residual_observer', **kwargs)
        self.base_bundle = Path(base_bundle)
        self.base_hash = digest(self.base_bundle/'metadata.json')
        self.model = ResidualModel.load(model_path, self.base_hash)
        self.model_hash = digest(model_path)
        self.pipeline = Compensator(self.model)
        self.envelope = np.deg2rad(np.array(json.loads((self.base_bundle/'metadata.json').read_text())['operating_envelope']['position_bounds_deg']))
        self.history = deque(maxlen=50)
        self.state, self.reason = 'warming', 'waiting_fresh_inputs'
        self.last_input = self.last_base = self.last_adapter = 0.
        self.base_state = self.adapter_ok = None
        self.identity_ok = True
        self.last_ros = None
        self.latest = None
        self.counts = dict(outputs=0, dropped=0, resets=0)
        self.times = deque(maxlen=100)
        self.clear_pending = False
        self.pubs = {key:self.create_publisher(JointState, ROOT+'/'+suffix, 10) for key,suffix in dict(
            raw='external_joint_torque/raw', final='external_joint_torque', free_total='free_joint_torque_pred',
            baseline_raw='upstream_external_joint_torque/raw', correction='correction',
            q_filtered='input/joint_pos_filtered', v_estimated='input/joint_vel_estimated',
            a_estimated='input/joint_acc_estimated', command_filtered='input/joint_cmd_filtered',
            motor_velocity_filtered='input/motor_vel_filtered', measured_filtered='input/joint_effort_filtered').items()}
        self.status_pub = self.create_publisher(DiagnosticArray, ROOT+'/status', qos_profile_sensor_data)
        self.packet_pub = self.create_publisher(String, ROOT+'/distribution', 10)
        self.mse_pub = self.create_publisher(Float32, ROOT+'/mse', 10)
        self.norm_pub = self.create_publisher(Float32, ROOT+'/normalized_contact_magnitude', 10)
        self.create_subscription(DiagnosticArray, BASE_ROOT+'/status', self.base_status, qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, '/factr2/right/adapter_status', self.adapter_status, qos_profile_sensor_data)
        reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        topics = ['/factr2/right/joint_pos', '/factr2/right/joint_vel', '/factr2/right/joint_cmd',
                  '/factr2/right/joint_effort', BASE_ROOT+'/free_joint_torque_pred']
        # The adapter and frozen-mean publishers are all RELIABLE. A lost mean
        # frame cannot be reconstructed by the exact-time synchronizer; use the
        # same bounded reliable history for all five paired inputs.
        self.subscribers = [Subscriber(self, JointState, t, qos_profile=reliable)
                            for i,t in enumerate(topics)]
        self.sync = TimeSynchronizer(self.subscribers, 100)
        self.sync.registerCallback(self.joined)
        self.create_timer(.02, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.last_identity_check = 0.

    def valid_age(self, msg):
        return 0 <= (self.get_clock().now().nanoseconds-stamp(msg))/1e9 <= .25

    def base_status(self, msg):
        statuses = [s for s in msg.status if s.name=='factr2/next/right']
        if len(statuses)!=1 or not self.valid_age(msg):
            self.base_state = None
            return
        self.base_state = {v.key:json.loads(v.value) for v in statuses[0].values}.get('state')
        self.last_base = time.monotonic()

    def adapter_status(self, msg):
        s = [s for s in msg.status if s.name=='factr2/adapter/right']
        self.adapter_ok = len(s)==1 and s[0].level==DiagnosticStatus.OK and self.valid_age(msg)
        self.last_adapter = time.monotonic()

    def sources_ok(self):
        now = time.monotonic()
        return (self.identity_ok and self.base_state in ('valid','out_of_scope') and self.adapter_ok
                and 0<=now-self.last_base<=.25 and 0<=now-self.last_adapter<=.25)

    def invalidate(self, reason, state='invalid'):
        if self.reason!=reason or self.state!=state or self.history:
            self.counts['resets'] += 1
        self.pipeline.reset(); self.history.clear(); self.latest=None
        self.state, self.reason = state, reason
        self.clear_pending = True

    def joined(self, pos, vel, cmd, tau, mean):
        now = time.monotonic()
        try:
            msgs = (pos, vel, cmd, tau, mean)
            if not self.sources_ok():
                raise ValueError('upstream_or_adapter_unavailable')
            if len({stamp(m) for m in msgs})!=1 or not self.valid_age(pos):
                raise ValueError('unpaired_or_expired_frame')
            arrays = []
            for m,field in zip(msgs, ('position','velocity','position','effort','position')):
                a = np.asarray(getattr(m,field), float)
                if m.name!=NAMES or a.shape!=(7,) or not np.isfinite(a).all():
                    raise ValueError('malformed_input')
                arrays.append(a)
            self.history.append(arrays[0])
            q = np.array(self.history)
            outside = np.any((q<self.envelope[:,0]) | (q>self.envelope[:,1]))
            f = self.pipeline.update(stamp(pos), *arrays,
                allow_compensation=not outside and self.base_state=='valid')
            if f['reset']:
                self.history.clear(); self.counts['resets'] += 1
            if not f['ready'] or len(self.history)<50:
                self.state, self.reason = 'warming', 'causal_filter_warmup'
            elif outside or self.base_state=='out_of_scope':
                self.state, self.reason = 'out_of_scope', 'outside_base_collection_range'
            else:
                self.state = 'valid'
                self.reason = 'ok' if f['supported'] else 'base_valid_compensation_bypassed_out_of_support'
            values = dict(raw=f['raw'], final=f['final'], free_total=f['free_total'],
                          baseline_raw=f['baseline_raw'], correction=f['correction'],
                          q_filtered=f['actual'][0], v_estimated=f['actual'][1], a_estimated=f['actual'][2],
                          command_filtered=f['command'][0], motor_velocity_filtered=f['motor_velocity'],
                          measured_filtered=f['measured'])
            scalars = [float(np.mean(f['raw']**2)), float(np.sum(abs(f['final'])))]
            if (any(not np.isfinite(a).all() or np.max(abs(a))>np.finfo(np.float32).max for a in values.values())
                    or not np.isfinite(scalars).all() or max(scalars)>np.finfo(np.float32).max):
                raise ValueError('output_nonfinite_or_float32_range')
            self.latest = dict(schema=SCHEMA,stamp_ns=stamp(pos),state=self.state,reason=self.reason,
                joint_names=NAMES,units='Nm',base_metadata_sha256=self.base_hash,
                compensation_model_sha256=self.model_hash,feature_mode=self.model.mode,
                compensation_applied=bool(f['supported'] and self.state=='valid'),
                enabled_joints=self.model.enabled.tolist(),calibration='uncalibrated',
                contact_validated=False,signal=self.model.artifact['signal'],
                raw_external_nm=f['raw'].tolist(),final_external_nm=f['final'].tolist(),
                upstream_external_nm=f['baseline_raw'].tolist(),correction_nm=f['correction'].tolist())
            # Record one coherent frame instead of rejoining thirteen ROS
            # messages. These are the same unrounded arrays used above; this
            # additional payload never enters prediction or motor control.
            self.latest['recording_frame'] = dict(
                q_deg=np.rad2deg(arrays[0]).tolist(),
                velocity_deg_s=np.rad2deg(arrays[1]).tolist(),
                command_deg=np.rad2deg(arrays[2]).tolist(),
                measured_tau_nm=arrays[3].tolist(),base_free_pred_nm=arrays[4].tolist(),
                measured_filtered_nm=f['measured'].tolist(),
                corrected_free_pred_nm=f['free_total'].tolist(),
                compensated_raw_nm=f['raw'].tolist(),external_filtered_nm=f['final'].tolist(),
                correction_nm=f['correction'].tolist(),base_external_raw_nm=f['baseline_raw'].tolist(),
                estimated_velocity_deg_s=np.rad2deg(f['actual'][1]).tolist(),
                estimated_acceleration_deg_s2=np.rad2deg(f['actual'][2]).tolist())
            self.packet_pub.publish(String(data=json.dumps(self.latest,allow_nan=False)))
            for key,a in values.items():
                m = JointState(); m.header.stamp=pos.header.stamp; m.name=NAMES; m.position=a.tolist()
                self.pubs[key].publish(m)
            self.mse_pub.publish(Float32(data=scalars[0]))
            self.norm_pub.publish(Float32(data=scalars[1]))
            self.last_input=now; self.counts['outputs']+=1; self.times.append(now)
        except (ValueError, KeyError, FloatingPointError) as exc:
            self.counts['dropped'] += 1
            self.invalidate(str(exc))

    def tick(self):
        now = time.monotonic(); ros = self.get_clock().now().nanoseconds
        if self.last_ros is not None and ros<self.last_ros:
            self.invalidate('clock_rollback')
        self.last_ros=ros
        if now-self.last_identity_check>=1:
            self.last_identity_check=now
            try:identity_ok=digest(self.base_bundle/'metadata.json')==self.base_hash
            except OSError:identity_ok=False
            if not identity_ok:
                self.invalidate('source_model_identity_changed')
                # A different bundle requires a fresh, explicitly bound process.
                self.base_state=None; self.adapter_ok=False
                self.identity_ok=False
        if now-self.last_input>.25:
            self.invalidate('input_timeout','stale')
        elif not self.sources_ok():
            self.invalidate('upstream_status_timeout')
        if self.clear_pending:
            for queue in self.sync.queues:queue.clear()
            self.clear_pending=False
        while self.times and now-self.times[0]>1:self.times.popleft()
        values=dict(self.counts,state=self.state,reason=self.reason,output_hz=len(self.times),
            calibration='uncalibrated',contact_validated=False,
            compensation_applied=bool(self.latest and self.latest['compensation_applied']),
            enabled_joints=self.model.enabled.tolist(),base_metadata_sha256=self.base_hash,
            input_cutoff_hz=self.model.artifact['signal']['input_torque_hz'],
            output_cutoff_hz=self.model.artifact['signal']['output_hz'])
        s=DiagnosticStatus();s.name='factr2/next/right'
        s.level=DiagnosticStatus.OK if self.state=='valid' else DiagnosticStatus.WARN
        s.message=self.state+':'+self.reason;s.values=[KeyValue(key=k,value=json.dumps(v)) for k,v in values.items()]
        msg=DiagnosticArray();msg.header.stamp=self.get_clock().now().to_msg();msg.status=[s];self.status_pub.publish(msg)
