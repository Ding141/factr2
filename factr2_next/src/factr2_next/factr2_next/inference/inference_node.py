from factr2_next.w3_config import validate_w3_config
from pathlib import Path
from collections import deque
import json
import time
from factr2_next.w3_samples import frame, stamp_ns
from rclpy.clock import Clock, ClockType
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32

from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.history_buffer import HistoryBuffer
from factr2_next.inference.operating_envelope import OperatingEnvelope


INPUT_KEYS = ("joint_pos", "joint_vel", "joint_cmd", "measured_joint_torque")


class TorqueFilter:
    def __init__(self, cfg):
        self.mode = str(cfg.get("mode", "none")).lower()
        if not bool(cfg.get("enabled", self.mode != "none")):
            self.mode = "none"
        self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
        self.cutoff_hz = float(cfg.get("cutoff_hz", 5.0))
        self.sample_hz = float(cfg.get("sample_hz", 50.0))
        self.y = None
        if not np.isfinite([self.ema_alpha,self.cutoff_hz,self.sample_hz]).all() or not 0 < self.ema_alpha <= 1 or self.cutoff_hz <= 0 or self.sample_hz <= 0:
            raise ValueError('smoothing_alpha_cutoff_rate')

        valid = {"none", "ema", "lowpass"}
        if self.mode not in valid:
            raise ValueError(
                f"Unsupported smoothing mode '{self.mode}'. Use: {sorted(valid)}"
            )

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "none" or self.y is None:
            self.y = x.copy()
            return x

        alpha = self._alpha()
        self.y = (1.0 - alpha) * self.y + alpha * x
        return self.y.astype(np.float32)

    def reset(self):
        self.y = None

    def _alpha(self):
        if self.mode == "ema":
            return self.ema_alpha
        dt = 1.0 / self.sample_hz
        rc = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        return float(dt / (rc + dt))


class InferenceNode(Node):
    def __init__(self, **kwargs):
        super().__init__("factr2_next_inference", **kwargs)
        default_config = (
            Path(get_package_share_directory("factr2_next")) / "config" / "inference.yaml"
        )
        config_file = self.declare_parameter("config_file", str(default_config)).value
        self.cfg = self._load_config(config_file)
        validate_w3_config(self.cfg, "inference")
        self.w3 = 'contract_version' in self.cfg
        threads = int(self.cfg.get('cpu_threads',1))
        if threads < 1:raise ValueError('cpu_threads_positive')
        torch.set_num_threads(threads)
        self.timing = self.cfg.get('timing',{})
        self.input_timeout = float(self.timing.get('input_timeout_seconds',.25))
        self.max_gap = float(self.timing.get('max_gap_seconds',.04))
        if not np.isfinite([self.input_timeout,self.max_gap]).all() or self.input_timeout != .25 or self.max_gap != .04:
            raise ValueError('W3 timing requires timeout=.25, gap=.04')
        self.require_adapter = bool(self.cfg.get('require_adapter_status',self.w3))
        self.state, self.reason = 'loading', 'checkpoint_loading'
        self.last_stamp = None
        self.epoch_start_ns = 0
        self.last_input_mono = time.monotonic()
        self.last_adapter_mono = None
        self.adapter_ok = False
        self.last_ros_ns = None
        self.clear_queues_pending = False
        self.counts = dict(outputs=0,dropped=0,resets=0)
        self.output_times = deque(maxlen=100)
        self.infer_times = deque(maxlen=512)
        self.last_output_stamp = None
        self.robot_topic_root = str(self.cfg.get("robot_topic_root", "/robot"))
        self.next_topic_root = str(self.cfg.get("next_topic_root", "/next"))

        self.loaded = load_checkpoint(
            Path(str(self.cfg["checkpoint_dir"])).expanduser(),
            device=self.cfg.get("device", "cpu"),
            expected=({'side':self.cfg['side'],'joint_order':self.cfg['joint_names'],
                       'sample_hz':self.cfg.get('sample_hz',50),'history':50} if self.w3 else None),
        )
        self.buffer = HistoryBuffer(self.loaded.history)
        self.operating_envelope = OperatingEnvelope(
            self.cfg.get('operating_envelope'), len(self.cfg.get('joint_names', [])))
        self.torque_filter = TorqueFilter(self.cfg.get("smoothing", {}))
        self.contact_magnitude_cfg = self._contact_magnitude_cfg()
        self.contact_cfg = self.cfg.get("contact", {})
        self.contact_state = False
        self.topic_cfg = self.cfg["topics"]
        self._validate_postprocessing()
        self.names = self.cfg.get('joint_names',[])
        self.status_pub = self.create_publisher(DiagnosticArray,self.next_topic_root+'/status',qos_profile_sensor_data)
        self._publish_status()
        self.state,self.reason = 'warming','waiting_50_fresh_frames'
        if self.w3:
            self.adapter_sub = self.create_subscription(DiagnosticArray,self.robot_topic_root+'/adapter_status',self._adapter_status,qos_profile_sensor_data)

        self.subscribers = [
            Subscriber(
                self,
                JointState,
                self._format_topic(self.topic_cfg[key]["topic"]),
                qos_profile=qos_profile_sensor_data,
            )
            for key in INPUT_KEYS
        ]
        sync_cfg = self.cfg.get("sync", {})
        self.sync = ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=int(sync_cfg.get("queue_size", 5)),
            slop=float(sync_cfg.get("slop_seconds", 0.03)),
        )
        self.sync.registerCallback(self._callback)

        self.outputs = self._output_topics(self.cfg["outputs"])
        self.pubs = {
            "free": self._pub(JointState, "free_joint_torque_pred"),
            "external": self._pub(JointState, "external_joint_torque"),
            "external_raw": self._pub(JointState, "external_joint_torque_raw"),
            "mse": self._pub(Float32, "mse"),
            "normalized_contact_magnitude": self._pub(
                Float32,
                "normalized_contact_magnitude",
            ),
            "contact": self._pub(Bool, "contact_state"),
        }

        self.watchdog = self.create_timer(.02,self._watchdog,clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"Loaded NEXT checkpoint: robot={self.robot_topic_root}, next={self.next_topic_root}"
        )
        self.get_logger().info(
            f"External torque smoothing: {self.torque_filter.mode}; "
            f"filtered={self.outputs['external_joint_torque']}, "
            f"raw={self.outputs['external_joint_torque_raw']}"
        )
        self.get_logger().info(
            "Normalized contact magnitude: "
            f"{self.contact_magnitude_cfg.get('norm', 'l1')} norm of "
            f"{self.contact_magnitude_cfg.get('source', 'filtered_external_joint_torque')}"
        )
        self.get_logger().info(
            "Contact hysteresis: "
            f"low={self.contact_cfg.get('low_threshold', 1.0)}, "
            f"high={self.contact_cfg.get('high_threshold', 2.0)}"
        )

    def _validate_postprocessing(self):
        c=self.contact_magnitude_cfg
        if c.get('source','filtered_external_joint_torque') not in ('filtered_external_joint_torque','raw_external_joint_torque'):
            raise ValueError('contact_source')
        if c.get('norm','l1') not in ('l1','l2','linf','inf','max'):
            raise ValueError('contact_norm')
        scale=float(c.get('scale',1.));low=float(self.contact_cfg.get('low_threshold',1.));high=float(self.contact_cfg.get('high_threshold',2.))
        if not np.isfinite([scale,low,high]).all() or scale<=0 or not 0<=low<=high:
            raise ValueError('contact_scale_thresholds')
        if self.w3 and self.torque_filter.sample_hz != self.loaded.metadata['sample_hz']:
            raise ValueError('W3 smoothing rate must match checkpoint sample_hz')

    def _invalidate(self, reason, state='invalid'):
        if self.state != state or self.reason != reason or len(self.buffer.rows):
            self.counts['resets']+=1
        self.buffer.clear();self.torque_filter.reset();self.contact_state=False
        self.last_stamp=None;self.state=state;self.reason=reason
        self.epoch_start_ns=self.get_clock().now().nanoseconds
        self.clear_queues_pending=True
        if hasattr(self,'pubs'):self.pubs['contact'].publish(Bool(data=False))

    def _adapter_status(self,msg):
        statuses=[s for s in msg.status if s.name==f"factr2/adapter/{self.cfg['side']}"]
        age=(self.get_clock().now().nanoseconds-stamp_ns(msg))*1e-9
        self.adapter_ok=len(statuses)==1 and statuses[0].level==DiagnosticStatus.OK and 0<=age<=self.input_timeout
        self.last_adapter_mono=time.monotonic()
        if not self.adapter_ok and self.state not in ('invalid','stale'):
            self._invalidate('adapter:'+ (statuses[0].message if statuses else 'missing_side'))

    def _watchdog(self):
        now=time.monotonic();ros=self.get_clock().now().nanoseconds
        if self.w3 and self.last_ros_ns is not None and ros<self.last_ros_ns:
            self._invalidate('clock_rollback')
        self.last_ros_ns=ros
        if now-self.last_input_mono>self.input_timeout:
            if self.state!='stale':self._invalidate('input_timeout','stale')
        elif self.require_adapter and (self.last_adapter_mono is None or now-self.last_adapter_mono>self.input_timeout):
            if self.state!='invalid':self._invalidate('adapter_status_timeout')
        if self.clear_queues_pending:
            # ATS deletes selected entries after callback; clear only outside it.
            for queue in self.sync.queues:queue.clear()
            self.clear_queues_pending=False
        self._publish_status()

    def _publish_status(self):
        now=time.monotonic();ros=self.get_clock().now().nanoseconds
        while self.output_times and now-self.output_times[0]>1:self.output_times.popleft()
        values=dict(self.counts,state=self.state,reason=self.reason,history_count=len(self.buffer.rows),
            source_age_seconds=(ros-self.last_stamp)*1e-9 if self.last_stamp is not None else None,
            receive_age_seconds=now-self.last_input_mono,last_output_stamp_ns=self.last_output_stamp,
            output_hz=len(self.output_times),infer_ms=self.infer_times[-1] if self.infer_times else None,
            infer_p95_ms=float(np.percentile(self.infer_times,95)) if self.infer_times else None,
            device=str(self.loaded.device) if hasattr(self,'loaded') else self.cfg.get('device','cpu'))
        msg=DiagnosticArray();msg.header.stamp=self.get_clock().now().to_msg()
        status=DiagnosticStatus();status.name='factr2/next/'+str(self.cfg.get('side','generic'))
        status.level=DiagnosticStatus.OK if self.state=='valid' else DiagnosticStatus.WARN
        status.message=self.state+':'+self.reason
        status.values=[KeyValue(key=k,value=json.dumps(v)) for k,v in values.items()]
        msg.status=[status];self.status_pub.publish(msg)

    def _callback(self, *msgs):
        if self.w3:
            try:
                ros=self.get_clock().now().nanoseconds
                stamp,samples=frame(msgs,INPUT_KEYS,self.topic_cfg,self.names,ros,self.input_timeout)
                if stamp<self.epoch_start_ns:raise ValueError('pre_reset_queued_frame')
                if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('stamp_not_advancing')
                if self.require_adapter and (not self.adapter_ok or self.last_adapter_mono is None or time.monotonic()-self.last_adapter_mono>self.input_timeout):
                    raise ValueError('adapter_not_ok')
                if self.last_stamp is not None and stamp-self.last_stamp>self.max_gap*1e9:
                    self._invalidate('gap_over_40ms')
                self.last_stamp=stamp;self.last_input_mono=time.monotonic()
                self.state,self.reason='warming','waiting_50_fresh_frames'
                joint_pos,joint_vel,joint_cmd,measured=[samples[k] for k in INPUT_KEYS]
            except ValueError as exc:
                self.counts['dropped']+=1
                self._invalidate(str(exc))
                return
        else:
            joint_pos,joint_vel,joint_cmd,measured=[self._extract(m,k) for m,k in zip(msgs,INPUT_KEYS)]
            self.last_stamp=stamp_ns(msgs[-1]);self.last_input_mono=time.monotonic()

        self.buffer.append(joint_pos, joint_vel, joint_cmd)
        # Paper Sec. 4, Eq. (3): f_theta consumes a full history window.
        if not self.buffer.ready:
            return

        started=time.monotonic()
        tau_free = self._predict_free_torque()
        self.infer_times.append((time.monotonic()-started)*1000)
        if not np.isfinite(tau_free).all():
            self._invalidate('prediction_nonfinite');return
        if self.w3 and (self.get_clock().now().nanoseconds-self.last_stamp)*1e-9>self.input_timeout:
            self.counts['dropped']+=1;self._invalidate('expired_after_inference','stale');return
        # Paper Sec. 4, Eq. (2): tau_ext_hat = tau_m - tau_free_hat.
        tau_ext_raw = measured - tau_free
        # Smoothing, contact magnitude, and hysteresis are runtime/demo
        # post-processing, not part of the learned NEXT free-space torque model.
        tau_ext = self.torque_filter.update(tau_ext_raw)
        mse = float(np.mean(tau_ext_raw.astype(np.float64) ** 2))
        normalized_contact_magnitude = self._normalized_contact_magnitude(
            tau_ext,
            tau_ext_raw,
        )
        if (not np.isfinite(tau_ext_raw).all() or not np.isfinite(tau_ext).all()
                or not np.isfinite([mse,normalized_contact_magnitude]).all()
                or max(abs(mse),abs(normalized_contact_magnitude))>np.finfo(np.float32).max):
            self.counts['dropped']+=1
            self._invalidate('output_nonfinite_or_float32_range');return
        contact = self._update_observation_state(normalized_contact_magnitude)
        self.counts["outputs"]+=1;self.output_times.append(time.monotonic())
        self.last_output_stamp=self.last_stamp
        stamp = msgs[-1].header.stamp
        self.pubs["free"].publish(self._joint_state(tau_free, stamp))
        self.pubs["external_raw"].publish(self._joint_state(tau_ext_raw, stamp))
        self.pubs["external"].publish(self._joint_state(tau_ext, stamp))
        self.pubs["mse"].publish(Float32(data=mse))
        self.pubs["normalized_contact_magnitude"].publish(
            Float32(data=normalized_contact_magnitude)
        )
        self.pubs["contact"].publish(Bool(data=contact))

    def _update_observation_state(self, magnitude):
        outside = self.operating_envelope.outside_joints(self.buffer.array())
        if outside:
            self.state = 'out_of_scope'
            self.reason = 'outside_collection_range:' + ','.join(self.names[i] for i in outside)
            self.contact_state = False
            return False
        self.state, self.reason = 'valid', 'ok'
        return self._update_contact_state(magnitude)

    def _predict_free_torque(self):
        norm = self.loaded.normalization
        x = (self.buffer.array() - norm["x_mean"]) / norm["x_std"]
        x = torch.from_numpy(x).unsqueeze(0).to(self.loaded.device)
        with torch.inference_mode():
            y = self.loaded.model(x).cpu().numpy()[0]
        return (y * norm["y_std"] + norm["y_mean"]).astype(np.float32)

    def _normalized_contact_magnitude(self, tau_ext, tau_ext_raw):
        source = str(
            self.contact_magnitude_cfg.get(
                "source",
                "filtered_external_joint_torque",
            )
        ).lower()
        values = tau_ext_raw if source == "raw_external_joint_torque" else tau_ext
        values = np.asarray(values, dtype=np.float32)
        norm = str(self.contact_magnitude_cfg.get("norm", "l1")).lower()

        if norm == "l1":
            magnitude = float(np.sum(np.abs(values)))
        elif norm == "l2":
            magnitude = float(np.linalg.norm(values))
        elif norm in ("linf", "inf", "max"):
            magnitude = float(np.max(np.abs(values))) if values.size else 0.0
        else:
            raise ValueError(
                f"Unsupported contact magnitude norm '{norm}'. Use: l1, l2, linf."
            )

        scale = float(self.contact_magnitude_cfg.get("scale", 1.0))
        return magnitude / scale

    def _update_contact_state(self, normalized_contact_magnitude):
        if not bool(self.contact_cfg.get("enabled", True)):
            self.contact_state = False
            return False

        low = float(self.contact_cfg.get("low_threshold", 1.0))
        high = float(self.contact_cfg.get("high_threshold", 2.0))

        if normalized_contact_magnitude >= high:
            self.contact_state = True
        elif normalized_contact_magnitude <= low:
            self.contact_state = False
        return self.contact_state

    def _extract(self, msg, key):
        field = self.topic_cfg[key].get("field", "position")
        if field == "position":
            return np.asarray(msg.position, dtype=np.float32)
        if field == "velocity":
            return np.asarray(msg.velocity, dtype=np.float32)
        if field == "effort":
            return np.asarray(msg.effort, dtype=np.float32)
        raise ValueError(f"Unsupported JointState field: {field}")

    def _joint_state(self, values, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(self.names)
        msg.position = [float(x) for x in values]
        return msg

    def _format_topic(self, topic):
        return str(topic).format(
            robot_topic_root=self.robot_topic_root,
            next_topic_root=self.next_topic_root,
        )

    def _pub(self, msg_type, key):
        return self.create_publisher(msg_type, self.outputs[key], 10)

    def _output_topics(self, outputs):
        outputs = dict(outputs)
        outputs.setdefault(
            "external_joint_torque_raw",
            outputs["external_joint_torque"] + "/raw",
        )
        if "normalized_contact_magnitude" not in outputs and "score" in outputs:
            outputs["normalized_contact_magnitude"] = outputs["score"]
        outputs.setdefault(
            "normalized_contact_magnitude",
            "{next_topic_root}/normalized_contact_magnitude",
        )
        outputs.setdefault("contact_state", "{next_topic_root}/contact_state")
        return {key: self._format_topic(topic) for key, topic in outputs.items()}

    def _contact_magnitude_cfg(self):
        return self.cfg.get(
            "normalized_contact_magnitude",
            self.cfg.get("score", {}),
        )

    def _load_config(self, path):
        with open(Path(path).expanduser(), "r") as f:
            return yaml.safe_load(f) or {}


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
