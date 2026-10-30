"""Joint torque dashboard with an atomic, explicitly uncalibrated sigma panel."""
import json
import time

import numpy as np
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

from factr2_next.inference.frozen_sigma_observer import ROOT, SCHEMA
from factr2_next.visualization.web_node import WebNode, page

PANEL = '''<section class="group" style="margin-top:16px;overflow:auto">
<h2>自由运动残差的不确定度参考 · 未校准</h2>
<p class="description">σ 是自由运动反馈力矩残差的预测尺度。±1.96σ 是以零为中心的名义参考带，
不是外力真值的 95% 置信区间；不会据此调零或改变力矩。</p>
<div class="status" id="uncertainty-status">等待同一帧的力矩与 σ…</div>
<table style="width:100%;text-align:right;border-spacing:12px 8px"><thead><tr>
<th style="text-align:left">关节</th><th>配对原始外力矩 N·m</th><th>学习 σ N·m</th>
<th>固定 σ 对照 N·m</th><th>名义参考带 ±N·m</th></tr></thead><tbody id="sigma-values"></tbody></table></section>'''
SCRIPT = '''
function updateUncertainty(snapshot){
 const u=snapshot.uncertainty||{},p=u.packet||{},ok=u.available===true&&snapshot.state==="valid";
 document.getElementById('uncertainty-status').textContent=ok?
  '研究观察 · 未校准 · 同帧参考数据 · 不用于接触判定':
  '参考数据不可用 · '+(u.reason||'等待新鲜数据');
 const show=(xs,i)=>ok&&xs&&Number.isFinite(xs[i])?xs[i].toFixed(4):'—';
 const showTorque=(xs,i)=>ok&&xs&&Number.isFinite(xs[i])?Number(xs[i].toFixed(2)).toFixed(2):'—';
 document.getElementById('sigma-values').innerHTML=(snapshot.joint_names||[]).map((name,i)=>
  `<tr><td style="text-align:left">J${i+1} · ${name}</td>`+
  `<td>${showTorque(p.raw_external_nm,i)}</td><td>${show(p.sigma_nm,i)}</td>`+
  `<td>${show(p.fixed_sigma_nm,i)}</td><td>${show(p.nominal_free_residual_upper_nm,i)}</td></tr>`).join('');
}
'''


def valid_packet(packet, names, now_ns):
    if (packet.get("schema") != SCHEMA or packet.get("joint_names") != names
            or type(packet.get("stamp_ns")) is not int or packet["stamp_ns"] <= 0
            or not 0 <= (now_ns-packet["stamp_ns"])*1e-9 <= .25
            or packet.get("calibration") != "uncalibrated"
            or packet.get("state") not in ("valid", "out_of_scope")):
        raise ValueError("uncertainty_packet_profile_or_time")
    for key in ("raw_external_nm", "free_mean_nm", "sigma_nm", "fixed_sigma_nm",
                "nominal_free_residual_lower_nm", "nominal_free_residual_upper_nm"):
        value = np.asarray(packet[key], dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("uncertainty_packet_values")
    if np.any(np.asarray(packet["sigma_nm"]) <= 0) or np.any(np.asarray(packet["fixed_sigma_nm"]) <= 0):
        raise ValueError("uncertainty_packet_sigma")
    np.testing.assert_allclose(packet["nominal_free_residual_upper_nm"],
                               1.959963984540054*np.asarray(packet["sigma_nm"]), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(packet["nominal_free_residual_lower_nm"],
                               -np.asarray(packet["nominal_free_residual_upper_nm"]), rtol=1e-6, atol=1e-8)
    return packet


class FrozenSigmaWeb(WebNode):
    def __init__(self, config, **kwargs):
        if config["next_topic_root"] != ROOT:
            raise ValueError("frozen_sigma_web_namespace")
        self.packet = None
        self.packet_received = None
        super().__init__(config=config, **kwargs)
        self.html = page(self.names, view="joint_torque", model_label=self.cfg["model_label"],
                         model_notice=self.cfg["model_notice"], extra_panel=PANEL, extra_script=SCRIPT)

    def _subscribe(self):
        super()._subscribe()
        self.subs.append(self.create_subscription(String, ROOT+"/distribution", self._distribution,
                                                  qos_profile_sensor_data))

    def _distribution(self, msg):
        try:
            packet = valid_packet(json.loads(msg.data), self.names, self.get_clock().now().nanoseconds)
        except (ValueError, KeyError, TypeError, AssertionError, OverflowError):
            self.packet = None
            self.packet_received = None
            return
        with self.cond:
            self.packet, self.packet_received = packet, time.monotonic()
            self.seq += 1
            self.cond.notify_all()

    def _snapshot(self):
        snapshot = json.loads(super()._snapshot())
        with self.cond:
            packet = self.packet
            age = time.monotonic()-self.packet_received if self.packet_received is not None else None
        source_age = ((self.get_clock().now().nanoseconds-packet["stamp_ns"])*1e-9
                      if packet is not None else None)
        available = (snapshot["state"] == "valid" and packet is not None and packet["state"] == "valid"
                     and age is not None and age <= .25 and source_age is not None and 0 <= source_age <= .25)
        snapshot["uncertainty"] = dict(available=available, calibration="uncalibrated",
            reason=snapshot["reason"] if snapshot["state"] != "valid" else
                   ("ok_uncalibrated" if available else "distribution_missing_or_stale"),
            receive_age_seconds=age, source_age_seconds=source_age, packet=packet if available else None,
            band_definition="zero-centered free-motion reference; not a true-external-torque confidence interval")
        return json.dumps(snapshot, allow_nan=False).encode()


def main():
    import rclpy
    import yaml
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, ros_args = parser.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rclpy.init(args=ros_args)
    node = None
    try:
        node = FrozenSigmaWeb(cfg, node_name="right_frozen_sigma_dashboard")
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
