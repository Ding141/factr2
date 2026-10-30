"""Explicit, read-only deployment of the uncalibrated frozen-mean sigma trial."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from factr2_next.data_collection.quality import sha256
from factr2_next.inference.checkpoint import LoadedCheckpoint
from factr2_next.inference.inference_node import InferenceNode
from factr2_next.training.frozen_variance import VarianceHead
from factr2_next.training.models import build_model

SCHEMA = "w3_frozen_sigma_observer_v1"
ROOT = "/next/right/frozen_sigma"


class FrozenSigmaModel(nn.Module):
    def __init__(self, base, head, hidden_norm, fixed_z):
        super().__init__()
        self.base, self.head = base, head
        self.register_buffer("hidden_mean", torch.as_tensor(hidden_norm["mean"], dtype=torch.float32))
        self.register_buffer("hidden_std", torch.as_tensor(hidden_norm["std"], dtype=torch.float32))
        self.register_buffer("fixed_z", torch.as_tensor(fixed_z, dtype=torch.float32))
        self.requires_grad_(False)
        self.eval()

    def distribution(self, x):
        sequence, _ = self.base.lstm(x)
        hidden = self.base.head[:-1](sequence[:, -1])
        mean_z = self.base.head[-1](hidden)
        variance_z = self.head((hidden-self.hidden_mean)/self.hidden_std)
        return mean_z, variance_z

    def forward(self, x):
        return self.distribution(x)[0]


def load_observer(path, device="cpu", expected=None):
    """Keep the production W3 loader strict; this research schema is opt-in."""
    path, device = Path(path), torch.device(device)
    if device.type != "cpu":
        raise ValueError("frozen_sigma_observer_cpu_only")
    meta = json.loads((path/"metadata.json").read_text())
    required = dict(schema=SCHEMA, side="right", joint_order=[f"right_joint_{i}" for i in range(7)],
                    contract_version="w3_next_v1", history=50, sample_hz=100, input_size=21, output_size=7,
                    deployment_mode="read_only_research_observation", calibration_state="uncalibrated",
                    deployment_ready=False)
    required.update(expected or {})
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError("frozen_sigma_observer_profile:"+key)
    expected_files = {"mean_checkpoint/model.pt", "mean_checkpoint/normalization.npz",
                      "mean_checkpoint/config.yaml", "mean_checkpoint/metadata.json", "mean_checkpoint/metrics.json",
                      "variance_head/head.pt", "variance_head/normalization.npz", "variance_head/metadata.json",
                      "variance_head/history.json", "provenance.json"}
    if set(meta["artifact_sha256"]) != expected_files:
        raise ValueError("frozen_sigma_observer_file_manifest")
    for name, digest in meta["artifact_sha256"].items():
        if sha256(path/name) != digest:
            raise ValueError("frozen_sigma_observer_hash:"+name)
    mean_path = path/"mean_checkpoint"
    mean_meta = json.loads((mean_path/"metadata.json").read_text())
    cfg = yaml.safe_load((mean_path/"config.yaml").read_text())
    if (mean_meta.get("schema") != "next_u_ablation_v1" or mean_meta.get("arm") != "NEXT_MSE"
            or mean_meta.get("side") != meta["side"] or mean_meta.get("joint_order") != meta["joint_order"]
            or cfg["joint_names"] != meta["joint_order"] or cfg["side"] != meta["side"]):
        raise ValueError("frozen_sigma_mean_profile")
    for file, digest in mean_meta["artifact_sha256"].items():
        if sha256(mean_path/file) != digest:
            raise ValueError("frozen_sigma_mean_hash")
    mean = torch.load(mean_path/"model.pt", map_location="cpu", weights_only=True)
    if ((mean["input_size"], mean["output_size"], mean["history"]) != (21, 7, 50)
            or mean["model"] != cfg["model"] or mean["arm"] != "NEXT_MSE"
            or cfg["model"].get("type") != "lstm"
            or cfg["model"].get("state_mode") != "stateless" or cfg["model"].get("bidirectional")):
        raise ValueError("frozen_sigma_mean_dimensions_or_model")
    with torch.random.fork_rng(devices=[]):
        base = build_model(cfg["model"], 21, 7, 50)
    state = mean["model_state_dict"]
    if any(not k.startswith(("base.", "variance_head.")) for k in state):
        raise ValueError("frozen_sigma_mean_state_keys")
    base.load_state_dict({k[len("base."):]:v for k,v in state.items() if k.startswith("base.")}, strict=True)
    with np.load(mean_path/"normalization.npz", allow_pickle=False) as h:
        norm = {k:h[k].astype(np.float32) for k in h.files}
    for key, size in (("x_mean", 21), ("x_std", 21), ("y_mean", 7), ("y_std", 7)):
        if norm[key].shape != (size,) or not np.isfinite(norm[key]).all():
            raise ValueError("frozen_sigma_mean_normalization")
        if key.endswith("std") and np.any(norm[key] <= 0):
            raise ValueError("frozen_sigma_mean_std")
    head_path = path/"variance_head"
    head_meta = json.loads((head_path/"metadata.json").read_text())
    if (head_meta.get("schema") != "next_frozen_variance_v1"
            or head_meta["mean_model_sha256"] != sha256(mean_path/"model.pt")):
        raise ValueError("frozen_sigma_head_mean_binding")
    for file, digest in head_meta["artifact_sha256"].items():
        if sha256(head_path/file) != digest:
            raise ValueError("frozen_sigma_head_hash")
    head_data = torch.load(head_path/"head.pt", map_location="cpu", weights_only=True)
    if (head_data["schema"] != head_meta["schema"]
            or head_data["mean_model_sha256"] != head_meta["mean_model_sha256"]
            or head_data["width"] != base.head[-1].in_features):
        raise ValueError("frozen_sigma_head_contract")
    with np.load(head_path/"normalization.npz", allow_pickle=False) as h:
        hidden_norm = {k:h[k].astype(np.float32) for k in ("mean", "std")}
        fixed_z = h["fixed_variance_z"].astype(np.float32)
    if (any(v.shape != (head_data["width"],) or not np.isfinite(v).all() for v in hidden_norm.values())
            or np.any(hidden_norm["std"] <= 0)):
        raise ValueError("frozen_sigma_hidden_normalization")
    with torch.random.fork_rng(devices=[]):
        head = VarianceHead(head_data["width"], fixed_z, head_data["floor"], head_data["ceiling"])
    head.load_state_dict(head_data["state"], strict=True)
    model = FrozenSigmaModel(base, head, hidden_norm, fixed_z)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("frozen_sigma_observer_weights_finite")
    from factr2_next.inference.operating_envelope import OperatingEnvelope
    envelope = OperatingEnvelope(meta["operating_envelope"], 7)
    if envelope.bounds is None:
        raise ValueError("frozen_sigma_envelope_required")
    return LoadedCheckpoint(model, norm, cfg, 50, 21, 7, device, meta)


class FrozenSigmaInferenceNode(InferenceNode):
    def __init__(self, **kwargs):
        self.latest_free = self.latest_sigma = None
        super().__init__(**kwargs)
        self.recording_metadata_sha256=sha256(Path(self.cfg['checkpoint_dir'])/'metadata.json')
        from std_msgs.msg import String
        from sensor_msgs.msg import JointState
        self.distribution_pub = self.create_publisher(String, ROOT+"/distribution", 10)
        self.sigma_pub = self.create_publisher(JointState, ROOT+"/free_joint_torque_sigma", 10)
        self.fixed_sigma_pub = self.create_publisher(JointState, ROOT+"/free_joint_torque_sigma_fixed", 10)
        self.fixed_sigma = np.sqrt(self.loaded.model.fixed_z.numpy())*self.loaded.normalization["y_std"]
        self.get_logger().warning("Read-only research observation: sigma is uncalibrated; no sigma-based gating or zeroing")

    def _load_checkpoint(self, path, **kwargs):
        if self.cfg.get("observation_mode") != "uncalibrated_research" or self.cfg["next_topic_root"] != ROOT:
            raise ValueError("frozen_sigma_explicit_observation_config_required")
        for name, topic in self.cfg["outputs"].items():
            if not self._format_topic(topic).startswith(ROOT+"/"):
                raise ValueError("frozen_sigma_output_namespace:"+name)
        return load_observer(path, **kwargs)

    def _predict_free_torque(self):
        norm = self.loaded.normalization
        x = (self.buffer.array()-norm["x_mean"])/norm["x_std"]
        with torch.inference_mode():
            mean_z, variance_z = self.loaded.model.distribution(torch.from_numpy(x).unsqueeze(0))
        self.latest_free = (mean_z.numpy()[0]*norm["y_std"]+norm["y_mean"]).astype(np.float32)
        self.latest_sigma = (np.sqrt(variance_z.numpy()[0])*norm["y_std"]).astype(np.float32)
        if not np.isfinite(self.latest_sigma).all() or np.any(self.latest_sigma <= 0):
            return np.full(7, np.nan, np.float32)
        return self.latest_free

    def _invalidate(self, reason, state="invalid"):
        self.latest_free = self.latest_sigma = None
        super()._invalidate(reason, state)

    def _callback(self, *msgs):
        before = self.counts["outputs"]
        super()._callback(*msgs)
        if self.counts["outputs"] == before:
            return
        from std_msgs.msg import String
        raw = self._extract(msgs[-1], "measured_joint_torque")-self.latest_free
        packet = dict(schema=SCHEMA, stamp_ns=self.last_output_stamp, joint_names=self.names,
            state=self.state, reason=self.reason, calibration="uncalibrated", deployment_ready=False,
            units="Nm", raw_external_nm=raw.tolist(), free_mean_nm=self.latest_free.tolist(),
            sigma_nm=self.latest_sigma.tolist(), fixed_sigma_nm=self.fixed_sigma.tolist(),
            nominal_free_residual_lower_nm=(-1.959963984540054*self.latest_sigma).tolist(),
            nominal_free_residual_upper_nm=(1.959963984540054*self.latest_sigma).tolist(),
            band_definition="zero-centered free-motion residual reference; not a confidence interval for true external torque")
        # These are the exact paired inputs and published outputs, with no
        # learned residual correction. Additional telemetry never enters NEXT.
        packet.update(base_metadata_sha256=self.recording_metadata_sha256,
            compensation_applied=False,correction_nm=[0.]*7,
            final_external_nm=self.last_external_output.tolist(),
            upstream_external_nm=raw.tolist(),
            recording_frame=dict(q_deg=np.rad2deg(self._extract(msgs[0],'joint_pos')).tolist(),
                velocity_deg_s=np.rad2deg(self._extract(msgs[1],'joint_vel')).tolist(),
                command_deg=np.rad2deg(self._extract(msgs[2],'joint_cmd')).tolist(),
                measured_tau_nm=self._extract(msgs[-1],'measured_joint_torque').tolist(),
                base_free_pred_nm=self.latest_free.tolist(),base_external_raw_nm=raw.tolist(),
                external_filtered_nm=self.last_external_output.tolist(),correction_nm=[0.]*7))
        self.distribution_pub.publish(String(data=json.dumps(packet, allow_nan=False)))
        self.sigma_pub.publish(self._joint_state(self.latest_sigma, msgs[-1].header.stamp))
        self.fixed_sigma_pub.publish(self._joint_state(self.fixed_sigma, msgs[-1].header.stamp))


def main():
    import rclpy
    rclpy.init()
    node = None
    try:
        node = FrozenSigmaInferenceNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
