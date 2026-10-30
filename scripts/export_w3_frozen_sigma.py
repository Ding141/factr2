#!/usr/bin/env python3
"""Export the completed sigma pilot as an explicit read-only observer bundle."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.frozen_sigma_observer import SCHEMA, ROOT, load_observer
from run_w3_frozen_sigma import verify, load_head, core
from factr2_next.training.frozen_variance import head_variance


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    lock, cfg, manifest, original, norm = verify(source)
    state = json.loads((source/"state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("frozen_sigma_export_requires_complete")
    head, hidden_norm, fixed_z = load_head(source)
    _, reports = core.validate_manifest(Path(lock["parent"])/"split.json")
    source_meta = reports["train"][0]["metadata"]
    ds = core.dataset(manifest, cfg, "train")
    positions = np.concatenate([x[:, :7] for x, _ in ds.episodes]).astype(np.float64)
    envelope = dict(position_bounds_deg=np.rad2deg(np.stack((positions.min(0),positions.max(0)),axis=1)).tolist(),
        position_margin_deg=0., source="observed training positions only; rectangular bounds do not certify pose combinations")
    output.mkdir(parents=True)
    shutil.copytree(source/"mean_checkpoint", output/"mean_checkpoint")
    shutil.copytree(source/"variance_head", output/"variance_head")
    report = json.loads((source/"evaluation/comparison.json").read_text())
    provenance = dict(source_experiment=str(source), experiment_lock_sha256=sha256(source/"lock.json"),
        source_evaluation_sha256=sha256(source/"evaluation/comparison.json"),
        source_mean_model_sha256=sha256(source/"mean_checkpoint/model.pt"),
        source_sigma_head_sha256=sha256(source/"variance_head/head.pt"),
        selected_mean_epoch=report["mean_selected_epoch"], selected_sigma_epoch=report["best_epoch"],
        test_results={arm:report["splits"]["test"]["arms"][arm]["overall"] for arm in report["splits"]["test"]["arms"]},
        purpose="human-requested observation deployment; no controller or automatic zeroing",
        calibration="uncalibrated", deployment_ready=False)
    write_json(output/"provenance.json", provenance)
    metadata = {k:copy.deepcopy(source_meta[k]) for k in
        ("source", "tool", "gripper", "load", "calibration", "control", "control_profile_sha256", "control_profile_scope")
        if k in source_meta}
    files = [str(p.relative_to(output)) for d in (output/"mean_checkpoint",output/"variance_head")
             for p in d.iterdir() if p.is_file()]+["provenance.json"]
    metadata.update(schema=SCHEMA, side="right", joint_order=cfg["joint_names"], contract_version="w3_next_v1",
        sample_hz=100, history=50, input_size=21, output_size=7, operating_envelope=envelope,
        feature_order=["q", "qdot", "q_cmd-q"], deployment_mode="read_only_research_observation",
        calibration_state="uncalibrated", deployment_ready=False, automatic_deployment=False,
        external_sign="measured minus predicted free mean", sigma_units="Nm",
        artifact_sha256={f:sha256(output/f) for f in files})
    write_json(output/"metadata.json", metadata)
    loaded = load_observer(output)
    # Compare online single-window inference against the original mean and
    # offline sigma head, rather than only testing a checkpoint roundtrip.
    ds.norm = norm
    items = np.stack([ds[i][0].numpy() for i in range(min(16,len(ds)))])
    with torch.inference_mode():
        old_mean = original(torch.from_numpy(items))[0]
        means, variances = zip(*(loaded.model.distribution(torch.from_numpy(x[None])) for x in items))
        np.testing.assert_allclose(torch.cat(means).numpy(), old_mean.numpy(), atol=2e-6, rtol=2e-6)
        sequence, _ = original.base.lstm(torch.from_numpy(items))
        hidden = original.base.head[:-1](sequence[:,-1]).numpy()
        expected = head_variance(head, dict(hidden=hidden), hidden_norm)
        np.testing.assert_allclose(torch.cat(variances).numpy(), expected, atol=2e-6, rtol=2e-6)
    runtime = yaml.safe_load((Path(__file__).resolve().parents[1]/"config/w3/right/inference_targeted_20261016.yaml").read_text())
    runtime.update(checkpoint_dir=str(output), next_topic_root=ROOT, operating_envelope=envelope,
                   observation_mode="uncalibrated_research", cpu_threads=1)
    runtime["outputs"]["distribution"] = "{next_topic_root}/distribution"
    runtime["outputs"]["free_joint_torque_sigma"] = "{next_topic_root}/free_joint_torque_sigma"
    runtime["outputs"]["free_joint_torque_sigma_fixed"] = "{next_topic_root}/free_joint_torque_sigma_fixed"
    (output/"inference.yaml").write_text(yaml.safe_dump(runtime,sort_keys=False))
    web = dict(contract_version="w3_next_v1", side="right", joint_names=cfg["joint_names"],
        next_topic_root=ROOT, observation_mode="uncalibrated_research",
        robot_topic_root="/factr2/right", outputs=runtime["outputs"],
        web=dict(host="127.0.0.1",port=8082,refresh_hz=10), plot=dict(view="joint_torque",max_points=600),
        model_label="NEXT · 关节外力矩与 sigma（研究观察）",
        model_notice="学习 sigma 未经独立校准，名义 95% 参考带不是外力真值置信区间。超出采集范围或数据过期时不可用；sigma 不改变力矩或接触判定。")
    (output/"web.yaml").write_text(yaml.safe_dump(web,sort_keys=False))
    write_json(output/"export_verification.json",dict(online_single_window_mean_matches_offline=True,
        online_single_window_variance_matches_offline=True, deployed=False,
        metadata_sha256=sha256(output/"metadata.json")))
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(export(args.source,args.output))
