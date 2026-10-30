"""Controlled NEXT-U checks; tiny synthetic fixtures, no ROS or real training."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from factr2_next.data_collection.campaign import control_profile
from factr2_next.data_collection.h5_writer import H5Writer
from factr2_next.data_collection.quality import make_manifest, sha256, sidecar_path, write_json
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.training.models import build_model
from factr2_next.training.next_u import (
    ARMS, NextU, distribution_metrics, objective, physical_distribution, train_epoch,
)
from factr2_next.training.train import run_epoch
from factr2_next.w3_config import SIGNALS
import run_w3_next_u_ablation as runner
from control_w3_next_u_budget import stop_ready

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(type="lstm", state_mode="stateless", hidden_size=8, num_layers=2,
             bidirectional=False, head_hidden=8, head_layers=2, dropout=.1)


def test_mean_initialization_dropout_and_rng_match_original_next():
    torch.manual_seed(42)
    original = build_model(SMALL, 21, 7, 50)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    u = NextU(SMALL)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, weight in original.state_dict().items():
        torch.testing.assert_close(weight, u.base.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(5, 50, 21)
    torch.manual_seed(9)
    expected = original(x)
    torch.manual_seed(9)
    actual, v = u(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(v, torch.ones_like(v))


def test_mse_arm_optimizer_matches_original_training_step_exactly():
    torch.manual_seed(2)
    u = NextU(SMALL).configure_arm("NEXT_MSE")
    original = copy.deepcopy(u.base)
    ds = TensorDataset(torch.randn(11, 50, 21), torch.randn(11, 7))
    scale = np.arange(1, 8, dtype=np.float32) * .2
    opt_a = torch.optim.AdamW(original.parameters(), lr=.001, weight_decay=1e-6)
    opt_b = torch.optim.AdamW([p for p in u.parameters() if p.requires_grad], lr=.001, weight_decay=1e-6)
    def loader():
        return DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(20)
    a = run_epoch(original, loader(), "cpu", opt_a, gradient_clip=1., output_scale=scale)
    torch.manual_seed(20)
    b = train_epoch(u, loader(), opt_b, dict(y_std=scale), "NEXT_MSE", 1.)
    assert b == a
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, u.base.state_dict()[key], atol=0, rtol=0)
    assert u.variance_head.weight.grad is None


def test_nll_restores_squared_units_and_learns_mean_and_variance():
    mu = torch.zeros(2, 7, requires_grad=True)
    v = torch.full((2, 7), 2., requires_grad=True)
    scale = torch.arange(1., 8.)
    norm = dict(y_std=scale.numpy(), y_mean=np.arange(7, dtype=np.float32))
    mean_nm, var_nm2 = physical_distribution(mu, v, norm)
    torch.testing.assert_close(var_nm2, 2 * scale.square().expand(2, 7))
    torch.testing.assert_close(mean_nm, torch.arange(7.).expand(2, 7))
    target = torch.full_like(mu, .5)
    loss = objective("NEXT_U", mu, v, target, scale)
    expected = .5 * (np.log(2*np.pi) + (2*scale.square()).log() + .25/2)
    assert loss.item() == pytest.approx(expected.mean().item())
    loss.backward()
    assert torch.isfinite(mu.grad).all() and torch.count_nonzero(mu.grad) == 14
    assert torch.isfinite(v.grad).all() and torch.count_nonzero(v.grad) == 14


def test_variance_bounds_hold_for_extreme_logits_and_reject_invalid_recipe():
    model = NextU(SMALL)
    with torch.no_grad():
        model.variance_head.bias.fill_(-1e5)
        _, low = model(torch.zeros(1, 50, 21))
        model.variance_head.bias.fill_(1e5)
        _, high = model(torch.zeros(1, 50, 21))
    torch.testing.assert_close(low, torch.full_like(low, 1e-4))
    torch.testing.assert_close(high, torch.full_like(high, 1e4))
    with pytest.raises(ValueError, match="bounds"):
        NextU(SMALL, dict(floor=0, ceiling=1, initial=.1))


def test_interval_diagnostics_detect_miscalibration_and_keep_bias():
    rng = np.random.default_rng(5)
    error = rng.normal(size=(20000, 7)) * np.arange(1, 8)
    v = np.broadcast_to(np.arange(1, 8)**2, error.shape)
    good = distribution_metrics(error, v)
    bad = distribution_metrics(error, v*.01)
    np.testing.assert_allclose(good["nominal_gaussian_intervals"]["95"]["per_joint_coverage"], .95, atol=.005)
    assert max(bad["nominal_gaussian_intervals"]["95"]["per_joint_coverage"]) < .18
    biased = distribution_metrics(np.full((100, 7), .3), np.ones((100, 7)))
    np.testing.assert_allclose(biased["signed_bias_nm"], .3)
    np.testing.assert_allclose(biased["per_joint_std_nm"], 0, atol=1e-14)


def test_production_loader_explicitly_rejects_new_schema(tmp_path):
    cfg = yaml.safe_load((ROOT/"config/w3/right/train.yaml").read_text())
    cfg["data"]["sample_hz"] = 100
    cfg["next_u"] = dict(floor=1e-4, ceiling=1e4, initial=1.)
    cfg["comparison"] = dict(selection="validation MSE")
    model = NextU(cfg["model"])
    norm = dict(x_mean=np.zeros(21), x_std=np.ones(21), y_mean=np.zeros(7), y_std=np.ones(7))
    (tmp_path/"lock.json").write_text("{}")
    path = tmp_path/"artifact"
    runner.save_artifact(path, model, cfg, norm, 0, "NEXT_U", {}, tmp_path)
    # Architecture/config pass the real production contract. The schema itself
    # must reject this artifact, before weights or deployment data are used.
    with pytest.raises(ValueError, match="checkpoint_profile:schema"):
        load_checkpoint(path)


@pytest.fixture
def fixture_campaign(tmp_path):
    """Synthetic data using the real-session validation schema, only in tmpdir."""
    calibration = tmp_path/"calibration.yaml"
    calibration.write_text("id: synthetic-test-only\n")
    gains = tmp_path/"gains.yaml"
    gains.write_text("id: synthetic-gains\n")
    controls = dict(gains=dict(position_params=str(gains)),
                    feedforward=dict(gravity_params=str(gains), friction_model=str(gains)))
    paths = {}
    for i, role in enumerate(("train", "val", "test")):
        path = tmp_path/(role+".h5")
        meta = dict(contract_version="w3_next_v1", side="right",
            joint_order=[f"right_joint_{j}" for j in range(7)], sample_hz=100,
            session_id=role, source="real", health_gate_enabled=True,
            topics={k:dict(topic="/factr2/right/"+v[0], field=v[1]) for k, v in SIGNALS.items()},
            software_commits={}, config_sha256="test", tool="mock", gripper="excluded", load="none",
            calibration=dict(path=str(calibration), sha256=sha256(calibration)), control=controls,
            contact=dict(present=False), temperature=dict(source="unavailable"),
            time_source="synthetic fixture only", trajectory_id="synthetic unit test",
            audit_paths=[], episodes={})
        meta["control_profile_sha256"] = control_profile(meta)
        writer = H5Writer(path, role, SIGNALS, meta)
        writer.start_episode()
        for step in range(63):
            q = np.sin((step+i*5)*.08+np.arange(7))*.05
            writer.append(10**15+step*10_000_000, dict(joint_pos=q, joint_vel=q*.3,
                joint_cmd=q+.02, measured_joint_torque=q*4+np.arange(7)*.1))
        writer.close()
        meta = json.loads(sidecar_path(path).read_text())
        meta["episodes"]["ep_0000"].update(kind="static_test", speed="synthetic")
        write_json(sidecar_path(path), meta)
        paths[role] = [path]
    manifest = tmp_path/"split.json"
    make_manifest(manifest, paths, "synthetic_next_u")
    value = json.loads(manifest.read_text())
    value["evaluation_scope"] = "independent_recording_sessions"
    write_json(manifest, value)
    cfg = dict(contract_version="w3_next_v1", side="right", joint_names=[f"right_joint_{i}" for i in range(7)],
        data=dict(arm_mode="single", history=50, sample_hz=100, keys={k:k for k in SIGNALS},
                  memory_budget_mb=32, manifest=str(manifest)),
        train=dict(device="cpu", seed=0, optimizer="adamw", weight_decay=1e-6,
                   learning_rate=.001, gradient_clip=1., loss_units="physical_nm"))
    cfg["model"] = SMALL.copy()
    cfg["train"].update(epochs=2, batch_size=8, cpu_threads=1)
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return tmp_path, config, manifest, paths


def test_parent_session_leak_fails_before_creating_output(fixture_campaign):
    tmp, config, manifest, paths = fixture_campaign
    meta = json.loads(sidecar_path(paths["val"][0]).read_text())
    meta["derivation"] = dict(parent_session_id="train")
    write_json(sidecar_path(paths["val"][0]), meta)
    with pytest.raises(ValueError, match="independent_session_overlap"):
        runner.prepare(config, manifest, tmp/"bad", [0])
    assert not (tmp/"bad").exists()


@pytest.mark.parametrize("target", ["experiment.yaml", "frozen_runtime/factr2_next/training/next_u.py", "sidecar"])
def test_frozen_inputs_reject_tampering(fixture_campaign, target):
    tmp, config, manifest, paths = fixture_campaign
    folder = runner.prepare(config, manifest, tmp/"experiment", [0])
    file = sidecar_path(paths["test"][0]) if target == "sidecar" else folder/target
    file.write_text(file.read_text()+"\n")
    with pytest.raises(ValueError, match="hash"):
        runner.verify(folder)


def test_frozen_cli_end_to_end_pair_and_timestamp_alignment(fixture_campaign):
    tmp, config, manifest, paths = fixture_campaign
    folder = runner.prepare(config, manifest, tmp/"experiment", [0, 1])
    assert json.loads((folder/"state.json").read_text())["status"] == "prepared"
    assert not (folder/"results").exists()
    with pytest.raises(ValueError, match="all_seeds_trained"):
        runner.evaluate(folder)
    for command in ("train", "evaluate"):
        result = subprocess.run([sys.executable, str(folder/"start.py"), command],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((folder/"evaluation/comparison.json").read_text())
    assert len(report["seeds"]) == 2
    for seed in (0, 1):
        for arm in ARMS:
            path = folder/"results"/f"seed_{seed}"/arm
            _, _, meta = runner.load_artifact(path)
            assert meta["deployment_ready"] is False
            history = json.loads((path.parent/(arm+"_history.json")).read_text())
            values = [e["validation"]["physical_mse_nm2"] for e in history["epochs"]]
            assert history["best_epoch"] == int(np.argmin(values))+1
            assert len(values) == 2
            with pytest.raises(ValueError):
                load_checkpoint(path)
    files = [folder/"evaluation"/f"seed_0_test_{arm}.npz" for arm in ARMS]
    with np.load(files[0]) as a, np.load(files[1]) as b:
        np.testing.assert_array_equal(a["timestamp_ns"], b["timestamp_ns"])
        assert len(a["timestamp_ns"]) == 63-49
        assert a["timestamp_ns"][0] == 10**15+49*10_000_000
        np.testing.assert_allclose(a["mean_free_nm"]+a["error_nm"],
                                   b["mean_free_nm"]+b["error_nm"], atol=1e-7)
        assert "variance_free_nm2" not in a and np.all(b["variance_free_nm2"] > 0)
    with pytest.raises(ValueError, match="already_started"):
        runner.train(folder)
    model_file = folder/"results/seed_0/NEXT_U/model.pt"
    with model_file.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="artifact_hash"):
        runner.load_artifact(model_file.parent)


def test_test_labels_do_not_influence_weights_or_normalization(fixture_campaign, monkeypatch):
    tmp, config, manifest, paths = fixture_campaign
    first = runner.prepare(config, manifest, tmp/"first", [0])
    roles = []
    actual_dataset = runner.dataset
    def tracked_dataset(m, cfg, role):
        roles.append(role)
        return actual_dataset(m, cfg, role)
    monkeypatch.setattr(runner, "dataset", tracked_dataset)
    runner.train(first)
    assert "test" not in roles
    p = paths["test"][0]
    with h5py.File(p, "r+") as h:
        h["ep_0000/measured_joint_torque/data"][:] += 100
    meta = json.loads(sidecar_path(p).read_text())
    meta["h5_sha256"] = sha256(p)
    write_json(sidecar_path(p), meta)
    m = json.loads(manifest.read_text())
    m["splits"]["test"][0]["h5_sha256"] = sha256(p)
    write_json(manifest, m)
    second = runner.prepare(config, manifest, tmp/"second", [0])
    runner.train(second)
    for arm in ARMS:
        a, an, _ = runner.load_artifact(first/"results/seed_0"/arm)
        b, bn, _ = runner.load_artifact(second/"results/seed_0"/arm)
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        for key in an:
            np.testing.assert_array_equal(an[key], bn[key])


def test_budget_stop_is_requested_only_after_mse_target_epoch_finishes():
    assert not stop_ready(dict(epochs=[]), 30)
    assert not stop_ready(dict(epochs=[dict(epoch=29)]), 30)
    assert stop_ready(dict(epochs=[dict(epoch=30)]), 30)


def test_budget_evaluation_preserves_lock_and_requires_paired_histories(fixture_campaign):
    tmp, config, manifest, paths = fixture_campaign
    folder = runner.prepare(config, manifest, tmp/"budget", [0])
    runner.train(folder)
    original_lock = sha256(folder/"lock.json")
    original_config = sha256(folder/"experiment.yaml")
    state = json.loads((folder/"state.json").read_text())
    state["status"] = "stopped"
    write_json(folder/"state.json", state)
    plan = folder/"budget_amendment.json"
    write_json(plan, dict(schema="next_u_budget_amendment_v1", seed=0, max_epochs=2,
        original_lock_sha256=original_lock))
    report = runner.evaluate(folder, budget_plan=plan)
    assert list(report["seeds"]) == ["0"]
    assert report["budget_amendment"]["completed_epochs"] == 2
    assert sha256(folder/"lock.json") == original_lock
    assert sha256(folder/"experiment.yaml") == original_config
    assert json.loads((folder/"state.json").read_text())["status"] == "stopped"
    with pytest.raises(ValueError, match="all_seeds_trained"):
        runner.evaluate(folder)
    history_path = folder/"results/seed_0/NEXT_U_history.json"
    history = json.loads(history_path.read_text())
    history["epochs"].pop()
    write_json(history_path, history)
    with pytest.raises(ValueError, match="unpaired_history"):
        runner.evaluate(folder, budget_plan=plan)


def test_budget_rejects_wrong_lock_and_exceeded_limit(fixture_campaign):
    tmp, config, manifest, paths = fixture_campaign
    folder = runner.prepare(config, manifest, tmp/"budget", [0])
    runner.train(folder)
    cfg, _ = runner.verify(folder)
    plan = folder/"budget_amendment.json"
    request = dict(schema="next_u_budget_amendment_v1", seed=0, max_epochs=1,
                   original_lock_sha256="wrong")
    write_json(plan, request)
    with pytest.raises(ValueError, match="invalid_budget"):
        runner.evaluation_budget(folder, cfg, plan)
    request["original_lock_sha256"] = sha256(folder/"lock.json")
    write_json(plan, request)
    with pytest.raises(ValueError, match="budget_exceeded"):
        runner.evaluation_budget(folder, cfg, plan)
