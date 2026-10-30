"""Variance-only controls, immutable mean, unit checks and frozen CLI replay."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from factr2_next.training.frozen_variance import (
    VarianceHead, fixed_variance, hidden_normalization, cache_mean, fit_head,
    head_variance, variance_metrics,
)
from factr2_next.training.next_u import NextU
from factr2_next.data_collection.quality import write_json
from diagnose_w3_next_u import aligned_states, load_previous, joint_summary, group_masks
from test_w3_next_u_ablation import fixture_campaign, SMALL
import run_w3_next_u_ablation as core
import run_w3_frozen_sigma as runner

torch.set_num_threads(1)


def test_constant_sigma_uses_second_moment_including_bias_and_squared_units():
    error = np.tile(np.arange(1, 8, dtype=np.float32)*.1, (100, 1))
    v = fixed_variance(error, 1e-4, 1e4)
    np.testing.assert_allclose(v, error[0]**2)
    metric = variance_metrics(dict(error_z=error), v, np.arange(1, 8, dtype=np.float32))
    expected_sigma = np.arange(1, 8)**2*.1
    np.testing.assert_allclose(metric["per_joint_mean_sigma_nm"], expected_sigma, rtol=1e-6)
    assert max(metric["per_joint_floor_fraction"]) == 0


def test_head_starts_at_exact_fixed_control_and_has_finite_gradient():
    initial = np.geomspace(.0002, 10, 7).astype(np.float32)
    head = VarianceHead(4, initial)
    x = torch.randn(20, 4)
    v = head(x)
    torch.testing.assert_close(v, torch.as_tensor(initial).expand(20, 7))
    loss = (v.log()+.3/v).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with pytest.raises(ValueError, match="initial"):
        VarianceHead(4, np.zeros(7))


def test_hidden_norm_is_train_only_and_handles_dead_neurons():
    x = np.c_[np.arange(10), np.ones(10)].astype(np.float32)
    norm = hidden_normalization(x)
    assert norm["mean"][0] == 4.5
    assert norm["std"][1] == pytest.approx(1e-6)
    assert np.isfinite((x-norm["mean"])/norm["std"]).all()


def test_known_heteroscedastic_data_is_learned_without_access_to_test():
    rng = np.random.default_rng(21)
    def cache(n):
        x = rng.normal(size=(n, 2)).astype(np.float32)
        sigma = np.exp(.6*x[:, :1])
        e = rng.normal(size=(n, 7)).astype(np.float32)*sigma
        return dict(hidden=x, error_z=e)
    train, val = cache(3000), cache(1500)
    recipe = dict(runner.DEFAULT_RECIPE, epochs=20, batch_size=128,
                  learning_rate=.02, cpu_threads=1, patience=20)
    head, norm, _, history = fit_head(train, val, np.ones(7, np.float32), recipe)
    points = dict(hidden=np.array([[-1., 0], [1., 0]], np.float32))
    v = head_variance(head, points, norm)
    assert np.all(v[1] > v[0]*3)
    assert history["best_validation_nll"] < history["epochs"][0]["validation"]["marginal_nll"]-.15


def test_mse_decomposition_accounts_for_mean_shift_and_bias():
    a = np.full((10, 7), .1)
    b = np.full((10, 7), .3)
    result = joint_summary(a, b, np.ones_like(b)*.01, 0)
    assert result["mse_change_nm2"] == pytest.approx(.08)
    assert result["mse_change_nm2"] == pytest.approx(
        result["mean_shift_squared_nm2"]+result["minus_twice_error_shift_covariance_nm2"])
    assert result["nominal95_coverage"] == 0


def test_direction_and_reversal_reset_at_episode_boundaries():
    x = np.zeros((8, 21), np.float32)
    x[:, 7] = np.deg2rad([1, 1, -1, -1, 1, 1, 1, 1])
    states = dict(x=x, kind=np.array(["static"]*8), approach=np.array(["low"]*8),
                  episode_index=np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    groups = dict(group_masks(states, x, 0))
    np.testing.assert_array_equal(groups["measured_sign_reversal:following_0.5s"], [0, 0, 1, 1, 0, 0, 0, 0])


@pytest.fixture
def completed_parent(fixture_campaign):
    tmp, config, manifest, paths = fixture_campaign
    parent = core.prepare(config, manifest, tmp/"parent", [0])
    core.train(parent)
    core.evaluate(parent)
    return tmp, parent


def test_mean_cache_disables_dropout_and_fit_cannot_change_mean(completed_parent):
    tmp, parent = completed_parent
    cfg, manifest = core.verify(parent)
    model, norm, _ = core.load_artifact(parent/"results/seed_0/NEXT_MSE")
    model.train()
    before = runner.mean_digest(model)
    caches = {}
    for role in ("train", "val"):
        ds = core.dataset(manifest, cfg, role)
        caches[role] = cache_mean(model, norm, ds, 8)
    assert all(not part.training for part in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    recipe = dict(runner.DEFAULT_RECIPE, epochs=2, cpu_threads=1, batch_size=8)
    head, hidden_norm, _, _ = fit_head(caches["train"], caches["val"], norm["y_std"], recipe)
    assert runner.mean_digest(model) == before
    assert all(p.grad is None for p in model.parameters())
    assert head_variance(head, caches["val"], hidden_norm).shape == caches["val"]["error_z"].shape


def test_timestamp_mismatch_is_rejected(completed_parent):
    tmp, parent = completed_parent
    cfg, manifest = core.verify(parent)
    ds = core.dataset(manifest, cfg, "val")
    states = aligned_states(ds, cfg["data"]["keys"])
    load_previous(parent, "val", states)
    states["timestamp_ns"][0] += 1
    with pytest.raises(AssertionError):
        load_previous(parent, "val", states)


def test_frozen_cli_end_to_end_identical_mean_and_bound_artifact(completed_parent):
    tmp, parent = completed_parent
    recipe = dict(runner.DEFAULT_RECIPE, epochs=2, batch_size=8, cpu_threads=1,
                  patience=2, warmup=1)
    folder = runner.prepare(parent, tmp/"sigma", recipe)
    proc = subprocess.run([sys.executable, str(folder/"start.py")], capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    state = json.loads((folder/"state.json").read_text())
    assert state["status"] == "complete" and state["mean_unchanged"]
    assert state["test_used_for_fit"] is False
    report = json.loads((folder/"evaluation/comparison.json").read_text())
    assert report["best_epoch"] <= 2
    assert report["head_parameters"] == 7*(8+1)
    for role in ("val", "test"):
        paths = [folder/"evaluation"/f"{role}_{arm}.npz" for arm in runner.ARMS]
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            for key in ("mean_free_nm", "error_nm", "timestamp_ns", "episode_index"):
                np.testing.assert_array_equal(a[key], b[key])
    head, norm, fixed = runner.load_head(folder)
    assert np.all(fixed > 0)
    runner.verify(folder)
    with (folder/"variance_head/head.pt").open("ab") as h:
        h.write(b"changed")
    with pytest.raises(ValueError, match="head_hash"):
        runner.load_head(folder)


def test_frozen_runtime_and_mean_artifact_tamper_detection(completed_parent):
    tmp, parent = completed_parent
    folder = runner.prepare(parent, tmp/"tamper", dict(runner.DEFAULT_RECIPE, epochs=1, cpu_threads=1))
    code = folder/"frozen_runtime/factr2_next/training/frozen_variance.py"
    code.write_text(code.read_text()+"\n")
    with pytest.raises(ValueError, match="frozen_sigma_hash"):
        runner.verify(folder)
