#!/usr/bin/env python3
"""Frozen-mean, fixed-sigma versus state-dependent sigma offline pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import h5py

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.training.frozen_variance import (
    SCHEMA, VarianceHead, cache_mean, fit_head, head_variance, variance_metrics,
)
import run_w3_next_u_ablation as core
from diagnose_w3_next_u import aligned_states, diagnose, load_previous

ARMS = ("FIXED_SIGMA", "LEARNED_SIGMA")
DEFAULT_RECIPE = dict(seed=0, cpu_threads=8, batch_size=2048, epochs=20,
    learning_rate=.001, weight_decay=1e-6, gradient_clip=1., floor=1e-4, ceiling=1e4,
    patience=5, warmup=5, min_delta=1e-5,
    selection="validation physical marginal NLL; epoch 0 included; first on ties",
    test_scope="previously inspected sessions; retrospective pilot, not a new blind test",
    calibration="none; nominal Gaussian intervals are diagnostic",
    velocity_deadband_deg_s=.1)


def mean_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.base.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prepare(parent, output, recipe=None):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg, _ = core.verify(parent)
    model, norm, meta = core.load_artifact(parent/"results/seed_0/NEXT_MSE")
    if meta["arm"] != "NEXT_MSE" or meta["seed"] != 0:
        raise ValueError("frozen_sigma_mean_source")
    recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)
    if recipe["epochs"] < 1 or recipe["batch_size"] < 1 or recipe["cpu_threads"] < 1:
        raise ValueError("frozen_sigma_budget")
    output.mkdir(parents=True)
    shutil.copytree(parent/"results/seed_0/NEXT_MSE", output/"mean_checkpoint")
    write_json(output/"recipe.json", recipe)
    frozen = output/"frozen_runtime"
    shutil.copytree(parent/"frozen_runtime/factr2_next", frozen/"factr2_next",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    import factr2_next.training.frozen_variance as module
    shutil.copy2(module.__file__, frozen/"factr2_next/training/frozen_variance.py")
    shutil.copy2(parent/"frozen_runtime/run_w3_next_u_ablation.py", frozen/"run_w3_next_u_ablation.py")
    for name in ("run_w3_frozen_sigma.py", "diagnose_w3_next_u.py"):
        shutil.copy2(Path(__file__).with_name(name), frozen/name)
    (output/"start.py").write_text(
        "from pathlib import Path\nimport sys,runpy\n"
        "r=Path(__file__).resolve().parent\n"
        "sys.path.insert(0,str(r/'frozen_runtime'))\n"
        "sys.argv=[str(r/'frozen_runtime/run_w3_frozen_sigma.py'),'run','--experiment',str(r)]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n")
    external = {}
    for file in (parent/"lock.json", parent/"evaluation/comparison.json"):
        external[str(file)] = sha256(file)
    for file in (parent/"evaluation").glob("seed_0_*.npz"):
        external[str(file)] = sha256(file)
    for file in (parent/"results/seed_0/NEXT_U").iterdir():
        if file.is_file():
            external[str(file)] = sha256(file)
    locked = {str(p.relative_to(output)):sha256(p) for p in output.rglob("*") if p.is_file()}
    write_json(output/"lock.json", dict(schema=SCHEMA, parent=str(parent), files=locked,
        external_files=external, mean_state_sha256=mean_digest(model),
        source_checkpoint_sha256=sha256(parent/"results/seed_0/NEXT_MSE/model.pt"),
        source_selected_epoch=json.loads((output/"mean_checkpoint/metrics.json").read_text())["best_epoch"],
        software=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                      h5py=h5py.__version__),
        changes="only a variance head; mean, data splits and normalization fixed",
        deployment_ready=False, automatic_deployment=False))
    write_json(output/"state.json", dict(status="prepared", deployment_ready=False))
    print(str(output))
    return output


def verify(folder):
    folder = Path(folder).resolve()
    lock = json.loads((folder/"lock.json").read_text())
    if lock["schema"] != SCHEMA:
        raise ValueError("frozen_sigma_schema")
    if lock["software"] != dict(python=platform.python_version(), numpy=np.__version__,
                                 torch=str(torch.__version__), h5py=h5py.__version__):
        raise ValueError("frozen_sigma_numerical_runtime_changed")
    for name, digest in lock["files"].items():
        if sha256(folder/name) != digest:
            raise ValueError("frozen_sigma_hash:"+name)
    for name, digest in lock["external_files"].items():
        if sha256(name) != digest:
            raise ValueError("frozen_sigma_parent_hash:"+name)
    cfg, manifest = core.verify(Path(lock["parent"]))
    model, norm, _ = core.load_artifact(folder/"mean_checkpoint")
    if mean_digest(model) != lock["mean_state_sha256"]:
        raise ValueError("frozen_sigma_mean_hash")
    return lock, cfg, manifest, model, norm


def save_head(folder, head, hidden_norm, fixed_z, recipe, history, mean_sha):
    result = folder/"variance_head"
    result.mkdir()
    torch.save(dict(schema=SCHEMA, state=head.state_dict(), width=head.linear.in_features,
                    floor=recipe["floor"], ceiling=recipe["ceiling"],
                    mean_model_sha256=mean_sha), result/"head.pt")
    np.savez(result/"normalization.npz", **hidden_norm, fixed_variance_z=fixed_z)
    write_json(result/"history.json", history)
    write_json(result/"metadata.json", dict(schema=SCHEMA, deployment_ready=False,
        automatic_deployment=False, mean_model_sha256=mean_sha,
        selection=history["selection"], artifact_sha256={f:sha256(result/f) for f in
            ("head.pt", "normalization.npz", "history.json")}))


def load_head(folder):
    result = Path(folder)/"variance_head"
    meta = json.loads((result/"metadata.json").read_text())
    if meta["schema"] != SCHEMA:
        raise ValueError("frozen_sigma_head_schema")
    for name in ("head.pt", "normalization.npz", "history.json"):
        if sha256(result/name) != meta["artifact_sha256"][name]:
            raise ValueError("frozen_sigma_head_hash")
    checkpoint = torch.load(result/"head.pt", map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or checkpoint["mean_model_sha256"] != meta["mean_model_sha256"]:
        raise ValueError("frozen_sigma_head_contract")
    if sha256(Path(folder)/"mean_checkpoint/model.pt") != meta["mean_model_sha256"]:
        raise ValueError("frozen_sigma_head_mean_binding")
    with np.load(result/"normalization.npz", allow_pickle=False) as h:
        norm, fixed = {k:h[k] for k in ("mean", "std")}, h["fixed_variance_z"]
    if (any(v.shape != (checkpoint["width"],) or not np.isfinite(v).all() for v in norm.values())
            or np.any(norm["std"] <= 0)):
        raise ValueError("frozen_sigma_head_normalization")
    head = VarianceHead(checkpoint["width"], fixed, checkpoint["floor"], checkpoint["ceiling"])
    head.load_state_dict(checkpoint["state"], strict=True)
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError("frozen_sigma_head_finite")
    head.eval()
    return head, norm, fixed


def grouped(cache, variance_z, y_std, states, recipe):
    values = {"overall":variance_metrics(cache, variance_z, y_std, recipe["floor"], recipe["ceiling"])}
    for field in ("kind", "approach", "episode_index"):
        for label in np.unique(states[field]):
            mask = states[field] == label
            c = {k:v[mask] for k,v in cache.items()}
            v = variance_z[mask] if np.asarray(variance_z).ndim == 2 else variance_z
            values[f"{field}:{label}"] = variance_metrics(c, v, y_std, recipe["floor"], recipe["ceiling"])
    return values


def run(folder):
    folder = Path(folder).resolve()
    lock, cfg, manifest, model, norm = verify(folder)
    if json.loads((folder/"state.json").read_text())["status"] != "prepared":
        raise ValueError("frozen_sigma_already_started")
    recipe = json.loads((folder/"recipe.json").read_text())
    torch.set_num_threads(recipe["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    write_json(folder/"state.json", dict(status="caching_train_val", test_used_for_fit=False))
    try:
        train_ds, val_ds = (core.dataset(manifest, cfg, r) for r in ("train", "val"))
        with np.load(Path(lock["parent"])/"normalization.npz") as original:
            for key in norm:
                np.testing.assert_array_equal(norm[key], original[key])
        cache_folder = folder/"cache"
        cache_folder.mkdir()
        caches = {}
        for role, ds in (("train", train_ds), ("val", val_ds)):
            caches[role] = cache_mean(model, norm, ds, recipe["batch_size"])
            np.savez_compressed(cache_folder/(role+".npz"), **caches[role])
            print(f"cached {role}: {len(ds)} frames", flush=True)
        initial_mean_sha = mean_digest(model)
        def epoch_callback(record, best_epoch):
            write_json(folder/"progress.json", dict(epoch=record["epoch"], best_epoch=best_epoch,
                       train_nll=record["train_nll"], validation_nll=record["validation"]["marginal_nll"]))
            print(f"variance-only epoch={record['epoch']} train_NLL={record['train_nll']:.6f} "
                  f"val_NLL={record['validation']['marginal_nll']:.6f} best={best_epoch}", flush=True)
        write_json(folder/"state.json", dict(status="training_variance_only", test_used_for_fit=False))
        head, hidden_norm, fixed_z, history = fit_head(caches["train"], caches["val"], norm["y_std"],
            recipe, on_epoch=epoch_callback, should_stop=lambda:(folder/"STOP_TRAINING").exists())
        if mean_digest(model) != initial_mean_sha or initial_mean_sha != lock["mean_state_sha256"]:
            raise ValueError("frozen_sigma_mean_changed")
        save_head(folder, head, hidden_norm, fixed_z, recipe, history, lock["source_checkpoint_sha256"])
        loaded, loaded_norm, loaded_fixed = load_head(folder)
        np.testing.assert_array_equal(loaded_fixed, fixed_z)
        probe = {k:v[:64] for k,v in caches["val"].items()}
        np.testing.assert_array_equal(head_variance(head, probe, hidden_norm), head_variance(loaded, probe, loaded_norm))
        # No test target or hidden representation enters head fitting/selection.
        write_json(folder/"state.json", dict(status="evaluating", best_epoch=history["best_epoch"], test_used_for_fit=False))
        test_ds = core.dataset(manifest, cfg, "test")
        caches["test"] = cache_mean(model, norm, test_ds, recipe["batch_size"])
        np.savez_compressed(cache_folder/"test.npz", **caches["test"])
        output = folder/"evaluation"
        output.mkdir()
        diagnosis = diagnose(Path(lock["parent"]), output, cfg, train_ds,
                             {"val":val_ds, "test":test_ds}, norm["y_std"])
        report = dict(schema=SCHEMA, scope=recipe["test_scope"], calibration=recipe["calibration"],
            mean_frozen=True, deployment_ready=False, best_epoch=history["best_epoch"],
            mean_selected_epoch=lock["source_selected_epoch"], fit_split="train", selection_split="val",
            test_used_for_fit=False, head_parameters=sum(p.numel() for p in head.parameters()), splits={})
        for role, ds in (("val", val_ds), ("test", test_ds)):
            states = aligned_states(ds, cfg["data"]["keys"])
            previous = load_previous(Path(lock["parent"]), role, states)["NEXT_MSE"]
            cache = caches[role]
            mean_delta = abs(cache["mean_free_nm"]-previous["mean_free_nm"])
            np.testing.assert_allclose(cache["mean_free_nm"], previous["mean_free_nm"], atol=1e-7, rtol=1e-6)
            learned_z = head_variance(head, cache, hidden_norm, recipe["batch_size"])
            split = dict(mean_max_abs_replay_difference_nm=float(mean_delta.max()), arms={})
            for arm, v in (("FIXED_SIGMA", fixed_z), ("LEARNED_SIGMA", learned_z)):
                split["arms"][arm] = grouped(cache, v, norm["y_std"], states, recipe)
                np.savez_compressed(output/f"{role}_{arm}.npz", mean_free_nm=cache["mean_free_nm"],
                    error_nm=cache["error_z"]*norm["y_std"],
                    variance_free_nm2=np.broadcast_to(v, cache["error_z"].shape)*norm["y_std"]**2,
                    timestamp_ns=states["timestamp_ns"], episode_index=states["episode_index"])
            report["splits"][role] = split
        write_json(output/"comparison.json", report)
        write_report(output, report, diagnosis, history)
        plot_comparison(output/"coverage_width.png", report)
        write_json(folder/"cache_sha256.json", {p.name:sha256(p) for p in cache_folder.glob("*.npz")})
        # Revalidate all bound source inputs after training and diagnostics.
        verify(folder)
        write_json(folder/"state.json", dict(status="complete", best_epoch=history["best_epoch"],
            completed_epochs=len(history["epochs"])-1, mean_unchanged=True,
            elapsed_seconds=time.monotonic()-started,
            report=str(output/"COMPARISON_ZH.md"), test_used_for_fit=False))
        print(str(output/"COMPARISON_ZH.md"), flush=True)
        return report
    except BaseException as exc:
        write_json(folder/"state.json", dict(status="failed", error=str(exc)))
        raise


def write_report(output, report, diagnosis, history):
    lines = ["# 冻结 NEXT 均值后的方差对照", "",
        "单 seed 回顾性试验；均值使用原 NEXT_MSE 验证最佳权重，未重新训练。固定方差仅由训练残差平方均值估计；状态相关方差仅训练线性头，按验证 NLL 选模，包含 epoch 0 固定方差候选。", "",
        f"方差头最佳 epoch：{history['best_epoch']}；实际训练 {len(history['epochs'])-1} 轮。测试集未参与拟合或选模，但这些历史会话此前已被查看，不能称为新盲测。", "",
        "|划分|方案|MSE（Nm²）|NLL|95%覆盖率范围|95%平均宽度范围（Nm）|", "|---|---|---:|---:|---|---|"]
    for role, split in report["splits"].items():
        for arm in ARMS:
            m = split["arms"][arm]["overall"]
            interval = m["nominal_gaussian_intervals"]["95"]
            c, w = interval["per_joint_coverage"], interval["per_joint_mean_width_nm"]
            lines.append(f"|{role}|{arm}|{m['physical_mse_nm2']:.8f}|{m['marginal_nll']:.6f}|"
                         f"{min(c)*100:.1f}%–{max(c)*100:.1f}%|{min(w):.4f}–{max(w):.4f}|")
    lines += ["", "## 逐关节测试区间", "", "|关节|固定 sigma 覆盖率|学习 sigma 覆盖率|固定宽度 Nm|学习平均宽度 Nm|", "|---|---:|---:|---:|---:|"]
    fixed, learned = (report["splits"]["test"]["arms"][arm]["overall"]["nominal_gaussian_intervals"]["95"] for arm in ARMS)
    for j in range(7):
        lines.append(f"|J{j+1}|{fixed['per_joint_coverage'][j]*100:.2f}%|{learned['per_joint_coverage'][j]*100:.2f}%|"
                     f"{fixed['per_joint_mean_width_nm'][j]:.5f}|{learned['per_joint_mean_width_nm'][j]:.5f}|")
    lines += ["", "## 原联合 NEXT-U 的 J1/J4 诊断", "",
              "以下分组来自旧试验，仅用于定位关联，不能证明因果。姿态与跟踪误差分箱由训练集定义，速度来自实测反馈；换向标签为超过 0.1°/s 死区的符号变化后 0.5 秒。", "",
              "|关节|分组|窗口数|NEXT RMSE Nm|联合 NEXT-U RMSE Nm|联合 U 95%覆盖率|", "|---|---|---:|---:|---:|---:|"]
    for joint, groups in diagnosis["splits"]["test"]["joints"].items():
        for label, m in groups.items():
            if label == "overall" or label.startswith(("kind:", "velocity:", "static:", "measured_sign_reversal:")):
                lines.append(f"|{joint}|{label}|{m['samples']}|{m['next_rmse_nm']:.5f}|{m['u_rmse_nm']:.5f}|{m['nominal95_coverage']*100:.1f}%|")
    bounds = diagnosis["splits"]["test"]["variance_bound_fractions"]
    lines += ["", f"旧联合 NEXT-U 的逐关节方差下限占比：{bounds['floor_per_joint']}；上限占比：{bounds['ceiling_per_joint']}。", "",
        "两组使用完全相同的均值和残差数组，MSE 相同是冻结均值的预期结果。覆盖率必须结合区间宽度、NLL、关节和会话一起判断；扩大区间本身不是精度改善。未做独立会话校准，也未验证已知接触，不能把 sigma 当成电流精度、真实外力置信度或 OOD 保证。", "",
        "详细逐组指标见 comparison.json；旧模型的姿态/方向/速度/跟踪误差/换向诊断与 MSE 变化分解见 previous_diagnosis.json。"]
    (output/"COMPARISON_ZH.md").write_text("\n".join(lines)+"\n")


def plot_comparison(path, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    x = np.arange(7)
    for row, role in enumerate(("val", "test")):
        for i, arm in enumerate(ARMS):
            m = report["splits"][role]["arms"][arm]["overall"]["nominal_gaussian_intervals"]["95"]
            axes[row, 0].bar(x+(i-.5)*.36, m["per_joint_coverage"], width=.36, label=arm)
            axes[row, 1].bar(x+(i-.5)*.36, m["per_joint_mean_width_nm"], width=.36, label=arm)
        axes[row, 0].axhline(.95, color="red", ls="--", lw=.8)
        axes[row, 0].set_ylim(0, 1.03)
        axes[row, 0].set_ylabel(role+" coverage")
        axes[row, 1].set_ylabel(role+" mean width (Nm)")
    for ax in axes.flat:
        ax.set_xticks(x, [f"J{j+1}" for j in x])
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Same frozen NEXT mean | uncalibrated nominal 95% intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent", required=True)
    p.add_argument("--output", required=True)
    for name in ("verify", "run"):
        sub.add_parser(name).add_argument("--experiment", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent, args.output)
    elif args.command == "verify":
        verify(args.experiment)
        print("PASS frozen-mean experiment")
    else:
        folder = Path(args.experiment).resolve()
        if Path(__file__).resolve() != folder/"frozen_runtime/run_w3_frozen_sigma.py":
            verify(folder)
            subprocess.run([sys.executable, str(folder/"start.py")], check=True)
        else:
            run(folder)


if __name__ == "__main__":
    main()
