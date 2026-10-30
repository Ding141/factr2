#!/usr/bin/env python3
"""Freeze, train and evaluate paired offline NEXT MSE / NEXT-U experiments.

Use `prepare` first; training/evaluation dispatch into the frozen source copy.
No ROS imports, controller commands, model deployment, or test-based selection.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from factr2_next.data_collection.campaign import validate_independent_manifest
from factr2_next.data_collection.quality import sha256, sidecar_path, validate_manifest, write_json
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.next_u import (
    ARMS, SCHEMA, DEFAULT_VARIANCE, NextU, distribution_metrics, error_metrics,
    predict_dataset, train_epoch,
)
from factr2_next.training.train import set_seed


def validate_config(cfg, seeds):
    if (cfg.get("contract_version") != "w3_next_v1" or cfg.get("side") not in ("left", "right")
            or cfg.get("joint_names") != [f'{cfg["side"]}_joint_{i}' for i in range(7)]):
        raise ValueError("next_u_contract")
    if cfg["data"]["history"] != 50 or cfg["data"]["sample_hz"] != 100:
        raise ValueError("next_u_requires_50_frames_100hz")
    if (cfg["data"].get("arm_mode") != "single"
            or cfg["data"]["keys"] != {k: k for k in
                ("joint_pos", "joint_vel", "joint_cmd", "measured_joint_torque")}):
        raise ValueError("next_u_original_feature_contract_required")
    if cfg["train"].get("device", "cpu") != "cpu":
        raise ValueError("next_u_paired_runner_cpu_only")
    if cfg["train"].get("loss_units") != "physical_nm":
        raise ValueError("next_u_baseline_requires_physical_mse")
    if cfg["train"].get("optimizer") not in ("adam", "adamw"):
        raise ValueError("next_u_optimizer")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**31 for s in seeds):
        raise ValueError("next_u_unique_nonnegative_seeds")
    t = cfg["train"]
    if any(int(t.get(k, 0)) < 1 for k in ("epochs", "batch_size", "cpu_threads")):
        raise ValueError("next_u_positive_training_budget")
    for key in ("learning_rate", "gradient_clip"):
        if not np.isfinite(float(t[key])) or float(t[key]) <= 0:
            raise ValueError("next_u_positive_" + key)
    if not np.isfinite(float(t.get("weight_decay", 0))) or float(t.get("weight_decay", 0)) < 0:
        raise ValueError("next_u_weight_decay")
    early = t.get("early_stopping", {})
    if (int(early.get("patience", 20)) < 1 or int(early.get("warmup", 10)) < 0
            or not np.isfinite(float(early.get("min_delta", 1e-5)))
            or float(early.get("min_delta", 1e-5)) < 0):
        raise ValueError("next_u_early_stopping")
    # Validate architecture and variance without perturbing caller RNG.
    with torch.random.fork_rng(devices=[]):
        NextU(cfg["model"], cfg.get("next_u", DEFAULT_VARIANCE))


def validate_data(manifest_path, cfg):
    manifest, reports = validate_manifest(manifest_path, dict(
        side=cfg["side"], joint_order=cfg["joint_names"],
        contract_version=cfg["contract_version"], sample_hz=100))
    independent = validate_independent_manifest(manifest, reports)
    return manifest, reports, independent


def dataset(manifest, cfg, role):
    return ManifestDataset(manifest["splits"][role], cfg["data"]["keys"], 50,
                           cfg["data"].get("memory_budget_mb", 512))


def prepare(config, manifest_path, output, seeds):
    """No training. Copy code/config; bind existing data and sidecars by SHA256."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    config = Path(config).expanduser().resolve()
    cfg = yaml.safe_load(config.read_text())
    cfg.setdefault("next_u", dict(DEFAULT_VARIANCE))
    validate_config(cfg, seeds)
    manifest_path = Path(manifest_path or cfg["data"]["manifest"]).expanduser().resolve()
    manifest, reports, independent = validate_data(manifest_path, cfg)
    train_ds = dataset(manifest, cfg, "train")
    norm = train_ds.fit_normalization()
    # Only train labels define normalization. Test is inspected for contract,
    # checksums and isolation here, never for fitting/selection/tuning.
    output.mkdir(parents=True)
    shutil.copy2(config, output/"source_config.yaml")
    shutil.copy2(manifest_path, output/"split.json")
    np.savez(output/"normalization.npz", **norm)
    cfg["data"]["manifest"] = str(output/"split.json")
    cfg["train"].pop("resolved_device", None)
    cfg["train"]["stop_file"] = str(output/"STOP_TRAINING")
    cfg["comparison"] = dict(schema=SCHEMA, arms=list(ARMS), seeds=list(seeds),
        fresh_paired_training=True, historical_checkpoint_reused=False,
        selection="minimum validation physical MSE; first epoch on ties, both arms",
        epoch_budget="lockstep; stop only when both arms meet common MSE patience",
        variance_units="normalized target squared; physical = variance_z * y_std**2",
        calibration="none; Gaussian coverage is an uncalibrated diagnostic",
        interpretation="learned dispersion of reported free torque, not current sensor accuracy",
        deployment_ready=False, automatic_deployment=False,
        test_scope="existing sessions; retrospective comparison, not a new blind test")
    (output/"experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    frozen = output/"frozen_runtime"
    package = Path(__import__("factr2_next").__file__).resolve().parent
    shutil.copytree(package, frozen/"factr2_next", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, frozen/"run_w3_next_u_ablation.py")
    # All future training/evaluation runs execute these frozen sources.
    (output/"start.py").write_text(
        "from pathlib import Path\nimport sys, runpy\n"
        "root=Path(__file__).resolve().parent\n"
        "sys.path.insert(0,str(root/'frozen_runtime'))\n"
        "sys.argv=[str(root/'frozen_runtime/run_w3_next_u_ablation.py'),"
        "*sys.argv[1:],'--experiment',str(root)]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n")
    locked = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    data_files = {}
    for entries in manifest["splits"].values():
        for entry in entries:
            p = Path(entry["path"]).resolve()
            for file in (p, sidecar_path(p)):
                data_files[str(file)] = sha256(file)
    write_json(output/"lock.json", dict(schema=SCHEMA, files=locked, data_files=data_files,
        independent=independent, original_manifest_sha256=sha256(manifest_path),
        source_config_sha256=sha256(config),
        software=dict(python=platform.python_version(), numpy=np.__version__,
                      torch=str(torch.__version__), h5py=h5py.__version__)))
    write_json(output/"state.json", dict(status="prepared", deployment_ready=False))
    print(json.dumps(dict(experiment=str(output), status="prepared", train_windows=len(train_ds),
                         seeds=seeds, arms=ARMS, training_started=False), indent=2))
    return output


def verify(folder):
    folder = Path(folder).resolve()
    lock = json.loads((folder/"lock.json").read_text())
    if lock["schema"] != SCHEMA:
        raise ValueError("next_u_lock_schema")
    for file, digest in lock["files"].items():
        if sha256(folder/file) != digest:
            raise ValueError("next_u_frozen_hash:" + file)
    for file, digest in lock["data_files"].items():
        if sha256(file) != digest:
            raise ValueError("next_u_data_hash:" + file)
    cfg = yaml.safe_load((folder/"experiment.yaml").read_text())
    validate_config(cfg, cfg["comparison"]["seeds"])
    manifest, _, independent = validate_data(folder/"split.json", cfg)
    if independent != lock["independent"]:
        raise ValueError("next_u_control_profile_changed")
    return cfg, manifest


def save_artifact(path, model, cfg, norm, seed, arm, metrics, folder):
    # The custom schema intentionally cannot load as a production NEXT checkpoint.
    path.mkdir(parents=True, exist_ok=True)
    tmp = path/"model.pending"
    torch.save(dict(schema=SCHEMA, model_state_dict=model.state_dict(), model=cfg["model"],
                    variance=cfg["next_u"], input_size=21, output_size=7, history=50,
                    arm=arm), tmp)
    tmp.replace(path/"model.pt")
    np.savez(path/"normalization.npz", **norm)
    artifact_cfg = copy.deepcopy(cfg)
    artifact_cfg["train"]["seed"] = seed
    (path/"config.yaml").write_text(yaml.safe_dump(artifact_cfg, sort_keys=False))
    write_json(path/"metrics.json", metrics)
    write_json(path/"metadata.json", dict(schema=SCHEMA, arm=arm, seed=seed,
        side=cfg["side"], joint_order=cfg["joint_names"], input_size=21, output_size=7,
        history=50, sample_hz=100, output="free torque mean Nm and variance Nm2",
        external_sign="measured minus predicted free mean",
        calibration="uncalibrated", deployment_ready=False, automatic_deployment=False,
        selection=cfg["comparison"]["selection"],
        lock_sha256=sha256(folder/"lock.json"),
        artifact_sha256={f: sha256(path/f) for f in
                         ("model.pt", "normalization.npz", "config.yaml", "metrics.json")}))


def load_artifact(path):
    path = Path(path)
    meta = json.loads((path/"metadata.json").read_text())
    if meta.get("schema") != SCHEMA or meta.get("arm") not in ARMS:
        raise ValueError("next_u_artifact_schema")
    for file in ("model.pt", "normalization.npz", "config.yaml", "metrics.json"):
        if sha256(path/file) != meta["artifact_sha256"].get(file):
            raise ValueError("next_u_artifact_hash:" + file)
    ckpt = torch.load(path/"model.pt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load((path/"config.yaml").read_text())
    if (ckpt.get("schema") != SCHEMA or ckpt.get("arm") != meta["arm"]
            or ckpt["model"] != cfg["model"] or ckpt["variance"] != cfg["next_u"]
            or cfg["train"]["seed"] != meta["seed"]
            or (ckpt["input_size"], ckpt["output_size"], ckpt["history"]) != (21, 7, 50)):
        raise ValueError("next_u_checkpoint_contract")
    with torch.random.fork_rng(devices=[]):
        model = NextU(ckpt["model"], ckpt["variance"]).configure_arm(meta["arm"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("next_u_weights_finite")
    model.eval()
    with np.load(path/"normalization.npz", allow_pickle=False) as h:
        norm = {k: h[k] for k in h.files}
    for key, size in (("x_mean", 21), ("x_std", 21), ("y_mean", 7), ("y_std", 7)):
        if norm[key].shape != (size,) or not np.isfinite(norm[key]).all():
            raise ValueError("next_u_normalization")
        if key.endswith("std") and np.any(norm[key] <= 0):
            raise ValueError("next_u_normalization_std")
    return model, norm, meta


def train(folder):
    folder = Path(folder).resolve()
    cfg, manifest = verify(folder)
    if json.loads((folder/"state.json").read_text())["status"] != "prepared":
        raise ValueError("next_u_already_started_use_new_experiment")
    if (folder/"STOP_TRAINING").exists():
        raise ValueError("next_u_stop_file_present")
    if (folder/"results").exists():
        raise ValueError("next_u_results_exist")
    # Require identical numerical runtime to the prepared experiment.
    versions = json.loads((folder/"lock.json").read_text())["software"]
    if versions != dict(python=platform.python_version(), numpy=np.__version__,
                        torch=str(torch.__version__), h5py=h5py.__version__):
        raise ValueError("next_u_numerical_runtime_changed")
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    train_ds, val_ds = (dataset(manifest, cfg, role) for role in ("train", "val"))
    norm = train_ds.fit_normalization()
    with np.load(folder/"normalization.npz") as h:
        for key, value in norm.items():
            np.testing.assert_array_equal(value, h[key])
    train_ds.norm = val_ds.norm = norm
    t = cfg["train"]
    early = t.get("early_stopping", {})
    started = time.monotonic()
    write_json(folder/"state.json", dict(status="training", test_used=False))
    try:
        for seed in cfg["comparison"]["seeds"]:
            if (folder/"STOP_TRAINING").exists():
                break
            set_seed(seed)
            prototype = NextU(cfg["model"], cfg["next_u"])
            models = {arm: copy.deepcopy(prototype).configure_arm(arm) for arm in ARMS}
            optimizers, loaders, records = {}, {}, {}
            for arm, model in models.items():
                cls = torch.optim.AdamW if t["optimizer"] == "adamw" else torch.optim.Adam
                optimizers[arm] = cls([p for p in model.parameters() if p.requires_grad],
                    lr=t["learning_rate"], weight_decay=t.get("weight_decay", 0))
                loaders[arm] = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
                records[arm] = dict(seed=seed, arm=arm, epochs=[], best_epoch=None,
                    best_validation_mse_nm2=None,
                    selection=cfg["comparison"]["selection"],
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            best = {arm: float("inf") for arm in ARMS}
            patience_best, stale = best.copy(), {arm: 0 for arm in ARMS}
            stop_reason = "epoch_budget"
            for epoch in range(1, t["epochs"] + 1):
                for arm in ARMS:
                    # Paired dropout streams independent of evaluation and head initialization.
                    torch.manual_seed(seed * 1_000_003 + epoch)
                    loss = train_epoch(models[arm], loaders[arm], optimizers[arm], norm,
                                       arm, t["gradient_clip"])
                    arrays = predict_dataset(models[arm], norm, val_ds, t["batch_size"])
                    diagnostic = (distribution_metrics(arrays["error_nm"], arrays["variance_free_nm2"])
                                  if arm == "NEXT_U" else error_metrics(arrays["error_nm"]))
                    va = diagnostic["physical_mse_nm2"]
                    if not np.isfinite(va):
                        raise ValueError("next_u_nonfinite_validation")
                    record = records[arm]
                    record["epochs"].append(dict(epoch=epoch, train_objective=loss,
                        objective_units="Nm2" if arm == "NEXT_MSE" else "marginal Gaussian NLL",
                        validation=diagnostic))
                    if va < best[arm]:
                        best[arm] = va
                        record.update(best_epoch=epoch, best_validation_mse_nm2=va)
                        save_artifact(folder/"results"/f"seed_{seed}"/arm,
                                      models[arm], cfg, norm, seed, arm, record, folder)
                    if va < patience_best[arm] - early.get("min_delta", 1e-5):
                        patience_best[arm], stale[arm] = va, 0
                    else:
                        stale[arm] += 1
                    write_json(folder/"results"/f"seed_{seed}"/(arm+"_history.json"), record)
                    print(f"seed={seed} epoch={epoch} {arm} objective={loss:.6f} val_MSE={va:.6f}", flush=True)
                write_json(folder/"state.json", dict(status="training", seed=seed,
                    completed_epoch=epoch, test_used=False, seconds=time.monotonic()-started))
                # Both arms always receive exactly the same number of epochs.
                if (folder/"STOP_TRAINING").exists():
                    stop_reason = "user_stop"
                    break
                if (early.get("enabled", False) and epoch >= early.get("warmup", 10)
                        and all(s >= early.get("patience", 20) for s in stale.values())):
                    stop_reason = "both_arms_validation_patience"
                    break
            write_json(folder/"results"/f"seed_{seed}"/"completion.json",
                       dict(completed_epoch=epoch, stop_reason=stop_reason, test_used=False))
            if stop_reason == "user_stop":
                break
        stopped = (folder/"STOP_TRAINING").exists()
        write_json(folder/"state.json", dict(status="stopped" if stopped else "trained",
            test_used=False, training_seconds=time.monotonic()-started))
    except BaseException:
        write_json(folder/"state.json", dict(status="failed", test_used=False))
        raise
    # Evaluation is a separate explicit stage, after all checkpoint selection.


def grouped_metrics(arrays, ds, with_variance):
    from factr2_next.training.coverage_audit import group_name
    indices = {"overall": np.arange(len(ds))}
    previous = 0
    groups = {}
    for end, info in zip(ds.ends, ds.episode_info):
        rows = np.arange(previous, end)
        previous = end
        for label in ("kind:" + group_name(info), "speed:" + str(info.get("speed", "unknown")),
                      "session:" + info["path"]):
            groups.setdefault(label, []).append(rows)
    indices.update({k: np.concatenate(v) for k, v in groups.items()})
    result = {}
    for label, rows in indices.items():
        error = arrays["error_nm"][rows]
        result[label] = (distribution_metrics(error, arrays["variance_free_nm2"][rows])
                         if with_variance else error_metrics(error))
    return result


def evaluation_budget(folder, cfg, plan_path=None):
    """Accept a documented compute-budget truncation without rewriting the lock.

    A stopped paired prefix is evaluable only when both full histories agree
    with its completion record and their validation-selected snapshots.
    """
    state = json.loads((folder/"state.json").read_text())
    if plan_path is None:
        if state["status"] != "trained":
            raise ValueError("next_u_evaluate_requires_all_seeds_trained")
        return cfg["comparison"]["seeds"], None
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    seed, limit = plan.get("seed"), plan.get("max_epochs")
    if (plan.get("schema") != "next_u_budget_amendment_v1"
            or plan.get("original_lock_sha256") != sha256(folder/"lock.json")
            or type(seed) is not int or seed != cfg["comparison"]["seeds"][0]
            or type(limit) is not int or not 1 <= limit <= cfg["train"]["epochs"]
            or state["status"] not in ("stopped", "trained")):
        raise ValueError("next_u_invalid_budget_amendment")
    completed = json.loads((folder/"results"/f"seed_{seed}"/"completion.json").read_text())
    n = completed["completed_epoch"]
    if not 1 <= n <= limit:
        raise ValueError("next_u_budget_exceeded")
    for arm in ARMS:
        base = folder/"results"/f"seed_{seed}"
        history = json.loads((base/(arm+"_history.json")).read_text())
        if [v["epoch"] for v in history["epochs"]] != list(range(1, n+1)):
            raise ValueError("next_u_budget_unpaired_history")
        expected = int(np.argmin([v["validation"]["physical_mse_nm2"] for v in history["epochs"]]))+1
        snapshot = json.loads((base/arm/"metrics.json").read_text())
        if snapshot["best_epoch"] != expected or history["best_epoch"] != expected:
            raise ValueError("next_u_budget_snapshot_selection")
    return [seed], dict(plan=plan, plan_sha256=sha256(plan_path), completed_epochs=n,
        original_seeds=cfg["comparison"]["seeds"], original_max_epochs=cfg["train"]["epochs"],
        evaluator_sha256=sha256(__file__),
        interpretation="compute budget reduced during training; retained common paired prefix; single-seed pilot")


def evaluate(folder, budget_plan=None):
    folder = Path(folder).resolve()
    cfg, manifest = verify(folder)
    seeds, amendment = evaluation_budget(folder, cfg, budget_plan)
    # Never compare a partial/failed seed or select by test errors.
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    report = dict(schema=SCHEMA, calibration="none", deployment_ready=False,
                  scope=cfg["comparison"]["test_scope"], seeds={})
    if amendment is not None:
        report["budget_amendment"] = amendment
    output = folder/"evaluation"
    output.mkdir(exist_ok=True)
    for seed in seeds:
        if not (folder/"results"/f"seed_{seed}"/"completion.json").exists():
            raise ValueError("next_u_incomplete_seed")
        pair = {}
        for role in ("val", "test"):
            ds = dataset(manifest, cfg, role)
            pair[role] = {}
            # Explicit output row identity: H5 checksum, episode, final source timestamp.
            stamps, episode_indices, episode_map = [], [], []
            for i, info in enumerate(ds.episode_info):
                with h5py.File(info["path"]) as h:
                    stamps.append(h[info["episode"]+"/"+cfg["data"]["keys"]["joint_pos"]+"/timestamps"][49:])
                episode_indices.append(np.full(len(stamps[-1]), i, dtype=np.int32))
                episode_map.append(dict(path=info["path"], episode=info["episode"],
                                        h5_sha256=sha256(info["path"])))
            common_norm = None
            for arm in ARMS:
                model, norm, meta = load_artifact(folder/"results"/f"seed_{seed}"/arm)
                if meta["lock_sha256"] != sha256(folder/"lock.json"):
                    raise ValueError("next_u_artifact_from_different_experiment")
                if meta["seed"] != seed or meta["arm"] != arm:
                    raise ValueError("next_u_artifact_seed_arm")
                if common_norm is not None:
                    for key in norm:
                        np.testing.assert_array_equal(norm[key], common_norm[key])
                common_norm = norm
                arrays = predict_dataset(model, norm, ds, cfg["train"]["batch_size"])
                pair[role][arm] = grouped_metrics(arrays, ds, arm == "NEXT_U")
                # NEXT_MSE's frozen dummy variance is not an uncertainty estimate.
                if arm == "NEXT_MSE":
                    arrays.pop("variance_free_nm2")
                np.savez_compressed(output/f"seed_{seed}_{role}_{arm}.npz", **arrays,
                    timestamp_ns=np.concatenate(stamps), episode_index=np.concatenate(episode_indices))
            write_json(output/f"seed_{seed}_{role}_episodes.json", episode_map)
        report["seeds"][str(seed)] = pair
    deltas = [v["test"]["NEXT_U"]["overall"]["physical_mse_nm2"]
              - v["test"]["NEXT_MSE"]["overall"]["physical_mse_nm2"]
              for v in report["seeds"].values()]
    report["paired_test_mse_delta_u_minus_mse_nm2"] = dict(
        per_seed=deltas, mean=float(np.mean(deltas)),
        seed_std=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
        inference="descriptive only; seeds and overlapping windows are not independent recording sessions")
    write_json(output/"comparison.json", report)
    lines = ["# NEXT-U controlled offline ablation", "",
             "Both checkpoints selected by validation physical MSE. Existing sessions; no calibrated force confidence.",
             "", "|Seed|NEXT MSE (Nm²)|NEXT-U MSE (Nm²)|U minus NEXT|", "|---|---:|---:|---:|"]
    if amendment is not None:
        lines[2:2] = [f"Budget reduced during training: seed {seeds[0]} only, "
                      f"{amendment['completed_epochs']} paired epochs completed. "
                      "Single-seed pilot; original frozen configuration retained.", ""]
    for seed, value in report["seeds"].items():
        a, b = (value["test"][arm]["overall"]["physical_mse_nm2"] for arm in ARMS)
        lines.append(f"|{seed}|{a:.6g}|{b:.6g}|{b-a:+.6g}|")
    lines += ["", "Full bias/MAE/RMSE/P95, per-session metrics and nominal Gaussian interval diagnostics are in comparison.json.",
              "Sigma is conditional dispersion of free reported torque; it is not current calibration, contact confidence or an OOD guarantee."]
    (output/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    write_json(folder/"evaluation_state.json", dict(status="evaluated", test_used=True,
        comparison_sha256=sha256(output/"comparison.json")))
    print(str(output/"COMPARISON.md"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="validate/freeze only; does not train")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    for command in ("verify", "train", "evaluate"):
        sub.add_parser(command).add_argument("--experiment", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.manifest, args.output, args.seeds)
        return
    folder = Path(args.experiment).expanduser().resolve()
    if args.command == "verify":
        verify(folder)
        print("PASS frozen experiment, data, sidecars and control profile")
        return
    frozen_script = folder/"frozen_runtime/run_w3_next_u_ablation.py"
    if Path(__file__).resolve() != frozen_script:
        verify(folder)
        sys.path.insert(0, str(folder/"frozen_runtime"))
        # A fresh process avoids live package modules already imported here.
        import subprocess
        subprocess.run([sys.executable, str(folder/"start.py"), args.command], check=True)
        return
    (train if args.command == "train" else evaluate)(folder)


if __name__ == "__main__":
    main()
