#!/usr/bin/env python3
"""Stop an already-running frozen NEXT-U experiment at a paired epoch boundary.

The original process, optimizer, shuffle and dropout state stay intact. No
configuration/lock/weights are rewritten. Evaluation records the amendment.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def stop_ready(mse_history, target_epoch):
    """MSE of the target epoch finished; U of the same epoch must finish next."""
    return bool(mse_history["epochs"] and mse_history["epochs"][-1]["epoch"] >= target_epoch)


def run(folder, plan_path):
    folder, plan_path = Path(folder).resolve(), Path(plan_path).resolve()
    plan = read(plan_path)
    seed, limit = plan["seed"], plan["max_epochs"]
    # Use the original frozen package and the explicitly snapshotted new evaluator.
    sys.path.insert(0, str(folder/"frozen_runtime"))
    evaluator_path = Path(__file__).with_name("run_w3_next_u_ablation.py")
    spec = importlib.util.spec_from_file_location("budget_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    from factr2_next.data_collection.quality import sha256, write_json
    cfg, _ = evaluator.verify(folder)
    if (plan.get("schema") != "next_u_budget_amendment_v1"
            or plan["original_lock_sha256"] != sha256(folder/"lock.json")
            or seed != cfg["comparison"]["seeds"][0]
            or type(limit) is not int or not 1 <= limit <= cfg["train"]["epochs"]):
        raise ValueError("invalid_budget_plan")
    for name, digest in plan["budget_runtime_sha256"].items():
        if sha256(folder/"budget_runtime"/name) != digest:
            raise ValueError("budget_runtime_hash:"+name)
    status_path = folder/"budget_status.json"
    def status(stage, **extra):
        write_json(status_path, dict(stage=stage, seed=seed, max_epochs=limit,
                                     updated_unix=time.time(), **extra))
    status("waiting_for_paired_budget_boundary")
    print(f"Budget amendment: retain seed={seed}, at most {limit} paired epochs", flush=True)
    requested = False
    try:
        while True:
            state = read(folder/"state.json")
            if state["status"] == "failed":
                raise RuntimeError("original_training_failed")
            history_path = folder/"results"/f"seed_{seed}"/"NEXT_MSE_history.json"
            if state.get("seed", seed) != seed:
                (folder/"STOP_TRAINING").write_text("budget amendment: unexpected next seed\n")
                raise RuntimeError("next_seed_already_started")
            if state["status"] == "training" and history_path.exists() and stop_ready(read(history_path), limit):
                if not requested:
                    (folder/"STOP_TRAINING").write_text("Budget amendment; finish current paired epoch, then stop.\n")
                    requested = True
                    status("stop_requested_at_epoch_boundary")
                    print("Stop requested; waiting for the paired NEXT_U epoch to finish", flush=True)
            if state["status"] in ("stopped", "trained"):
                pipeline = read(folder/"execution_status.json")
                if pipeline["status"] in ("stopped", "complete"):
                    break
                if pipeline["status"] == "failed":
                    raise RuntimeError("original_pipeline_failed")
            time.sleep(1)
        evaluator.evaluation_budget(folder, cfg, plan_path)
        status("evaluating")
        with (folder/"evaluation.log").open("x", buffering=1) as output:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(output), redirect_stderr(output):
                evaluator.evaluate(folder, budget_plan=plan_path)
        status("complete", report=str(folder/"evaluation/COMPARISON.md"))
        print(str(folder/"evaluation/COMPARISON.md"), flush=True)
    except BaseException as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    run(args.experiment, args.plan)
