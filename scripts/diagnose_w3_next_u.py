"""Exact-frame, retrospective diagnosis of the completed joint NEXT-U trial."""
import json
from pathlib import Path

import h5py
import numpy as np

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.training.coverage_audit import group_name


def aligned_states(ds, keys):
    xs, stamps, identities, kinds, directions = [], [], [], [], []
    for i, ((x, _), info) in enumerate(zip(ds.episodes, ds.episode_info)):
        rows = x[49:]
        with h5py.File(info["path"]) as h:
            t = h[info["episode"]+"/"+keys["joint_pos"]+"/timestamps"][49:]
        xs.append(rows)
        stamps.append(t)
        identities.append(np.full(len(t), i, np.int32))
        kinds.append(np.full(len(t), group_name(info)))
        directions.append(np.full(len(t), str(info.get("approach_direction") or "unspecified")))
    return dict(x=np.concatenate(xs), timestamp_ns=np.concatenate(stamps),
                episode_index=np.concatenate(identities), kind=np.concatenate(kinds),
                approach=np.concatenate(directions))


def joint_summary(base_error, u_error, u_variance, joint):
    a, b, v = (np.asarray(x, np.float64)[:, joint] for x in (base_error, u_error, u_variance))
    sigma = np.sqrt(v)
    delta = a-b  # mu_U - mu_NEXT; measured signal is identical.
    return dict(samples=len(a), next_bias_nm=float(a.mean()), u_bias_nm=float(b.mean()),
        next_rmse_nm=float(np.sqrt(np.mean(a*a))), u_rmse_nm=float(np.sqrt(np.mean(b*b))),
        next_abs_p95_nm=float(np.percentile(abs(a), 95)), u_abs_p95_nm=float(np.percentile(abs(b), 95)),
        mean_sigma_nm=float(sigma.mean()), nominal95_coverage=float(np.mean(abs(b) <= 1.959963984540054*sigma)),
        nominal95_mean_width_nm=float(np.mean(2*1.959963984540054*sigma)),
        standardized_rms=float(np.sqrt(np.mean(b*b/v))),
        mse_change_nm2=float(np.mean(b*b-a*a)),
        mean_shift_squared_nm2=float(np.mean(delta*delta)),
        minus_twice_error_shift_covariance_nm2=float(-2*np.mean(a*delta)))


def group_masks(states, train_x, joint):
    x = states["x"]
    for kind in np.unique(states["kind"]):
        yield "kind:"+kind, states["kind"] == kind
    for ep in np.unique(states["episode_index"]):
        yield f"episode:{ep}", states["episode_index"] == ep
    for direction in np.unique(states["approach"]):
        yield "approach:"+direction, states["approach"] == direction
    velocity = np.rad2deg(x[:, 7+joint])
    yield "velocity:negative_below_minus_0.1deg_s", velocity < -.1
    yield "velocity:near_zero_abs_le_0.1deg_s", abs(velocity) <= .1
    yield "velocity:positive_above_0.1deg_s", velocity > .1
    for lo, hi in ((0., .1), (.1, 2.), (2., 6.), (6., float("inf"))):
        yield f"abs_speed_deg_s:({lo},{hi}]", (abs(velocity) > lo) & (abs(velocity) <= hi)
    q_edges = np.unique(np.quantile(train_x[:, joint], [0., .25, .5, .75, 1.]))
    for i in range(len(q_edges)+1):
        lo = q_edges[i-1] if i else -np.inf
        hi = q_edges[i] if i < len(q_edges) else np.inf
        yield f"pose_rad:({lo:.8g},{hi:.8g}]", (x[:, joint] > lo) & (x[:, joint] <= hi)
    thresholds = np.unique(np.quantile(abs(train_x[:, 14+joint]), [.5, .9, .99]))
    for value in thresholds:
        yield f"abs_tracking_error_rad:above_train_quantile_{value:.8g}", abs(x[:, 14+joint]) > value
    # A label for the settled tail, not an assertion of true zero velocity.
    settled = np.zeros(len(x), bool)
    for ep in np.unique(states["episode_index"]):
        rows = np.flatnonzero((states["episode_index"] == ep) & (states["kind"] == "static"))
        settled[rows[-100:]] = True
    yield "static:last_at_most_1s_per_episode", settled
    # Causal sign changes above a declared deadband, reset at every episode.
    reversal = np.zeros(len(x), bool)
    for ep in np.unique(states["episode_index"]):
        rows = np.flatnonzero(states["episode_index"] == ep)
        previous = 0
        for offset, row in enumerate(rows):
            sign = int(np.sign(velocity[row])) if abs(velocity[row]) > .1 else 0
            if sign and previous and sign != previous:
                reversal[rows[offset:offset+50]] = True
            if sign:
                previous = sign
    yield "measured_sign_reversal:following_0.5s", reversal


def load_previous(parent, role, states):
    values = {}
    for arm in ("NEXT_MSE", "NEXT_U"):
        with np.load(parent/"evaluation"/f"seed_0_{role}_{arm}.npz", allow_pickle=False) as h:
            values[arm] = {k:h[k] for k in h.files}
        for key in ("timestamp_ns", "episode_index"):
            np.testing.assert_array_equal(values[arm][key], states[key])
    np.testing.assert_allclose(values["NEXT_MSE"]["mean_free_nm"]+values["NEXT_MSE"]["error_nm"],
                               values["NEXT_U"]["mean_free_nm"]+values["NEXT_U"]["error_nm"], atol=1e-6)
    return values


def diagnose(parent, output, cfg, train_ds, datasets, y_std):
    parent, output = Path(parent), Path(output)
    train_x = np.concatenate([x[49:] for x, _ in train_ds.episodes])
    bounds = np.stack((train_x.min(0), train_x.max(0)), axis=1)
    result = dict(scope="retrospective association, not proof of causality", input_bounds=bounds.tolist(),
        bound_scope="axis-aligned ranges of training-window final states; not an OOD certificate",
        grouping=dict(velocity_deadband_deg_s=.1, reversal_after_seconds=.5,
                      position_bins="training quantiles", tracking_bins="training absolute-error quantiles"), splits={})
    for role, ds in datasets.items():
        states = aligned_states(ds, cfg["data"]["keys"])
        old = load_previous(parent, role, states)
        a, b = old["NEXT_MSE"]["error_nm"], old["NEXT_U"]["error_nm"]
        v = old["NEXT_U"]["variance_free_nm2"]
        outside = (states["x"] < bounds[:, 0]) | (states["x"] > bounds[:, 1])
        split = dict(samples=len(a), outside_per_feature_fraction=outside.mean(0).tolist(),
                     outside_any_final_state_fraction=float(outside.any(1).mean()), joints={})
        vz = v / np.asarray(y_std)**2
        split["variance_bound_fractions"] = dict(
            floor_per_joint=(vz <= cfg["next_u"]["floor"]*(1+1e-3)).mean(0).tolist(),
            ceiling_per_joint=(vz >= cfg["next_u"]["ceiling"]*(1-1e-3)).mean(0).tolist())
        for joint in (0, 3):
            groups = {"overall": joint_summary(a, b, v, joint)}
            for label, mask in group_masks(states, train_x, joint):
                if mask.any():
                    groups[label] = joint_summary(a[mask], b[mask], v[mask], joint)
            split["joints"][f"J{joint+1}"] = groups
        result["splits"][role] = split
        if role == "test":
            plot_diagnosis(output/"previous_test_diagnosis.png", states, old)
    result["source_evaluation_sha256"] = sha256(parent/"evaluation/comparison.json")
    write_json(output/"previous_diagnosis.json", result)
    return result


def plot_diagnosis(path, states, old):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for col, j in enumerate((0, 3)):
        q = np.rad2deg(states["x"][:, j])
        e = old["NEXT_U"]["error_nm"][:, j]
        sigma = np.sqrt(old["NEXT_U"]["variance_free_nm2"][:, j])
        row = np.arange(len(q))
        axes[0, col].plot(row, q, lw=.6)
        axes[0, col].set_title(f"J{j+1}")
        axes[0, col].set_ylabel("Position (deg)")
        axes[1, col].fill_between(row, -1.959963984540054*sigma, 1.959963984540054*sigma, alpha=.2, label="U nominal 95%")
        axes[1, col].plot(row, old["NEXT_MSE"]["error_nm"][:, j], lw=.5, label="NEXT error")
        axes[1, col].plot(row, e, lw=.5, label="joint U error")
        axes[1, col].set_ylabel("Residual (Nm)")
        axes[1, col].legend(fontsize=8)
        axes[2, col].plot(row, abs(e)/sigma, lw=.5)
        axes[2, col].axhline(1.959963984540054, color="red", ls="--", lw=.8)
        axes[2, col].set_ylabel("Absolute standardized error")
        axes[2, col].set_xlabel("Ordered final-frame index (episodes separated)")
    boundaries = np.flatnonzero(np.diff(states["episode_index"]))+1
    for ax in axes.flat:
        for boundary in boundaries:
            ax.axvline(boundary, color="black", lw=.5, alpha=.3)
        ax.grid(alpha=.2)
    fig.suptitle("Previous joint mean/variance NEXT-U: retrospective aligned-frame diagnosis")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
