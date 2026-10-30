"""Offline NEXT-U ablation. Variance describes reported free torque, in Nm².

The original NEXT mean path, feature order and stateless window are unchanged.
This is distribution learning on NEXT, not a momentum observer or MOB-Net.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from factr2_next.training.models import build_model

SCHEMA = "next_u_ablation_v1"
ARMS = ("NEXT_MSE", "NEXT_U")
DEFAULT_VARIANCE = dict(floor=1e-4, ceiling=1e4, initial=1.0)


class NextU(nn.Module):
    """Original LSTM/head with a parallel linear variance output.

    Both experimental arms have this architecture. NEXT_MSE freezes the extra
    output; it cannot affect its mean path. No extra dropout or recurrent state.
    Initialization of the variance head does not consume the mean/dropout RNG.
    """
    def __init__(self, model_cfg, variance=None):
        super().__init__()
        if (model_cfg.get("type", "lstm") != "lstm"
                or model_cfg.get("state_mode", "stateless") != "stateless"
                or model_cfg.get("bidirectional", False)):
            raise ValueError("next_u_requires_stateless_unidirectional_lstm")
        self.variance = dict(DEFAULT_VARIANCE if variance is None else variance)
        lo, hi, initial = (float(self.variance[k]) for k in ("floor", "ceiling", "initial"))
        if not all(math.isfinite(v) for v in (lo, hi, initial)) or not 0 < lo < initial < hi:
            raise ValueError("next_u_variance_bounds")
        self.base = build_model(model_cfg, 21, 7, 50)
        width = self.base.head[-1].in_features
        with torch.random.fork_rng(devices=[]):
            self.variance_head = nn.Linear(width, 7)
            nn.init.zeros_(self.variance_head.weight)
            value = initial - lo
            nn.init.constant_(self.variance_head.bias, value + math.log(-math.expm1(-value)))

    def forward(self, x):
        sequence, _ = self.base.lstm(x)
        hidden = self.base.head[:-1](sequence[:, -1])
        mean = self.base.head[-1](hidden)
        variance = (F.softplus(self.variance_head(hidden)) + self.variance["floor"])
        return mean, variance.clamp(max=self.variance["ceiling"])

    def configure_arm(self, arm):
        if arm not in ARMS:
            raise ValueError("next_u_unknown_arm")
        self.variance_head.requires_grad_(arm == "NEXT_U")
        return self


def physical_distribution(mean_z, variance_z, norm):
    scale = torch.as_tensor(norm["y_std"], dtype=mean_z.dtype, device=mean_z.device)
    center = torch.as_tensor(norm["y_mean"], dtype=mean_z.dtype, device=mean_z.device)
    return mean_z * scale + center, variance_z * scale.square()


def gaussian_nll(error_nm, variance_nm2):
    """Full marginal Gaussian NLL, mean over samples and joints (Nm reference)."""
    return .5 * (math.log(2 * math.pi) + variance_nm2.log()
                 + error_nm.square() / variance_nm2)


def objective(arm, mean_z, variance_z, target_z, scale):
    error = (target_z - mean_z) * scale
    if arm == "NEXT_MSE":
        # Keep the original run_epoch arithmetic as well as its objective.
        return F.mse_loss(mean_z * scale, target_z * scale)
    if arm == "NEXT_U":
        return gaussian_nll(error, variance_z * scale.square()).mean()
    raise ValueError("next_u_unknown_arm")


def train_epoch(model, loader, optimizer, norm, arm, gradient_clip):
    model.train()
    scale = torch.as_tensor(norm["y_std"], dtype=torch.float32)
    total = 0.
    for x, y in loader:
        mean, variance = model(x)
        loss = objective(arm, mean, variance, y, scale)
        if not torch.isfinite(loss):
            raise ValueError("next_u_nonfinite_loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], gradient_clip,
            error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(x)
    return total / len(loader.dataset)


def error_metrics(error):
    a = np.asarray(error, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 7 or not len(a) or not np.isfinite(a).all():
        raise ValueError("next_u_metric_shape_finite")
    return dict(samples=len(a), physical_mse_nm2=float(np.mean(a*a)),
                signed_bias_nm=a.mean(0).tolist(), per_joint_std_nm=a.std(0).tolist(),
                per_joint_mae_nm=np.abs(a).mean(0).tolist(),
                per_joint_rmse_nm=np.sqrt((a*a).mean(0)).tolist(),
                per_joint_abs_p95_nm=np.percentile(abs(a), 95, axis=0).tolist())


def distribution_metrics(error, variance):
    a, v = np.asarray(error, np.float64), np.asarray(variance, np.float64)
    if a.shape != v.shape or not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("next_u_metric_variance")
    result = error_metrics(a)
    sigma = np.sqrt(v)
    z = a / sigma
    result.update(marginal_nll=float(np.mean(.5 * (np.log(2*np.pi*v) + a*a/v))),
                  per_joint_mean_sigma_nm=sigma.mean(0).tolist(),
                  per_joint_standardized_bias=z.mean(0).tolist(),
                  per_joint_standardized_rms=np.sqrt((z*z).mean(0)).tolist(),
                  nominal_gaussian_intervals={})
    for label, quantile in (("68.27", 1.), ("95", 1.959963984540054), ("99", 2.5758293035489004)):
        result["nominal_gaussian_intervals"][label] = dict(
            per_joint_coverage=(np.abs(z) <= quantile).mean(0).tolist(),
            per_joint_mean_width_nm=(2 * quantile * sigma).mean(0).tolist())
    return result


def predict_dataset(model, norm, dataset, batch_size):
    """Preserves window order and final-timestep alignment; no EMA/zeroing."""
    from torch.utils.data import DataLoader
    dataset.norm = norm
    model.eval()
    errors, means, variances = [], [], []
    # DataLoader consumes its own RNG, including during validation.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        for x, y in loader:
            mean, variance = model(x)
            physical_mean, physical_variance = physical_distribution(mean, variance, norm)
            errors.append(((y - mean) * torch.as_tensor(norm["y_std"])).numpy())
            means.append(physical_mean.numpy())
            variances.append(physical_variance.numpy())
    return dict(error_nm=np.concatenate(errors), mean_free_nm=np.concatenate(means),
                variance_free_nm2=np.concatenate(variances))
