"""Variance-only learning on cached, deterministic NEXT hidden representations.

The mean model runs in eval/inference mode once. Its hidden states and residuals
are cached; no optimizer ever receives a parameter of the mean model.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from factr2_next.training.next_u import distribution_metrics, gaussian_nll

SCHEMA = "next_frozen_variance_v1"


def fixed_variance(error_z, floor, ceiling):
    """Zero-mean Gaussian MLE: mean squared residual, including mean bias."""
    a = np.asarray(error_z, np.float64)
    if a.ndim != 2 or a.shape[1] != 7 or not len(a) or not np.isfinite(a).all():
        raise ValueError("frozen_variance_training_residuals")
    return np.clip(np.mean(a*a, axis=0), floor*(1+1e-3), ceiling*(1-1e-3)).astype(np.float32)


class VarianceHead(nn.Module):
    def __init__(self, width, initial_variance_z, floor=1e-4, ceiling=1e4):
        super().__init__()
        if not 0 < floor < ceiling or not math.isfinite(ceiling):
            raise ValueError("frozen_variance_bounds")
        initial = torch.as_tensor(initial_variance_z, dtype=torch.float32)
        if initial.shape != (7,) or not torch.isfinite(initial).all() or not torch.all((initial > floor) & (initial < ceiling)):
            raise ValueError("frozen_variance_initial")
        self.floor, self.ceiling = float(floor), float(ceiling)
        self.linear = nn.Linear(int(width), 7)
        with torch.no_grad():
            self.linear.weight.zero_()
            value = initial - self.floor
            self.linear.bias.copy_(value + torch.log(-torch.expm1(-value)))

    def forward(self, hidden):
        return (F.softplus(self.linear(hidden)) + self.floor).clamp(max=self.ceiling)


def hidden_normalization(hidden):
    a = np.asarray(hidden, np.float64)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError("frozen_variance_hidden")
    return dict(mean=a.mean(0).astype(np.float32), std=(a.std(0)+1e-6).astype(np.float32))


def cache_mean(model, norm, dataset, batch_size):
    """One inference pass, no training-time dropout, same mean arithmetic as NEXT."""
    from torch.utils.data import DataLoader
    model.eval()
    model.requires_grad_(False)
    dataset.norm = norm
    hidden, errors, means = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    scale, center = torch.as_tensor(norm["y_std"]), torch.as_tensor(norm["y_mean"])
    with torch.inference_mode():
        for x, target_z in loader:
            sequence, _ = model.base.lstm(x)
            h = model.base.head[:-1](sequence[:, -1])
            mu = model.base.head[-1](h)
            hidden.append(h.numpy().copy())
            errors.append((target_z-mu).numpy().copy())
            means.append((mu*scale+center).numpy().copy())
    return dict(hidden=np.concatenate(hidden), error_z=np.concatenate(errors),
                mean_free_nm=np.concatenate(means))


def head_variance(head, cache, normalization, batch_size=4096):
    values = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(cache["hidden"]), batch_size):
            x = (cache["hidden"][start:start+batch_size]-normalization["mean"])/normalization["std"]
            values.append(head(torch.from_numpy(x.astype(np.float32))).numpy())
    return np.concatenate(values)


def variance_metrics(cache, variance_z, y_std, floor=1e-4, ceiling=1e4):
    variance_z = np.broadcast_to(variance_z, cache["error_z"].shape)
    scale = np.asarray(y_std, np.float32)
    error, variance = cache["error_z"]*scale, variance_z*scale**2
    result = distribution_metrics(error, variance)
    result.update(per_joint_nll=np.mean(.5*(np.log(2*np.pi*variance.astype(np.float64))
                    + error.astype(np.float64)**2/variance), axis=0).tolist(),
                  per_joint_floor_fraction=(variance_z <= floor*(1+1e-3)).mean(0).tolist(),
                  per_joint_ceiling_fraction=(variance_z >= ceiling*(1-1e-3)).mean(0).tolist())
    return result


def fit_head(train_cache, val_cache, y_std, recipe, on_epoch=None, should_stop=None):
    """Selection uses validation NLL, including the fixed-sigma epoch-0 control."""
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(recipe["seed"])
    torch.use_deterministic_algorithms(True)
    norm = hidden_normalization(train_cache["hidden"])
    initial = fixed_variance(train_cache["error_z"], recipe["floor"], recipe["ceiling"])
    head = VarianceHead(train_cache["hidden"].shape[1], initial, recipe["floor"], recipe["ceiling"])
    tx = ((train_cache["hidden"]-norm["mean"])/norm["std"]).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(train_cache["error_z"]))
    loader = DataLoader(ds, batch_size=recipe["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(recipe["seed"]))
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["learning_rate"],
                                 weight_decay=recipe["weight_decay"])
    scale = torch.as_tensor(y_std, dtype=torch.float32)
    def validate():
        v = head_variance(head, val_cache, norm, recipe["batch_size"])
        return variance_metrics(val_cache, v, y_std, recipe["floor"], recipe["ceiling"])
    first = validate()
    best, best_epoch = first["marginal_nll"], 0
    state = {k:v.detach().clone() for k,v in head.state_dict().items()}
    history = [dict(epoch=0, train_nll=None, validation=first)]
    early_best, stale = best, 0
    stop_reason = "epoch_budget"
    for epoch in range(1, recipe["epochs"]+1):
        head.train()
        total = 0.
        for x, error_z in loader:
            v = head(x)
            loss = gaussian_nll(error_z*scale, v*scale.square()).mean()
            if not torch.isfinite(loss):
                raise ValueError("frozen_variance_nonfinite_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(x)
        metric = validate()
        nll = metric["marginal_nll"]
        history.append(dict(epoch=epoch, train_nll=total/len(ds), validation=metric))
        if nll < best:
            best, best_epoch = nll, epoch
            state = {k:v.detach().clone() for k,v in head.state_dict().items()}
        if nll < early_best-recipe["min_delta"]:
            early_best, stale = nll, 0
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(history[-1], best_epoch)
        if should_stop is not None and should_stop():
            stop_reason = "user_stop"
            break
        if epoch >= recipe["warmup"] and stale >= recipe["patience"]:
            stop_reason = "validation_nll_patience"
            break
    head.load_state_dict(state)
    head.eval()
    return head, norm, initial, dict(best_epoch=best_epoch, best_validation_nll=best,
        epochs=history, stop_reason=stop_reason, selection="minimum validation marginal NLL, including epoch 0; first on ties")
