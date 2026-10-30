"""Independent optimization and validation selection for scalar torque models.

Input windows contain all joints. Each branch has its own optimizer, clipping,
validation minimum and early stopping. Test labels are never consumed here.
"""
from pathlib import Path

import numpy as np
import torch


def train_joint_epoch(model, loader, device, optimizers, active, scale, clip):
    model.train()
    total = np.zeros(len(model.joints), dtype=np.float64)
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for j, (branch, optimizer) in enumerate(zip(model.joints, optimizers)):
            if not active[j]:
                continue
            error = (branch(x)[:, 0] - y[:, j]) * scale[j]
            loss = error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError('joint_lstm_nonfinite_training_loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(branch.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            total[j] += float(loss.detach()) * len(x)
        count += len(x)
    if not count:
        raise ValueError('joint_lstm_empty_training')
    return [float(total[j] / count) if active[j] else None for j in range(len(total))]


def validate_joints(model, loader, device, scale):
    model.eval()
    squared = torch.zeros(len(model.joints), dtype=torch.float64, device=device)
    count = 0
    with torch.inference_mode():
        for x, y in loader:
            error = (model(x.to(device)) - y.to(device)) * scale
            squared += error.double().square().sum(0)
            count += len(x)
    values = (squared / max(count, 1)).cpu().numpy()
    if not count or not np.isfinite(values).all():
        raise ValueError('joint_lstm_invalid_validation')
    return values


class JointSelection:
    """Compose independently selected branches, including different epochs."""
    def __init__(self, count, early):
        self.best = np.full(count, np.inf)
        self.early_best = np.full(count, np.inf)
        self.stale = np.zeros(count, dtype=int)
        self.active = np.ones(count, dtype=bool)
        self.epochs = [None] * count
        self.stopped = [None] * count
        self.states = [None] * count
        self.early = early

    def update(self, model, values, epoch):
        changed = False
        for j, branch in enumerate(model.joints):
            if not self.active[j]:
                continue
            if values[j] < self.best[j]:
                self.best[j], self.epochs[j] = float(values[j]), epoch
                self.states[j] = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
                changed = True
            if values[j] < self.early_best[j] - self.early.get('min_delta', 1e-5):
                self.early_best[j], self.stale[j] = values[j], 0
            else:
                self.stale[j] += 1
            if (self.early.get('enabled', False) and epoch >= self.early.get('warmup', 10)
                    and self.stale[j] >= self.early.get('patience', 20)):
                self.active[j], self.stopped[j] = False, epoch
                branch.load_state_dict(self.states[j])
        return changed

    def state_dict(self):
        return {f'joints.{j}.{key}': value for j, state in enumerate(self.states)
                for key, value in state.items()}


def fit_joint_lstm(model, loaders, device, norm, cfg, metrics, on_best):
    settings = cfg['train']
    optimizer_cls = torch.optim.AdamW if settings.get('optimizer', 'adam') == 'adamw' else torch.optim.Adam
    optimizers = [optimizer_cls(branch.parameters(), lr=float(settings['learning_rate']),
        weight_decay=float(settings.get('weight_decay', 0))) for branch in model.joints]
    scale = torch.as_tensor(norm['y_std'] if settings.get('loss_units') == 'physical_nm'
                            else np.ones(len(model.joints)), dtype=torch.float32, device=device)
    selection = JointSelection(len(model.joints), settings.get('early_stopping', {}))
    metrics.update(selection_rule='independent per-joint minimum validation MSE; first epoch on ties',
                   per_joint_train_loss=[], per_joint_val_loss=[],
                   per_joint_best_epoch=selection.epochs, per_joint_stopped_epoch=selection.stopped,
                   optimizer_scope='one optimizer and gradient clip per joint',
                   test_used_for_fit=False)
    stop_file = settings.get('stop_file')
    if stop_file and Path(stop_file).exists():
        raise ValueError('joint_lstm_stop_file_already_exists')
    for epoch in range(1, int(settings['epochs']) + 1):
        tr = train_joint_epoch(model, loaders[0], device, optimizers, selection.active,
                               scale, settings.get('gradient_clip'))
        va = validate_joints(model, loaders[1], device, scale)
        metrics['per_joint_train_loss'].append(tr)
        metrics['per_joint_val_loss'].append(va.tolist())
        metrics['train_loss'].append(float(np.mean([v for v in tr if v is not None])))
        metrics['val_loss'].append(float(va.mean()))
        if settings.get('validation_diagnostics', False):
            from factr2_next.training.validation_metrics import validation_epoch
            _, diagnostic = validation_epoch(model, loaders[1], device, norm, settings.get('loss_units', 'normalized'))
            metrics['validation_diagnostics'].append(diagnostic)
        changed = selection.update(model, va, epoch)
        metrics['per_joint_best_validation_mse'] = selection.best.tolist()
        if changed:
            # best_epoch denotes the composite update, not a shared selected epoch.
            metrics['best_epoch'] = epoch
            on_best(selection.state_dict())
        print(f"joint LSTM epoch {epoch}: validation MSE={va.tolist()}; "
              f"selected epochs={selection.epochs}", flush=True)
        if stop_file and Path(stop_file).exists():
            metrics.update(stop_reason='user_requested', stopped_epoch=epoch)
            break
        if not selection.active.any():
            metrics.update(stop_reason='all_joints_early_stopped', early_stopped_epoch=epoch)
            break
    else:
        metrics['stop_reason'] = 'epoch_budget'
    model.load_state_dict(selection.state_dict())
    selected = validate_joints(model, loaders[1], device, scale)
    np.testing.assert_allclose(selected, selection.best, rtol=1e-5, atol=1e-8)
    metrics['selected_validation_mse'] = selected.tolist()
    return selection.state_dict()
