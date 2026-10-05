from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NextTorqueDataset(Dataset):
    def __init__(self, h5_paths, key_templates, history, arm=None, episodes="all"):
        self.history = int(history)
        if self.history < 1:
            raise ValueError("history must be >= 1.")
        self.arm = arm
        self.keys = {
            name: template.format(arm=arm) if arm is not None else template
            for name, template in key_templates.items()
        }
        self.x, self.y = self._load(h5_paths, episodes)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

    @property
    def input_size(self):
        return self.x.shape[-1]

    @property
    def output_size(self):
        return self.y.shape[-1]

    def _load(self, h5_paths, episodes):
        xs, ys = [], []
        for path in h5_paths:
            with h5py.File(Path(path).expanduser(), "r") as h5:
                for ep in self._episode_names(h5, episodes):
                    x_step, y_step = self._episode_arrays(h5[ep])
                    if len(x_step) < self.history:
                        continue
                    xs.append(self._windows(x_step))
                    # Paper Sec. 4, Eq. (3): each history window predicts tau_m
                    # at its final timestep, not at a future timestep.
                    ys.append(y_step[self.history - 1 :])

        if not xs:
            raise ValueError(
                "No training windows found. Check H5 paths, episodes, keys, and history."
            )
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)

    def _episode_arrays(self, episode):
        pos = self._read(episode, "joint_pos")
        vel = self._read(episode, "joint_vel")
        cmd = self._read(episode, "joint_cmd")
        torque = self._read(episode, "measured_joint_torque")
        n = len(pos)
        if any(len(a) != n for a in (vel, cmd, torque)):
            raise ValueError("stream_lengths: truncation is forbidden")
        # Paper Sec. 4, Eq. (3): x_i = [q, qdot, q_cmd - q] over history.
        from factr2_next.w3_samples import features
        x_step = features(pos, vel, cmd)
        # Paper Sec. 4: in D_free, tau_m supervises tau_free_hat.
        return x_step, torque[:n]

    def _read(self, episode, name):
        key = self.keys[name]
        if key not in episode:
            raise KeyError(f"Missing H5 key '{key}' for dataset field '{name}'.")
        return np.asarray(episode[key]["data"], dtype=np.float32)

    def _windows(self, x_step):
        return np.stack(
            [x_step[i : i + self.history] for i in range(len(x_step) - self.history + 1)]
        )

    def _episode_names(self, h5, episodes):
        if episodes == "all":
            return sorted(h5.keys())
        return list(episodes)


class ManifestDataset(Dataset):
    """Cache only episode steps; materialize one window per DataLoader access."""
    def __init__(self, entries, keys, history=50, memory_budget_mb=512):
        from factr2_next.w3_samples import features
        from factr2_next.data_collection.quality import require
        for entry in entries:
            require(entry['path'], entry['episodes'])
        self.history, self.keys = int(history), keys
        self.input_size, self.output_size = 21, 7
        self.episodes, self.ends, self.norm = [], [], None
        estimate = 0
        for entry in entries:
            with h5py.File(entry['path'], 'r') as h:
                for ep in entry['episodes']:
                    n = h[ep][keys['joint_pos']]['data'].shape[0]
                    estimate += n * 28 * 4
        if estimate > memory_budget_mb * 1024**2:
            raise ValueError(f'dataset_memory_budget: {estimate} bytes')
        self.step_bytes = estimate
        total = 0
        for entry in entries:
            with h5py.File(entry['path'], 'r') as h:
                for ep in entry['episodes']:
                    arrays = [np.asarray(h[ep][keys[k]]['data'], dtype=np.float32) for k in
                              ('joint_pos', 'joint_vel', 'joint_cmd', 'measured_joint_torque')]
                    if any(a.shape != (len(arrays[0]), 7) or not np.isfinite(a).all() for a in arrays):
                        raise ValueError('dataset_shape_finite_lengths')
                    x = features(*arrays[:3]); y = arrays[3]
                    windows = len(x) - self.history + 1
                    if windows <= 0:
                        raise ValueError('no_windows:' + ep)
                    self.episodes.append((x, y))
                    total += windows; self.ends.append(total)
        if not total:
            raise ValueError('no_windows')

    def __len__(self):
        return self.ends[-1]

    def raw_item(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = int(np.searchsorted(self.ends, index, side='right'))
        start = index - (self.ends[ep-1] if ep else 0)
        x, y = self.episodes[ep]
        return x[start:start+self.history], y[start+self.history-1]

    def __getitem__(self, index):
        x, y = self.raw_item(index)
        if self.norm is not None:
            x = (x-self.norm['x_mean'])/self.norm['x_std']
            y = (y-self.norm['y_mean'])/self.norm['y_std']
        return torch.from_numpy(np.array(x, dtype=np.float32)), torch.from_numpy(np.array(y, dtype=np.float32))

    def fit_normalization(self):
        # Each step has the same multiplicity as in eagerly stacked windows.
        sx, sy = np.zeros(21), np.zeros(7)
        count = 0
        weights = []
        for x, y in self.episodes:
            n = len(x); rows = np.arange(n)
            w = np.maximum(0, np.minimum(rows, n-self.history) - np.maximum(0, rows-self.history+1) + 1)
            weights.append(w)
            sx += (x.astype(np.float64)*w[:,None]).sum(axis=0)
            sy += y[self.history-1:].astype(np.float64).sum(axis=0)
            count += n-self.history+1
        mx, my = sx/(count*self.history), sy/count
        vx, vy = np.zeros(21), np.zeros(7)
        for (x,y),w in zip(self.episodes,weights):
            vx += (((x.astype(np.float64)-mx)**2)*w[:,None]).sum(axis=0)
            vy += ((y[self.history-1:].astype(np.float64)-my)**2).sum(axis=0)
        return {k:v.astype(np.float32) for k,v in dict(x_mean=mx,x_std=np.sqrt(vx/(count*self.history))+1e-6,
            y_mean=my,y_std=np.sqrt(vy/count)+1e-6).items()}
