from collections import deque

import numpy as np


class HistoryBuffer:
    def __init__(self, history):
        self.history = int(history)
        if self.history < 1:
            raise ValueError("history_positive")
        self.rows = deque(maxlen=self.history)

    @property
    def ready(self):
        return len(self.rows) == self.history

    def append(self, joint_pos, joint_vel, joint_cmd):
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        joint_cmd = np.asarray(joint_cmd, dtype=np.float32)
        # Match the NEXT training input exactly: [q, qdot, q_cmd - q].
        from factr2_next.w3_samples import features
        if any(a.ndim != 1 or a.shape != joint_pos.shape or not np.isfinite(a).all() for a in (joint_pos,joint_vel,joint_cmd)):
            raise ValueError("history_shape_finite")
        self.rows.append(features(joint_pos,joint_vel,joint_cmd))

    def clear(self):
        self.rows.clear()

    def array(self):
        if not self.ready:
            raise ValueError(f"HistoryBuffer needs {self.history} rows, has {len(self.rows)}.")
        return np.stack(self.rows).astype(np.float32)
