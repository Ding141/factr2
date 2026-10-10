"""Flag positions outside the collection workspace without hiding residuals."""
import numpy as np


class OperatingEnvelope:
    def __init__(self, cfg, joint_count):
        self.bounds = None
        if cfg is None:
            return
        bounds = np.asarray(cfg['position_bounds_deg'], dtype=float)
        margin = float(cfg.get('position_margin_deg', 0.0))
        if (bounds.shape != (joint_count, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])
                or not np.isfinite(margin) or margin < 0):
            raise ValueError('operating_envelope_position_bounds_or_margin')
        self.bounds = np.deg2rad(bounds + np.array([-margin, margin]))

    def outside_joints(self, history):
        if self.bounds is None:
            return []
        q = np.asarray(history)[:, :len(self.bounds)]
        outside = (~np.isfinite(q) | (q < self.bounds[:, 0])
                   | (q > self.bounds[:, 1])).any(axis=0)
        return np.flatnonzero(outside).tolist()
