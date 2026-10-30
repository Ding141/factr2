"""Frozen kinematics-only correction for the independent joint LSTM.

The correction is subtracted BEFORE the existing identical RC12/RC5 filters.
No measured torque or LSTM residual enters its features; no online adaptation.
"""
from pathlib import Path
import json
import numpy as np
from factr2_next.inference.kinematic_compensation import SignalProcessor, ResidualModel, features, digest


class JointResidual:
    def __init__(self, path, bundle):
        self.path=Path(path); self.bundle=Path(bundle)
        a=json.loads(self.path.read_text())
        if (a.get('mean_model_sha256')!=digest(self.bundle/'mean_checkpoint/model.pt')
                or a.get('application')!='subtract_before_identical_RC12_RC5'):
            raise ValueError('joint_residual_mean_or_filter_mismatch')
        self.model=ResidualModel.load(self.path,digest(self.bundle/'mean_checkpoint/metadata.json'))
        self.hash=digest(self.path);self.reset()

    def reset(self):
        self.signal=SignalProcessor(self.model.artifact['signal']);self.frame=None

    def update(self, stamp, q, v, cmd):
        # The two zero vectors are ignored by the feature computation. Torque
        # and model predictions are deliberately absent from this interface.
        self.frame=self.signal.update(int(stamp),q,v,cmd,np.zeros(7),np.zeros(7))
        return self.frame

    def correction(self, allow=True):
        if self.frame is None:return np.zeros(7,dtype=np.float32),False
        c,supported=self.model.predict(self.frame)
        supported=bool(supported and allow)
        return (c if supported else np.zeros(7)).astype(np.float32),supported
