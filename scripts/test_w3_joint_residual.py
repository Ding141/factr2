"""Correction cannot see injected torque; output filters retain unity DC gain."""
import json
from pathlib import Path
import numpy as np
import pytest
from test_w3_kinematic_compensation import artifact
from w3_joint_residual import JointResidual
from factr2_next.inference.kinematic_compensation import digest
from factr2_next.inference.joint_lstm_observer import ResidualCascade

def test_frozen_correction_contact_transfer_and_gap_reset(tmp_path):
    bundle=Path(__file__).resolve().parents[1]/'models/w3/right/joint_lstm_observer_20261028_v1'
    a=artifact(digest(bundle/'mean_checkpoint/metadata.json'))
    a.update(mean_model_sha256=digest(bundle/'mean_checkpoint/model.pt'),application='subtract_before_identical_RC12_RC5')
    p=tmp_path/'model.json';p.write_text(json.dumps(a));branch=JointResidual(p,bundle)
    before,after=ResidualCascade(),ResidualCascade();delta=[]
    for k in range(500):
        q=np.full(7,.02*np.sin(k*.01));branch.update(1_000_000_000+k*10_000_000,q,np.zeros(7),q+.001)
        c,active=branch.correction();tau=np.zeros(7) if k<100 else np.full(7,.3)
        delta.append(after.update(tau-c)-before.update(-c))
    np.testing.assert_allclose(delta[-1],.3,atol=1e-6)
    assert 0<delta[100][0]<.3
    branch.update(9_000_000_000,q,np.zeros(7),q)
    c,active=branch.correction();assert not active;np.testing.assert_array_equal(c,0.)
    a['mean_model_sha256']='wrong';p.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='mismatch'):JointResidual(p,bundle)
