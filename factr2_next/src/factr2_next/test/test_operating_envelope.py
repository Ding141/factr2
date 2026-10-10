import numpy as np
import pytest
from factr2_next.inference.operating_envelope import OperatingEnvelope
from factr2_next.inference.inference_node import InferenceNode
from factr2_next.inference.history_buffer import HistoryBuffer


def test_out_of_scope_contact_suppression_and_full_window_recovery():
    n = InferenceNode.__new__(InferenceNode)
    n.names = ['right_joint_0', 'right_joint_3']
    n.operating_envelope = OperatingEnvelope(
        {'position_bounds_deg': [[-16, 16], [-40, -8]], 'position_margin_deg': 3}, 2)
    n.buffer = HistoryBuffer(50)
    n.contact_cfg = {'low_threshold': 1., 'high_threshold': 2.}
    n.contact_state = True
    for _ in range(50):
        n.buffer.append(np.zeros(2), np.zeros(2), np.zeros(2))
    assert not n._update_observation_state(10.)
    assert n.state == 'out_of_scope' and 'right_joint_3' in n.reason
    for _ in range(49):
        q = np.deg2rad([0., -24.])
        n.buffer.append(q, np.zeros(2), q)
    assert not n._update_observation_state(10.)
    n.buffer.append(q, np.zeros(2), q)
    assert n._update_observation_state(10.)
    assert n.state == 'valid'
    history = n.buffer.array()
    history[-1, 0] = np.nan
    assert n.operating_envelope.outside_joints(history) == [0]


@pytest.mark.parametrize('cfg', [
    {'position_bounds_deg': [[1, 0]]},
    {'position_bounds_deg': [[0, 1]], 'position_margin_deg': -1},
    {'position_bounds_deg': [[0, float('inf')]]},
    {'position_bounds_deg': [[0, 1], [0, 1]]},
])
def test_reject_invalid_envelope(cfg):
    with pytest.raises(ValueError):
        OperatingEnvelope(cfg, 1)


def test_optional_envelope_preserves_existing_profiles():
    assert OperatingEnvelope(None, 7).outside_joints(np.zeros((50, 21))) == []
