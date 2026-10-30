"""Dual-arm discovery regression; no ROS initialization or actuation."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from w3_coverage_motion import CoverageClient


def client():
    c = CoverageClient.__new__(CoverageClient)
    c.stationary_left_deg = np.zeros(7)
    c.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=1_050_000_000)))
    c.state = dict(ok=True, left_healthy=True)
    for key in ('q', 'cmd', 'health', 'left_q', 'left_cmd', 'left_health'):
        c.state[key] = np.zeros(7)
        c.state[key+'_received'] = 1.
        c.state[key+'_stamp'] = 1_000_000_000
    return c


def test_right_ready_does_not_accept_missing_or_stale_left_health():
    c = client()
    with patch('w3_motion_sequence.time.monotonic', return_value=1.05):
        assert c.ready()
        for key in ('left_q', 'left_cmd', 'left_health'):
            stamp = c.state.pop(key+'_stamp')
            assert not c.ready()
            c.state[key+'_stamp'] = stamp
        c.state['left_health_stamp'] = 700_000_000
        assert not c.ready()
        with pytest.raises(RuntimeError, match='left arm feedback/target stale'):
            c.guard_stationary_left()


def test_unhealthy_left_refuses_readiness_and_single_arm_is_unaffected():
    c = client()
    with patch('w3_motion_sequence.time.monotonic', return_value=1.05):
        c.state['left_healthy'] = False
        assert not c.ready()
        with pytest.raises(RuntimeError, match='left arm motor feedback unhealthy'):
            c.guard_stationary_left()
        c.stationary_left_deg = None
        assert c.ready()
        c.state['q_stamp'] = 700_000_000
        assert not c.ready()
