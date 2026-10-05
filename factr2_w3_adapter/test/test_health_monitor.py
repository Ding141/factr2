from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from factr2_w3_adapter.health import HealthGate, motor_health

T = 10_000_000_000


def raw(sides=('left', 'right'), stamp=T):
    motors = [NS(channel=0 if side == 'left' else 1, motor_index=i, position=0.1*i,
                 velocity=0.0, torque=1.0, online=True, enabled=True, error_flags=1)
              for side in sides for i in range(7)]
    return NS(header=NS(stamp=NS(sec=stamp//10**9, nanosec=stamp%10**9)), motors=motors)


@pytest.mark.parametrize('damage,key', [
    (lambda m: m.motors.pop(0), 'missing_slots'),
    (lambda m: m.motors.append(deepcopy(m.motors[0])), 'duplicate_slots'),
    (lambda m: setattr(m.motors[0], 'online', False), 'offline_slots'),
    (lambda m: setattr(m.motors[0], 'enabled', False), 'disabled_slots'),
    (lambda m: setattr(m.motors[0], 'error_flags', 8), 'fault_slots'),
    (lambda m: setattr(m.motors[0], 'torque', float('nan')), 'nonfinite_slots'),
])
def test_ADP09_bad_left_does_not_affect_right(damage, key):
    msg = raw()
    damage(msg)
    g = HealthGate()
    assert g.ingest(msg, T, 100)
    values = g.assess(T, 100)
    assert not values['left']['healthy']
    assert values['left'][key] == [0]
    assert values['right']['healthy']


def test_ADP09_enabled_ERR1_and_right_only_channel1():
    msg = raw(('right',))
    g = HealthGate(sides=('right',))
    assert g.ingest(msg, T, 100)
    assert g.assess(T, 100)['right']['healthy']
    assert g.assess(T, 100)['right']['channel'] == 1
    assert not motor_health(msg.motors, 'left')['healthy']


def test_raw_timeout_cannot_be_hidden_by_new_diagnostic_header():
    g = HealthGate()
    assert not g.assess(T, 100)['left']['healthy']
    assert g.ingest(raw(), T, 100)
    assert g.assess(T, 100)['left']['healthy']
    v = g.assess(T+260_000_000, 100.26)['left']
    assert not v['healthy'] and v['raw_source_stamp_ns'] == T and v['reason'] == 'raw_stale'
    assert g.assess(T, 100.27)['left']['healthy'] is False  # Clock rollback clears raw.


@pytest.mark.parametrize('stamp', [0, T-300_000_000, T+1])
def test_invalid_raw_source_stamp(stamp):
    g = HealthGate()
    assert not g.ingest(raw(stamp=stamp), T, 100)
    assert not g.assess(T, 100)['left']['healthy']


def test_repeated_raw_does_not_refresh_receive_time():
    g = HealthGate()
    assert g.ingest(raw(), T, 100)
    assert not g.ingest(raw(), T+20_000_000, 100.02)
    assert not g.assess(T+20_000_000, 100.02)['left']['healthy']


@pytest.mark.parametrize('kwargs', [dict(timeout_seconds=0),dict(timeout_seconds=float('inf')),
                                   dict(sides=()),dict(sides=('right','right')),dict(sides=('both',))])
def test_health_parameters(kwargs):
    with pytest.raises(ValueError):
        HealthGate(**kwargs)
