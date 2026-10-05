"""Deterministic contract tests, with separately controlled ROS/steady clocks."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from factr2_w3_adapter.validation import AdapterConfig, AdapterGate, joint_vectors

T = 10_000_000_000


def js(side='left', stamp=T, names=None, command=False):
    names = names or [f'{side}_joint_{i}' for i in range(7)]
    indices = [int(n.rsplit('_', 1)[-1]) if '_joint_' in n else 99 for n in names]
    return NS(header=NS(stamp=NS(sec=stamp // 10**9, nanosec=stamp % 10**9)),
              name=list(names), position=[float(i) for i in indices],
              velocity=[float(i + 10) for i in indices],
              effort=[] if command else [float(i + 20) for i in indices])


def inputs(g, stamp=T, mono=100.0):
    assert g.ingest('state', js(stamp=stamp), stamp, mono)
    assert g.ingest('command', js(stamp=stamp, command=True), stamp, mono)


def health(side='left', stamp=T, level=0):
    return NS(header=NS(stamp=NS(sec=stamp // 10**9, nanosec=stamp % 10**9)),
              status=[NS(name=f'factr2/w3/{side}', level=level)])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_ADP01_independent_orders_dual_arm_and_gripper(side):
    names = [f'{s}_joint_{i}' for s in ('left', 'right') for i in range(7)] + ['gripper']
    state = js(names=list(reversed(names)))
    command = js(names=names[5:] + names[:5], command=True)
    # Give the other arm different data to detect side mixing.
    for m in (state, command):
        for i, name in enumerate(m.name):
            if name.startswith('right_'):
                m.position[i] += 100
    g = AdapterGate(AdapterConfig(side))
    assert g.ingest('state', state, T, 100)
    assert g.ingest('command', command, T, 100)
    stamp, values = g.sample(T, 100)
    assert stamp == T
    assert values['joint_pos'] == ('position', tuple(float(i + (100 if side == 'right' else 0)) for i in range(7)))
    assert values['joint_vel'] == ('velocity', tuple(float(i + 10) for i in range(7)))
    assert values['joint_effort'] == ('effort', tuple(float(i + 20) for i in range(7)))
    assert values['joint_cmd'][1] == values['joint_pos'][1]


def test_ADP02_static_advancing_stamps_and_ADP07_no_resampling():
    g = AdapterGate(AdapterConfig('left'))
    inputs(g)
    assert g.sample(T, 100)
    assert g.sample(T + 20_000_000, 100.02) is None
    assert g.ingest('command', js(stamp=T+20_000_000, command=True), T+20_000_000, 100.02)
    assert g.sample(T+20_000_000, 100.02) is None
    inputs(g, T+40_000_000, 100.04)
    assert g.sample(T+40_000_000, 100.04)
    assert g.counts['outputs'] == 2


@pytest.mark.parametrize('kind,damage,reason', [
    ('state', lambda m: m.name.pop(), 'array_length'),
    ('state', lambda m: m.name.__setitem__(0, 'other_joint'), 'missing_joints'),
    ('state', lambda m: m.name.__setitem__(0, m.name[1]), 'duplicate_names'),
    ('state', lambda m: m.velocity.pop(), 'array_length'),
    ('state', lambda m: m.effort.pop(), 'array_length'),
    ('state', lambda m: m.position.__setitem__(0, float('nan')), 'nonfinite'),
    ('state', lambda m: m.effort.__setitem__(0, float('inf')), 'nonfinite'),
    ('command', lambda m: m.velocity.pop(), 'array_length'),
    ('command', lambda m: m.position.__setitem__(0, float('-inf')), 'nonfinite'),
    ('command', lambda m: m.effort.append(1.0), 'command_effort_not_empty'),
])
def test_ADP03_invalid_evicts_both_and_needs_both_new(kind, damage, reason):
    g = AdapterGate(AdapterConfig('left'))
    inputs(g)
    bad = js(stamp=T+10_000_000, command=kind == 'command')
    damage(bad)
    assert not g.ingest(kind, bad, T+10_000_000, 100.01)
    assert g.last_rejection.endswith(reason)
    assert g.counts['invalid'] == 1
    assert g.sample(T+10_000_000, 100.01) is None
    assert g.ingest('state', js(stamp=T+20_000_000), T+20_000_000, 100.02)
    assert g.sample(T+20_000_000, 100.02) is None
    assert g.ingest('command', js(stamp=T+20_000_000, command=True), T+20_000_000, 100.02)
    assert g.sample(T+20_000_000, 100.02)


def test_ADP04_missing_and_independent_ages():
    g = AdapterGate(AdapterConfig('left'))
    assert g.sample(T, 100) is None
    assert g.ingest('state', js(), T, 100)
    assert g.sample(T, 100) is None
    assert g.ingest('command', js(command=True), T, 100)
    assert g.sample(T, 100.251) is None  # ROS time stopped, steady receive timeout.
    assert g.counts['stale'] == 1
    g = AdapterGate(AdapterConfig('left'))
    inputs(g)
    assert g.sample(T+251_000_000, 100.001) is None  # source timeout independent of receipt.
    assert g.last_rejection == 'state_stale'


@pytest.mark.parametrize('stamp,reason,counter', [
    (0, 'zero_stamp', 'invalid'), (T, 'duplicate_stamp', 'duplicate'),
    (T-1, 'stamp_rollback', 'duplicate'), (T+1, 'future_stamp', 'invalid'),
    (T-300_000_000, 'source_stale', 'stale'),
])
def test_ADP05_stamp_rejection(stamp, reason, counter):
    g = AdapterGate(AdapterConfig('left'))
    inputs(g)
    assert not g.ingest('state', js(stamp=stamp), T, 100.01)
    assert reason in g.reason
    assert g.counts[counter] == 1
    assert g.sample(T, 100.01) is None


def test_ADP05_skew_and_clock_rollback_recovery():
    g = AdapterGate(AdapterConfig('left'))
    assert g.ingest('state', js(stamp=T), T+31_000_000, 100)
    assert g.ingest('command', js(stamp=T+31_000_000, command=True), T+31_000_000, 100)
    assert g.sample(T+31_000_000, 100) is None
    assert g.counts['skew'] == 1
    # A genuine ROS clock rollback resets all history and waits for two new sources.
    assert g.sample(T-1_000_000_000, 100.01) is None
    assert g.counts['clock_reset'] == 1
    inputs(g, T-1_000_000_000, 100.02)
    assert g.sample(T-1_000_000_000, 100.02)


def test_ADP08_health_gate_side_timeout_error_recovery():
    g = AdapterGate(AdapterConfig('left', require_hardware_health=True))
    inputs(g)
    assert g.sample(T, 100) is None
    assert g.ingest_health(health(side='right'), T, 100) is False
    assert g.sample(T, 100) is None
    assert g.ingest_health(health(stamp=T+1), T+1, 100.01)
    assert g.sample(T+1, 100.01) is None
    inputs(g, T+2, 100.02)
    assert g.sample(T+2, 100.02)
    assert not g.ingest_health(health(stamp=T+3, level=2), T+3, 100.03)
    assert g.sample(T+3, 100.03) is None
    inputs(g, T+4, 100.04)
    assert g.ingest_health(health(stamp=T+5), T+5, 100.05)
    assert g.sample(T+5, 100.05) is None  # does not revive the previous inputs
    inputs(g, T+6, 100.06)
    assert g.sample(T+6, 100.06)
    assert g.sample(T+7, 100.301) is None  # health receive timeout even fresh ROS time
    assert g.counts['health_rejected'] >= 3


@pytest.mark.parametrize('kwargs', [dict(side=''),dict(side='both'),dict(side='left',publish_hz=0),
    dict(side='left',publish_hz=float('nan')),dict(side='left',input_timeout_seconds=-1),
    dict(side='left',hardware_health_timeout_seconds=0),dict(side='left',max_source_skew_seconds=-1),
    dict(side='left',output_root='/factr2/right'),dict(side='left',health_topic='relative')])
def test_parameters_reject_at_startup(kwargs):
    with pytest.raises(ValueError):
        AdapterConfig(**kwargs)
