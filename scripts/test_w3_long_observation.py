import copy
import json
from pathlib import Path

import numpy as np
import pytest

from w3_long_observation import build_plan,PROFILES
from w3_targeted_protocol import validate,bounds
from test_w3_coverage_motion import fixture_description


@pytest.fixture(scope='module')
def plan():
    root=Path(__file__).resolve().parents[1]
    envelope=json.loads((root/'models/w3/right/targeted_candidate_20261016/metadata.json').read_text())['operating_envelope']
    return build_plan(fixture_description(),[0,50,0,-55,0,0,0],[0]*7,envelope)


def test_long_motion_returns_home_with_bounded_native_polynomials(plan):
    assert sum(s['duration_seconds'] for s in plan['steps'])>=600
    np.testing.assert_allclose(plan['steps'][-1]['positions_deg'][-1],plan['initial_deg'],atol=1e-8)
    q=np.concatenate([s['positions_deg'] for s in plan['steps']])
    assert np.ptp(q[:,0])>=16 and np.ptp(q[:,1])>=9 and np.ptp(q[:,3])>=7
    envelope=np.array(plan['config']['working_bounds_deg'])
    for s in plan['steps']:
        low,high,peak=bounds(s['knots'])
        assert np.all(low>=envelope[:,0]) and np.all(high<=envelope[:,1])
        limits=np.array([s['profile'][k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])
        assert np.all(peak<=limits[:,None]+1e-7)


def test_every_profile_has_all_probes_before_larger_motion(plan):
    for p in PROFILES:
        same=[s for s in plan['steps'] if s['speed']==p['id']]
        first=next(i for i,s in enumerate(same) if s['family']=='continuous')
        assert {j for s in same[:first] if s['family']=='speed_probe' for j in s['active_joints']}==set(range(7))
        assert all(s['dwell_seconds'] is None for s in same if s['family']=='continuous')
    assert max(p['velocity_deg_s'] for p in plan['config']['profiles'])==6


def test_tampered_long_motion_cannot_bypass_probe_gate(plan):
    changed=copy.deepcopy(plan)
    changed['steps']=[s for s in changed['steps'] if not(s['family']=='speed_probe' and s['active_joints']==[6])]
    with pytest.raises(ValueError,match='Missing speed probe'):validate(changed)
