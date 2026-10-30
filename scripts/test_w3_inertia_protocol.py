import numpy as np
import pytest
from w3_inertia_protocol import matched_stroke,ACCELERATIONS,DISTANCE,VELOCITY,JERK,PATTERNS
from w3_targeted_protocol import sample,bounds


@pytest.mark.parametrize('pattern',list(PATTERNS))
def test_strokes_have_same_path_speed_and_jerk_with_distinct_acceleration(pattern):
    direction=np.array(PATTERNS[pattern],float);start=np.array([0,50,0,-55,0,0,0.])
    for acceleration in ACCELERATIONS:
        knots=matched_stroke(start,direction,acceleration)
        q,v,a=sample(knots,np.linspace(0,knots[-1]['time_seconds'],10001))
        progress=(q-start)@direction/(direction@direction)
        np.testing.assert_allclose(q,start+progress[:,None]*direction,atol=1e-12)
        np.testing.assert_allclose(q[-1],start+DISTANCE*direction,atol=1e-12)
        assert np.min(np.diff(progress))>-1e-10
        np.testing.assert_allclose(np.max(abs(v)),VELOCITY,atol=1e-8)
        np.testing.assert_allclose(np.max(abs(a)),acceleration,atol=1e-8)
        np.testing.assert_allclose(bounds(knots)[2].max(axis=1),[VELOCITY,acceleration,JERK],atol=1e-8)
        np.testing.assert_allclose(v[[0,-1]],0,atol=1e-10);np.testing.assert_allclose(a[[0,-1]],0,atol=1e-10)
        reverse=matched_stroke(q[-1],-direction,acceleration)
        np.testing.assert_allclose(reverse[-1]['position_deg'],start,atol=1e-12)


def test_invalid_acceleration_is_rejected():
    with pytest.raises(ValueError,match='Unknown acceleration'):matched_stroke(np.zeros(7),[1,0,0,0,0,0,0],20.)


def test_full_plan_safe_yaml_roundtrip_and_balanced_conditions():
    import json,yaml
    from pathlib import Path
    from w3_inertia_protocol import build_plan,ROOT
    from w3_targeted_protocol import validate
    source=ROOT/'log/md_force_observation_20261016_223403'
    spec=json.loads((source/'test_spec.json').read_text())
    plan=build_plan((source/'robot.urdf').read_text(),spec['initial_command_deg'],
        list(np.asarray(spec['left_pose_deg'])),spec['training_envelope'],'test_inertia')
    decoded=yaml.safe_load(yaml.dump(plan,Dumper=yaml.CSafeDumper));validate(decoded)
    first=next(i for i,s in enumerate(decoded['steps']) if s['kind'].startswith('inertia_'))
    assert decoded['steps'][first-1]['kind']=='hold_after_probes'
    assert decoded['steps'][first-1]['dwell_seconds']==2.
    trials=[s['inertia_condition'] for s in decoded['steps'] if s['kind'].startswith('inertia_')]
    assert len(trials)==72
    for pattern in PATTERNS:
        for direction in ('out','return'):
            for acceleration in ACCELERATIONS:
                assert sum(c['pattern']==pattern and c['direction']==direction and c['acceleration_deg_s2']==acceleration for c in trials)==3
