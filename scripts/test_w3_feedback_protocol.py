import numpy as np
import pytest
from w3_feedback_protocol import plateau_stroke, build_plan, ROOT, DIRECTIONS
from w3_targeted_protocol import sample, bounds, validate


@pytest.mark.parametrize('distance,velocity,acceleration,jerk',[(2.4,.6,1.5,7.5),(6.,1.2,3.,15.),(6.,2.4,3.,15.)])
def test_exact_constant_velocity_and_limits(distance,velocity,acceleration,jerk):
    start=np.array([0,50,0,-55,0,0,0.]); direction=np.array([0,0,0,-1,0,0,0.])
    knots,plateau=plateau_stroke(start,direction,distance,velocity,acceleration,jerk)
    t=np.linspace(*plateau,100);q,v,a=sample(knots,t)
    np.testing.assert_allclose(v,direction[None,:]*np.full((100,1),velocity),atol=1e-10)
    np.testing.assert_allclose(a,0,atol=1e-9)
    np.testing.assert_allclose(bounds(knots)[2].max(1),[velocity,acceleration,jerk],atol=1e-8)
    np.testing.assert_allclose(knots[-1]['position_deg'],start+distance*direction)
    reverse,_=plateau_stroke(knots[-1]['position_deg'],-direction,distance,velocity,acceleration,jerk)
    np.testing.assert_allclose(reverse[-1]['position_deg'],start,atol=1e-10)
    assert plateau[1]-plateau[0]>=1.5-1e-10


def test_whole_plan_probes_before_trials_all_axes_both_directions_and_return():
    import json,yaml
    source=ROOT/'log/next_inertia_trial_20261016_v1'
    spec=json.loads((source/'test_spec.json').read_text())
    plan=build_plan((source/'robot.urdf').read_text(),spec['initial_command_deg'],spec['left_pose_deg'],spec['training_envelope'],'test_feedback')
    plan=yaml.load(yaml.dump(plan,Dumper=yaml.CSafeDumper),Loader=yaml.CSafeLoader);validate(plan)
    first=next(i for i,s in enumerate(plan['steps']) if s['kind'].startswith('diagnostic_'))
    assert plan['steps'][first-1]['kind']=='hold_after_probes'
    assert len([s for s in plan['steps'][:first] if s['family']=='speed_probe'])==7
    trials=[s for s in plan['steps'] if s['kind'].startswith('diagnostic_')]
    assert len(trials)==26
    for joint in range(7):
        group=[s['feedback_condition'] for s in trials if s['feedback_condition']['joint']==joint+1]
        assert {c['direction'] for c in group}=={'out','return'}
        assert all(c['signed_direction']==DIRECTIONS[joint]*(1 if c['direction']=='out' else -1) for c in group)
    np.testing.assert_allclose(plan['steps'][-1]['knots'][-1]['position_deg'],spec['initial_command_deg'],atol=1e-9)
    assert plan['feedback_protocol']['model_changes'] is False


def test_short_stroke_rejected_instead_of_silent_triangulation():
    with pytest.raises(ValueError,match='Insufficient'):plateau_stroke(np.zeros(7),np.array([1.,0,0,0,0,0,0]),.1,2.4,3.,15.)
