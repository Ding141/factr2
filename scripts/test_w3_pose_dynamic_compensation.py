"""Nonlinear causal compensation regression tests; never initializes hardware."""
import numpy as np
import pytest

from factr2_next.inference.kinematic_compensation import (
    SCHEMA, DEFAULT_SIGNAL, POSE_MODES, augment_position_features, features,
    feature_names, Compensator, ResidualModel,
)


@pytest.mark.parametrize('mode',POSE_MODES)
def test_batch_features_match_single_causal_frame_and_have_unique_names(mode):
    rng=np.random.default_rng(42);a=rng.normal(size=(20,3,7));c=rng.normal(size=(20,3,7))
    base=np.array([features(x,y,'velocity_acceleration') for x,y in zip(a,c)])
    batch=augment_position_features(base,a[:,0],mode)
    single=np.array([features(x,y,mode) for x,y in zip(a,c)])
    np.testing.assert_array_equal(batch,single)
    names=feature_names(mode)
    assert batch.shape==(20,len(names)) and len(set(names))==len(names)
    np.testing.assert_array_equal(batch[:,:70],base)


def pose_artifact():
    mode=POSE_MODES[-1];n=len(feature_names(mode));weights=np.zeros((n,7))
    weights[70:77]=np.eye(7)*.4;weights[63:70]=np.eye(7)*.1
    return dict(schema=SCHEMA,feature_mode=mode,feature_names=feature_names(mode),
        feature_scale=[1.]*n,feature_min=[-100.]*n,feature_max=[100.]*n,
        weights=weights.tolist(),active_features=[True]*n,correction_cap_nm=[2.]*7,
        enabled_joints=[True]*7,position_bounds_rad=[[-3.,3.]]*7,signal=DEFAULT_SIGNAL,
        uses_torque_as_feature=False,online_learning=False,base_metadata_sha256='test')


def test_pose_correction_preserves_sustained_external_torque_and_causality():
    one,two=[Compensator(ResidualModel(pose_artifact())) for _ in range(2)]
    for k in range(400):
        q=np.full(7,.5+.1*np.sin(k*.02));v=np.zeros(7);t=1_000_000_000+k*10_000_000
        x=one.update(t,q,v,q,np.full(7,.2),v)
        y=two.update(t,q,v,q,np.full(7,.2+(.3 if k>=100 else 0)),v)
        np.testing.assert_array_equal(x['correction'],y['correction'])
        if k<100:np.testing.assert_array_equal(x['final'],y['final'])
    np.testing.assert_allclose(y['final']-x['final'],.3,atol=1e-10)
    assert np.all(abs(x['correction'])>.01)


def test_pose_model_out_of_scope_does_not_hide_large_external_torque():
    a=pose_artifact();a['position_bounds_rad']=[[-.1,.1]]*7
    p=Compensator(ResidualModel(a));q=np.full(7,.5);zero=np.zeros(7)
    for k in range(100):r=p.update(1_000_000_000+k*10_000_000,q,zero,q,np.full(7,4.),zero)
    np.testing.assert_array_equal(r['correction'],zero)
    np.testing.assert_allclose(r['final'],4.)
    assert not r['supported']


def nonlinear_artifact():
    a=pose_artifact()
    a['motion_gate_deg_s']=[1.,2.,4.,8.,1.,2.,4.]
    a['nonlinear_residual']=dict(activation='tanh',input_indices=[0,7,70],
        input_center=[0.,0.,0.],input_scale=[1.,1.,1.],layers=[
            dict(weight=[[.1,.2],[.3,.4],[.5,.6]],bias=[.01,.02]),
            dict(weight=[[.2]*7,[.3]*7],bias=[.01]*7)])
    return a


def test_nonlinear_batch_matches_scalar_and_zero_speed_preserves_output():
    m=ResidualModel(nonlinear_artifact());rng=np.random.default_rng(7)
    x=rng.normal(0,.1,(25,len(m.scale)))
    np.testing.assert_allclose(m.correction_from_features(x),
        np.array([m.correction_from_features(row) for row in x]),atol=1e-15)
    p=Compensator(m);q=np.full(7,.5);zero=np.zeros(7)
    for k in range(100):r=p.update(1_000_000_000+k*10_000_000,q,zero,q,np.full(7,.4),zero)
    np.testing.assert_array_equal(r['correction'],zero)
    np.testing.assert_allclose(r['final'],.4)


def test_nonlinear_does_not_absorb_a_fixed_kinematics_contact_step():
    one,two=[Compensator(ResidualModel(nonlinear_artifact())) for _ in range(2)]
    for k in range(400):
        q=np.full(7,.5+.1*np.sin(k*.02));v=np.zeros(7);t=1_000_000_000+k*10_000_000
        x=one.update(t,q,v,q,np.full(7,.2),v)
        y=two.update(t,q,v,q,np.full(7,.2+(.3 if k>=100 else 0)),v)
        np.testing.assert_array_equal(x['correction'],y['correction'])
        if k<100:np.testing.assert_array_equal(x['final'],y['final'])
    np.testing.assert_allclose(y['final']-x['final'],.3,atol=1e-10)


@pytest.mark.parametrize('bad',['nan','bad_shape','unsupported_input','gate'])
def test_malformed_network_is_rejected(bad):
    a=nonlinear_artifact()
    if bad=='nan':a['nonlinear_residual']['layers'][0]['bias'][0]=float('nan')
    elif bad=='bad_shape':a['nonlinear_residual']['layers'][1]['weight']=[[0.]*7]
    elif bad=='unsupported_input':a['active_features'][70]=False
    else:a['motion_gate_deg_s']=[1.,2.]
    with pytest.raises(ValueError):ResidualModel(a)


def test_speed_blend_is_convex_and_does_not_scale_the_measured_external_signal():
    import copy
    slow=nonlinear_artifact();fast=pose_artifact();a=copy.deepcopy(fast)
    a['weights']=np.zeros_like(a['weights']).tolist()
    a['speed_blend']=dict(slow=slow,fast=fast,transition_deg_s=[2.]*7)
    m=ResidualModel(a);x=np.zeros(len(m.scale));x[7:14]=.04;x[70:77]=.5
    s=ResidualModel(slow).correction_from_features(x);f=ResidualModel(fast).correction_from_features(x)
    actual=m.correction_from_features(x)
    assert np.all(actual>=np.minimum(s,f)-1e-12) and np.all(actual<=np.maximum(s,f)+1e-12)
    one,two=[Compensator(ResidualModel(a)) for _ in range(2)]
    for k in range(400):
        q=np.full(7,.5+.1*np.sin(k*.02));v=np.zeros(7);t=1_000_000_000+k*10_000_000
        r=one.update(t,q,v,q,np.full(7,.2),v);twin=two.update(t,q,v,q,np.full(7,.5),v)
        np.testing.assert_array_equal(r['correction'],twin['correction'])
    np.testing.assert_allclose(twin['final']-r['final'],.3,atol=1e-10)


def test_blend_rejects_different_base_or_nested_models():
    import copy
    a=pose_artifact();a['speed_blend']=dict(slow=nonlinear_artifact(),fast=pose_artifact(),transition_deg_s=2.)
    a['speed_blend']['fast']['base_metadata_sha256']='different'
    with pytest.raises(ValueError,match='share mean'):ResidualModel(a)
    b=pose_artifact();b['speed_blend']=dict(slow=copy.deepcopy(a),fast=pose_artifact(),transition_deg_s=2.)
    with pytest.raises(ValueError,match='nested'):ResidualModel(b)


def test_outside_nonlinear_support_uses_frozen_incumbent_instead_of_zero():
    import copy
    a=pose_artifact();fallback=copy.deepcopy(a)
    fallback['weights']=(2*np.array(fallback['weights'])).tolist()
    # Restrict only the primary posture feature; the prior model remains in domain.
    a['feature_min'][70]=-.1;a['feature_max'][70]=.1
    a['support_fallback']=fallback;m=ResidualModel(a)
    q=np.full(7,.5);z=np.zeros(7);f=dict(actual=np.stack([q,z,z]),command=np.stack([q,z,z]),ready=True)
    expected,okay=ResidualModel(fallback).predict(f);actual,supported=m.predict(f)
    assert okay and supported and np.max(abs(expected))>.1
    np.testing.assert_array_equal(actual,expected)
    x=features(f['actual'],f['command'],a['feature_mode'])
    batched,scope=m.predict_features(np.stack([x,x]),np.stack([q,q]),np.array([True,False]))
    np.testing.assert_array_equal(batched[0],expected);np.testing.assert_array_equal(batched[1],z)
    assert scope.tolist()==[True,False]


def test_fallback_cannot_change_mean_identity_or_chain_indefinitely():
    import copy
    a=pose_artifact();a['support_fallback']=copy.deepcopy(a)
    a['support_fallback']['base_metadata_sha256']='wrong'
    with pytest.raises(ValueError,match='fallback identity'):ResidualModel(a)
    b=pose_artifact();b['support_fallback']=copy.deepcopy(b)
    b['support_fallback']['support_fallback']=pose_artifact()
    with pytest.raises(ValueError,match='single frozen'):ResidualModel(b)
