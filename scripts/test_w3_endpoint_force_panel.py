"""Real-geometry force axes, unavailable gaps and independent old/new validity."""
import hashlib
import numpy as np
from test_w3_endpoint_wrench import model
from w3_endpoint_force_panel import EndpointForcePanel


def fixture():
    m=model()
    cfg=dict(tool_frame='right_attachment_point',root_frame=m.root_frame,
        robot_description_sha256=hashlib.sha256(m.description.encode()).hexdigest(),history_points=10,
        solver=dict(characteristic_length_m=.4,damping=0.,max_condition=100.,max_relative_residual=.35,mode='full_wrench'))
    panel=EndpointForcePanel(cfg,dict(old='old',new='new'));panel.started=100.
    panel.description(m.description)
    q=np.deg2rad([10,20,10,-30,5,8,5]);pose,j=m.fk(q,True)
    tool=np.array([2.,-1.,.5,.2,-.3,.1]);rotation=pose[:3,:3]
    base=np.r_[rotation@tool[:3],rotation@tool[3:]];tau=j.T@base
    packet=dict(schema='w3_joint_lstm_comparison_v1',model_sha256=dict(old='old',new='new'),units='Nm',
        joint_names=m.names,same_input_history=True,stamp_ns=10_000_000_000,
        old_state='valid',new_state='valid',old_reason='ok',new_reason='ok',
        recording_frame=dict(q_deg=np.rad2deg(q).tolist(),old_filtered_nm=(2*tau).tolist(),new_filtered_nm=tau.tolist()))
    return panel,m,packet,tool


def test_known_tool_force_and_independent_old_new_curves():
    panel,m,p,tool=fixture();panel.ingest(p,100.);panel.tick(p['stamp_ns'],100.)
    for kind,scale in (('old',2.),('new',1.)):
        s=panel.snapshot(kind,100.);assert s['state']=='valid' and s['units']=='N'
        np.testing.assert_allclose([s['force'][k] for k in ('fx','fy','fz')],scale*tool[:3],atol=1e-12)
        np.testing.assert_allclose(s['force']['norm'],scale*np.linalg.norm(tool[:3]),atol=1e-12)
        assert s['frame_id']=='right_attachment_point' and s['source_stamp_ns']==p['stamp_ns']
    # A newer torque packet must not relabel a previously computed force timestamp.
    p=dict(p,stamp_ns=p['stamp_ns']+10_000_000);panel.ingest(p,100.01)
    assert panel.snapshot('new',100.01)['source_stamp_ns']==10_000_000_000


def test_singular_invalid_stale_and_model_mismatch_leave_gaps():
    panel,m,p,_=fixture();panel.ingest(p,100.);panel.tick(p['stamp_ns'],100.)
    p['old_state']='out_of_scope';p['old_reason']='outside_collection_range:right_joint_1'
    panel.tick(p['stamp_ns'],100.1)
    assert panel.snapshot('old',100.1)['force'] is None
    assert panel.snapshot('new',100.1)['force'] is not None
    p['old_state']='valid';p['recording_frame']['q_deg']=[0.]*7
    panel.tick(p['stamp_ns'],100.2)
    s=panel.snapshot('new',100.2)
    assert s['reason']=='near_singular' and s['force'] is None and s['history']['fx'][-1] is None
    panel.tick(p['stamp_ns'],100.3)
    assert panel.snapshot('new',100.3)['reason']=='paired_input_stale'
    panel.description(m.description+' ');panel.ingest(p,100.4);panel.tick(p['stamp_ns'],100.4)
    assert panel.snapshot('new',100.4)['reason']=='robot_description_mismatch'
    assert panel.snapshot('new',101.)['reason']=='endpoint_update_stale'


def test_non_endpoint_torque_is_rejected_and_does_not_modify_joint_values():
    panel,m,p,_=fixture();_,j=m.fk(np.deg2rad(p['recording_frame']['q_deg']),True)
    u,_,_=np.linalg.svd(j.T,full_matrices=True);tau=u[:,-1]
    p['recording_frame']['new_filtered_nm']=tau.tolist()
    panel.ingest(p,100.);panel.tick(p['stamp_ns'],100.)
    assert panel.snapshot('new',100.)['reason']=='inconsistent_with_tool_contact'
    assert panel.snapshot('old',100.)['state']=='valid'
    np.testing.assert_array_equal(p['recording_frame']['new_filtered_nm'],tau)
