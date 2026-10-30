import json

import numpy as np

from evaluate_w3_highspeed_candidate import replay
from fit_w3_kinematic_compensation import series,metrics
from test_w3_kinematic_compensation import artifact
from process_w3_continuous_campaign import mean_test_gate


def test_vectorized_and_stepwise_causal_replay_agree_for_nonzero_motion_and_torque():
    n=250;t=1_000_000_000+np.arange(n)*10_000_000
    x=np.zeros((n,21));x[:,0]=.05*np.sin(np.arange(n)*.03);x[:,7]=.02;x[:,14]=.01
    y=np.tile(np.linspace(.1,.7,7),(n,1));mu=y*.8
    a=artifact();item=series(t,x,y,mu,np.full(n,'one_recording'))
    _,v,_,_=metrics(item,np.ones(n,bool),a)
    output,_=replay(t,x,y,mu,a,[[-10000,10000]]*7)
    np.testing.assert_allclose(output,v['final'],atol=1e-12,rtol=1e-12)


def test_test_gate_preserves_incumbent_on_low_speed_regression(tmp_path):
    (tmp_path/'evaluation').mkdir()
    criteria=dict(minimum_new_highspeed_test_mse_improvement_fraction=.05,maximum_old_test_mse_ratio=1.15)
    (tmp_path/'deployment_criteria.json').write_text(json.dumps(criteria))
    metric=lambda mse:dict(physical_mse_nm2=mse,per_joint_rmse_nm=[.1]*7)
    report=dict(splits=dict(test=dict(INCUMBENT=dict(old=metric(1),high=metric(1)),NEXT_MSE=dict(old=metric(1.1),high=metric(.8)))))
    p=tmp_path/'evaluation/comparison.json';p.write_text(json.dumps(report))
    assert mean_test_gate(tmp_path)['passed']
    report['splits']['test']['NEXT_MSE']['old']['physical_mse_nm2']=1.2;p.write_text(json.dumps(report))
    assert not mean_test_gate(tmp_path)['passed']
