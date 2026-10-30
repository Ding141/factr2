"""Deterministic sampling/holdout and actual position pause screen tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from train_w3_highspeed import groups,subset_indices,validation_eligible
from audit_w3_continuous_recording import longest_run


def test_fixed_balanced_sampling_keeps_every_target_in_its_original_group():
    ds=SimpleNamespace(ends=[1000,1500,1700],episode_info=[dict(speed='v06_a15'),
        dict(speed='v10_a20'),dict(speed='continuous_v12')])
    g=groups(ds);a=subset_indices(ds);b=subset_indices(ds)
    assert len(a[False])==len(a[True])==40
    for high in (False,True):
        np.testing.assert_array_equal(a[high],b[high])
        assert set(a[high])<=set(g[high])
    assert not set(a[False])&set(a[True])


def test_missing_new_or_old_stratum_cannot_silently_change_validation_objective():
    with pytest.raises(ValueError,match='Both old and new'):
        groups(SimpleNamespace(ends=[100],episode_info=[dict(speed='continuous_v12')]))


def test_pause_screen_counts_boundary_runs_and_does_not_join_separate_events():
    assert longest_run([True,True,False,True],.01)==pytest.approx(.02)
    assert longest_run([False]*100,.01)==0.
    assert longest_run([True]*31,.01)==pytest.approx(.31)


def test_lower_total_error_cannot_hide_a_regressing_wrist():
    import copy
    baseline={k:dict(physical_mse_nm2=.1,per_joint_rmse_nm=[.3]*4+[.03]*3) for k in ('old','high')}
    candidate=copy.deepcopy(baseline)
    candidate['high']['physical_mse_nm2']=.05
    assert validation_eligible(candidate,baseline)
    candidate['old']['per_joint_rmse_nm'][6]=.1
    assert not validation_eligible(candidate,baseline)


def test_anchored_loss_can_update_recurrent_weights_without_mutating_teacher():
    import copy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    from train_w3_highspeed import anchored_epoch
    class TinyMean(nn.Module):
        def __init__(self):
            super().__init__();self.lstm=nn.LSTM(3,4,batch_first=True);self.head=nn.Linear(4,7)
        def forward(self,x):return self.head(self.lstm(x)[0][:,-1]),None
    torch.manual_seed(0);model=TinyMean();teacher=copy.deepcopy(model).requires_grad_(False)
    before=model.lstm.weight_ih_l0.detach().clone()
    teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    data=TensorDataset(torch.randn(8,5,3),torch.randn(8,7),torch.tensor([False,True]*4))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
    baseline={k:dict(physical_mse_nm2=.1) for k in ('old','high')}
    anchored_epoch(model,teacher,DataLoader(data,batch_size=8),opt,{'y_std':np.ones(7)},baseline,1.)
    assert not torch.equal(before,model.lstm.weight_ih_l0)
    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,teacher_before[k],atol=0,rtol=0)
