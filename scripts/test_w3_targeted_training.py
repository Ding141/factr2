"""Whole-recording isolation and acceleration-specific hysteresis diagnostics."""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset,DataLoader
from factr2_next.training.coverage_audit import group_name
from factr2_next.training.validation_metrics import validation_epoch
from prepare_w3_targeted_training import check_assignment


def test_parent_identity_and_raw_hash_cannot_cross_splits_or_repeat():
    sessions={};hashes={};check_assignment(sessions,hashes,'recording_a','hash_a','train')
    with pytest.raises(ValueError):check_assignment(sessions,hashes,'recording_a','another_hash','val')
    with pytest.raises(ValueError):check_assignment(sessions,hashes,'another_id','hash_a','test')


@pytest.mark.parametrize('family', ['continuous','hysteresis'])
def test_targeted_grouping_distinguishes_single_multi_and_static(family):
    assert group_name(dict(family=family,active_joints=[2],kind='return'))=='single_joint'
    assert group_name(dict(family=family,active_joints=list(range(7)),kind='cycle'))=='multi_joint'
    assert group_name(dict(family=family,active_joints=[2],kind='static_anchor0_pattern2_from_low'))=='static'


def test_targeted_return_pairs_keep_acceleration_and_pattern_separate():
    class Ordered(Dataset):
        ends=list(range(2,18,2))
        episode_info=[dict(kind=f'static_anchor0_pattern{pattern}_from_{direction}',
            family='hysteresis',path='session',episode=str(i),phase='hysteresis',speed=speed)
            for i,(speed,pattern,direction) in enumerate((s,p,d) for s in ('v10_a20','v10_a40')
                for p in (0,1) for d in ('low','high'))]
        def __len__(self):return 16
        def __getitem__(self,i):
            # Distinct gaps at each profile and pattern; fake prediction is zero.
            ep=i//2;base=10*(ep//4)+4*((ep//2)%2);gap=1+ep//4+((ep//2)%2)
            return torch.zeros(1),torch.full((7,),float(base+(gap if ep%2 else 0)))
    class Zero(torch.nn.Module):
        def forward(self,x):return torch.zeros((len(x),7))
    loss,diag=validation_epoch(Zero(),DataLoader(Ordered(),batch_size=3),torch.device('cpu'),dict(y_std=np.ones(7)),'physical_nm')
    pairs=diag['matched_return_baselines'];assert len(pairs)==4 and np.isfinite(loss)
    gaps={(p['speed'],p['pattern_id']):p['baseline_gap_nm'][0] for p in pairs}
    assert gaps=={('v10_a20','0'):1.,('v10_a20','1'):2.,('v10_a40','0'):2.,('v10_a40','1'):3.}
