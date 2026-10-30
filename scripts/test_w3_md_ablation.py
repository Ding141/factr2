"""Causality, real-parent history, split isolation and frozen MD artifact tests."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from factr2_next.data_collection.quality import sha256, sidecar_path
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.models import build_model
from factr2_next.training.motion_discriminator import (
    MotionDiscriminator, MDDataset, PAPER_THRESHOLDS, stationary_floor, parent_features)
from evaluate_w3_md_ablation import error_metrics, load_md
from run_w3_md_ablation import check_parent_isolation, paired_initial_model


def test_algorithm1_stores_crossing_velocity_in_threshold_major_order():
    md=MotionDiscriminator([[.1,.1],[.3,.3]])
    np.testing.assert_allclose(md.step([.2,-.4],1),[.2,-.4,0,-.4])
    np.testing.assert_allclose(md.step([0,0],2),[.2,-.4,0,-.4])
    np.testing.assert_allclose(md.step([-.3,.1],3),[-.3,.1,-.3,-.4])


def test_memory_survives_hundreds_of_stationary_frames():
    md=MotionDiscriminator([[.1]*7]);md.step([.2]*7,1)
    memory=md.transform(np.zeros((700,7)),np.arange(1,701,dtype=np.int64)*10_000_000+1)
    np.testing.assert_allclose(memory,.2)


def test_future_velocity_does_not_change_prefix_and_sessions_are_independent():
    velocities=np.zeros((100,7));velocities[10]=.2
    stamps=np.arange(100,dtype=np.int64)*10_000_000+1
    a=MotionDiscriminator([[.1]*7]).transform(velocities,stamps)
    changed=velocities.copy();changed[60:]=-.4
    b=MotionDiscriminator([[.1]*7]).transform(changed,stamps)
    np.testing.assert_array_equal(a[:60],b[:60])
    np.testing.assert_array_equal(MotionDiscriminator([[.1]*7]).step(np.zeros(7),1),np.zeros(7))


def test_streaming_chunks_and_restored_state_equal_single_causal_pass():
    velocities=np.random.default_rng(3).normal(size=(100,7))*.2
    stamps=np.arange(100,dtype=np.int64)*10_000_000+1
    expected=MotionDiscriminator([[.1]*7,[.3]*7]).transform(velocities,stamps)
    md=MotionDiscriminator([[.1]*7,[.3]*7]);prefix=md.transform(velocities[:41],stamps[:41])
    other=MotionDiscriminator([[.1]*7,[.3]*7]);other.restore(json.loads(json.dumps(md.snapshot())))
    actual=np.r_[prefix,other.transform(velocities[41:],stamps[41:])]
    np.testing.assert_array_equal(expected,actual)
    with pytest.raises(ValueError,match='recipe_mismatch'):
        MotionDiscriminator([[.2]*7]).restore(md.snapshot())


def test_source_gap_resets_history_and_rejects_reordered_or_float_time():
    md=MotionDiscriminator([[.1]*7]);md.step([.2]*7,1)
    np.testing.assert_array_equal(md.step(np.zeros(7),50_000_002),np.zeros(7))
    assert not md.known.any()
    with pytest.raises(ValueError,match='nonmonotonic'):md.step(np.zeros(7),50_000_002)
    with pytest.raises(ValueError,match='integer'):md.step(np.zeros(7),50_000_003.)


def write_streams(path,values,stamps):
    with h5py.File(path,'w') as h:
        for key in ('joint_pos','joint_vel','joint_cmd','measured_joint_torque'):
            g=h.create_group('ep_0000/'+key)
            g.create_dataset('data',data=values if key=='joint_vel' else np.zeros_like(values))
            g.create_dataset('timestamps',data=stamps)


def test_parent_history_is_computed_before_derived_episode_selection(tmp_path):
    source=tmp_path/'raw.h5';derived=tmp_path/'derived.h5'
    v=np.zeros((200,7),dtype=np.float32);v[20]=.3
    stamps=np.arange(200,dtype=np.int64)*10_000_000+1
    write_streams(source,v,stamps);write_streams(derived,v[100:160],stamps[100:160])
    sidecar_path(derived).write_text(json.dumps(dict(derivation=dict(source_h5=str(source),source_sha256=sha256(source)),
        episodes=dict(ep_0000=dict(source_episode='ep_0000',source_row_start=100,source_row_stop_exclusive=160)))))
    entry=dict(path=str(derived),h5_sha256=sha256(derived),episodes=['ep_0000'])
    values,_=parent_features(entry,[[.1]*7])
    np.testing.assert_allclose(values['ep_0000'],.3)
    # A wrong per-supervised-episode reset would entirely lose this signal.
    np.testing.assert_array_equal(MotionDiscriminator([[.1]*7]).transform(v[100:160],stamps[100:160]),0)
    with h5py.File(derived,'r+') as h:h['ep_0000/joint_vel/data'][0,0]=.5
    entry['h5_sha256']=sha256(derived)
    with pytest.raises(ValueError,match='source_range_mismatch'):parent_features(entry,[[.1]*7])


def test_stationary_threshold_floor_cannot_fit_validation_and_does_not_copy_noise(tmp_path):
    path=tmp_path/'train.h5';v=np.full((600,7),.01,dtype=np.float32)
    write_streams(path,v,np.arange(600,dtype=np.int64)*10_000_000+1)
    with h5py.File(path,'r+') as h:h.create_dataset('ep_0000/labels/phase_offset_seconds',data=np.arange(600)*.01)
    meta=dict(training_split=dict(role='train'),episodes=dict(ep_0000=dict(kind='static_anchor0_pattern0_from_low',dwell_seconds=6)))
    sidecar_path(path).write_text(json.dumps(meta));entry=dict(path=str(path),episodes=['ep_0000'])
    recipe=stationary_floor([entry]);threshold=np.array(recipe['thresholds_rad_s'])
    assert min(recipe['stationary_samples_per_joint'])==300
    np.testing.assert_allclose(threshold,np.array(recipe['stationary_abs_velocity_p995_rad_s'])[None,:]+PAPER_THRESHOLDS[:,None])
    assert not MotionDiscriminator(threshold).step(v[0],1).any()
    meta['training_split']['role']='val';sidecar_path(path).write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='train_only'):stationary_floor([entry])


def test_extra_features_do_not_change_base_normalization_or_targets():
    rng=np.random.default_rng(0);episodes=[];original=[]
    for n in (61,93):
        x=rng.normal(size=(n,21)).astype(np.float32);y=rng.normal(size=(n,7)).astype(np.float32)
        original.append((x,y));episodes.append((np.c_[x,rng.normal(size=(n,105)).astype(np.float32)],y))
    base=ManifestDataset.__new__(ManifestDataset);base.episodes=original;base.history=50
    md=MDDataset.__new__(MDDataset);md.episodes=episodes;md.history=50;md.input_size=126
    a=base.fit_normalization();b=md.fit_normalization()
    for key in ('x_mean','x_std'):np.testing.assert_array_equal(a[key],b[key][:21])
    for key in ('y_mean','y_std'):np.testing.assert_array_equal(a[key],b[key])


def test_parent_ids_and_raw_hashes_must_be_disjoint_even_if_child_ids_differ():
    def item(identity,digest):return dict(metadata=dict(derivation=dict(parent_session_id=identity,parent_h5_sha256=digest)))
    check_parent_isolation(dict(train=[item('a','ha')],val=[item('b','hb')],test=[]))
    with pytest.raises(ValueError,match='duplicate_parent'):
        check_parent_isolation(dict(train=[item('a','ha')],val=[item('a','other')]))
    with pytest.raises(ValueError,match='duplicate_parent'):
        check_parent_isolation(dict(train=[item('a','ha')],test=[item('b','ha')]))


def test_md_model_checkpoint_roundtrip_checks_hash_and_dimension(tmp_path):
    cfg=dict(type='lstm',hidden_size=8,num_layers=2,head_hidden=8,head_layers=2,state_mode='stateless')
    model=build_model(cfg,126,7,50).eval()
    torch.save(dict(model_state_dict=model.state_dict(),model=cfg,input_size=126,history=50,output_size=7),tmp_path/'model.pt')
    np.savez(tmp_path/'normalization.npz',x_mean=np.zeros(126,np.float32),x_std=np.ones(126,np.float32),y_mean=np.zeros(7,np.float32),y_std=np.ones(7,np.float32))
    meta=dict(schema='next_md_ablation_v1',input_size=126,artifact_sha256={f:sha256(tmp_path/f) for f in ('model.pt','normalization.npz')})
    (tmp_path/'metadata.json').write_text(json.dumps(meta));loaded,_=load_md(tmp_path)
    x=torch.randn(2,50,126)
    torch.testing.assert_close(model(x),loaded(x))
    with (tmp_path/'normalization.npz').open('ab') as h:h.write(b'changed')
    with pytest.raises(ValueError,match='hash'):load_md(tmp_path)


def test_metrics_distinguish_constant_bias_from_noise():
    result=error_metrics(np.tile([.2,-.3],(100,1)))
    np.testing.assert_allclose(result['signed_bias_nm'],[.2,-.3])
    np.testing.assert_allclose(result['per_joint_std_nm'],0,atol=1e-14)
    assert result['physical_mse_nm2']==pytest.approx(.065)


def test_shared_initial_weights_and_minibatch_rng_match_historical_baseline():
    cfg=dict(type='lstm',hidden_size=8,num_layers=2,head_hidden=8,head_layers=2,state_mode='stateless')
    torch.manual_seed(0);original=build_model(cfg,21,7,50);expected_order=torch.randperm(100)
    torch.manual_seed(0);augmented=paired_initial_model(cfg);actual_order=torch.randperm(100)
    torch.testing.assert_close(actual_order,expected_order)
    for name,value in original.state_dict().items():
        actual=augmented.state_dict()[name]
        torch.testing.assert_close(actual[:,:21] if name=='lstm.weight_ih_l0' else actual,value)
