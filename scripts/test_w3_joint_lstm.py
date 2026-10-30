"""Independent-joint training, selection and production-loader roundtrip."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from factr2_next.training.models import build_model, JointLSTMRegressor
from factr2_next.training.joint_lstm import train_joint_epoch, JointSelection
from factr2_next.training.w3_train import train
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.data_collection.quality import sha256, sidecar_path, write_json
from factr2_next.w3_config import validate_w3_config
from test_w3_next_u_ablation import fixture_campaign

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(type='joint_lstm', state_mode='stateless', hidden_size=8, num_layers=2,
             head_hidden=8, head_layers=2, dropout=0., bidirectional=False)
torch.set_num_threads(1)


def test_independent_weights_all_joint_inputs_and_output_order():
    model = build_model(SMALL, 21, 7, 50).eval()
    x = torch.randn(2, 50, 21, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 7)
    pointers = [{p.data_ptr() for p in branch.parameters()} for branch in model.joints]
    assert sum(map(len, pointers)) == len(set.union(*pointers))
    for j, branch in enumerate(model.joints):
        torch.testing.assert_close(out[:, j:j+1], branch(x), atol=0, rtol=0)
    out[:, 0].sum().backward()
    assert torch.all(x.grad.abs().sum((0, 1)) > 0)  # J1 sees every joint's state.
    assert any(p.grad.abs().sum() > 0 for p in model.joints[0].parameters())
    for branch in model.joints[1:]:
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in branch.parameters())


def test_stateless_and_invalid_input():
    model = build_model(SMALL, 21, 7, 50).eval()
    x = torch.randn(2, 50, 21)
    first = model(x)
    model(torch.randn_like(x) * 50)
    torch.testing.assert_close(model(x), first, rtol=0, atol=0)
    for bad in (torch.zeros(2, 50, 3), torch.zeros(50, 21), torch.zeros(2, 0, 21)):
        with pytest.raises(ValueError, match='all_features'):
            model(bad)
    with pytest.raises(NotImplementedError):
        build_model(dict(SMALL, state_mode='stateful'), 21, 7, 50)


def test_large_error_in_one_joint_does_not_change_other_optimizer_updates():
    torch.manual_seed(4)
    a = build_model(SMALL, 21, 7, 50)
    b = copy.deepcopy(a)
    x, y = torch.randn(8, 50, 21), torch.randn(8, 7)
    shifted = y.clone(); shifted[:, 0] += 1e6
    for model, labels in ((a, y), (b, shifted)):
        opts = [torch.optim.AdamW(branch.parameters(), lr=.001) for branch in model.joints]
        train_joint_epoch(model, DataLoader(TensorDataset(x, labels), batch_size=4),
                          'cpu', opts, [True]*7, torch.arange(1., 8.), 1.)
    for j in range(1, 7):
        for key, value in a.joints[j].state_dict().items():
            torch.testing.assert_close(value, b.joints[j].state_dict()[key], rtol=0, atol=0)
    assert any(not torch.equal(v, b.joints[0].state_dict()[k]) for k, v in a.joints[0].state_dict().items())


def test_per_joint_selection_and_early_stop_restore_different_epochs():
    model = build_model(SMALL, 21, 7, 50)
    selector = JointSelection(7, dict(enabled=True, warmup=1, patience=1, min_delta=0.))
    def fill(value):
        with torch.no_grad():
            for p in model.parameters(): p.fill_(value)
    fill(1.)
    selector.update(model, np.ones(7), 1)
    fill(2.)
    selector.update(model, np.array([2., .5, 2., .5, 2., .5, 2.]), 2)
    assert selector.epochs == [1, 2, 1, 2, 1, 2, 1]
    assert selector.active.tolist() == [False, True, False, True, False, True, False]
    model.load_state_dict(selector.state_dict())
    for j, branch in enumerate(model.joints):
        for p in branch.parameters():
            assert torch.all(p == (1. if j % 2 == 0 else 2.))


def test_explicit_profile_required_and_baseline_unchanged():
    cfg = yaml.safe_load((ROOT/'config/w3/right/train_joint_lstm.yaml').read_text())
    validate_w3_config(cfg, 'train', require_paths=False)
    del cfg['trial']
    with pytest.raises(ValueError, match='explicit_research_trial'):
        validate_w3_config(cfg, 'train', require_paths=False)
    original = yaml.safe_load((ROOT/'config/w3/right/train.yaml').read_text())
    validate_w3_config(original, 'train', require_paths=False)


def test_manifest_training_checkpoint_reload_and_test_label_isolation(fixture_campaign):
    tmp, _, manifest, paths = fixture_campaign
    cfg = yaml.safe_load((ROOT/'config/w3/right/train_joint_lstm.yaml').read_text())
    cfg['data']['manifest'] = str(manifest)
    cfg['save']['output_dir'] = str(tmp/'runs')
    cfg['train'].update(epochs=2, batch_size=8, cpu_threads=1, stop_file=str(tmp/'STOP'))
    path = train(cfg)
    loaded = load_checkpoint(path, expected={'side': 'right', 'sample_hz': 100})
    assert isinstance(loaded.model, JointLSTMRegressor)
    metrics = json.loads((path/'metrics.json').read_text())
    assert loaded.metadata['deployment_ready'] is False
    assert loaded.metadata['architecture'] == 'independent_joint_lstm_v1'
    losses = np.asarray(metrics['per_joint_val_loss'])
    assert metrics['per_joint_best_epoch'] == (losses.argmin(0)+1).tolist()
    np.testing.assert_allclose(metrics['selected_validation_mse'], losses.min(0), rtol=1e-5)
    assert len(loaded.metadata['per_joint_best_epoch']) == 7
    # Test records may be audited by the manifest checker, but cannot train/select weights.
    p = paths['test'][0]
    with h5py.File(p, 'r+') as h:
        h['ep_0000/measured_joint_torque/data'][:] += 100
    meta = json.loads(sidecar_path(p).read_text()); meta['h5_sha256'] = sha256(p)
    write_json(sidecar_path(p), meta)
    m = json.loads(manifest.read_text()); m['splits']['test'][0]['h5_sha256'] = sha256(p)
    write_json(manifest, m)
    second = load_checkpoint(train(cfg))
    for key, weight in loaded.model.state_dict().items():
        torch.testing.assert_close(weight, second.model.state_dict()[key], rtol=0, atol=0)
    for key, value in loaded.normalization.items():
        np.testing.assert_array_equal(value, second.normalization[key])
    # Wrong architecture identity cannot be silently loaded as a shared LSTM.
    meta = json.loads((path/'metadata.json').read_text()); meta['architecture'] = 'lstm'
    write_json(path/'metadata.json', meta)
    with pytest.raises(ValueError, match='checkpoint_profile:architecture'):
        load_checkpoint(path)


def test_parallel_mixed_precision_workers_save_fp32_selected_models(fixture_campaign):
    tmp,_,manifest,_=fixture_campaign
    cfg=yaml.safe_load((ROOT/'config/w3/right/train_joint_lstm.yaml').read_text())
    cfg['data']['manifest']=str(manifest)
    cfg['save']['output_dir']=str(tmp/'parallel')
    cfg['train'].update(epochs=1,batch_size=8,cpu_threads=1,parallel_joints=7,
                        worker_threads=1,autocast_bfloat16=True,stop_file=str(tmp/'STOP'))
    result=train(cfg)
    loaded=load_checkpoint(result)
    metrics=json.loads((result/'metrics.json').read_text())
    assert metrics['per_joint_best_epoch']==[1]*7
    assert metrics['validation_precision']=='float32'
    assert all(p.dtype==torch.float32 for p in loaded.model.parameters())
    for j in range(7):
        state=torch.load(result/f'joint_training/joint_{j+1}/best.pt',weights_only=True)['state']
        for key,value in loaded.model.joints[j].state_dict().items():
            torch.testing.assert_close(value,state[key],rtol=0,atol=0)


def test_epoch_resume_restores_optimizer_dropout_and_window_order(tmp_path, monkeypatch):
    from factr2_next.training import parallel_joint_lstm as parallel
    cfg=dict(model=dict(SMALL,dropout=.2),train=dict(seed=0,worker_threads=1,
        batch_size=8,optimizer='adamw',learning_rate=.001,epochs=4,
        loss_units='physical_nm',autocast_bfloat16=True))
    torch.manual_seed(22)
    states=[copy.deepcopy(branch.state_dict()) for branch in build_model(cfg['model'],21,7,50).joints]
    dataset=TensorDataset(torch.randn(24,50,21),torch.randn(24,7))
    norm={'y_std':np.ones(7,dtype=np.float32)}
    uninterrupted=tmp_path/'uninterrupted';uninterrupted.mkdir()
    recovered=tmp_path/'recovered';recovered.mkdir()
    parallel.worker(0,states,[dataset,dataset],norm,cfg,uninterrupted)
    original=parallel.scalar_epoch
    train_calls=0
    def interrupt(model,loader,scale,optimizer=None,*args,**kwargs):
        nonlocal train_calls
        if optimizer is not None:
            train_calls+=1
            if train_calls==3:raise RuntimeError('simulated power interruption')
        return original(model,loader,scale,optimizer,*args,**kwargs)
    monkeypatch.setattr(parallel,'scalar_epoch',interrupt)
    with pytest.raises(RuntimeError,match='power interruption'):
        parallel.worker(0,states,[dataset,dataset],norm,cfg,recovered)
    assert torch.load(recovered/'joint_1/resume.pt',weights_only=True)['epoch']==2
    monkeypatch.setattr(parallel,'scalar_epoch',original)
    parallel.worker(0,states,[dataset,dataset],norm,cfg,recovered)
    a,b=[torch.load(p/'joint_1/final.pt',weights_only=True) for p in (uninterrupted,recovered)]
    assert a['history']==b['history']
    assert a['best_epoch']==b['best_epoch']
    for key,value in a['state'].items():
        torch.testing.assert_close(value,b['state'][key],rtol=0,atol=0)
    a,b=[torch.load(p/'joint_1/resume.pt',weights_only=True) for p in (uninterrupted,recovered)]
    for key,value in a['model'].items():
        torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)


def test_legacy_recovery_reuses_complete_and_verifies_replay_contract(tmp_path):
    from factr2_next.training.parallel_joint_lstm import prepare_recovery
    source=tmp_path/'source';source.mkdir();work=tmp_path/'work';work.mkdir()
    contract={'example':'pinned'}
    write_json(source/'training_contract.json',contract)
    for j in (1,3):
        folder=source/'joint_training'/f'joint_{j}';folder.mkdir(parents=True)
        write_json(folder/'progress.json',dict(status='complete' if j==1 else 'training'))
        torch.save({'original':j},folder/('final.pt' if j==1 else 'best.pt'))
    before=sha256(source/'joint_training/joint_1/final.pt')
    audit=prepare_recovery(source,work,contract)
    assert [x['mode'] for x in audit]==['reuse_complete','deterministic_replay_from_seed']
    assert sha256(work/'joint_1/final.pt')==before
    assert sha256(source/'joint_training/joint_1/final.pt')==before
    assert (work/'joint_3/replay_reference/best.pt').is_file()
    with pytest.raises(ValueError,match='contract_mismatch'):
        prepare_recovery(source,tmp_path/'wrong',{'example':'changed'})
