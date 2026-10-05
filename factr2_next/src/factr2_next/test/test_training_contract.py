"""Shared features, lazy equivalence, reproducibility and strict deployment."""
import copy
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
import pytest
import torch
import yaml
from test_w3_quality import fixture, refresh
from factr2_next.data_collection.quality import make_manifest, sha256, write_json, sidecar_path
from factr2_next.training.dataset import NextTorqueDataset, ManifestDataset
from factr2_next.training.w3_train import train
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.offline_eval import predict_sequence, read_episode, physical_metrics, evaluate, acceptance
from factr2_next.inference.history_buffer import HistoryBuffer
from factr2_next.w3_config import SIGNALS, validate_w3_config

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[4]


def config(tmp):
    paths=[fixture(tmp/f'{s}.h5',phase=i*20) for i,s in enumerate(('train','val','test'))]
    manifest=tmp/'manifest.json'
    make_manifest(manifest,dict(zip(('train','val','test'),[[p] for p in paths])),'unit_synthetic')
    cfg=yaml.safe_load((ROOT/'config/w3/left/train.yaml').read_text())
    cfg['data']['manifest']=str(manifest)
    cfg['train'].update(batch_size=32,epochs=1,cpu_threads=1)
    cfg['save'].update(output_dir=str(tmp/'runs'),run_name='smoke')
    return cfg


def test_lazy_feature_label_normalization_and_boundaries(tmp_path):
    path=fixture(tmp_path/'training.h5',n=70)
    # Explicit second episode: last-to-first discontinuity cannot enter a window.
    with h5py.File(path,'r+') as h:h.copy('ep_0000','ep_0001')
    meta=json.loads(sidecar_path(path).read_text());meta['episodes']['ep_0001']={'rows':70};sidecar_path(path).write_text(json.dumps(meta));refresh(path)
    keys={k:k for k in SIGNALS}
    eager=NextTorqueDataset([path],keys,50)
    lazy=ManifestDataset([{'path':str(path),'episodes':['ep_0000','ep_0001']}],keys)
    assert len(lazy)==42
    for i in range(len(lazy)):
        x,y=lazy.raw_item(i)
        np.testing.assert_array_equal(x,eager.x[i]);np.testing.assert_array_equal(y,eager.y[i])
    q=read_episode(path,'ep_0000')
    np.testing.assert_array_equal(lazy.raw_item(0)[0][:,14:],q['joint_cmd'][:50]-q['joint_pos'][:50])
    np.testing.assert_array_equal(lazy.raw_item(20)[1],q['measured_joint_torque'][69])
    np.testing.assert_array_equal(lazy.raw_item(21)[1],q['measured_joint_torque'][49])
    norm=lazy.fit_normalization()
    for k,ref in dict(x_mean=eager.x.astype(float).mean((0,1)),x_std=eager.x.astype(float).std((0,1))+1e-6,
                     y_mean=eager.y.astype(float).mean(0),y_std=eager.y.astype(float).std(0)+1e-6).items():
        np.testing.assert_allclose(norm[k],ref,atol=1e-7,rtol=1e-6)
    with pytest.raises(ValueError,match='memory_budget'):ManifestDataset([{'path':str(path),'episodes':['ep_0000']}],keys,memory_budget_mb=0)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('training');cfg=config(tmp)
    validate_w3_config(cfg,'train')
    run=train(cfg)
    return tmp,cfg,run


def test_repeatability_and_test_not_used(trained):
    tmp,cfg,run=trained
    # Change only test values and regenerate the manifest hash; train/val are identical.
    path=tmp/'test.h5'
    with h5py.File(path,'r+') as h:h['ep_0000/measured_joint_torque/data'][:]+=100
    refresh(path)
    make_manifest(tmp/'manifest.json',{s:[tmp/(s+'.h5')] for s in ('train','val','test')},'unit_synthetic')
    again=train(cfg)
    a=load_checkpoint(run);b=load_checkpoint(again)
    for k in a.model.state_dict():torch.testing.assert_close(a.model.state_dict()[k],b.model.state_dict()[k],rtol=0,atol=0)
    for k in a.normalization:np.testing.assert_array_equal(a.normalization[k],b.normalization[k])
    norm=ManifestDataset(json.loads((tmp/'manifest.json').read_text())['splits']['train'],cfg['data']['keys']).fit_normalization()
    for k in norm:np.testing.assert_array_equal(norm[k],a.normalization[k])
    m=json.loads((run/'metrics.json').read_text());assert m['best_epoch']==int(np.argmin(m['val_loss']))+1
    assert m['device']=='cpu' and m['train_loss'][0]>0


def test_history_reload_offline_equivalence(trained):
    tmp,cfg,run=trained;loaded=load_checkpoint(run)
    arrays=read_episode(tmp/'train.h5','ep_0000',keys=cfg['data']['keys'])
    pred,measured=predict_sequence(loaded,arrays)
    ds=NextTorqueDataset([tmp/'train.h5'],cfg['data']['keys'],50)
    n=loaded.normalization
    with torch.inference_mode():batch_pred=loaded.model(torch.from_numpy((ds.x-n['x_mean'])/n['x_std'])).numpy()*n['y_std']+n['y_mean']
    np.testing.assert_allclose(pred,batch_pred,atol=1e-5,rtol=1e-5)
    np.testing.assert_array_equal(measured,ds.y)
    buf=HistoryBuffer(50)
    for i in range(50):buf.append(*(arrays[k][i] for k in ('joint_pos','joint_vel','joint_cmd')))
    np.testing.assert_array_equal(buf.array(),ds.x[0]);buf.clear();assert not buf.ready


@pytest.mark.parametrize('bad',['missing','side','order','rate','dims','std','norm_shape','weights'])
def test_checkpoint_rejection(trained,tmp_path,bad):
    _,_,run=trained;dest=tmp_path/'run';shutil.copytree(run,dest)
    meta=json.loads((dest/'metadata.json').read_text())
    if bad=='missing':(dest/'model.pt').unlink()
    elif bad in ('side','order','rate'):
        meta[{'side':'side','order':'joint_order','rate':'sample_hz'}[bad]]='wrong'
    elif bad in ('std','norm_shape'):
        with np.load(dest/'normalization.npz') as f:n={k:f[k] for k in f.files}
        n['x_std']=np.zeros(21) if bad=='std' else np.ones(20)
        np.savez(dest/'normalization.npz',**n);meta['artifact_sha256']['normalization.npz']=sha256(dest/'normalization.npz')
    else:
        ck=torch.load(dest/'model.pt',weights_only=True)
        if bad=='dims':ck['input_size']=20
        else:ck['model_state_dict']['lstm.weight_ih_l0']=torch.zeros(1)
        torch.save(ck,dest/'model.pt');meta['artifact_sha256']['model.pt']=sha256(dest/'model.pt')
    write_json(dest/'metadata.json',meta)
    with pytest.raises(ValueError):load_checkpoint(dest)
    with pytest.raises(ValueError,match='checkpoint_profile:side'):load_checkpoint(run,expected={'side':'right'})


def test_metrics_and_acceptance():
    measured=np.array([[1,2],[3,6]],float);pred=np.zeros_like(measured)
    m=physical_metrics(measured,pred)
    np.testing.assert_allclose(m['per_joint_mse_nm2'],[5,20])
    np.testing.assert_allclose(m['signed_bias_nm'],[2,4]);np.testing.assert_allclose(m['residual_std_nm'],[1,2])
    assert acceptance({'model':m,'baseline':m},[measured],{'version':None},np.ones(2))['result']=='UNCONFIGURED'


def test_full_and_prefixed_evaluation(trained,tmp_path):
    _,cfg,run=trained;loaded=load_checkpoint(run)
    path=fixture(tmp_path/'long.h5',n=1105)
    report,_=evaluate(loaded,[{'path':str(path),'episodes':['ep_0000']}],tmp_path/'eval',make_plots=False)
    assert report['model']['samples']==1056
    with h5py.File(path,'r+') as h:
        for key in list(h['ep_0000']):h['ep_0000'].move(key,'left_'+key)
    generic=copy.copy(loaded);generic.metadata={};generic.config=copy.deepcopy(loaded.config)
    generic.config['data']['keys']={k:'{arm}_'+k for k in SIGNALS}
    arrays=read_episode(path,'ep_0000',arm='left',keys=generic.config['data']['keys'])
    pred,measured=predict_sequence(generic,arrays);assert len(pred)==1056
    report,_=evaluate(generic,[{'path':str(path),'episodes':['ep_0000']}],tmp_path/'prefix_eval',arm='left',make_plots=False)
    assert report['model']['samples']==1056


def test_bad_dataset_not_truncated(tmp_path):
    path=fixture(tmp_path/'bad.h5')
    with h5py.File(path,'r+') as h:h['ep_0000/joint_vel/data'].resize(100,axis=0)
    with pytest.raises(ValueError,match='stream_lengths'):NextTorqueDataset([path],{k:k for k in SIGNALS},50)
    with pytest.raises(ValueError,match='lengths'):read_episode(path,'ep_0000')


def test_best_epoch_is_saved(tmp_path,monkeypatch):
    cfg=config(tmp_path);cfg['train']['epochs']=2
    epoch=[0]
    def fake_epoch(model,loader,device,opt=None):
        if opt is not None:
            epoch[0]+=1
            with torch.no_grad():
                for p in model.parameters():p.fill_(epoch[0]*.001)
            return 1.
        return float(epoch[0])  # epoch 1 best; last epoch 2 deliberately worse
    monkeypatch.setattr('factr2_next.training.train.run_epoch',fake_epoch)
    run=train(cfg);ck=torch.load(run/'model.pt',weights_only=True)
    assert json.loads((run/'metrics.json').read_text())['best_epoch']==1
    for value in ck['model_state_dict'].values():torch.testing.assert_close(value,torch.full_like(value,.001),atol=0,rtol=0)
