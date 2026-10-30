"""Evaluate frozen joint LSTMs and export a hash-pinned read-only bundle."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from factr2_next.data_collection.quality import sha256, write_json, validate_manifest
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.frozen_sigma_observer import load_observer
from factr2_next.inference.joint_lstm_observer import ResidualCascade, ROOT, SCHEMA, load_bundle
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.next_u import error_metrics

REPO = Path(__file__).resolve().parents[1]
OLD_BASE = REPO/'models/w3/right/wide_highspeed_observer_20261026_v1'


def prediction(loaded, ds):
    ds.norm = loaded.normalization
    values = []
    with torch.inference_mode():
        for x, _ in DataLoader(ds, batch_size=256, shuffle=False):
            mu = loaded.model(x).numpy()*loaded.normalization['y_std']+loaded.normalization['y_mean']
            if not np.isfinite(mu).all(): raise ValueError('nonfinite_mean')
            values.append(mu)
    return np.concatenate(values)


def configs(bundle, envelope):
    cfg = yaml.safe_load((REPO/'config/w3/right/inference.yaml').read_text())
    cfg.update(checkpoint_dir=str(bundle/'mean_checkpoint'),next_topic_root=ROOT,
        observation_mode='independent_joint_lstm_research',sample_hz=100,cpu_threads=4,
        operating_envelope=envelope,require_adapter_status=True)
    cfg['contact']['enabled'] = False
    cfg['smoothing'] = dict(enabled=True,mode='lowpass',cutoff_hz=5.,sample_hz=100.)
    web = dict(contract_version=cfg['contract_version'],side='right',joint_names=cfg['joint_names'],
        next_topic_root=ROOT,observation_mode=cfg['observation_mode'],outputs=copy.deepcopy(cfg['outputs']),
        web=dict(host='127.0.0.1',port=8083,refresh_hz=10),
        plot=dict(view='joint_torque',max_points=1000,torque_display=dict(primary='ext',
            storage_prefix='next.joint_lstm.',raw_label='新模型 · 原始外力矩残差（N·m）',
            filtered_label='新模型 · 平滑外力矩（N·m）',
            description='七个独立 LSTM 各自读取全部关节的 50 帧状态，只预测对应关节空载力矩。'
                        '外力矩为反馈减去预测；默认显示 12 Hz、5 Hz 两级因果低通结果，可展开原始残差对照。'
                        'J1–J7 单位 N·m，显示保留两位小数。')),
        model_label='NEXT · 七关节独立 LSTM · 完整重训',
        model_notice='当前使用新训练的七个 LSTM。附加残差补偿关闭；旧 sigma 未复用。'
                     '独立记录空载误差已离线评估，真实接触及新实机运动尚未验证。')
    return cfg, web


def evaluate(checkpoint, output, report):
    checkpoint, output, report = [Path(p).resolve() for p in (checkpoint, output, report)]
    if output.exists() or report.exists(): raise FileExistsError('preserve existing export/evaluation')
    loaded = load_checkpoint(checkpoint)
    if loaded.config['model']['type'] != 'joint_lstm': raise ValueError('joint_lstm_required')
    torch.set_num_threads(4)
    manifest_path = Path(loaded.config['data']['manifest'])
    manifest, _ = validate_manifest(manifest_path)
    if sha256(manifest_path) != loaded.metadata['manifest_sha256']:
        raise ValueError('training_manifest_changed')
    report.mkdir(parents=True)
    # Derive the workspace exclusively from actual training positions.
    ds = ManifestDataset(manifest['splits']['train'], loaded.config['data']['keys'], 50, 512)
    lo = np.min([x[:, :7].min(0) for x, _ in ds.episodes], axis=0)
    hi = np.max([x[:, :7].max(0) for x, _ in ds.episodes], axis=0)
    envelope = dict(position_bounds_deg=np.rad2deg(np.stack((lo,hi),axis=1)).tolist(),
        position_margin_deg=0.,source='actual training positions only; rectangular bounds do not certify pose combinations')
    del ds
    result = dict(checkpoint=str(checkpoint),mean_sha256=sha256(checkpoint/'model.pt'),
        architecture='independent_joint_lstm_v1',splits={},compensation_enabled=False,
        test_used_for_fit=False,test_scope='Existing independent recording roles retained; retrospectively inspected, not a new blind test',
        physical_contact_validated=False)
    wide_data_root = Path(manifest_path).parent.resolve()
    result['group_definitions'] = dict(old='all inherited files outside '+str(wide_data_root),
        high='only files in '+str(wide_data_root),overall='all eligible validation/test windows')
    old_loaded = load_observer(OLD_BASE)
    result['incumbent_metadata_sha256'] = sha256(OLD_BASE/'metadata.json')
    result['incumbent_compensation_enabled'] = False
    result['comparison_signal'] = 'both raw residuals and identical LP12 then LP5; no learned compensation'
    example = []
    for role in ('val','test'):
        ds = ManifestDataset(manifest['splits'][role], loaded.config['data']['keys'],50,512)
        mu = prediction(loaded, ds)
        old_mu = prediction(old_loaded, ds)
        if mu.shape != old_mu.shape: raise ValueError('incumbent_cache_alignment')
        arrays={k:[] for k in ('new_raw','old_raw','new_final','old_filter_only','high')}
        offset=0
        for (x,y),info in zip(ds.episodes,ds.episode_info):
            x,y=x[49:],y[49:];n=len(x)
            with h5py.File(info['path']) as h: t=np.array(h[info['episode']+'/joint_pos/timestamps'][49:])
            fresh=mu[offset:offset+n];old=old_mu[offset:offset+n];offset+=n
            new_filter,old_filter=ResidualCascade(),ResidualCascade()
            new_final=[];old_filtered=[]
            for tau,nn,oo in zip(y,fresh,old):
                new_final.append(new_filter.update(tau-nn));old_filtered.append(old_filter.update(tau-oo))
            keep=t>=t[0]+1_000_000_000
            values=dict(new_raw=y-fresh,old_raw=y-old,new_final=np.asarray(new_final),
                old_filter_only=np.asarray(old_filtered),
                high=np.full(n,Path(info['path']).resolve().parent == wide_data_root))
            for key,value in values.items():arrays[key].append(value[keep])
            if role=='test' and str(info.get('speed','')).startswith('continuous_v'):
                example.append(dict(name=info['speed'],t=(t-t[0])*1e-9,new=np.asarray(new_final),old=np.asarray(old_filtered)))
        combined={k:np.concatenate(v) for k,v in arrays.items()}
        result['splits'][role]={}
        for name,mask in (('overall',np.ones(len(combined['high']),bool)),
                          ('old',~combined['high']),('high',combined['high'])):
            result['splits'][role][name]={key:error_metrics(value[mask]) for key,value in combined.items() if key!='high'}
        np.savez_compressed(report/(role+'_predictions.npz'),mean_free_nm=mu,**combined)
        write_json(report/'comparison.json',result)
    # No model/threshold tuning after test evaluation.
    probe=torch.randn(1,50,21);timings=[]
    with torch.inference_mode():
        for _ in range(20):loaded.model(probe)
        for _ in range(300):
            start=time.perf_counter();loaded.model(probe);timings.append((time.perf_counter()-start)*1000)
    result['inference_ms']=dict(median=float(np.median(timings)),p99=float(np.percentile(timings,99)))
    result['per_joint_best_epoch']=loaded.metadata['per_joint_best_epoch']
    write_json(report/'comparison.json',result)
    output.mkdir(parents=True)
    shutil.copytree(checkpoint, output/'mean_checkpoint',ignore=shutil.ignore_patterns('best_epoch_*.pt','*.pending'))
    cfg,web=configs(output,envelope)
    (output/'inference.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    (output/'web.yaml').write_text(yaml.safe_dump(web,sort_keys=False,allow_unicode=True))
    files=['inference.yaml','web.yaml']+['mean_checkpoint/'+name for name in
        ('model.pt','config.yaml','metrics.json','metadata.json','normalization.npz')]
    write_json(output/'bundle.json',dict(schema=SCHEMA,architecture='independent_joint_lstm_v1',
        operating_envelope=envelope,controller_changes=False,compensation_enabled=False,
        mean_model_sha256=result['mean_sha256'],evaluation=str(report),evaluation_sha256=sha256(report/'comparison.json'),
        deployment_ready=False,physical_contact_validated=False,files={f:sha256(output/f) for f in files}))
    load_bundle(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if example:
        fig,axes=plt.subplots(7,len(example),figsize=(15,15),sharex='col',squeeze=False)
        for col,item in enumerate(example):
            for j in range(7):
                ax=axes[j,col];ax.plot(item['t'],item['old'][:,j],lw=.6,label='shared LSTM + same filters')
                ax.plot(item['t'],item['new'][:,j],lw=.6,label='joint LSTM + filters');ax.set_ylabel(f'J{j+1} (Nm)');ax.grid(alpha=.2)
            axes[0,col].set_title(item['name']);axes[0,col].legend();axes[-1,col].set_xlabel('seconds')
        fig.tight_layout();fig.savefig(report/'heldout_waveforms.png',dpi=130);plt.close(fig)
    print(json.dumps(dict(bundle=str(output),report=str(report),inference_ms=result['inference_ms'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','output','report'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();evaluate(args.checkpoint,args.output,args.report)
