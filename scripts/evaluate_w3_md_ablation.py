"""Frozen-model offline torque diagnostics and identical raw-session replay."""
from collections import Counter
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.endpoint_wrench import estimate_wrench, rotate_wrench_at_same_origin
from factr2_next.training.coverage_audit import group_name
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.models import build_model
from factr2_next.training.motion_discriminator import MDDataset, MotionDiscriminator
from factr2_next.w3_samples import features


def error_metrics(error):
    a = np.asarray(error, dtype=float)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError('md_metric_shape_finite_nonempty')
    return dict(samples=len(a), physical_mse_nm2=float(np.mean(a*a)),
        signed_bias_nm=a.mean(0).tolist(), per_joint_rmse_nm=np.sqrt((a*a).mean(0)).tolist(),
        per_joint_std_nm=a.std(0).tolist(), per_joint_abs_p95_nm=np.percentile(abs(a),95,axis=0).tolist(),
        norm_mean=float(np.linalg.norm(a,axis=1).mean()),
        norm_p95=float(np.percentile(np.linalg.norm(a,axis=1),95)))


def load_md(path):
    meta=json.loads((path/'metadata.json').read_text())
    if meta['schema']!='next_md_ablation_v1' or meta['input_size']!=126:
        raise ValueError('md_checkpoint_schema')
    for file,digest in meta['artifact_sha256'].items():
        if sha256(path/file)!=digest:raise ValueError('md_checkpoint_hash:'+file)
    ckpt=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    model=build_model(ckpt['model'],126,7,50)
    model.load_state_dict(ckpt['model_state_dict']);model.eval()
    with np.load(path/'normalization.npz',allow_pickle=False) as h:
        norm={k:h[k] for k in h.files}
    for key,size in [('x_mean',126),('x_std',126),('y_mean',7),('y_std',7)]:
        if norm[key].shape!=(size,) or not np.isfinite(norm[key]).all():raise ValueError('md_normalization')
        if key.endswith('std') and np.any(norm[key]<=0):raise ValueError('md_normalization_std')
    return model,norm


def dataset_metrics(model, norm, dataset, batch_size):
    dataset.norm=norm;predictions=[]
    with torch.inference_mode():
        for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            predictions.append(((y-model(x))*torch.as_tensor(norm['y_std'])).numpy().astype(float))
    residual=np.concatenate(predictions);assert len(residual)==len(dataset)
    group_rows={};static={};previous=0
    for (x,_),end,info in zip(dataset.episodes,dataset.ends,dataset.episode_info):
        r=residual[previous:end];previous=end;group=group_name(info)
        for key in [group,group+':'+info.get('speed','unknown')]:group_rows.setdefault(key,[]).append(r)
        if group=='static':
            match=re.fullmatch(r'static_anchor(\d+)_pattern(\d+)_from_(low|high)',info['kind'])
            if match:
                pose,pattern,direction=match.groups();key=(info['path'],pose,pattern,info['speed'])
                if direction in static.setdefault(key,{}):raise ValueError('md_duplicate_static_pair')
                n=min(100,len(r));tail=x[-n:]
                static[key][direction]=dict(bias=r[-n:].mean(0).tolist(),
                    actual_position_deg=np.rad2deg(tail[:,:7]).mean(0).tolist(),
                    native_command_deg=np.rad2deg(tail[:,:7]+tail[:,14:21]).mean(0).tolist())
    pairs=[]
    for (path,pose,pattern,speed),pair in static.items():
        if set(pair)=={'low','high'}:
            delta=lambda key: (np.asarray(pair['high'][key])-pair['low'][key]).tolist()
            pairs.append(dict(path=path,pose=pose,pattern=pattern,speed=speed,
                signed_gap_nm=delta('bias'),actual_position_gap_deg=delta('actual_position_deg'),
                native_command_gap_deg=delta('native_command_deg')))
    return dict(overall=error_metrics(residual),groups={k:error_metrics(np.concatenate(v)) for k,v in group_rows.items()},
        static_return_pairs_last_1s=pairs,
        max_static_return_gap_nm=max((max(abs(np.asarray(p['signed_gap_nm']))) for p in pairs),default=None)),residual


def sequence_prediction(model,norm,x,history=50,batch_size=2048):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(x)-history+1,batch_size):
            batch=np.stack([x[i:i+history] for i in range(start,min(start+batch_size,len(x)-history+1))])
            batch=(batch-norm['x_mean'])/norm['x_std']
            prediction=model(torch.from_numpy(batch.astype(np.float32))).numpy()*norm['y_std']+norm['y_mean']
            result.append(prediction)
    return np.concatenate(result)


def replay(folder, models, recipe, cfg):
    observation=Path('/home/venom/factr2/log/targeted_force_observation_20261016')
    state=json.loads((observation/'test_state.json').read_text());spec=json.loads((observation/'test_spec.json').read_text())
    session=Path(state['session']);run=json.loads((session/'audit/motion_run.json').read_text())
    files=list(session.glob('*.h5'));assert len(files)==1
    with h5py.File(files[0]) as h:
        if len(h)!=1:raise ValueError('Replay requires an explicitly continuous source episode')
        ep=next(iter(h));arrays=[h[ep+'/'+k+'/data'][:] for k in ('joint_pos','joint_vel','joint_cmd','measured_joint_torque')]
        stamps=h[ep+'/joint_pos/timestamps'][:]
        for k in ('joint_vel','joint_cmd','measured_joint_torque'):
            np.testing.assert_array_equal(stamps,h[ep+'/'+k+'/timestamps'][:])
    if np.any(np.diff(stamps)>40_000_000):raise ValueError('Replay has a source gap; do not filter across it')
    memory=MotionDiscriminator(recipe['thresholds_rad_s'],recipe['max_gap_ns']).transform(arrays[1],stamps)
    base=features(*arrays[:3]);xs=dict(NEXT=base,NEXT_MD=np.c_[base,memory])
    stamps=stamps[49:];q=arrays[0][49:];torque=arrays[3][49:]
    masks={};motion=np.zeros(len(stamps),bool)
    for event in run['events']:
        a,b=event['command_start_ros_ns'],event['command_end_ros_ns']
        mask=(stamps>=a)&(stamps<=b)
        if event['kind'] in ('initial_settle','final_settle'):
            label='before' if event['kind']=='initial_settle' else 'after'
            masks[label]=mask&(stamps>=b-5_000_000_000)
        elif event['family'] in ('speed_probe','continuous'):motion|=mask
    masks['motion']=motion
    from w3_coverage_plan import ArmModel
    description=(observation/'robot.urdf').read_text()
    if sha256(observation/'robot.urdf')!=spec['endpoint_config']['robot_description_sha256']:
        raise ValueError('Replay robot description mismatch')
    arm=ArmModel(description,'right',spec['endpoint_config']['tool_frame'])
    jacobians=arm.jacobians(q);rotation=np.stack([arm.fk(v)[:3,:3] for v in q])
    bounds=np.array(spec['training_envelope']['position_bounds_deg'])
    envelope=np.all((np.rad2deg(q)>=bounds[:,0])&(np.rad2deg(q)<=bounds[:,1]),axis=1)
    result={};plot_data={}
    for label,(model,norm) in models.items():
        pred=sequence_prediction(model,norm,xs[label],batch_size=cfg['train']['batch_size'])
        residual=torque-pred;filtered=np.empty_like(residual);value=None
        # Exact float32 EMA recurrence, unchanged alpha=.2 at 100Hz.
        for i,r in enumerate(residual):
            value=r.copy() if value is None else .8*value+.2*r
            filtered[i]=value
        wrench=np.full((len(q),6),np.nan);reasons=[]
        for i,(j,r) in enumerate(zip(jacobians,filtered)):
            value,diag=estimate_wrench(j,r,**spec['endpoint_config']['solver'])
            if not envelope[i]:reasons.append('outside_collection_range');continue
            reasons.append(diag['reason'])
            if value is not None:wrench[i]=rotate_wrench_at_same_origin(value,rotation[i])
        valid=np.isfinite(wrench).all(axis=1);groups={}
        for group,mask in masks.items():
            selected=mask&valid;force=wrench[selected,:3];norms=np.linalg.norm(force,axis=1)
            groups[group]=dict(rows=int(mask.sum()),endpoint_available_fraction=float(valid[mask].mean()),
                unavailable_reasons=dict(Counter(reasons[i] for i in np.flatnonzero(mask&~valid))),
                raw_joint=error_metrics(residual[mask]),filtered_joint=error_metrics(filtered[mask]),
                force_n=dict(samples=len(force),bias=force.mean(0).tolist(),
                    norm_mean=float(norms.mean()),norm_p95=float(np.percentile(norms,95)),norm_peak=float(norms.max()))
                if len(force) else dict(samples=0))
        result[label]=dict(groups=groups,zero_initial_history=True,
            initialize_ema_at_first_common_50_frame_output=True,
            source_h5_sha256=sha256(files[0]),replay_scope='post-hoc diagnostic, not blind prospective validation')
        plot_data[label]=dict(raw=residual,filtered=filtered,wrench=wrench,valid=valid)
    # Compare after-before joint bias on all frames, rather than only on the
    # different endpoint-valid subsets selected by each model.
    for label,data in plot_data.items():
        result[label]['joint_bias_after_minus_before_nm']=(data['filtered'][masks['after']].mean(0)-data['filtered'][masks['before']].mean(0)).tolist()
    np.savez_compressed(folder/'replay_arrays.npz',stamps=stamps,
        **{label+'_'+key:value for label,data in plot_data.items() for key,value in data.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin=min(e['command_start_ros_ns'] for e in run['events']);t=(stamps-origin)*1e-9
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for label,data in plot_data.items():
        axes[0].plot(t,np.linalg.norm(data['filtered'],axis=1),lw=.8,label=label)
        axes[1].plot(t,np.linalg.norm(data['wrench'][:,:3],axis=1),lw=.8,label=label)
        axes[2].step(t,data['valid'].astype(float),where='post',lw=.7,label=label)
    for ax in axes:ax.grid(alpha=.2);ax.legend(loc='upper left');ax.set_xlim(0,t.max())
    axes[0].set_ylabel('Joint residual norm (Nm)');axes[1].set_ylabel('Endpoint force norm (N)')
    axes[2].set_ylabel('Endpoint available');axes[2].set_xlabel('Time from initial commanded hold (s)')
    fig.suptitle('Frozen NEXT vs NEXT+MD | same raw unloaded session, EMA and endpoint solver')
    fig.tight_layout();fig.savefig(folder/'replay_comparison.png',dpi=150);plt.close(fig)
    return result


def evaluate(folder):
    folder=Path(folder);cfg=yaml.safe_load((folder/'experiment.yaml').read_text())
    manifest=json.loads((folder/'split.json').read_text());recipe=json.loads((folder/'md_recipe.json').read_text())
    baseline=load_checkpoint(Path(cfg['comparison']['baseline_run']));md,norm=load_md(folder/'next_md_seed0')
    models=dict(NEXT=(baseline.model,baseline.normalization),NEXT_MD=(md,norm))
    results={}
    for split in ('val','test'):
        results[split]={}
        for label,(model,normalization) in models.items():
            if label=='NEXT':ds=ManifestDataset(manifest['splits'][split],cfg['data']['keys'])
            else:ds=MDDataset(manifest['splits'][split],cfg['data']['keys'],cfg['comparison']['feature_cache'])
            values,_=dataset_metrics(model,normalization,ds,cfg['train']['batch_size'])
            results[split][label]=values
    historical=json.loads((Path(cfg['comparison']['baseline_run'])/'trial_evaluation.json').read_text())
    np.testing.assert_allclose(results['test']['NEXT']['overall']['physical_mse_nm2'],
        historical['test_physical_mse_nm2'],rtol=1e-5,atol=1e-8)
    results['replay']=replay(folder,models,recipe,cfg)
    results['limitations']=[
        'single seed 0; increased input projection parameter count is a capacity confound',
        'whole-session holdouts are post-hoc; held-out protocols cover only 6 deg/s at the reference anchor',
        'test and observation logs have already been inspected; comparison is exploratory, not a new blind test',
        'no contact/release or dedicated cold/warm validation; no automatic deployment',
        'same commanded return pose does not imply identical actual pose',
        'zero MD at recording start means earlier friction history is unknown']
    write_json(folder/'comparison.json',results)
    lines=['# 原版 NEXT 与 NEXT+MD 对照','',
        'MD 按 Shan & Pham (ICRA 2024) Algorithm 1 实现；作者主页与论文未找到可确认的官方代码仓库。本实验为 MD 特征消融，不是完整 HRDL-MD 复现。',
        '', '两组使用相同整会话划分、50 帧原始位置/速度/目标误差、实际力矩标签、训练归一化、LSTM 隐层和回归头、AdamW、物理 MSE、seed=0 与早停。NEXT 复用冻结模型；MD 增加 105 个输入，阈值只由训练静止段校准；源会话先因果计算 MD 再取监督片段。',
        '', '| 集合 | 模型 | 总体力矩 MSE Nm² | 静止最大绝对 bias Nm | 静止最大关节 RMSE Nm | 正反到位最后1秒最大差 Nm |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for split in ('val','test'):
        for label,value in results[split].items():
            static=value['groups']['static']
            lines.append(f'| {split} | {label} | {value["overall"]["physical_mse_nm2"]:.7f} | {max(abs(np.array(static["signed_bias_nm"]))):.5f} | {max(static["per_joint_rmse_nm"]):.5f} | {value["max_static_return_gap_nm"]:.5f} |')
    lines+=['','静止均值可能抵消相反方向偏差，须同时看每段返回差、P95 和按速度分组结果（comparison.json）。',
        '', '| 空载回放条件 | 模型 | 有效末端比例 | 力模长均值 N | P95 N | 峰值 N |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for group in ('before','motion','after'):
        for label,result in results['replay'].items():
            v=result['groups'][group];f=v['force_n']
            lines.append(f'| {group} | {label} | {v["endpoint_available_fraction"]:.1%} | {f.get("norm_mean",float("nan")):.3f} | {f.get("norm_p95",float("nan")):.3f} | {f.get("norm_peak",float("nan")):.3f} |')
    lines+=['','末端统计只覆盖有效结果，无效样本不填零。此回放从原始日志开始初始化，首个共同50帧输出初始化EMA；与此前已持续运行的在线估计可能有初始历史差别。',
            '', '## 适用范围','',*['- '+v for v in results['limitations']],
            '', '未修改机器人当前部署模型或控制参数。','', '![相同空载日志回放](replay_comparison.png)']
    (folder/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({split:{label:value['overall']['physical_mse_nm2'] for label,value in results[split].items()}
                      for split in ('val','test')},ensure_ascii=False),flush=True)
    return results
