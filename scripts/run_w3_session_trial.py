#!/usr/bin/env python3
"""Train the existing W3 LSTM, freeze the best validation epoch, then evaluate."""
import argparse
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from factr2_next.w3_config import validate_w3_config
from factr2_next.data_collection.quality import validate_manifest,write_json,sha256
from factr2_next.training.w3_train import train
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.offline_eval import evaluate


def run_trial(dataset):
    state_path=dataset/'trial_state.json'
    if state_path.exists():raise ValueError('This trial already started; use a new dataset directory')
    audit_input=json.loads((dataset/'split_audit.json').read_text())
    state={'status':'training','evaluation_scope':audit_input.get('scope','within-session trial, not independent-session generalization'),
           'config':str(dataset/'train.yaml'),'started_unix':time.time()}
    write_json(state_path,state)
    try:
        root=Path(__file__).resolve().parents[1]
        snapshot=dataset/'training_source'
        sources=list((root/'factr2_next/src/factr2_next/factr2_next').rglob('*.py'))
        sources += [root/'scripts'/name for name in ('run_w3_session_trial.py','prepare_w3_session_trial.py',
            'w3_smooth_coverage.py','w3_coverage_plan.py','w3_coverage_motion.py','quick_capture.py',
            'analyze_w3_coverage_session.py','audit_w3_controller_stream.py','combine_w3_session_trials.py')]
        source_manifest={}
        for path in sources:
            relative=path.relative_to(root); target=snapshot/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            source_manifest[str(relative)]=sha256(target)
        write_json(snapshot/'source_sha256.json',source_manifest)
        cfg=yaml.safe_load((dataset/'train.yaml').read_text())
        validate_w3_config(cfg,'train')
        manifest,_=validate_manifest(cfg['data']['manifest'])
        audit=json.loads((dataset/'split_audit.json').read_text())
        session=Path(audit['parent_h5']).parent
        capture_snapshot=snapshot/'collection_config'
        capture_snapshot.mkdir()
        for path in [session/'record.yaml',*[session/'audit'/name for name in
                ('position.yaml','gravity.yaml','friction_model.yaml','robot_description.yaml','controllers.txt')]]:
            if path.is_file():
                shutil.copy2(path,capture_snapshot/path.name)
        if (session/'audit/source_snapshot').is_dir():
            shutil.copytree(session/'audit/source_snapshot',capture_snapshot/'controller_source')
        for path in [session/'audit/motion_run.json',session/'analysis/controller_smoothness.json',
                     session/'analysis/capture_diagnostics.json']:
            if path.is_file():
                shutil.copy2(path,capture_snapshot/path.name)
        write_json(capture_snapshot/'sha256.json',
            {str(p.relative_to(capture_snapshot)):sha256(p)
             for p in capture_snapshot.rglob('*') if p.is_file()})
        for i,parent in enumerate(audit.get('source_sessions',[])):
            original=Path(parent['parent_h5']).parent
            destination=snapshot/'source_sessions'/str(i)
            shutil.copytree(original/'audit/source_snapshot',destination/'controller_source')
            for path in [original/'record.yaml',original/'audit/gravity.yaml',original/'audit/position.yaml',
                         original/'audit/friction_model.yaml',original/'audit/motion_run.json',
                         original/'analysis/controller_smoothness.json']:
                if path.is_file():shutil.copy2(path,destination/path.name)
        run=train(cfg)
        (dataset/'run_directory.txt').write_text(str(run)+'\n')
        state.update(status='evaluating',run_directory=str(run));write_json(state_path,state)
        shutil.copytree(snapshot,run/'training_source')
        for name in ('split.json','split_audit.json'):
            shutil.copy2(dataset/name,run/name)
        shutil.copy2(Path(__file__),run/'trial_runner.py')
        torch.set_num_threads(cfg['train']['cpu_threads'])
        loaded=load_checkpoint(run)
        summary={'run_directory':str(run),'model_sha256':sha256(run/'model.pt'),
                 'evaluation_scope':state['evaluation_scope'],
                 'training_metrics':json.loads((run/'metrics.json').read_text()),'splits':{}}
        # Test is read only after the validation-selected checkpoint is saved/reloaded.
        for split in ('test','val','train'):
            report,_=evaluate(loaded,manifest['splits'][split],run/'evaluation'/split,split=split,make_plots=(split!='train'))
            report['evaluation_scope']=state['evaluation_scope']
            report['acceptance']={'result':'UNCONFIGURED','reason':'No physical accuracy thresholds or independent sessions supplied'}
            write_json(run/'evaluation'/split/'report.json',report)
            summary['splits'][split]={'samples':report['model']['samples'],
                'rmse_nm':report['model']['overall_rmse_nm'],'baseline_rmse_nm':report['baseline']['overall_rmse_nm'],
                'per_joint_rmse_nm':report['model']['per_joint_rmse_nm'],
                'baseline_per_joint_rmse_nm':report['baseline']['per_joint_rmse_nm'],
                'signed_bias_nm':report['model']['signed_bias_nm'],
                'abs_residual_p95_nm':report['model']['abs_residual_p95_nm'],
                'rmse_improvement':1-report['model']['overall_rmse_nm']/report['baseline']['overall_rmse_nm']}
            print(split,summary['splits'][split],flush=True)
        parents=audit.get('source_sessions',[audit])
        summary['source_data_unchanged']=all(sha256(p['parent_h5'])==p['parent_sha256_before'] for p in parents)
        summary['source_recordings']=[p['parent_h5'] for p in parents]
        summary.update(side=cfg['side'],sample_hz=cfg['data'].get('sample_hz',50),parent_h5=audit['parent_h5'])
        write_json(run/'trial_summary.json',summary)
        metrics=summary['training_metrics']
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        epochs=np.arange(1,len(metrics['train_loss'])+1)
        axes[0].plot(epochs,metrics['train_loss'],label='train');axes[0].plot(epochs,metrics['val_loss'],label='validation')
        axes[0].axvline(metrics['best_epoch'],ls='--',color='gray',label='saved epoch');axes[0].set(xlabel='Epoch',ylabel='Normalized torque MSE');axes[0].legend();axes[0].grid(alpha=.3)
        test=summary['splits']['test'];x=np.arange(7)
        axes[1].bar(x-.18,test['baseline_per_joint_rmse_nm'],.36,label='Training mean baseline')
        axes[1].bar(x+.18,test['per_joint_rmse_nm'],.36,label='LSTM')
        axes[1].set_xticks(x,[f'J{i+1}' for i in x]);axes[1].set_ylabel('Test torque RMSE (Nm)');axes[1].legend();axes[1].grid(axis='y',alpha=.3)
        fig.suptitle('Repeated-motion torque regression trial');fig.tight_layout();fig.savefig(run/'training_summary.png',dpi=150);fig.savefig(run/'training_summary.pdf');plt.close(fig)
        rows=['# 本次重复运动试训结果','', f'使用 {summary["side"]} 臂录制 {Path(audit["parent_h5"]).parent.name}，{summary["sample_hz"]} Hz；训练/验证/测试按重复动作完整时间段划分并剔除边界。各划分没有共享原始帧，也不跨录制断点构造历史窗口；数据仍共享机械臂状态与相似重复轨迹，不能代替独立会话泛化测试。','',
              f'两层 LSTM（hidden=128、dropout=0.1），历史 50 帧，CPU、batch={cfg["train"]["batch_size"]}、{cfg["train"].get("optimizer","adam")} lr=0.001，实际训练 {len(metrics["train_loss"])} epochs、seed=0。按验证集选择第 {metrics["best_epoch"]} 个 epoch。', '',
              '| 划分 | 窗口数 | 模型 RMSE (Nm) | 训练均值基线 RMSE (Nm) |','|---|---:|---:|---:|']
        for split in ('train','val','test'):
            s=summary['splits'][split];rows.append(f'| {split} | {s["samples"]} | {s["rmse_nm"]:.6f} | {s["baseline_rmse_nm"]:.6f} |')
        rows+=['','| 关节 | 测试 RMSE (Nm) | 基线 RMSE (Nm) |','|---|---:|---:|']
        for i,(value,base) in enumerate(zip(test['per_joint_rmse_nm'],test['baseline_per_joint_rmse_nm'])):rows.append(f'| J{i+1} | {value:.6f} | {base:.6f} |')
        rows+=['',f'测试总体 RMSE 相对基线改善 {test["rmse_improvement"]*100:.1f}%。报告的是实测电机力矩与预测自由空间力矩的差。自由运动时它也是外力估计的零接触残差诊断，但本次没有接触外力参考，不能宣称真实接触外力精度。', '',
               '模型、归一化、配置和元数据已保存并通过 checkpoint 重新加载检查；没有向机器人部署或启动在线外力估计。原始 H5 未修改。物理验收阈值未配置，状态为 UNCONFIGURED。','',
               '模型文件：model.pt；归一化：normalization.npz；训练损失：metrics.json；划分审计：split_audit.json；评估：evaluation/{train,val,test}/report.json 和 samples.csv；总览：training_summary.png。']
        (run/'TRIAL_REPORT.md').write_text('\n'.join(rows)+'\n')
        state.update(status='completed',completed_unix=time.time());write_json(state_path,state)
        print('TRIAL COMPLETED',run,flush=True)
        return run
    except BaseException as exc:
        state.update(status='failed',error=str(exc),completed_unix=time.time());write_json(state_path,state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);args=p.parse_args();run_trial(args.dataset.resolve())
