"""Render frozen ablations with separate moving and static held-out metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from factr2_next.data_collection.quality import write_json
from fit_w3_kinematic_compensation import recent, predict, smooth


def run(experiment):
    recipe=json.loads((experiment/'recipe.json').read_text())
    comparison=json.loads((experiment/'comparison.json').read_text())
    model=json.loads((experiment/'model.json').read_text())
    item=recent(Path(recipe['observation'])/'labeled_samples.csv')
    with np.load(experiment/'replay.npz') as h:values={k:h[k] for k in ('original','input_filtered','filter_only','compensated_unfiltered_output','final','correction_nm','supported')}
    for mode in ('velocity','velocity_acceleration'):
        a=json.loads((experiment/(mode+'.json')).read_text())
        correction,_=predict(item,a)
        values[mode]=smooth(item['target']-correction,item['t'],model['signal']['output_hz'])
    masks=dict(validation=item['role']=='val',test_all=item['role']=='test',
        test_motion=(item['role']=='test')&np.char.startswith(item['labels'],'cycle3'),
        test_static=(item['role']=='test')&(item['labels']=='baseline_after'))
    summary={}
    for name,mask in masks.items():
        mask &= item['eligible']; group=dict(rows=int(mask.sum()),variants={})
        continuous=mask[1:]&mask[:-1]&(np.diff(item['t'])<=40_000_000)
        for label,arr in values.items():
            if arr.ndim!=2 or label=='correction_nm':continue
            a=arr[mask]
            group['variants'][label]=dict(rms_nm=np.sqrt(np.mean(a*a,axis=0)).tolist(),
                bias_nm=a.mean(0).tolist(),absolute_p95_nm=np.percentile(abs(a),95,axis=0).tolist(),
                first_difference_rms_nm=np.sqrt(np.mean(np.diff(arr,axis=0)[continuous]**2,axis=0)).tolist())
        summary[name]=group
    write_json(experiment/'heldout_detail.json',summary)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    start=np.flatnonzero(item['labels']=='baseline_before')[0]
    t=(item['t']-item['t'][start])*1e-9;show=t>=0
    fig,axs=plt.subplots(2,1,figsize=(14,7),sharex=True)
    axs[0].plot(t[show],np.rad2deg(item['x'][show,0]+item['x'][show,14]),'--',lw=1,label='J1 command')
    axs[0].plot(t[show],np.rad2deg(item['x'][show,0]),lw=1,label='J1 measured')
    axs[0].set_ylabel('Angle (deg)');axs[0].legend(ncol=2)
    for name,color,lw in [('original','#a8adb5',.55),('filter_only','#af7ac5',.8),('final','#1670ca',1.1)]:
        axs[1].plot(t[show],values[name][show,0],color=color,lw=lw,label=name)
    for role,color in [('train','#269ab5'),('val','#e4b21b'),('test','#27ae60')]:
        mask=(item['role']==role)&show
        axs[1].fill_between(t,0,1,where=mask,transform=axs[1].get_xaxis_transform(),color=color,alpha=.08,label=role)
    extrema=np.concatenate([values[k][show,0] for k in ('original','filter_only','final')])
    axs[1].set_ylim(extrema.min()-.05,extrema.max()+.05)
    axs[1].axhline(0,color='black',lw=.6);axs[1].set_ylabel('J1 residual (Nm)')
    axs[1].set_xlabel('Time from in-range baseline (s)');axs[1].legend(ncol=6)
    axs[0].set_title('Causal velocity/acceleration residual compensation; retrospective cycle holdout')
    fig.tight_layout();fig.savefig(experiment/'j1_comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(7,1,figsize=(14,14),sharex=True)
    for j,ax in enumerate(axs):
        for name,color,lw in [('original','#a8adb5',.5),('filter_only','#af7ac5',.7),('final','#1670ca',1.)]:
            ax.plot(t[show],values[name][show,j],color=color,lw=lw,label=name)
        ax.axhline(0,color='black',lw=.5);ax.set_ylabel(f'J{j+1} (Nm)')
    axs[0].legend(ncol=3);axs[-1].set_xlabel('Time from in-range baseline (s)')
    fig.tight_layout();fig.savefig(experiment/'all_joints_comparison.png',dpi=160);plt.close(fig)
    m=summary['test_motion']['variants'];st=summary['test_static']['variants']
    lines=['# 速度/加速度补偿与因果滤波：留出回放结果','',
        '这是原 NEXT 均值冻结后的附加观测分支；不改变机械臂控制器、增益、重力或摩擦前馈。没有执行新的机械臂运动。','',
        '历史数据保留原记录级 train/val/test。最新空载记录按第 1 次往复训练、第 2 次选择、第 3 次回顾性留出评估，边界两侧各排除 1 秒。训练还包含预先指定的静止前段和小幅试走，验证含试走后的停留，静止后段单独作为测试。测试轨迹以前已经看过，且三次往复相似，不能称为盲测或跨工况验证。','',
        f'验证选中：{model["feature_mode"]}，范围 {model["scope"]}，ridge={model["ridge"]}。启用关节：'+', '.join(f'J{j+1}' for j,v in enumerate(model['enabled_joints']) if v)+'；其余关节仅滤波。','',
        '以下只统计第 3 次往复的运动段（不混入静止段）。输入、样本、边界和原始模型完全相同。','',
        '|关节|原始 RMS Nm|仅滤波 RMS Nm|仅速度补偿＋滤波|速度/加速度补偿＋滤波|最终相对原始改善|','|---|---:|---:|---:|---:|---:|']
    for j in range(7):
        a=m['original']['rms_nm'][j];b=m['filter_only']['rms_nm'][j];v=m['velocity']['rms_nm'][j];f=m['final']['rms_nm'][j]
        lines.append(f'|J{j+1}|{a:.5f}|{b:.5f}|{v:.5f}|{f:.5f}|{100*(1-f/a):.1f}%|')
    rough0=m['original']['first_difference_rms_nm'][0];rough1=m['final']['first_difference_rms_nm'][0]
    lines += ['',f'J1 相邻样本一阶差分 RMS 从 {rough0:.6f} 降到 {rough1:.6f} N·m，降低 {100*(1-rough1/rough0):.1f}%。它是曲线粗糙度指标，不是独立噪声真值。',
        '',f'独立统计的停止后静止段，J1 原始 RMS {st["original"]["rms_nm"][0]:.5f}，最终 RMS {st["final"]["rms_nm"][0]:.5f} N·m。未使用静止后目标拟合补偿或在线调零。','',
        '速度/方向项贡献主要改善；加速度项总体额外改善很小，并不是每个关节都优于仅速度。不能从这次相关性拟合声称辨识出了真实转动惯量、真实摩擦力矩或电流误差。','',
        '选中的局部模型只覆盖当前姿态附近，尤其 J3≈−2.77°。旧验证/测试的姿态不符合该局部范围，补偿回退，补偿前后 RMS 相同；这只能说明回退有效，不能证明跨姿态泛化。新关节运动和更高速度需要独立采集与验证。','',
        '## 信号链和延迟','',
        '原 LSTM 继续使用训练时的原始 q、速度反馈和跟踪误差。新增支路对实际位置和指令位置使用位置观测的常加速度卡尔曼滤波，因果估计速度和加速度；电机速度反馈另做 8 Hz 低通供诊断，不把静止时半个量化码的反馈当成真实运动。',
        '', '12 Hz 相同因果低通同时作用于实测力矩与原 NEXT 自由预测，随后减去离线学到的运动残差，最后进行 5 Hz 因果低通。无未来帧、filtfilt、居中差分或峰值裁剪；只对补偿量设置训练数据范围上限，不裁剪外力残差。',
        '', '公式：r_final = H5[ H12(tau_measured) − H12(mu_NEXT) − c(filtered_motion) ]。c 没有力矩/残差输入，没有常数截距，不在线学习、不按静止标志归零。',
        '',f'固定运动状态的合成 0.3 N·m 阶跃验证：稳态增益 {comparison["contact_transfer"]["settled_gain"]:.6f}，达到 90% 约 {comparison["contact_transfer"]["rise_90pct_ms"]} ms，20 ms 时约保留 {100*comparison["contact_transfer"]["response_at_20ms"]:.1f}% 幅值。短碰撞脉冲会被削弱；本结果不是完整 LSTM/接触链的物理延迟标定。','',
        '真实接触时运动状态也可能变化，本次固定状态注入不能证明所有接触均不被削弱。需要独立接触真值验证；新输出不接入控制、急停或碰撞保护。原 sigma 对补偿后残差不再有效，新窗口不提供该置信带。','',
        '## 实现与验证','',
        '训练中没有充分激励的特征列固定为零权重，不用于外推判据，避免未激励关节的编码抖动使补偿频繁开关。v1 保留为初版记录，v2 修正此训练可辨识性问题；两版都是回顾性研究，不把已查看的第 3 次往复包装成新盲测。',
        '', '数值验证包括：因果性、速度估计的量化偏置隔离、成对力矩滤波一致性、持续外力阶跃、超范围回退、时间回退/断流重置、21,945 帧逐帧训练/实时实现一致性。隔离 ROS domain 199 验证传输和超时失效，并检查只有观测发布器。','',
        '新窗口：http://127.0.0.1:8083/ ，默认最终平滑输出，可勾选补偿前 NEXT 原始曲线与输出滤波前曲线。原始模型窗口 http://127.0.0.1:8082/ 保留。','',
        '![J1对比](j1_comparison.png)','','![各关节对比](all_joints_comparison.png)','',
        '逐关节数值与粗糙度：heldout_detail.json；候选、选择与历史验证：comparison.json；完整回放：replay.npz；模型与姿态范围：model.json。']
    (experiment/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(test_motion_rows=summary['test_motion']['rows'],j1_original_rms=m['original']['rms_nm'][0],j1_final_rms=m['final']['rms_nm'][0],j1_roughness_reduction=1-rough1/rough0,report=str(experiment/'REPORT_ZH.md')),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    run(p.parse_args().experiment.resolve())
