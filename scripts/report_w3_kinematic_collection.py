"""Report new physical block holdouts without selecting on test results."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fit_w3_kinematic_compensation import recent, predict, smooth


def run(experiment):
    recipe = json.loads((experiment/'recipe.json').read_text())
    model = json.loads((experiment/'model.json').read_text())
    comparison = json.loads((experiment/'comparison.json').read_text())
    item = recent(Path(recipe['observation'])/'labeled_samples.csv', recipe['block_roles'])
    with np.load(experiment/'replay.npz') as replay:
        variants = {k:replay[k] for k in ('original','filter_only','final')}
        supported = replay['supported']
        np.testing.assert_array_equal(replay['roles'], item['role'])
        np.testing.assert_array_equal(replay['timestamp_ns'], item['t'])
    for mode in ('velocity', 'velocity_acceleration'):
        artifact = json.loads((experiment/(mode+'.json')).read_text())
        correction, _ = predict(item, artifact)
        variants[mode] = smooth(item['target']-correction, item['t'], model['signal']['output_hz'])
    test = (item['role']=='test') & item['eligible']
    masks = {'validation':(item['role']=='val')&item['eligible'], 'test_all':test,
             'test_motion':test&(item['labels']!='baseline_after'),
             'test_static':test&(item['labels']=='baseline_after')}
    for profile in ('identify_slow', 'identify_fast'):
        masks['test_'+profile] = test & np.char.startswith(item['labels'],profile)

    def correlation(x,y):
        if len(x)<2 or x.std()<1e-10 or y.std()<1e-10:
            return None
        return float(np.corrcoef(x,y)[0,1])

    report = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError('empty evaluation slice: '+name)
        continuous = mask[1:] & mask[:-1] & (np.diff(item['t'])<=40_000_000)
        result = dict(rows=int(mask.sum()), supported_fraction=float(supported[mask].mean()), variants={})
        for label, values in variants.items():
            selected = values[mask]
            result['variants'][label] = dict(
                rms_nm=np.sqrt(np.mean(selected**2,axis=0)).tolist(),
                bias_nm=selected.mean(0).tolist(), absolute_p95_nm=np.percentile(abs(selected),95,axis=0).tolist(),
                absolute_peak_nm=abs(selected).max(0).tolist(),
                first_difference_rms_nm=np.sqrt(np.mean(np.diff(values,axis=0)[continuous]**2,axis=0)).tolist(),
                own_velocity_correlation=[correlation(item['features'][mask,j],selected[:,j]) for j in range(7)],
                own_acceleration_correlation=[correlation(item['features'][mask,56+j],selected[:,j]) for j in range(7)])
        report[name] = result
    (experiment/'heldout_detail.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

    start = np.flatnonzero(item['labels']=='baseline_before')[0]
    t = (item['t']-item['t'][start])*1e-9
    show = np.flatnonzero(t>=0)[::max(1,int((t>=0).sum()/20000))]
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    fig, axes = plt.subplots(7,1,figsize=(14,14),sharex=True)
    for j, axis in enumerate(axes):
        for label,color,width in [('original','#a8adb5',.55),('filter_only','#af7ac5',.7),('final','#1670ca',1.)]:
            axis.plot(t[show],variants[label][show,j],label=label,color=color,lw=width)
        for role,color in [('train','#269ab5'),('val','#e4b21b'),('test','#27ae60')]:
            axis.fill_between(t[show],0,1,where=item['role'][show]==role,
                              transform=axis.get_xaxis_transform(),color=color,alpha=.06,
                              label=role if j==0 else None)
        extent=np.concatenate([variants[label][show,j] for label in ('original','filter_only','final')])
        low,high=min(0.,float(extent.min())),max(0.,float(extent.max()))
        margin=.08*max(.02,high-low)
        axis.set_ylim(low-margin,high+margin)
        axis.axhline(0,color='black',lw=.5); axis.set_ylabel(f'J{j+1} (Nm)')
    axes[0].legend(ncol=6); axes[0].set_title('New no-load collection: predeclared whole-block holdout replay')
    axes[-1].set_xlabel('Time from working-pose baseline (s)')
    fig.tight_layout(); fig.savefig(experiment/'all_joints_comparison.png',dpi=150); plt.close(fig)
    fig, axes = plt.subplots(7,1,figsize=(14,12),sharex=True)
    for j, axis in enumerate(axes):
        axis.plot(t[show],np.rad2deg(item['x'][show,j]),label='measured',lw=.8)
        axis.plot(t[show],np.rad2deg(item['x'][show,j]+item['x'][show,14+j]),'--',label='command',lw=.7)
        axis.set_ylabel(f'J{j+1} (deg)')
    axes[0].legend(ncol=2); axes[-1].set_xlabel('Time from working-pose baseline (s)')
    fig.tight_layout(); fig.savefig(experiment/'joint_motion.png',dpi=150); plt.close(fig)

    motion = report['test_motion']['variants']
    enabled = ', '.join(f'J{j+1}' for j,value in enumerate(model['enabled_joints']) if value)
    lines = ['# 新采空载数据：速度与加速度补偿', '',
             '本次右臂执行新采集轨迹，模型拟合使用原始 ROS 精度，不使用窗口四舍五入值。'
             '空载由现场人员确认；无独立外力矩传感器，以下衡量空载残差，不能当作已标定的接触测量精度。', '',
             '训练、验证、测试角色在采集前按完整运动块冻结；每段首尾各排除 1 秒。'
             '只用训练段拟合、验证段选择模型与启用关节，测试结果未参与参数选择。'
             '历史记录级划分保持不变，已查看过的历史测试仅作回顾性比较。', '',
             f'选中：{model["feature_mode"]}，{model["scope"]}，ridge={model["ridge"]}；启用 {enabled}。'
             '未通过验证的关节只滤波，补偿不增加常数截距、不输入力矩残差、不在线调零。', '',
             f'新测试运动段 {report["test_motion"]["rows"]} 帧，处于补偿支持范围的比例 '
             f'{100*report["test_motion"]["supported_fraction"]:.2f}%。', '',
             '|关节|原始 RMS N·m|仅滤波|仅速度＋滤波|速度加速度＋滤波|选中输出|相对原始改善|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for j in range(7):
        raw = motion['original']['rms_nm'][j]
        selected = motion['final']['rms_nm'][j]
        cells = [motion[k]['rms_nm'][j] for k in ('original','filter_only','velocity','velocity_acceleration','final')]
        lines.append('|J'+str(j+1)+'|'+'|'.join(f'{value:.5f}' for value in cells)+f'|{100*(1-selected/max(raw,1e-12)):.1f}%|')
    weaker = [f'J{j+1}' for j in range(7) if motion['final']['rms_nm'][j]>motion['velocity']['rms_nm'][j]]
    static = report['test_static']['variants']
    slow = report['test_identify_slow']['variants']
    low_j1_gain = 100*(1-slow['final']['rms_nm'][0]/slow['original']['rms_nm'][0])
    lines += ['', '加速度相对仅速度的额外收益有限；本次测试 '+', '.join(weaker)+
              ' 的速度＋加速度结果略差于仅速度。模型仍按验证集选择，未用测试结果重新挑选关节参数。', '',
              f'分档统计中，低速 J1 的 RMS 改善仅 {low_j1_gain:.1f}%，不能用整体平均掩盖这一点。'
              f'静止测试 J1 最终 RMS {static["final"]["rms_nm"][0]:.5f} N·m，'
              f'J4 为 {static["final"]["rms_nm"][3]:.5f} N·m；'
              '速度/加速度补偿不能保证清除静止偏置。']
    lines += ['', '负改善表示变差，不应通过抹平或自动归零隐藏。以上是同次采集中留出的新运动块，'
              '仍不能证明不同姿态、温度、速度或真实接触时的泛化。', '',
              '补偿形式为 H5[H12(实测力矩)−H12(NEXT 自由预测)−c(因果速度、加速度与方向特征)]。'
              '实际位置与指令位置经常加速度卡尔曼滤波得到运动学量，保留输入/输出滤波既有配置。'
              '超出训练支持范围时补偿回退为零，保留残差；新姿态需要补充数据。', '',
              f'固定运动学的 0.3 N·m 合成阶跃稳态增益 {comparison["contact_transfer"]["settled_gain"]:.6f}，'
              f'90% 响应约 {comparison["contact_transfer"]["rise_90pct_ms"]} ms。'
              '这是信号链测试，未代替实机接触实验；新输出不用于安全联锁或力控。', '',
              '完整的偏置、峰值、运动/静止分项、两档速度分项及速度/加速度相关系数见 heldout_detail.json。', '',
              '![七关节补偿对比](all_joints_comparison.png)', '', '![七关节运动](joint_motion.png)', '',
              f'采集记录：{recipe["observation"]}', f'冻结轨迹 SHA256：{recipe["plan_sha256"]}']
    (experiment/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.experiment.resolve())
    print(json.dumps(result['test_motion'],ensure_ascii=False,indent=2))
