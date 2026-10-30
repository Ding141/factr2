"""Render the fixed diagnostic results; no estimator/controller tuning."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from w3_targeted_protocol import sample


def render(stage):
    r=json.loads((stage/'feedback_analysis.json').read_text());a=np.load(stage/'feedback_arrays.npz')
    plan=yaml.load((stage/'motion.yaml').read_text(),Loader=yaml.CSafeLoader)
    trial=next(t for t in r['trials'] if t['joint']==1 and t['velocity_deg_s']==2.4 and t['direction']=='out')
    step=next(s for s in plan['steps'] if s['name']==trial['name']);start=trial['start_ros_ns'];stop=trial['end_ros_ns']
    gt=(a['grid_stamps']-start)*1e-9;t=(a['stamps']-start)*1e-9;nt=(a['native_stamps']-start)*1e-9
    gm=(gt>=-1)&(gt<=(stop-start)*1e-9+2);sm=(t>=-1)&(t<=(stop-start)*1e-9+2);nm=(nt>=-1)&(nt<=(stop-start)*1e-9+2)
    expected=sample(step['knots'],np.clip(gt[gm],0,step['duration_seconds']))
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    axes[0].plot(nt[nm],a['native_q_deg'][nm,0],label='native target')
    axes[0].plot(t[sm],a['q_deg'][sm,0],label='reported position',linewidth=1)
    axes[0].set_ylabel('J1 position (deg)');axes[0].legend(loc='upper left')
    axes[1].plot(t[sm],a['v_deg_s'][sm,0],label='reported velocity',alpha=.5,linewidth=.8)
    axes[1].plot(nt[nm],a['native_v_deg_s'][nm,0],label='native target',color='black',linestyle='--')
    for w in [11,21,51]: axes[1].plot(gt[gm],a[f'SG{w}_velocity_deg_s'][gm,0],label=f'position SG{w}',linewidth=1)
    axes[1].set_ylabel('Velocity (deg/s)');axes[1].legend(ncol=3,fontsize=8)
    axes[2].plot(gt[gm],expected[2][:,0],label='commanded acceleration',color='black',linestyle='--')
    for w in [11,21,51]: axes[2].plot(gt[gm],a[f'SG{w}_acceleration_deg_s2'][gm,0],label=f'position SG{w}',linewidth=1)
    axes[2].set_ylabel('Acceleration (deg/s²)');axes[2].legend(ncol=2,fontsize=8)
    axes[3].plot(t[sm],a['raw_nm'][sm,0],label='J1 NEXT raw residual')
    axes[3].set_ylabel('Residual (Nm)');axes[3].set_xlabel('Time from trajectory start (s)');axes[3].legend()
    for ax in axes:
        ax.axvspan(*[(v-start)*1e-9 for v in trial['plateau_trimmed_ros_ns']],color='green',alpha=.07)
        ax.grid(alpha=.25)
    fig.suptitle('Unloaded J1, target 2.4 deg/s: feedback, derivative sensitivity and residual\nNo time shift fitted; centered SG curves are offline estimates')
    fig.tight_layout();fig.savefig(stage/'tracking_detail.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    x=np.arange(7)
    for w in [11,21,31,51]:
        axes[0].plot(x+1,[r['stationary'][f'J{j+1}']['before']['SG'][str(w)]['acceleration_std_deg_s2'] for j in x],marker='o',label=f'SG{w}')
    axes[0].set_yscale('symlog',linthresh=.01);axes[0].set_xlabel('Joint');axes[0].set_ylabel('Static estimated acceleration std (deg/s²)');axes[0].legend()
    ts=r['trials'];levels=[.6,1.2,2.4];colors=['tab:blue','tab:orange','tab:green']
    for level,color in zip(levels,colors):
        g=[t for t in ts if t['velocity_deg_s']==level]
        axes[1].scatter([t['encoder_displacement_deg'] for t in g],[t['velocity_integral_deg'] for t in g],label=f'target {level} deg/s',color=color,alpha=.7)
    lim=axes[1].get_xlim();axes[1].plot(lim,lim,'k--',label='equal displacement');axes[1].set_xlabel('Position displacement over plateau (deg)');axes[1].set_ylabel('Reported velocity integral (deg)');axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle('Quantization/derivative sensitivity and independent kinematic consistency checks')
    fig.tight_layout();fig.savefig(stage/'feedback_consistency.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for stamps,position,velocity,label in [
        ('CAN_j1_stamps','CAN_j1_q_deg','CAN_j1_v_deg_s','raw CAN'),
        ('bridge_state_stamps','bridge_state_q_deg','bridge_state_v_deg_s','bridge state'),
        ('software_state_stamps','software_state_q_deg','software_state_v_deg_s','joint states')]:
        lt=(a[stamps]-start)*1e-9;lm=(lt>=0)&(lt<=4.)
        lq=a[position] if a[position].ndim==1 else a[position][:,0]
        lv=a[velocity] if a[velocity].ndim==1 else a[velocity][:,0]
        axes[0].plot(lt[lm],lq[lm],label=label,linewidth=1)
        axes[1].plot(lt[lm],lv[lm],label=label,linewidth=.8,alpha=.7)
    lm=(nt>=0)&(nt<=4.);axes[0].plot(nt[lm],a['native_q_deg'][lm,0],'k--',label='native target',alpha=.6)
    idx=np.searchsorted(a['bridge_state_stamps'],a['software_state_stamps'],side='right')-1;m=idx>=0
    lt=(a['software_state_stamps'][m]-start)*1e-9;lm=(lt>=0)&(lt<=4.)
    error=a['software_state_q_deg'][m,0]-a['bridge_state_q_deg'][idx[m],0]
    axes[2].plot(lt[lm],error[lm]);axes[0].set_ylabel('J1 position (deg)');axes[1].set_ylabel('J1 velocity (deg/s)')
    axes[2].set_ylabel('Joint - preceding bridge (deg)');axes[2].set_xlabel('Time from command start (s)')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(ncol=4);axes[1].legend(ncol=3)
    fig.suptitle('Cross-layer comparison at original timestamps\nNo lag search or timestamp correction; RX/state headers do not establish device sample time')
    fig.tight_layout();fig.savefig(stage/'cross_layer_detail.png',dpi=160);plt.close(fig)
    maximum=max(c['native_vs_frozen_position_max_deg'] for c in r['native_command_checks'])
    j1=next(t for t in ts if t['joint']==1 and t['velocity_deg_s']==2.4 and t['direction']=='out')
    raw_max=max(max(abs(s['CAN_to_bridge']['position_error_deg'][k]) for k in ['minimum','maximum']) for s in r['raw_to_software'].values())
    joint_max=max(max(abs(s['bridge_to_joint']['position_error_deg'][k]) for k in ['minimum','maximum']) for s in r['raw_to_software'].values())
    lines=['# 位置、速度反馈与底层跟踪专项验证','',
        f"执行完成：{len(plan['steps'])} 段，{r['trial_count']} 次正式往返单程，{r['sample_rows']} 行 NEXT 同步结果。",
        '左臂保持不动；右臂回到原目标。原版 NEXT、控制增益、摩擦前馈及在线滤波未修改。本批为独立诊断数据，不混入训练。',
        '', '## 本轮发现及处理优先级','',
        f'1. **优先排查底层反馈积压。** 按原时间戳、只匹配此前最近消息，CAN 与驱动发布状态的最大位置差约 {raw_max:.4f}°，但驱动状态到 `/joint_states` 的最大位置差约 {joint_max:.3f}°。原生目标和驱动入口命令吻合。这把较大的反馈阶段性落后定位在驱动发布之后、关节状态输出之前；不能把全部目标—反馈相位差当作机械响应。',
        '   真实日志中驱动状态约350 Hz、硬件循环约300 Hz。当前硬件接口订阅 `KeepLast(50)`，每次 read 调用 `rclcpp::spin_some`，随后用控制周期时间戳重新发布关节状态，没有保留被消费反馈的原始时间戳。这与队列积压相符；应优先检查最新样本消费和源时间戳传播。这里没有在测试日志上估计或应用新的补偿延迟。',
        '   adapter记录的状态/命令源时间戳相同并不推翻这个发现：硬件接口的 `on_motor_state` 把位置、速度、力矩写入缓存时没有传播接收消息原header；关节广播使用新的控制周期时间戳。因此下游 `source_skew=0` 不能证明真实采样同步。旧状态与新目标混合会影响NEXT的跟踪误差输入，可能被解释为外力；本轮没有做参数修正来证明其贡献比例。',
        '2. **速度存在接近半个码的基线偏差。** 不同方向的“速度平台均值−位置位移均速”与静止速度基线接近，不能仅按随机噪声解释。以下是原始差值的描述统计，没有从数据中拟合修正系数或修改输入：',
        '', '|关节|平台速度差范围 (°/s)|静止前速度均值 (°/s)|', '|---|---:|---:|']
    for joint in range(1,8):
        g=[t for t in ts if t['joint']==joint];d=[t['reported_velocity_average_deg_s']-t['encoder_average_speed_deg_s'] for t in g]
        lines.append(f"|J{joint}|{min(d):+.3f} 至 {max(d):+.3f}|{r['stationary'][f'J{joint}']['before']['raw_velocity_deg_s']['mean']:+.3f}|")
    if r.get('source_feedback_available'):
        first=next(i for i,line in enumerate(lines) if line.startswith('1. **优先排查底层反馈积压。**'))
        source=r['layer_rates']['source_state']['bag_receive_minus_header_ms']
        lines[first:first+3]=[
            f'1. **已部署最新反馈消费，并保留源时间。** 原时间戳对照：CAN→桥接最大位置差 {raw_max:.4f}°，桥接→关节状态最大位置差 {joint_max:.4f}°。新源时间话题收录 {r["layer_rates"]["source_state"]["rows"]} 帧，入袋时源年龄 P95={source["p95"]:.3f} ms。',
            '   硬件接口采用 KeepLast(1)、持久执行器、源时间与新鲜度检查；没有重新拟合或补偿本批数据的延迟。源时间来自桥接发布，仍不能证明电机实际采样时刻。具体修复前后结果见 comparison.md。']
    lines += ['', '   需核实电机固件的零速度码定义和协议映射。修复时须统一训练、采集和推理的速度定义；本轮没有自动扣除静止基线。',
        f"3. **跟踪脉动与求导放大同时存在。** J1 目标峰速2.4°/s时，原始 CAN 速度峰值约 {j1['raw_CAN_peak_speed_deg_s']:.2f}°/s；原始 CAN 位置和速度都出现脉动，说明不能全部归为数值求导伪影。但位置求导的加速度峰值随SG11/21/51分别为 {j1['SG']['11']['peak_acceleration_deg_s2']:.1f}/{j1['SG']['21']['peak_acceleration_deg_s2']:.1f}/{j1['SG']['51']['peak_acceleration_deg_s2']:.1f}°/s²，而目标加速度仅3°/s²。这些峰值不是独立测得的真实加速度。",
        '   若 CAN 层仍有脉动，需要进一步分辨摩擦、死区、重力前馈和底层闭环动态。当前证据不支持直接依据某个求导窗口增加惯性扣除项。']
    probe=stage/'queue_probe_result.json'
    if probe.exists():
        q=json.loads(probe.read_text())
        lines += ['', '隔离的软件复现（专用ROS域、无机器人话题）采用350 Hz发布、300 Hz调用相同的 `spin_some` 读取方式：',
            f"深度50的缓存样本年龄中位数 {q['depth50']['cache_age_p50_ms']:.2f} ms；深度1为 {q['depth1']['cache_age_p50_ms']:.2f} ms。该结果支持积压机制，但只是模拟消息实验，不是对真机时延的拟合或新补偿标定。"]
    lines += [
        '', '## 运动与分析方法','',
        '起止各静止15秒；七关节安全探测后再静止10秒；七关节分别以0.6°/s走2.4°往返；J1/J2/J4另以1.2与2.4°/s走6°往返。每次到位单独标注3秒停留。每个运动段一次发送完整原生轨迹，不由上位机分段插值发送。',
        '采集前冻结 SG11/21/31/51（100 Hz、三阶多项式）和有限差分对照；匀速平台两端各裁0.25秒，用位置位移与速度积分核对。所有跨层匹配只取此前最近样本，没有拟合、搜索或应用新的时间平移。',
        '', '## 信号链与量化','',
        f"原始 CAN 共 {r['can']['frames']} 帧，内核时间戳缺失和接收丢帧均为0；CAN 时间是主机接收时间，不是电机采样时间。NEXT七路同步CSV最大间隔 {r['source_max_gap_ms']:.2f} ms，共 {len(r['source_gaps'])} 个≥40 ms缺口；已排除所有触及缺口的求导窗口。正式运动段的窗口完整性由分析程序逐段校验。",
        '', '|关节|位置 LSB (°)|速度 LSB (°/s)|静止前位置跨度 (°)|静止前速度均值 (°/s)|SG11/SG51 静止加速度标准差 (°/s²)|',
        '|---|---:|---:|---:|---:|---:|']
    for j in range(1,8):
        q=r['quantization'][f'J{j}'];s=r['stationary'][f'J{j}']['before'];sg=s['SG']
        lines.append(f"|J{j}|{q['position_lsb_deg']:.5f}|{q['velocity_lsb_deg_s']:.3f}|{s['raw_position_range_deg']:.4f}|{s['raw_velocity_deg_s']['mean']:.3f}|{sg['11']['acceleration_std_deg_s2']:.3f} / {sg['51']['acceleration_std_deg_s2']:.3f}|")
    lines+=['', '速度映射为偶数个离散码，在当前标称解码下零附近为半个速度码；不能把静止时的小非零值直接当作真实运动。量化只是误差来源之一，内部滤波、编码器与电机固件尚未独立标定。求导的量化界仅覆盖位置量化误差，不覆盖真实动态与平滑偏差。',
        '', '## 原生目标与底层跟踪','',
        f"原生目标与冻结轨迹的最大位置差 {maximum:.6f}°。桥接命令和原生目标按原时间戳比较，各轴最大位置差为 {[round(x,6) for x in r['bridge_command_check']['position_max_error_deg']]}°。",
        '记录到了硬件接口发布给驱动的命令；原始 CAN 没有独立可用的发送回显，因此重建的命令量化值只是按当前驱动映射计算，不能证明实际发送时刻或电机内部执行延迟。',
        '', '|关节/方向|目标速度 (°/s)|平台位移均速 (°/s)|反馈速度均值 (°/s)|速度积分−位移 (°)|SG11/SG51 速度峰值 (°/s)|SG11/SG51 加速度峰值 (°/s²)|跟踪误差绝对峰值 (°)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in ts:
        if t['velocity_deg_s']!=2.4: continue
        sg=t['SG'];e=t['actual_position_vs_native_target_error_deg']
        lines.append(f"|J{t['joint']} {t['direction']}|{t['velocity_deg_s']:.1f}|{t['encoder_average_speed_deg_s']:.3f}|{t['reported_velocity_average_deg_s']:.3f}|{t['velocity_integral_minus_displacement_deg']:.3f}|{sg['11']['peak_speed_deg_s']:.2f} / {sg['51']['peak_speed_deg_s']:.2f}|{sg['11']['peak_acceleration_deg_s2']:.1f} / {sg['51']['peak_acceleration_deg_s2']:.1f}|{max(abs(e['minimum']),abs(e['maximum'])):.3f}|")
    lines+=['', '## 解释边界','',
        '单一求导窗口给出的尖峰不能作为独立真实加速度证据。宽窗口降低噪声，也会衰减真实短时动态；不同窗口的范围是方法敏感性，不是置信区间。若原始速度和位置增量都支持脉冲运动，且原生/桥接目标保持平滑，则更支持实际跟踪或底层反馈链存在脉动；仍需独立编码器才能排除共同反馈误差。',
        '本次无外部速度/力传感器，无接触真值来自操作者说明，未重新估计延迟、修改零偏、在线滤波、模型或控制参数。下一步应根据下面曲线及定量结果区分求导不确定性和真实跟踪问题，再设计干预，不能直接把残差按求导加速度扣除。',
        '', '![跨层核查](cross_layer_detail.png)', '', '![跟踪细节](tracking_detail.png)', '', '![一致性与静止求导](feedback_consistency.png)',
        '', '原始会话：`'+r['session']+'`。完整数字见 `feedback_analysis.json`，冻结运动/方法见 `motion.yaml`、`test_spec.json`。']
    (stage/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);render(p.parse_args().stage.resolve())
