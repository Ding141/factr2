"""Summarize joint torque observation, tracking, timing and thermal evidence.

Offline only. Never fits delays, zeros residuals, trains or transmits commands.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0)
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),rms=float(np.sqrt(np.mean(x*x))),
        abs_p95=float(np.percentile(abs(x),95)),abs_max=float(max(abs(x))),minimum=float(x.min()),maximum=float(x.max()))


def report(stage,allow_partial=False):
    state=json.loads((stage/'test_state.json').read_text());assert not state['recording_active']
    complete=state['status']=='completed'
    assert complete or allow_partial,'Use --allow-partial to label interrupted data explicitly'
    assert json.loads((stage/'recorder_state.json').read_text())['status']=='closed'
    spec=json.loads((stage/'test_spec.json').read_text());session=Path(state['session']);audit=session/'audit'
    run=json.loads((audit/'motion_run.json').read_text());assert run['status']=='completed' or allow_partial
    assert hashlib.sha256((stage/'motion.yaml').read_bytes()).hexdigest()==spec['plan_sha256']
    assert hashlib.sha256((Path(spec['model_dir'])/'model.pt').read_bytes()).hexdigest()==spec['model_hash']
    plan=yaml.load((stage/'motion.yaml').read_text(),Loader=yaml.CSafeLoader)
    can=json.loads((audit/'can_record_state.json').read_text());quality=json.loads((session/'quality.json').read_text())
    assert can['status']=='closed' and can['socket_dropped_frames']==can['kernel_timestamps_missing']==0
    with (stage/'samples.csv').open() as f:rows=list(csv.DictReader(f))
    t=np.array([int(x['source_timestamp_ns']) for x in rows],np.int64);assert np.all(np.diff(t)>0)
    def array(key):return np.array([[float(x[f'{key}_j{j+1}']) for j in range(7)] for x in rows])
    q=array('q_deg');v=array('velocity_deg_s');cmd=array('command_deg');raw=array('external_raw_nm');filtered=array('external_filtered_nm')
    measured=array('measured_tau_nm');pred=array('free_pred_tau_nm');np.testing.assert_allclose(measured-pred,raw,atol=1e-5)
    def window(mask):
        assert mask.any()
        return dict(rows=int(mask.sum()),raw_joint_nm=[stats(raw[mask,j]) for j in range(7)],
            filtered_joint_nm=[stats(filtered[mask,j]) for j in range(7)],
            raw_vector_rms_nm=float(np.sqrt(np.mean(np.sum(raw[mask]**2,axis=1)))),
            filtered_vector_rms_nm=float(np.sqrt(np.mean(np.sum(filtered[mask]**2,axis=1)))),
            tracking_error_deg=[stats(q[mask,j]-cmd[mask,j]) for j in range(7)],
            reported_velocity_deg_s=[stats(v[mask,j]) for j in range(7)],
            measured_tau_nm=[stats(measured[mask,j]) for j in range(7)],
            free_pred_tau_nm=[stats(pred[mask,j]) for j in range(7)])
    events=run['events'];assert len(events)==len(plan['steps']) or allow_partial
    segments=[];groups={};moving=np.zeros(len(t),bool)
    for step,event in zip(plan['steps'],events):
        assert step['name']==event['name']
        a,b=event['command_start_ros_ns'],event['command_end_ros_ns'];mask=(t>=a)&(t<=b)
        if event['status']!='completed':continue  # Interrupted trajectory is not a completed segment.
        if step['family'] in ('continuous','speed_probe'):moving|=mask
        key=step['speed']+'/'+step['family'];groups[key]=groups.get(key,np.zeros(len(t),bool))|mask
        segments.append(dict(name=step['name'],family=step['family'],profile=step['speed'],
            start_ros_ns=a,end_ros_ns=b,duration_seconds=(b-a)*1e-9,telemetry=event['telemetry'],**window(mask)))
    static={}
    for label,kind in [('before','initial_settle'),('after','final_settle')]:
        event=next((e for e,s in zip(events,plan['steps']) if s['kind']==kind and e['status']=='completed'),None)
        end=event['command_end_ros_ns'] if event else t[-1]
        seconds=15 if event else 5
        mask=(t>=end-seconds*1_000_000_000)&(t<=end);static[label]=dict(window(mask),actual_pose_mean_deg=q[mask].mean(0).tolist(),native_target_mean_deg=cmd[mask].mean(0).tolist(),
            window_seconds=seconds,protocol_settle_completed=event is not None)
    # Exact source-frame coverage in the temporal overlap of the two recorders.
    source=[]
    with h5py.File(quality['path'],'r') as h:
        for key in h:source.extend(h[key]['joint_pos/timestamps'][:].tolist())
    source=np.array(source,np.int64);overlap=source[(source>=t[0])&(source<=t[-1])]
    joined=np.intersect1d(overlap,t)
    # Separate cumulative lifetime counters from changes during this session.
    counters=[];messages=Counter();infer_ms=[];adapter_ages=[]
    for line in (stage/'diagnostics.jsonl').open():
        e=json.loads(line)
        for st in e['status']:
            x=st['values'];messages[(st['name'],st['message'])]+=1
            if st['name']=='factr2/next/right':
                counters.append((e['source_timestamp_ns'],x.get('dropped',0),x.get('resets',0)))
                if x.get('infer_ms') is not None:infer_ms.append(x['infer_ms'])
            elif st['name']=='factr2/adapter/right':
                if x.get('state_source_age_seconds') is not None:adapter_ages.append(x['state_source_age_seconds']*1000)
    temperatures={}
    with (audit/'motor_temperature.csv').open() as f:
        temp_rows=list(csv.DictReader(f))
    # Capture stores nominal Celsius from firmware, not calibrated temperature.
    for row in temp_rows:
        motor=row.get('motor_index',row.get('motor_id',row.get('motor','')))
        for field in ('mos_c','rotor_c'):
            if row.get(field) not in (None,''):
                key=str(motor)+'/'+field;temperatures.setdefault(key,[]).append(float(row[field]))
    temp={k:dict(start_nominal_c=a[0],end_nominal_c=a[-1],min_nominal_c=min(a),max_nominal_c=max(a)) for k,a in temperatures.items()}
    result=dict(status='completed' if complete else 'interrupted_partial',stop_reason=run.get('error'),stage=str(stage),session=str(session),model_sha256=spec['model_hash'],
        window_view='joint_torque',units='Nm; no endpoint force conversion',ground_truth=spec['ground_truth'],
        planned_command_seconds=spec['command_seconds'],recorded_span_seconds=float((t[-1]-t[0])*1e-9),
        completed_steps=sum(e['status']=='completed' for e in events),attempted_steps=len(events),planned_steps=len(plan['steps']),sample_rows=len(t),quality_accepted=quality['accepted'],
        source_interval_ms=stats(np.diff(t)*1e-6),observation_receive_minus_source_ms=stats((np.array([int(x['received_wall_ns']) for x in rows],np.int64)-t)*1e-6),
        source_overlap_rows=len(overlap),exact_joined_rows=len(joined),source_join_fraction=float(len(joined)/len(overlap)),
        next_states=dict(Counter(x['next_state'] for x in rows)),
        recorded_position_range_deg=np.c_[q.min(0),q.max(0)].tolist(),native_position_range_deg=np.c_[cmd.min(0),cmd.max(0)].tolist(),
        whole=window(np.ones(len(t),bool)),motion=window(moving),stationary=static,
        groups={k:window(mask) for k,mask in groups.items()},segments=segments,
        next_lifetime_counters_first=list(counters[0]) if counters else None,
        next_lifetime_counters_last=list(counters[-1]) if counters else None,
        next_counter_change=dict(dropped=counters[-1][1]-counters[0][1],resets=counters[-1][2]-counters[0][2]) if counters else None,
        diagnostic_messages={str(k):v for k,v in messages.items()},inference_ms=stats(infer_ms),adapter_source_age_ms=stats(adapter_ages),
        can=can,temperature=temp,model_or_controller_changes=False,time_shift_fit_or_zero_subtraction=False,
        limitations=['Unloaded truth is operator declaration, no independent torque/force sensor',
            'Recorded velocity is motor feedback including quantization; not independent physical speed truth',
            'ROS/control source stamps are not motor acquisition timestamps',
            'No direct-contact, direction, sensitivity or unloading accuracy certification',
            'Profiles cap different motions; group differences do not isolate causal speed/acceleration effects'])
    (stage/'observation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    time=(t-t[0])*1e-9;colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(7,1,figsize=(14,15),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(time,raw[:,j],color=colors[j],linewidth=.5,label=f'J{j+1} raw')
        ax.plot(time,filtered[:,j],color='black',linewidth=.7,alpha=.7,label='EMA')
        ax.axhline(0,color='gray',linewidth=.6);ax.set_ylabel('Nm');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
    boundaries=[]
    for p in plan['config']['profiles']:
        event=next((e for e,s in zip(events,plan['steps']) if s['speed']==p['id']),None)
        if event is None:continue
        x=(event['command_start_ros_ns']-t[0])*1e-9;boundaries.append(x)
        for ax in axes:ax.axvline(x,color='gray',linestyle='--',alpha=.5)
        axes[0].text(x,1.02,f'{p["velocity_deg_s"]} deg/s cap',transform=axes[0].get_xaxis_transform(),fontsize=8)
    axes[-1].set_xlabel('Time from joined recording start (s)')
    fig.suptitle(('COMPLETED' if complete else 'INTERRUPTED / partial data')+' unloaded right arm: external joint torque\nOriginal NEXT; no bias subtraction; changes in pose, profile and temperature coexist')
    fig.tight_layout();fig.savefig(stage/'joint_torque_full.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
    for j in range(7):
        axes[0].plot(time,q[:,j]-cmd[0,j],color=colors[j],linewidth=.7,label=f'J{j+1}')
        axes[1].plot(time,q[:,j]-cmd[:,j],color=colors[j],linewidth=.6)
    axes[0].set_ylabel('Measured - home (deg)');axes[0].legend(ncol=7,fontsize=8)
    axes[1].set_ylabel('Measured - native (deg)')
    axes[2].plot(time,np.linalg.norm(raw,axis=1),label='Raw residual vector norm',linewidth=.7)
    axes[2].plot(time,np.linalg.norm(filtered,axis=1),label='Filtered residual vector norm',linewidth=.7)
    axes[2].set_ylabel('Vector norm (Nm)');axes[2].set_xlabel('Time (s)');axes[2].legend()
    for ax in axes:
        for x in boundaries:ax.axvline(x,color='gray',linestyle='--',alpha=.5)
        ax.grid(alpha=.2)
    fig.suptitle(('Completed' if complete else 'Interrupted')+' larger-amplitude motion: tracking and unloaded NEXT residual')
    fig.tight_layout();fig.savefig(stage/'tracking_overview.png',dpi=140);plt.close(fig)
    lines=['# 较大幅度、长时间空载观测','',
        f'完成 {result["completed_steps"]}/{len(plan["steps"])} 段，尝试 {len(events)} 段；规划命令时间 {spec["command_seconds"]:.2f} 秒；七路同步结果 {len(t)} 行，跨度 {result["recorded_span_seconds"]:.2f} 秒。',
        '最新窗口显示七关节外部力矩（N·m），没有雅可比换算或自动归零。原版NEXT模型、增益、重力和摩擦前馈保持固定。','']
    if not complete:lines+=['',f'**实验中断，未完成长时间协议。停止原因：{run.get("error")}。未完成运动段不计入完成段统计；最后静止值取停止后记录末尾5秒，不是规划的最终30秒静止段。**','']
    lines+=['|阶段|七轴原始残差向量 RMS (Nm)|七轴滤波残差向量 RMS (Nm)|','|---|---:|---:|']
    for label,x in [('初始静止最后15秒',static['before']),('已完成运动（含小幅验证）',result['motion']),('最终静止最后15秒' if complete else '停止后记录末尾5秒',static['after'])]:
        lines.append(f'|{label}|{x["raw_vector_rms_nm"]:.4f}|{x["filtered_vector_rms_nm"]:.4f}|')
    lines+=['','向量RMS是七个关节力矩的统计量，不是末端力。','',
        '|关节|实际角度范围 (°)|运动原始残差 RMS (Nm)|运动绝对残差 P95 (Nm)|运动绝对峰值 (Nm)|初始静止均值 (Nm)|最终静止均值 (Nm)|最大跟踪偏差 (°)|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for j in range(7):
        x=result['motion']['raw_joint_nm'][j];angle=result['recorded_position_range_deg'][j]
        lines.append(f'|J{j+1}|{angle[0]:.2f}～{angle[1]:.2f}|{x["rms"]:.4f}|{x["abs_p95"]:.4f}|{x["abs_max"]:.4f}|{static["before"]["raw_joint_nm"][j]["mean"]:+.4f}|{static["after"]["raw_joint_nm"][j]["mean"]:+.4f}|{result["whole"]["tracking_error_deg"][j]["abs_max"]:.3f}|')
    lines+=['','## 数据质量与解释边界','',
        f'H5质量检查：{quality["accepted"]}；CSV最大间隔 {result["source_interval_ms"]["maximum"]:.2f} ms；共同时间范围内，七路完整记录覆盖H5源帧 {result["source_join_fraction"]:.2%}。',
        f'原始CAN {can["frames"]} 帧，接收丢帧 {can["socket_dropped_frames"]}，内核时间戳缺失 {can["kernel_timestamps_missing"]}；本次NEXT计数变化 {result["next_counter_change"]}。累计计数包含启动前的恢复，不能当作本次丢帧。',
        '空载真值来自操作者说明，仍无独立外部力矩传感器。不同速度档同时改变了轨迹和时间/温度，不能仅靠档位均值宣称惯性或热漂移的因果关系。反馈速度、源时间与电机采样时间也必须区分。',
        '温度为固件提供的标称MOS/转子摄氏度；温度、各轴实测/预测力矩、分段误差、推理耗时和状态统计见 observation_summary.json。','',
        '![关节外部力矩全程](joint_torque_full.png)','', '![运动跟踪概览](tracking_overview.png)','',
        f'原始会话：{session}。保留H5、完整压缩bag、原始CAN、温度、实际参数、模型标识和冻结轨迹。',
        '本次为独立诊断数据，不自动转入训练；没有拟合或补偿新时延。']
    (stage/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Saved report',stage/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args();report(args.stage.resolve(),args.allow_partial)
