"""Compare two frozen unloaded diagnostics without fitting delays or models."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def peak(s):
    return max(abs(s['minimum']), abs(s['maximum']))


def static_inputs(stage,report):
    run=json.loads((Path(report['session'])/'audit/motion_run.json').read_text())
    with (stage/'samples.csv').open() as f: rows=list(csv.DictReader(f))
    stamps=np.array([int(x['source_timestamp_ns']) for x in rows],np.int64)
    result={}
    for label,kind in [('before','initial_settle'),('after','final_settle')]:
        event=next(x for x in run['events'] if x['kind']==kind)
        end=event['command_end_ros_ns'];mask=(stamps>=end-8_000_000_000)&(stamps<=end)
        selected=[x for x,m in zip(rows,mask) if m]
        assert selected,'Missing stationary input samples'
        result[label]={k:np.mean([[float(x[f'{k}_j{j}']) for j in range(1,8)] for x in selected],axis=0).tolist()
            for k in ('q_deg','command_deg','measured_tau_nm','free_pred_tau_nm')}
    (stage/'static_input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def endpoint_summary(stage, report):
    with (stage/'samples.csv').open() as f:
        rows=list(csv.DictReader(f))
    stamps=np.array([int(r['source_timestamp_ns']) for r in rows],np.int64)
    moving=np.zeros(len(rows),bool)
    for trial in report['trials']:
        moving |= (stamps>=trial['start_ros_ns']) & (stamps<=trial['end_ros_ns'])
    result={}
    for name,mask in [('whole_record',np.ones(len(rows),bool)),('formal_motion',moving)]:
        selected=[r for r,m in zip(rows,mask) if m]
        valid=[r for r in selected if r['endpoint_valid']=='True']
        force=np.array([[float(r[k]) for k in ('fx_n','fy_n','fz_n')] for r in valid]).reshape(-1,3)
        norms=np.linalg.norm(force,axis=1)
        result[name]=dict(rows=len(selected),valid_rows=len(valid),
            valid_fraction=len(valid)/len(selected) if selected else None,
            reasons=dict(Counter(r['endpoint_reason'] for r in selected)),
            force_norm_rms_n=float(np.sqrt(np.mean(norms**2))) if len(norms) else None,
            force_norm_p95_n=float(np.percentile(norms,95)) if len(norms) else None,
            force_norm_max_n=float(norms.max()) if len(norms) else None)
    return result


def compare(before,after):
    stages=[before,after]
    reports=[json.loads((s/'feedback_analysis.json').read_text()) for s in stages]
    specs=[json.loads((s/'test_spec.json').read_text()) for s in stages]
    assert specs[0]['model_hash']==specs[1]['model_hash'],'Model must remain frozen'
    assert specs[0]['feedback_protocol']==specs[1]['feedback_protocol'],'Analysis recipe differs'
    assert [t['name'] for t in reports[0]['trials']]==[t['name'] for t in reports[1]['trials']]
    static_inputs(after,reports[1])
    out=dict(before=str(before),after=str(after),model_sha256=specs[0]['model_hash'],
        no_time_shift_fit_or_applied=True,independent_force_sensor=False,
        causal_effect_identified=False,metrics=[])
    for stage,r in zip(stages,reports):
        source=r['layer_rates'].get('source_state')
        static={}
        for phase in ('before','after'):
            s=[r['stationary'][f'J{j}'][phase] for j in range(1,8)]
            static[phase]=dict(raw_residual_vector_rms_nm=float(np.sqrt(sum(x['residual_mean_nm']**2+x['residual_std_nm']**2 for x in s))),
                joint_mean_nm=[x['residual_mean_nm'] for x in s],joint_std_nm=[x['residual_std_nm'] for x in s])
        out['metrics'].append(dict(stage=str(stage),can_frames=r['can']['frames'],
            can_drops=r['can_health']['socket_dropped_frames'],
            bridge_to_joint_position_max_deg=max(peak(x['bridge_to_joint']['position_error_deg']) for x in r['raw_to_software'].values()),
            CAN_to_bridge_position_max_deg=max(peak(x['CAN_to_bridge']['position_error_deg']) for x in r['raw_to_software'].values()),
            source_bag_age_ms=source['bag_receive_minus_header_ms'] if source else None,
            hardware_feedback_metadata=r.get('hardware_feedback_metadata',{}),
            formal_motion_mean_residual_vector_rms_nm=float(np.mean([t['residual_vector_rms_nm'] for t in r['trials']])),
            stationary=static,endpoint=endpoint_summary(stage,r)))
    (after/'comparison.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    b,n=out['metrics'];lines=['# 最新反馈消费真机对照','',
        '原版 NEXT 模型、增益、滤波和分析窗口保持固定。右臂空载、左臂保持；无独立力传感器。两次实验日期、温度和初始实际姿态不同，因此以下为描述性前后对照，不能单凭残差变化归因于代码修复。','',
        '|指标|修复前|修复后|','|---|---:|---:|']
    for label,key in [('桥接→关节状态最大位置差 (°)','bridge_to_joint_position_max_deg'),('CAN→桥接最大位置差 (°)','CAN_to_bridge_position_max_deg'),('正式运动残差向量 RMS 的单程均值 (Nm)','formal_motion_mean_residual_vector_rms_nm')]:
        lines.append(f'|{label}|{b[key]:.6f}|{n[key]:.6f}|')
    for phase,label in [('before','初始静止'),('after','最终静止')]:
        lines.append(f'|{label}残差向量 RMS (Nm)|{b["stationary"][phase]["raw_residual_vector_rms_nm"]:.4f}|{n["stationary"][phase]["raw_residual_vector_rms_nm"]:.4f}|')
    if n['source_bag_age_ms']:
        age=n['source_bag_age_ms'];lines+=['',f'新源时间话题入袋年龄：P50={age["p50"]:.3f} ms，P95={age["p95"]:.3f} ms，最大={age["maximum"]:.3f} ms。该年龄包含发布、调度和入袋，不是电机采样到控制的完整物理时延。旧版本没有保留这一路源时间，不能构造同名时延前后差。']
    if n['hardware_feedback_metadata']:
        h=n['hardware_feedback_metadata'];lines+=['',f'硬件读周期内，各右臂关节源年龄 P95 最大值={max(x["source_age_ms"]["p95"] for x in h.values()):.3f} ms；源年龄最大值={max(x["source_age_ms"]["maximum"] for x in h.values()):.3f} ms；有效性监测共记录 {sum(x["invalid_rows"] for x in h.values())} 个关节失效样本。']
    lines+=['','|正式运动末端力输出|修复前|修复后|','|---|---:|---:|']
    for key,label in [('valid_fraction','有效输出覆盖率'),('force_norm_rms_n','有效帧力范数 RMS (N)'),('force_norm_p95_n','有效帧力范数 P95 (N)')]:
        vals=[m['endpoint']['formal_motion'][key] for m in (b,n)]
        lines.append('|'+label+'|'+'|'.join('无有效帧' if v is None else f'{v:.4f}' for v in vals)+'|')
    lines+=['','末端力统计只覆盖通过求解器和状态检查的有效帧；缺失/奇异帧没有填零。力矩残差使用原始 NEXT 输出，末端力使用现有发布的滤波输出，两者不能直接互换。','',
        '|关节|修复前跨层位置差峰值 (°)|修复后跨层位置差峰值 (°)|修复前静止速度 (°/s)|修复后静止速度 (°/s)|','|---|---:|---:|---:|---:|']
    for j in range(1,8):
        k=f'J{j}';v=[peak(r['raw_to_software'][k]['bridge_to_joint']['position_error_deg']) for r in reports]
        z=[r['stationary'][k]['before']['raw_velocity_deg_s']['mean'] for r in reports]
        lines.append(f'|{k}|{v[0]:.5f}|{v[1]:.5f}|{z[0]:+.3f}|{z[1]:+.3f}|')
    tr=next(t for t in reports[1]['trials'] if t['joint']==1 and t['velocity_deg_s']==2.4 and t['direction']=='out')
    lines+=['','## 结果解释','',
        '反馈消费这一段的积压已明显改善，但旧 NEXT 的空载误报并未通过验收。时序修复会同时改变闭环跟踪及模型历史输入；模型在旧通路上训练，即使控制参数相同，也不能保证新输入分布仍匹配。本轮没有在独立日志上重拟合或补偿这一变化。',
        f'本轮 J1 目标峰速 2.4°/s，原始 CAN 速度峰值仍达 {tr["raw_CAN_peak_speed_deg_s"]:.2f}°/s；位置出现停住再追赶的阶梯。SG11/21/51 的加速度峰值为 {tr["SG"]["11"]["peak_acceleration_deg_s2"]:.1f}/{tr["SG"]["21"]["peak_acceleration_deg_s2"]:.1f}/{tr["SG"]["51"]["peak_acceleration_deg_s2"]:.1f}°/s²，说明求导不确定性仍然很大。原始反馈同时支持脉动现象，需要排查摩擦、死区、前馈与底层闭环；缺少独立编码器，不能把这些峰值当作真实加速度标定。',
        f'正式运动中末端力拒绝原因计数：{n["endpoint"]["formal_motion"]["reasons"]}。本次拒绝均标为力矩残差与工具接触模型不一致，不能把这批缺失输出全部归于奇异点。',
        '静止时仍有接近半个速度码的非零读数，且速度积分与位置位移不一致。应先核对固件和驱动的零速度定义，再统一采集、训练与推理；仅对在线速度减去本批均值会引入新的不一致。','',
        '|关节|初始残差均值 ± 标准差 (Nm)|最终残差均值 ± 标准差 (Nm)|','|---|---:|---:|']
    for j in (1,6):
        x=reports[1]['stationary'][f'J{j}'];lines.append(f'|J{j}|{x["before"]["residual_mean_nm"]:+.4f} ± {x["before"]["residual_std_nm"]:.4f}|{x["after"]["residual_mean_nm"]:+.4f} ± {x["after"]["residual_std_nm"]:.4f}|')
    lines+=['','静止残差均值远大于短时波动，单纯加强低通无法解决不归零问题。']
    static_path=after/'static_input_audit.json'
    if static_path.exists():
        inputs=json.loads(static_path.read_text());delta=max(abs(np.asarray(inputs['after']['q_deg'])-np.asarray(inputs['before']['q_deg'])))
        lines += [f'起止目标相同，但实际静止角度最大相差 {delta:.3f}°。因此当前前后偏置还混有姿态变化与重力变化，不能单凭同一指令目标证明摩擦滞回。实际位置、目标、实测力矩和模型预测均值见 static_input_audit.json。']
    lines+=['','下一步建议：先确认固件速度码与单位，再在新反馈通路下检查摩擦/死区及底层跟踪；信号定义固定后，另采训练集和验证集重新训练 NEXT。本次诊断及旧对照继续留作独立测试集，延迟只允许在训练数据上估计后冻结。']
    lines+=['','未拟合新延迟，未扣除零偏，未修改模型输入或补偿残差。若反馈新鲜度恢复但 CAN 速度偏置和空载残差仍存在，应分别核实固件速度码与训练输入定义、摩擦和热漂移；新反馈时序下模型可能需要重新验证或重新训练。','',
        '![反馈差与残差对照](comparison.png)','',f'修复前：{before}；修复后：{after}。完整覆盖率、拒绝原因和静止每轴均值/标准差见 comparison.json。']
    (after/'comparison.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    for s,r,label in zip(stages,reports,['Before','After']):
        a=np.load(s/'feedback_arrays.npz');tr=next(t for t in r['trials'] if t['joint']==1 and t['velocity_deg_s']==2.4 and t['direction']=='out')
        ti=a['software_state_stamps'];idx=np.searchsorted(a['bridge_state_stamps'],ti,side='right')-1
        mask=(idx>=0)&(ti>=tr['start_ros_ns'])&(ti<=tr['end_ros_ns'])
        axes[0].plot((ti[mask]-tr['start_ros_ns'])*1e-9,a['software_state_q_deg'][mask,0]-a['bridge_state_q_deg'][idx[mask],0],label=label,linewidth=1)
    axes[0].set(xlabel='Time from J1 2.4 deg/s trajectory start (s)',ylabel='Joint - preceding bridge (deg)')
    x=np.arange(len(reports[0]['trials']))
    for offset,r,label in zip([-.18,.18],reports,['Before','After']):
        axes[1].bar(x+offset,[t['residual_vector_rms_nm'] for t in r['trials']],.36,label=label)
    axes[1].set(xlabel='Matched formal motion index',ylabel='Raw residual vector RMS (Nm)')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Unloaded diagnostics: unchanged NEXT model, no lag fitting\nDifferent sessions: descriptive comparison, not causal force validation')
    fig.tight_layout();fig.savefig(after/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    a=p.parse_args();compare(a.before.resolve(),a.after.resolve())
