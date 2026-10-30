"""Read-only friction/dead-zone/tracking evidence; never identifies control gains.

Only original timestamps and preceding samples are used. Reconstructed MIT
commands are NOT independently measured CAN TX or motor internal torque.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml

from analyze_w3_feedback import raw_can, read_layers, preceding
from w3_signal_audit import stats
from w3_targeted_protocol import sample


def parameters(path):
    return next(iter(yaml.safe_load(path.read_text()).values()))['ros__parameters']


def quantize(value,lo,hi,bits):
    """Match driver's ordered float32 arithmetic, rounding and clipping."""
    lo,hi=np.float32(lo),np.float32(hi);maximum=np.float32(2**bits-1)
    x=np.clip(np.asarray(value,np.float32),lo,hi)
    code=((x-lo)/(hi-lo)*maximum+np.float32(.5)).astype(np.int64)
    decoded=code.astype(np.float32)*(hi-lo)/maximum+lo
    return code,decoded.astype(float)


def smooth_target(v,positive,negative,viscous,gain,transition,limit):
    v=np.asarray(v,float)
    return np.clip(gain*(np.tanh(v/transition)*np.where(v>=0,positive,negative)+viscous*v),-limit,limit)


def constant_runs(codes):
    starts=np.r_[0,np.flatnonzero(np.diff(codes)!=0)+1]
    return starts,np.r_[starts[1:]-1,len(codes)-1]


def scalar_peak(x):
    return float(np.max(np.abs(x)))


def analyze(stage,output):
    spec=json.loads((stage/'test_spec.json').read_text());state=json.loads((stage/'test_state.json').read_text())
    assert state['status']=='completed' and not state['recording_active']
    session=Path(state['session']);audit=session/'audit';run=json.loads((audit/'motion_run.json').read_text())
    assert run['status']=='completed'
    plan=yaml.load((stage/'motion.yaml').read_text(),Loader=yaml.CSafeLoader)
    assert hashlib.sha256((stage/'motion.yaml').read_bytes()).hexdigest()==spec['plan_sha256']
    pos=parameters(audit/'position.yaml');gc=parameters(audit/'gravity.yaml')
    calibration=yaml.safe_load((stage/'calibration.yaml').read_text())['offsets']
    calibration=[next(c for c in calibration if c['name']==f'right_joint_{j}') for j in range(7)]
    profiles=[yaml.safe_load((audit/f'motor_config/can1_id{j+1}.yaml').read_text()) for j in range(7)]
    initial=np.deg2rad(spec['initial_command_deg'])
    can,can_info=raw_can(audit/'raw_can.bin',profiles,calibration,initial)
    layers=read_layers(audit/'bag',calibration,initial)
    bridge=layers['bridge'];js=layers['state'];bt=bridge['t'];indices=preceding(js['t'],bt)
    good=(indices>=0)&((bt-js['t'][np.maximum(indices,0)])<=20_000_000)
    right=[pos['joints'].index(f'right_joint_{j}') for j in range(7)]
    kp=np.array(pos['kp_gains'])[right];kd=np.array(pos['kd_gains'])[right]
    names=gc['joints'];gi=[names.index(f'right_joint_{j}') for j in range(7)]
    gravity_gains=np.array(gc['gravity_gains'])[gi]
    model=pin.buildModelFromXML(plan['robot_description']);data=model.createData();base=pin.neutral(model)
    q_indices=[model.joints[model.getJointId(f'right_joint_{j}')].idx_q for j in range(7)]
    v_indices=[model.joints[model.getJointId(f'right_joint_{j}')].idx_v for j in range(7)]
    for j,v in enumerate(spec['left_pose_deg']):base[model.joints[model.getJointId(f'left_joint_{j}')].idx_q]=np.deg2rad(v)
    gravity=np.full((len(bt),7),np.nan)
    for i in np.flatnonzero(good):
        q=base.copy();q[q_indices]=np.deg2rad(js['q'][indices[i]])
        gravity[i]=pin.computeGeneralizedGravity(model,data,q)[v_indices]
    physical_model_gravity=gravity.copy();gravity*=gravity_gains
    ff=bridge['raw'][:,:,4]*np.array([c['axis_sign'] for c in calibration])
    inferred_friction=ff-gravity
    fm=yaml.safe_load((audit/'friction_model.yaml').read_text())
    coeff=[];quantization=[];decoded={};nominal_target=np.empty_like(ff)
    for j in range(7):
        i=gi[j];f=dict(fm[gc['motor_types'][i]])
        for name,key in [('coulomb_pos','friction_coulomb_pos_overrides'),('coulomb_neg','friction_coulomb_neg_overrides')]:
            if gc.get(key) and gc[key][i]>0:f[name]=gc[key][i]
        f['gain']=gc['friction_gains'][i];coeff.append(f)
        nominal_target[:,j]=smooth_target(np.deg2rad(bridge['v'][:,j]),f['coulomb_pos'],f['coulomb_neg'],f['viscous'],f['gain'],gc['friction_transition_velocity'],gc['friction_torque_limit'])
        p=profiles[j];sg=calibration[j]['axis_sign'];d={};codes={}
        for k,name,bits in [(0,'position',16),(1,'velocity',12),(2,'kp',12),(3,'kd',12),(4,'torque',12)]:
            x=bridge['raw'][:,j,k].copy()
            if name=='position':x=(x-p['position_offset'])*p['signs']['position']
            elif name in ('velocity','torque'):x*=p['signs'][name]
            rg=p['ranges'][name];codes[name],d[name]=quantize(x,rg['min'],rg['max'],bits)
            if name=='position':d[name]=sg*(d[name]*p['signs'][name]+p['position_offset']-calibration[j]['zero_offset'])
            elif name in ('velocity','torque'):d[name]*=sg*p['signs'][name]
        decoded[j]=(d,codes)
        vlsb=np.rad2deg((p['ranges']['velocity']['max']-p['ranges']['velocity']['min'])/4095)
        tau_lsb=(p['ranges']['torque']['max']-p['ranges']['torque']['min'])/4095
        examples=[]
        for v in (0.,.1,-.1,.6,-.6,1.2,-1.2,2.4,-2.4):
            r=p['ranges']['velocity'];code,value=quantize(sg*np.deg2rad(v)*p['signs']['velocity'],r['min'],r['max'],12)
            examples.append(dict(requested_deg_s=v,code=int(code),reconstructed_deg_s=float(np.rad2deg(sg*value*p['signs']['velocity']))))
        quantization.append(dict(joint=j+1,position_lsb_deg=float(can[j]['lsb'][0]),velocity_lsb_deg_s=float(vlsb),
            torque_lsb_nm=float(tau_lsb),nominal_kp=float(kp[j]),nominal_kd=float(kd[j]),
            one_position_code_PD_nm=float(kp[j]*np.deg2rad(can[j]['lsb'][0])),
            half_velocity_code_KD_nm=float(kd[j]*np.deg2rad(vlsb/2)),velocity_examples=examples,
            command_saturation_rows={name:int(np.count_nonzero(np.abs(bridge['raw'][:,j,k])>p['limits'][limit]))
                for name,k,limit in [('velocity',1,'velocity_max'),('kp',2,'kp_max'),('kd',3,'kd_max'),('torque',4,'torque_max')]}))
    events={e['name']:e for e in run['events']};trials=[];dwells=[];flat_events=[]
    for i,step in enumerate(plan['steps']):
        e=events[step['name']];a,b=e['command_start_ros_ns'],e['command_end_ros_ns']
        if not step['kind'].startswith('diagnostic_'):continue
        c=step['feedback_condition'];j=c['joint']-1;motor=can[j]
        pa,pb=[a+round(x*1e9) for x in c['plateau_seconds']];pa+=250_000_000;pb-=250_000_000
        lo,hi=preceding(motor['t'],[pa,pb]);assert lo>=0 and hi>lo
        t=motor['t'][lo:hi+1];q=motor['q'][lo:hi+1];tau=motor['tau'][lo:hi+1];codes=motor['code'][lo:hi+1,0]
        starts,ends=constant_runs(codes);longest=max(zip(starts,ends),key=lambda p:t[p[1]]-t[p[0]])
        bm=(bt>=pa)&(bt<=pb)&good
        d,dc=decoded[j]
        plateau_codes=np.unique(dc['velocity'][bm])
        row=dict(name=step['name'],**c,
            plateau_ros_ns=[pa,pb],raw_encoder_speed_deg_s=float((q[-1]-q[0])/((t[-1]-t[0])*1e-9)),
            raw_velocity_peak_deg_s=scalar_peak(motor['v'][(motor['t']>=a)&(motor['t']<=b)]),
            longest_same_position_code_seconds=float((t[longest[1]]-t[longest[0]])*1e-9),
            longest_same_code_command_travel_deg=float(c['velocity_deg_s']*(t[longest[1]]-t[longest[0]])*1e-9),
            expected_one_code_travel_seconds=float(motor['lsb'][0]/c['velocity_deg_s']),
            reconstructed_velocity_codes=plateau_codes.tolist(),reconstructed_velocity_deg_s=np.unique(np.rad2deg(d['velocity'][bm])).tolist(),
            inferred_friction_from_command_minus_scaled_gravity_nm=stats(inferred_friction[bm,j]),
            expected_smooth_friction_nm=stats(nominal_target[bm,j]),
            ff_reconstruction_error_nm=stats(inferred_friction[bm,j]-nominal_target[bm,j]),
            command_model_extra_gravity_nm=stats((gravity-physical_model_gravity)[bm,j]),
            warnings=['Command-space reconstruction, not measured motor control torque or joint friction identification'])
        for first,last in zip(starts,ends):
            duration=(t[last]-t[first])*1e-9;travel=c['velocity_deg_s']*duration
            if duration<.10 or travel<4*motor['lsb'][0]:continue
            time_ns=int(t[last]);later=np.searchsorted(motor['t'],time_ns+150_000_000,side='right')-1
            follow=float(c['signed_direction']*(motor['q'][later]-q[last]))
            flat_events.append(dict(name=step['name'],joint=j+1,velocity_deg_s=c['velocity_deg_s'],direction=c['direction'],
                start_ros_ns=int(t[first]),end_ros_ns=time_ns,duration_seconds=float(duration),
                target_travel_deg=float(travel),target_travel_position_codes=float(travel/motor['lsb'][0]),
                raw_tau_change_nm=float(tau[last]-tau[first]),
                nominal_position_PD_buildup_nm=float(kp[j]*np.deg2rad(travel)),
                position_advance_next_150ms_deg=follow,
                advance_exceeds_2_codes=bool(follow>=2*motor['lsb'][0]-1e-8)))
        next_step=plan['steps'][i+1];assert next_step['dwell_seconds']
        de=events[next_step['name']];da,db=de['command_end_ros_ns']-1_000_000_000,de['command_end_ros_ns']
        dm=(bt>=da)&(bt<=db)&good;measured=js['q'][indices[dm],j];target=bridge['q'][dm,j]
        dwell=dict(name=next_step['name'],joint=j+1,velocity_deg_s=c['velocity_deg_s'],approach_direction=c['signed_direction'],
            same_target_key=round(float(target.mean()),4),actual_position_mean_deg=float(measured.mean()),
            target_mean_deg=float(target.mean()),target_minus_actual_deg=float((target-measured).mean()),
            nominal_position_PD_nm=float((kp[j]*np.deg2rad(target-measured)).mean()),
            measured_position_range_deg=float(np.ptp(measured)),
            motor_feedforward_nm=stats(ff[dm,j]),inferred_friction_nm=stats(inferred_friction[dm,j]),
            command_model_extra_gravity_nm=stats((gravity-physical_model_gravity)[dm,j]))
        dwells.append(dwell);trials.append(row)
    saturations={f'J{x["joint"]}':x['command_saturation_rows'] for x in quantization}
    result=dict(stage=str(stage),session=str(session),status='completed',can=can_info,
        method=dict(no_new_time_shift_estimated_or_applied=True,no_model_or_control_tuning=True,
            latency_matching='preceding original timestamps, no fitting or shift',
            gravity='captured URDF, gravity [0,0,-9.81], captured gains; not independently calibrated gravity',
            plateau_guard_seconds=.25,stall_descriptor='exact unchanged CAN position code >=0.10s while target travels >=4 position codes',
            breakaway_validation='subsequent 150ms signed displacement >=2 position codes; descriptive, not identified static friction',
            saturation='recorded bridge command vs configured software limits; motor current and firmware saturation unavailable'),
        control_parameters={k:gc[k] for k in gc if k!='robot_description'},
        quantization=quantization,trials=trials,dwells=dwells,flat_intervals=flat_events,
        software_saturation_rows=saturations,
        sample_alignment=dict(bridge_rows=len(bt),gravity_reconstructed_rows=int(good.sum()),
            state_header_age_ms=stats((bt[good]-js['t'][indices[good]])*1e-6)),
        limitations=['Only no-contact logs, no independent friction/torque/encoder ground truth',
            'No independent transmitted CAN packets; driver encoding reconstructed under captured profile',
            'Motor firmware internal position/velocity/current loop, filtering, compensation and saturation unobserved',
            'No joint friction parameter or mechanical backlash/dead-zone width is fitted on these held-out data',
            'Left pose frozen; fixed-base right-arm gravity must be independent of left-arm configuration',
            'Command-minus-gravity isolates configured feedforward approximately; timing and URDF errors remain'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Illustrate the most pronounced plateau, without tuning thresholds from it.
    chosen=max([x for x in trials if x['joint'] in (1,2,4)],key=lambda x:x['longest_same_position_code_seconds'])
    e=events[chosen['name']];a,b=e['command_start_ros_ns'],e['command_end_ros_ns'];j=chosen['joint']-1
    bm=(bt>=a-500_000_000)&(bt<=b+1_000_000_000)&good
    cm=(can[j]['t']>=a-500_000_000)&(can[j]['t']<=b+1_000_000_000)
    xt=(bt[bm]-a)*1e-9;ct=(can[j]['t'][cm]-a)*1e-9
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    axes[0].plot(xt,bridge['q'][bm,j],label='Native/bridge target')
    axes[0].plot(ct,can[j]['q'][cm],label='Raw CAN position',linewidth=1)
    axes[0].set_ylabel('Position (deg)')
    axes[1].plot(xt,bridge['v'][bm,j],label='Requested velocity')
    axes[1].step(xt,np.rad2deg(decoded[j][0]['velocity'][bm]),where='post',label='Reconstructed encoded velocity')
    axes[1].plot(ct,can[j]['v'][cm],alpha=.45,label='Raw CAN reported velocity',linewidth=.7)
    axes[1].set_ylabel('Velocity (deg/s)')
    axes[2].plot(xt,ff[bm,j],label='Recorded total feedforward')
    axes[2].plot(xt,gravity[bm,j],label='Reconstructed scaled gravity')
    axes[2].plot(xt,inferred_friction[bm,j],label='Feedforward - gravity')
    axes[2].plot(xt,nominal_target[bm,j],'--',label='Expected smooth friction')
    axes[2].set_ylabel('Command torque (Nm)')
    axes[3].plot(ct,can[j]['tau'][cm],label='Raw CAN torque feedback')
    axes[3].plot(xt,kp[j]*np.deg2rad(bridge['q'][bm,j]-js['q'][indices[bm],j]),label='Nominal Kp * position error')
    axes[3].set_ylabel('Torque / proxy (Nm)');axes[3].set_xlabel('Time from native motion start (s)')
    for x in flat_events:
        if x['name']==chosen['name']:
            for ax in axes:ax.axvspan((x['start_ros_ns']-a)*1e-9,(x['end_ros_ns']-a)*1e-9,color='orange',alpha=.15)
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f'Unloaded {chosen["name"]}: unchanged position codes despite moving target\nCommand reconstruction is not independently measured CAN TX or motor internal torque')
    fig.tight_layout();fig.savefig(output/'stiction_tracking.png',dpi=150);plt.close(fig)
    lines=['# 摩擦、死区和底层跟踪排查','',f'来源：{stage}。只读分析，无新运动、增益调整、模型修改或时延拟合。','',
        '## 量化与软件限幅','', '|关节|速度一个码 (°/s)|位置一个码 (°)|一位置码对应 Kp 力矩 (Nm)|零速半码对应 Kd 力矩量级 (Nm)|','|---|---:|---:|---:|---:|']
    for x in quantization:lines.append(f'|J{x["joint"]}|{x["velocity_lsb_deg_s"]:.3f}|{x["position_lsb_deg"]:.5f}|{x["one_position_code_PD_nm"]:.4f}|{x["half_velocity_code_KD_nm"]:.4f}|')
    lines+=['','上表Kd项仅是按软件解码计算的量级，不证明电机内部速度偏置。限幅计数是桥接入口相对配置限幅的计数，不覆盖电机电流/温度限幅。','',
        '## 匀速平台核查','', '|关节/方向|目标速度 (°/s)|同一位置码最长持续 (s)|这段时间目标走过 (°)|相当于位置码数|前馈减重力均值 (Nm)|平滑摩擦理论值 (Nm)|','|---|---:|---:|---:|---:|---:|---:|']
    for x in trials:
        lines.append(f'|J{x["joint"]} {x["direction"]}|{x["velocity_deg_s"]:.1f}|{x["longest_same_position_code_seconds"]:.3f}|{x["longest_same_code_command_travel_deg"]:.3f}|{x["longest_same_code_command_travel_deg"]/can[x["joint"]-1]["lsb"][0]:.1f}|{x["inferred_friction_from_command_minus_scaled_gravity_nm"]["mean"]:+.4f}|{x["expected_smooth_friction_nm"]["mean"]:+.4f}|')
    lines+=['','## 到位后的残余位置误差','', '|关节/来向|前段速度 (°/s)|目标 (°)|实际平均 (°)|目标−实际 (°)|Kp误差力矩量级 (Nm)|额外重力前馈 (Nm)|','|---|---:|---:|---:|---:|---:|---:|']
    for x in dwells:
        lines.append(f'|J{x["joint"]} {x["approach_direction"]:+g}|{x["velocity_deg_s"]:.1f}|{x["target_mean_deg"]:.3f}|{x["actual_position_mean_deg"]:.3f}|{x["target_minus_actual_deg"]:+.3f}|{x["nominal_position_PD_nm"]:+.3f}|{x["command_model_extra_gravity_nm"]["mean"]:+.3f}|')
    lines+=['','同目标、不同来向/路径历史的实际位置差混有重力偏差、静摩擦、控制器状态及机械回差，不能直接叫作机械死区宽度。','',
        f'检测到 {len(flat_events)} 段“反馈位置码不变且目标走过至少4个位置码”的区间；其中 {sum(x["advance_exceeds_2_codes"] for x in flat_events)} 段在随后150 ms沿目标方向至少前进2个位置码。它是阻滞与释放的证据，未识别静摩擦系数。','',
        '![跟踪与前馈分解](stiction_tracking.png)','',
        '所有逐段数值、软件限幅、编码示例、来源与解释边界见 diagnosis.json。原始CAN、实测力矩和前馈使用各自原时间戳，没有搜索或校正相位。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Saved',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.stage.resolve(),args.output.resolve())
