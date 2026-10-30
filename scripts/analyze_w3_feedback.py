"""Read-only, unshifted feedback diagnostic. No fitting, tuning or ROS nodes.

CAN timestamps are host kernel reception, not device sample time. Motor command
quantization is reconstructed from the captured bridge input, not measured TX.
"""
import argparse
import csv
import ctypes
import ctypes.util
import json
import hashlib
import gzip
import socket
import sqlite3
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter, savgol_coeffs

from w3_signal_audit import save, stats, read_bag, diagnostics_summary
from w3_targeted_protocol import sample


def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))
def region(t,a,b): return (t>=a)&(t<=b)
def stamp(m): return m.header.stamp.sec*10**9+m.header.stamp.nanosec


def preceding(reference, queries):
    """No future samples, interpolation, lag search or timestamp changes."""
    return np.searchsorted(reference,queries,side='right')-1


def canonical(q, anchor):
    # These small diagnostic paths stay away from wrap boundaries. Select the
    # equivalent revolution nearest the frozen start, without fitting offsets.
    return anchor+(q-anchor+np.pi)%(2*np.pi)-np.pi


def raw_can(path,profiles,calibration,initial):
    dtype=np.dtype([('t','<i8'),('wall','<i8'),('mono','<i8'),('id','<u4'),
        ('dlc','<u2'),('flags','<u2'),('data','u1',(8,))])
    # Current W3 uses 8-byte CAN payloads. A changed layout must not silently
    # corrupt timing; use explicit variable-record decoding as a fallback.
    raw=np.fromfile(path,dtype=dtype)
    if path.stat().st_size%dtype.itemsize or not np.all(raw['dlc']==8):
        head=struct.Struct('<qqqIHH'); records=[]
        with path.open('rb') as f:
            while data:=f.read(head.size):
                if len(data)!=head.size: raise ValueError('Truncated CAN header')
                t,w,m,i,n,flags=head.unpack(data); payload=f.read(n)
                if len(payload)!=n: raise ValueError('Truncated CAN payload')
                if n==8: records.append((t,w,m,i,n,flags,tuple(payload)))
        raw=np.array(records,dtype=dtype)
    ids=raw['id']&0x1fffffff
    tx=(raw['flags']&socket.MSG_DONTROUTE)!=0
    eligible=(~tx)&((raw['id']&(0x20000000|0x40000000))==0)&(ids==0)
    # This robot's shared-master feedback is CAN ID 0, motor in data[0]&15.
    # Reject a changed route rather than guessing whether data are commands.
    unexpected=(~tx)&((raw['id']&(0x20000000|0x40000000))==0)&(ids!=0)
    result={};d=raw['data'].astype(np.uint32); motors=d[:,0]&15
    for j in range(7):
        p=profiles[j];c=calibration[j];m=eligible&(motors==j+1)
        t=raw['t'][m];payload=d[m];order=np.argsort(t,kind='stable');t=t[order];payload=payload[order]
        if len(t)<100 or np.any(t<=0) or np.any(np.diff(t)<=0): raise ValueError('Invalid raw joint reception timing')
        code=np.c_[payload[:,1]*256+payload[:,2],payload[:,3]*16+(payload[:,4]>>4),(payload[:,4]&15)*256+payload[:,5]]
        values=np.empty(code.shape,float);lsb=[]
        for k,(name,bits) in enumerate(zip(('position','velocity','torque'),(16,12,12))):
            r=p['ranges'][name];step=(r['max']-r['min'])/(2**bits-1);lsb.append(step)
            values[:,k]=r['min']+code[:,k]*step
        s=c['axis_sign'];ps=p['signs']
        q=canonical(s*(values[:,0]*ps['position']+p['position_offset']-c['zero_offset']),initial[j])
        v=s*values[:,1]*ps['velocity'];tau=s*values[:,2]*ps['torque']
        result[j]=dict(t=t,q=np.rad2deg(q),v=np.rad2deg(v),tau=tau,code=code,
            lsb=np.array([np.rad2deg(lsb[0]),np.rad2deg(lsb[1]),lsb[2]]),error=payload[:,0]>>4)
    return result,dict(frames=len(raw),local_tx_echo_frames=int(tx.sum()),
        nonshared_id_receive_frames=int(unexpected.sum()),
        frame_routes={str(k):int(v) for k,v in Counter(ids.tolist()).items()},
        timestamp='Linux kernel software RX; device acquisition timestamp unavailable')


def read_layers(folder,calibration,initial,hardware_metadata=None):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    from w3_robot_bridge.msg import MotorCommandArray,MotorStateArray
    lib=ctypes.CDLL(ctypes.util.find_library('zstd'))
    lib.ZSTD_getFrameContentSize.argtypes=(ctypes.c_void_p,ctypes.c_size_t);lib.ZSTD_getFrameContentSize.restype=ctypes.c_ulonglong
    lib.ZSTD_decompress.argtypes=(ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t);lib.ZSTD_decompress.restype=ctypes.c_size_t
    result={k:[] for k in ('state','bridge','bridge_state','source_state')}
    metadata={f'J{j+1}':{k:[] for k in ('feedback_source_age','feedback_receive_age','feedback_valid')} for j in range(7)}
    for path in sorted(folder.glob('*.db3')):
        con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        topics={i:n for i,n in con.execute('select id,name from topics')}
        ids=[i for i,n in topics.items() if n in ('/joint_states','/w3_robot_bridge_node/commands','/w3_robot_bridge_node/state','/dm_hardware_interface/feedback_state')]
        if hardware_metadata is not None:
            ids += [i for i,n in topics.items() if n=='/dynamic_joint_states']
        if not ids: raise ValueError('Missing raw software layers')
        query='select timestamp,topic_id,data from messages where topic_id in ('+','.join('?' for _ in ids)+') order by timestamp'
        for received,tid,data in con.execute(query,ids):
            if data[:4]==b'\x28\xb5\x2f\xfd':
                size=lib.ZSTD_getFrameContentSize(data,len(data))
                if not 0<size<2**20: raise ValueError('Invalid compressed message length')
                dest=ctypes.create_string_buffer(size);n=lib.ZSTD_decompress(dest,size,data,len(data))
                if n!=size: raise ValueError('Invalid compressed message')
                data=dest.raw
            if topics[tid]=='/dynamic_joint_states':
                from control_msgs.msg import DynamicJointState
                m=deserialize_message(data,DynamicJointState)
                for j in range(7):
                    name=f'right_joint_{j}'
                    if name not in m.joint_names: continue
                    entry=m.interface_values[m.joint_names.index(name)]
                    values=dict(zip(entry.interface_names,entry.values))
                    for key in metadata[f'J{j+1}']:
                        if key in values: metadata[f'J{j+1}'][key].append(values[key])
                continue
            if topics[tid] in ('/joint_states','/dm_hardware_interface/feedback_state'):
                m=deserialize_message(data,JointState);indices=[m.name.index(f'right_joint_{j}') for j in range(7)]
                name='state' if topics[tid]=='/joint_states' else 'source_state'
                result[name].append((stamp(m),received,[m.position[i] for i in indices],[m.velocity[i] for i in indices]))
            elif topics[tid]=='/w3_robot_bridge_node/commands':
                m=deserialize_message(data,MotorCommandArray);commands={c.motor_index:c for c in m.commands if c.channel==1}
                if not all(j in commands for j in range(7)): raise ValueError('Incomplete right-arm bridge command')
                if any(commands[j].mode!=3 for j in range(7)): raise ValueError('Unexpected bridge motor mode')
                q=[canonical(calibration[j]['axis_sign']*(commands[j].position-calibration[j]['zero_offset']),initial[j]) for j in range(7)]
                v=[calibration[j]['axis_sign']*commands[j].velocity for j in range(7)]
                result['bridge'].append((stamp(m),received,q,v,
                    [[commands[j].position,commands[j].velocity,commands[j].kp,commands[j].kd,commands[j].torque] for j in range(7)]))
            else:
                m=deserialize_message(data,MotorStateArray);motors={c.motor_index:c for c in m.motors if c.channel==1}
                if not all(j in motors for j in range(7)): raise ValueError('Incomplete bridge state')
                q=[canonical(calibration[j]['axis_sign']*(motors[j].position-calibration[j]['zero_offset']),initial[j]) for j in range(7)]
                v=[calibration[j]['axis_sign']*motors[j].velocity for j in range(7)]
                result['bridge_state'].append((stamp(m),received,q,v))
        con.close()
    parsed={}
    for name,rows in result.items():
        if name=='source_state' and not rows: continue  # Legacy captures lack source stamps.
        if len(rows)<100: raise ValueError('Insufficient '+name+' layer data')
        t=np.array([r[0] for r in rows],np.int64);o=np.argsort(t,kind='stable')
        if np.any(np.diff(t[o])<=0): raise ValueError('Nonmonotone '+name+' headers')
        parsed[name]=dict(t=t[o],received=np.array([r[1] for r in rows],np.int64)[o],
            q=np.rad2deg(np.array([r[2] for r in rows]))[o],v=np.rad2deg(np.array([r[3] for r in rows]))[o])
        if name=='bridge': parsed[name]['raw']=np.array([r[4] for r in rows])[o]
    if hardware_metadata is not None:
        for joint,values in metadata.items():
            if not values['feedback_valid']: continue
            hardware_metadata[joint]=dict(source_age_ms=stats(np.asarray(values['feedback_source_age'])*1000),
                receive_age_ms=stats(np.asarray(values['feedback_receive_age'])*1000),
                recorded_rows=len(values['feedback_valid']),
                invalid_rows=int(np.count_nonzero(np.asarray(values['feedback_valid'])!=1.)))
    return parsed


def quantized_velocity(command,profile,axis_sign):
    # Reconstruction of configured float->uint12 mapping. Not an observation
    # of transmitted packets or firmware interpretation.
    r=profile['ranges']['velocity'];ps=profile['signs']['velocity']
    x=np.asarray(command)*ps
    code=np.floor(np.clip((x-r['min'])/(r['max']-r['min']),0,1)*4095+.5)
    return np.rad2deg(axis_sign*(r['min']+code*(r['max']-r['min'])/4095)*ps)


def analyze(stage):
    spec=json.loads((stage/'test_spec.json').read_text());state=json.loads((stage/'test_state.json').read_text())
    assert state['status']=='completed' and not state['recording_active']
    assert json.loads((stage/'recorder_state.json').read_text())['status']=='closed'
    session=Path(state['session']);audit=session/'audit'
    meta=json.loads((Path(spec['model_dir'])/'metadata.json').read_text())
    assert hashlib.sha256((stage/'calibration.yaml').read_bytes()).hexdigest()==meta['calibration']['sha256']
    assert hashlib.sha256((stage/'motion.yaml').read_bytes()).hexdigest()==spec['plan_sha256']
    assert hashlib.sha256((Path(spec['model_dir'])/'model.pt').read_bytes()).hexdigest()==spec['model_hash']
    can_state=json.loads((audit/'can_record_state.json').read_text())
    assert can_state['status']=='closed' and can_state['socket_dropped_frames']==can_state['kernel_timestamps_missing']==0
    plan=yaml.load((stage/'motion.yaml').read_text(),Loader=yaml.CSafeLoader)
    run=json.loads((audit/'motion_run.json').read_text());assert run['status']=='completed'
    events={e['name']:e for e in run['events']};assert len(events)==len(plan['steps'])
    calibration_yaml=yaml.safe_load((stage/'calibration.yaml').read_text())
    entries=calibration_yaml['offsets']
    calibration=[next(c for c in entries if c['name']==f'right_joint_{j}') for j in range(7)]
    profiles=[yaml.safe_load((audit/f'motor_config/can1_id{j+1}.yaml').read_text()) for j in range(7)]
    initial=np.deg2rad(spec['initial_command_deg'])
    can,can_summary=raw_can(audit/'raw_can.bin',profiles,calibration,initial)
    print('Raw CAN parsed',can_summary,flush=True)
    hardware_metadata={}
    layers=read_layers(audit/'bag',calibration,initial,hardware_metadata);print('Software layers parsed',flush=True)
    start=min(e['command_start_ros_ns'] for e in events.values());end=max(e['command_end_ros_ns'] for e in events.values())
    native,diagnostics=read_bag(audit/'bag',[(start-5_000_000_000,end+5_000_000_000)])
    native=dict(t=native['t'],q=np.rad2deg(native['q']),v=np.rad2deg(native['v']))
    with (stage/'samples.csv').open() as f: rows=list(csv.DictReader(f))
    st=np.array([int(r['source_timestamp_ns']) for r in rows],np.int64)
    assert np.all(np.diff(st)>0)
    gap_indices=np.flatnonzero(np.diff(st)>=40_000_000)
    source_gaps=[dict(before_ros_ns=int(st[i]),after_ros_ns=int(st[i+1]),gap_ms=float((st[i+1]-st[i])*1e-6)) for i in gap_indices]
    def matrix(key): return np.array([[float(r[f'{key}_j{j+1}']) for j in range(7)] for r in rows])
    q=matrix('q_deg');v=matrix('velocity_deg_s');raw=matrix('external_raw_nm');command=matrix('command_deg')
    np.testing.assert_allclose(matrix('measured_tau_nm')-matrix('free_pred_tau_nm'),raw,atol=1e-5)
    origin=st[0];t=(st-origin)*1e-9;grid=np.arange(int(t[-1]/.01)+1)*.01;gt=origin+np.rint(grid*1e9).astype(np.int64)
    gq=np.column_stack([np.interp(grid,t,q[:,j]) for j in range(7)])
    gv=np.column_stack([np.interp(grid,t,v[:,j]) for j in range(7)])
    derivatives={w:(savgol_filter(gq,w,3,deriv=1,delta=.01,axis=0),
        savgol_filter(gq,w,3,deriv=2,delta=.01,axis=0)) for w in spec['feedback_protocol']['position_derivatives']['centered_windows_samples']}
    naive_v=np.gradient(gq,.01,axis=0);naive_a=np.gradient(naive_v,.01,axis=0)
    # Interpolation is only an implementation convenience on the uniform grid.
    # Any derivative window touching a missing interval is discarded, along
    # with record edges; no number spanning a large gap enters the analysis.
    for w,arrays in list(derivatives.items())+[(5,(naive_v,naive_a))]:
        half=(w//2)*10_000_000;good=(gt>=st[0]+half)&(gt<=st[-1]-half)
        for gap in source_gaps:
            good &= (gt<gap['before_ros_ns']-half)|(gt>gap['after_ros_ns']+half)
        for array in arrays: array[~good]=np.nan
    transport={};quantization={};stationary={}
    for j in range(7):
        c=can[j];key=f'J{j+1}';joint_state=layers['state'];idx=preceding(c['t'],joint_state['t']);m=idx>=0
        indices=idx[m];qt=joint_state['q'][m,j]-c['q'][indices];vt=joint_state['v'][m,j]-c['v'][indices]
        transport[key]=dict(raw_can_interval_ms=stats(np.diff(c['t'])*1e-6),
            motor_status_counts={str(k):int(v) for k,v in Counter(c['error'].tolist()).items()},
            joint_state_minus_latest_CAN_ms=stats((joint_state['t'][m]-c['t'][indices])*1e-6),
            joint_state_minus_latest_CAN_position_deg=stats(qt),joint_state_minus_latest_CAN_velocity_deg_s=stats(vt),
            position_equal_within_0p0001deg_fraction=float(np.mean(abs(qt)<1e-4)),
            velocity_equal_within_0p0001deg_s_fraction=float(np.mean(abs(vt)<1e-4)))
        comparisons=[('CAN_to_bridge',layers['bridge_state'],c),
                     ('bridge_to_joint',joint_state,layers['bridge_state'])]
        if 'source_state' in layers:
            comparisons.append(('bridge_to_source',layers['source_state'],layers['bridge_state']))
        for name,target,reference in comparisons:
            indices=preceding(reference['t'],target['t']);m=indices>=0
            reference_q=reference['q'][indices[m]] if name=='CAN_to_bridge' else reference['q'][indices[m],j]
            reference_v=reference['v'][indices[m]] if name=='CAN_to_bridge' else reference['v'][indices[m],j]
            transport[key][name]=dict(position_error_deg=stats(target['q'][m,j]-reference_q),
                velocity_error_deg_s=stats(target['v'][m,j]-reference_v),
                header_age_ms=stats((target['t'][m]-reference['t'][indices[m]])*1e-6))
        lq,lv,lt=c['lsb']
        quantization[key]=dict(position_lsb_deg=lq,velocity_lsb_deg_s=lv,torque_lsb_nm=lt,
            centered_zero_velocity_half_bin_deg_s=lv/2,
            naive_first_difference_quantization_bound_deg_s=lq/.01,
            naive_second_difference_quantization_bound_deg_s2=2*lq/.01**2,
            SG_quantization_only_worst_case={str(w):dict(
                velocity_deg_s=float(lq/2*sum(abs(savgol_coeffs(w,3,deriv=1,delta=.01)))),
                acceleration_deg_s2=float(lq/2*sum(abs(savgol_coeffs(w,3,deriv=2,delta=.01))))) for w in derivatives})
        stationary[key]={}
        for label,kind in [('before','initial_settle'),('after','final_settle')]:
            event=next(events[s['name']] for s in plan['steps'] if s['kind']==kind)
            a,b=event['command_end_ros_ns']-8_000_000_000,event['command_end_ros_ns']
            cm=region(c['t'],a,b);gm=region(gt,a,b);sm=region(st,a,b)
            stationary[key][label]=dict(raw_position_range_deg=float(np.ptp(c['q'][cm])),raw_position_std_deg=float(c['q'][cm].std()),
                raw_velocity_deg_s=stats(c['v'][cm]),velocity_code_counts={str(k):int(n) for k,n in Counter(c['code'][cm,1].tolist()).items()},
                residual_mean_nm=float(raw[sm,j].mean()),residual_std_nm=float(raw[sm,j].std()),
                naive_velocity_std_deg_s=float(np.nanstd(naive_v[gm,j])),naive_acceleration_std_deg_s2=float(np.nanstd(naive_a[gm,j])),
                SG={str(w):dict(valid_samples=int(np.isfinite(d[0][gm,j]).sum()),velocity_std_deg_s=float(np.nanstd(d[0][gm,j])),
                    acceleration_std_deg_s2=float(np.nanstd(d[1][gm,j])),acceleration_abs_p99_deg_s2=float(np.nanpercentile(abs(d[1][gm,j]),99))) for w,d in derivatives.items()})
    command_checks=[];trials=[]
    for step in plan['steps']:
        event=events[step['name']];a,b=event['command_start_ros_ns'],event['command_end_ros_ns'];nm=region(native['t'],a,b)
        expected_q,expected_v,_=sample(step['knots'],(native['t'][nm]-a)*1e-9)
        command_checks.append(dict(name=step['name'],samples=int(nm.sum()),
            native_vs_frozen_position_max_deg=float(abs(native['q'][nm]-expected_q).max()),
            native_vs_frozen_velocity_max_deg_s=float(abs(native['v'][nm]-expected_v).max())))
        if not step['kind'].startswith('diagnostic_'):continue
        condition=step['feedback_condition'];j=condition['joint']-1;c=can[j];direction=condition['signed_direction']
        pa,pb=[a+round(x*1e9) for x in condition['plateau_seconds']];pa+=250_000_000;pb-=250_000_000
        indices=preceding(c['t'],[pa,pb]);assert indices[0]>=0 and indices[1]>indices[0]
        lo,hi=indices;ct=(c['t'][lo:hi+1]-c['t'][lo])*1e-9
        displacement=c['q'][hi]-c['q'][lo];duration=ct[-1]
        integral=float(np.trapz(c['v'][lo:hi+1],ct));encoder_speed=displacement/duration
        codes=c['code'][lo:hi+1,0];runs=np.r_[0,np.flatnonzero(np.diff(codes.astype(np.int64))!=0)+1]
        ends=np.r_[runs[1:]-1,len(codes)-1]
        longest_flat=float(max(ct[ends]-ct[runs]))
        gm=region(gt,a,b+500_000_000);pm=region(gt,pa,pb);sm=region(st,a,b)
        assert all(np.isfinite(d[0][gm]).all() and np.isfinite(d[1][gm]).all() for d in derivatives.values()),'Motion trial intersects excluded derivative gap'
        qm=region(layers['state']['t'],a,b);ci=preceding(native['t'],layers['state']['t'][qm]);valid=ci>=0
        error=layers['state']['q'][qm,j][valid]-native['q'][ci[valid],j]
        bridge=layers['bridge'];bm=region(bridge['t'],pa,pb)
        qv=quantized_velocity(bridge['raw'][bm,j,1],profiles[j],calibration[j]['axis_sign'])
        following=events[plan['steps'][event['index']]['name']]
        end_mask=region(st,following['command_end_ros_ns']-500_000_000,following['command_end_ros_ns'])
        trial=dict(name=step['name'],**condition,start_ros_ns=a,end_ros_ns=b,plateau_trimmed_ros_ns=[pa,pb],
            plateau_duration_seconds=float(duration),encoder_displacement_deg=float(displacement),
            encoder_average_speed_deg_s=float(encoder_speed),reported_velocity_average_deg_s=float(np.mean(c['v'][lo:hi+1])),
            velocity_integral_deg=integral,velocity_integral_minus_displacement_deg=float(integral-displacement),
            plateau_longest_unchanged_position_seconds=longest_flat,
            quantization_only_integral_error_bound_deg=float(c['lsb'][1]/2*duration+c['lsb'][0]),
            nominal_bridge_quantized_velocity_deg_s=sorted(set(qv.tolist())),
            actual_position_vs_native_target_error_deg=stats(error),
            settled_endpoint_error_deg=float((q[end_mask]-command[end_mask]).mean(0)[j]),
            actual_displacement_direction_matches=bool(encoder_speed*direction>0),
            residual_vector_rms_nm=rms(np.linalg.norm(raw[sm],axis=1)),
            native_peak_speed_deg_s=float(abs(native['v'][nm,j]).max()),
            raw_CAN_peak_speed_deg_s=float(abs(c['v'][region(c['t'],a,b+500_000_000)]).max()),
            SG={str(w):dict(peak_speed_deg_s=float(abs(d[0][gm,j]).max()),
                peak_acceleration_deg_s2=float(abs(d[1][gm,j]).max()),
                plateau_speed_mean_deg_s=float(d[0][pm,j].mean()),plateau_speed_std_deg_s=float(d[0][pm,j].std()),
                velocity_feedback_RMSE_deg_s=rms(d[0][pm,j]-gv[pm,j])) for w,d in derivatives.items()})
        trials.append(trial)
    bridge=layers['bridge'];idx=preceding(native['t'],bridge['t']);m=idx>=0
    bridge_check=dict(command_age_ms=stats((bridge['t'][m]-native['t'][idx[m]])*1e-6),
        position_max_error_deg=abs(bridge['q'][m]-native['q'][idx[m]]).max(0),
        velocity_max_error_deg_s=abs(bridge['v'][m]-native['v'][idx[m]]).max(0),
        kp_range=np.c_[bridge['raw'][:,:,2].min(0),bridge['raw'][:,:,2].max(0)],
        kd_range=np.c_[bridge['raw'][:,:,3].min(0),bridge['raw'][:,:,3].max(0)],
        CAN_TX_verified=False,meaning='bridge input follows native target; packet encoding reconstructed, physical TX not independently observed')
    layer_rates={name:dict(rows=len(layer['t']),interval_ms=stats(np.diff(layer['t'])*1e-6),
        bag_receive_minus_header_ms=stats((layer['received']-layer['t'])*1e-6)) for name,layer in layers.items()}
    next_changes=[];previous=None
    diagnostic_path=stage/'diagnostics.jsonl'
    diagnostic_lines=diagnostic_path.open() if diagnostic_path.exists() else gzip.open(str(diagnostic_path)+'.gz','rt')
    for line in diagnostic_lines:
        entry=json.loads(line)
        for status in entry['status']:
            if status['name']!='factr2/next/right':continue
            value=status['values'];counts=(value.get('dropped'),value.get('resets'))
            if counts!=previous:
                next_changes.append(dict(stamp_ros_ns=entry['source_timestamp_ns'],message=status['message'],values=value))
                previous=counts
    result=dict(status='completed',stage=str(stage),session=str(session),trial_count=len(trials),
        recipe=spec['feedback_protocol'],no_new_time_shift_fit_or_applied=True,can=can_summary,
        can_health=can_state,layer_rates=layer_rates,adapter_timing=diagnostics_summary(diagnostics),
        raw_to_software=transport,quantization=quantization,stationary=stationary,
        native_command_checks=command_checks,bridge_command_check=bridge_check,trials=trials,
        sample_rows=len(rows),source_max_gap_ms=float(np.diff(st).max()*1e-6),source_gaps=source_gaps,
        raw_adapter_capture_quality=json.loads((session/'quality.json').read_text()),
        next_drop_reset_counter_changes=next_changes,
        source_feedback_available='source_state' in layers,
        hardware_feedback_metadata=hardware_metadata,
        limitations=['Unloaded/no contact is operator declaration, no independent wrench sensor',
            'No external encoder or acquisition timestamp; feedback consistency is not physical calibration',
            'Centered SG is acausal offline analysis; peak values depend on window and attenuation',
            'Quantization bounds exclude mechanical dynamics, model error and filtering bias',
            'No new lag fit, model changes, controller tuning or zero-offset subtraction',
            'Bridge commands are software inputs; no independent evidence of actual CAN TX or motor internal loop timing'])
    save(stage/'feedback_analysis.json',result)
    np.savez_compressed(stage/'feedback_arrays.npz',stamps=st,q_deg=q,v_deg_s=v,raw_nm=raw,grid_stamps=gt,
        grid_q_deg=gq,grid_v_deg_s=gv,native_stamps=native['t'],native_q_deg=native['q'],native_v_deg_s=native['v'],
        software_state_stamps=layers['state']['t'],software_state_q_deg=layers['state']['q'],software_state_v_deg_s=layers['state']['v'],
        bridge_state_stamps=layers['bridge_state']['t'],bridge_state_q_deg=layers['bridge_state']['q'],bridge_state_v_deg_s=layers['bridge_state']['v'],
        **{f'CAN_j{j+1}_stamps':c['t'] for j,c in can.items()},
        **{f'CAN_j{j+1}_q_deg':c['q'] for j,c in can.items()},
        **{f'CAN_j{j+1}_v_deg_s':c['v'] for j,c in can.items()},
        **{f'SG{w}_velocity_deg_s':d[0] for w,d in derivatives.items()},
        **{f'SG{w}_acceleration_deg_s2':d[1] for w,d in derivatives.items()})
    print('Analysis saved:',stage/'feedback_analysis.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True)
    analyze(parser.parse_args().stage.resolve())
