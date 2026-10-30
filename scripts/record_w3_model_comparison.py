"""Full precision recording of coherent dual-model frames; no control API."""
import argparse,csv,json,os,signal,time
from pathlib import Path
import numpy as np
from w3_motion_sequence import write_json

KEYS=('q_deg','velocity_deg_s','command_deg','measured_tau_nm','old_free_nm','new_free_nm',
      'old_raw_nm','new_raw_nm','old_filtered_nm','new_filtered_nm')


def validate_packet(packet,hashes,compensation_sha256=None):
    if (packet['schema']!='w3_joint_lstm_comparison_v1' or packet['model_sha256']!=hashes
        or packet['joint_names']!=[f'right_joint_{j}' for j in range(7)] or packet['units']!='Nm'
        or packet['compensation_enabled'] is not (compensation_sha256 is not None) or packet['sigma_used'] is not False
        or packet['same_input_history'] is not True or packet['history_frames']!=50
        or packet['filter']!='LP12_then_LP5_100Hz'):raise ValueError('Comparison identity changed')
    values={k:np.asarray(packet['recording_frame'][k],float) for k in KEYS}
    if any(x.shape!=(7,) or not np.isfinite(x).all() for x in values.values()):raise ValueError('Malformed comparison frame')
    for kind in ('old','new'):
        expected=values['measured_tau_nm'].astype(np.float32)-values[kind+'_free_nm'].astype(np.float32)
        np.testing.assert_array_equal(expected,values[kind+'_raw_nm'])
    if compensation_sha256 is not None:
        if (packet.get('compensation_sha256')!=compensation_sha256
            or packet.get('comparison_kind')!='same_joint_lstm_residual_ablation_v1'
            or hashes['old']!=hashes['new']):raise ValueError('Residual comparison identity changed')
        for key in ('correction_nm','estimated_velocity_deg_s','estimated_acceleration_deg_s2'):
            a=np.asarray(packet['recording_frame'][key],float)
            if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Malformed residual frame')
            values[key]=a
        expected=(values['old_free_nm'].astype(np.float32)+values['correction_nm'].astype(np.float32)).astype(np.float32)
        np.testing.assert_array_equal(expected,values['new_free_nm'])
    return values


def main(stage):
    import rclpy
    from std_msgs.msg import String
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.qos import qos_profile_sensor_data
    stage=Path(stage).resolve();spec=json.loads((stage/'test_spec.json').read_text())
    compensation_sha256=spec.get('compensation_sha256')
    keys=KEYS+(('correction_nm','estimated_velocity_deg_s','estimated_acceleration_deg_s2') if compensation_sha256 else ())
    rclpy.init();node=rclpy.create_node('right_model_comparison_recorder')
    rows=0;last=None;last_stamp=None;phase='startup';index=0;start=time.monotonic();last_save=0.
    fields=['source_timestamp_ns','received_wall_ns','received_mono','elapsed_seconds','phase','step_index',
            'old_state','new_state','history_sha256','compensation_applied']+[f'{k}_j{j+1}' for k in keys for j in range(7)]
    f=(stage/'samples.csv').open('x',newline='');packets=(stage/'frames.jsonl').open('x');diag=(stage/'diagnostics.jsonl').open('x')
    endpoint=(stage/'endpoint_force.jsonl').open('x')
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    def receive(msg):
        nonlocal rows,last,last_stamp
        p=json.loads(msg.data);values=validate_packet(p,spec['model_sha256'],compensation_sha256);stamp=int(p['stamp_ns'])
        if last_stamp is not None and stamp<=last_stamp:raise ValueError('Source time not increasing')
        now=time.monotonic();row=dict(source_timestamp_ns=stamp,received_wall_ns=time.time_ns(),
            received_mono=now,elapsed_seconds=now-start,phase=phase,step_index=index,
            old_state=p['old_state'],new_state=p['new_state'],history_sha256=p['history_sha256'],
            compensation_applied=p.get('compensation_applied',False))
        for k,v in values.items():row.update({f'{k}_j{j+1}':float(x) for j,x in enumerate(v)})
        writer.writerow(row);packets.write(json.dumps(p,separators=(',',':'))+'\n')
        rows+=1;last=now;last_stamp=stamp
    def diagnostics(topic,msg):
        value=[dict(name=s.name,level=int.from_bytes(s.level,'little') if isinstance(s.level,bytes) else int(s.level),
            message=s.message,values={v.key:v.value for v in s.values}) for s in msg.status]
        diag.write(json.dumps(dict(received_wall_ns=time.time_ns(),topic=topic,status=value))+'\n')
    node.create_subscription(String,spec['packet_topic'],receive,100)
    def receive_endpoint(msg):
        p=json.loads(msg.data)
        if p['schema']!='w3_comparison_endpoint_force_v1' or p['model_sha256']!=spec['model_sha256']:
            raise ValueError('Endpoint comparison identity changed')
        endpoint.write(json.dumps(dict(received_wall_ns=time.time_ns(),packet=p),separators=(',',':'))+'\n')
    node.create_subscription(String,spec.get('endpoint_topic','/next/right/model_comparison/endpoint_force'),receive_endpoint,100)
    for t in ('/next/right/joint_lstm/status','/next/right/frozen_sigma/status','/factr2/right/adapter_status','/factr2/w3_health'):
        node.create_subscription(DiagnosticArray,t,lambda m,t=t:diagnostics(t,m),qos_profile_sensor_data)
    def state(status,error=None):write_json(stage/'recorder_state.json',dict(status=status,error=error,rows=rows,
        updated_mono=time.monotonic(),last_join_mono=last,last_stamp_ns=last_stamp,pid=os.getpid()))
    def stop(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:
        state('starting')
        while not (stage/'STOP_RECORDER').exists():
            rclpy.spin_once(node,timeout_sec=.002);now=time.monotonic()
            if now-last_save>=.1:
                path=stage/'test_state.json'
                if path.exists():s=json.loads(path.read_text());phase=s.get('phase','startup');index=s.get('step_index',0)
                state('ready' if rows>=100 and last is not None and 0<=now-last<=.25 else 'starting')
                f.flush();packets.flush();diag.flush();endpoint.flush();last_save=now
        state('closed')
    except KeyboardInterrupt:state('closed')
    except BaseException as exc:state('failed',str(exc));raise
    finally:
        f.close();packets.close();diag.close();endpoint.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);main(p.parse_args().stage)
