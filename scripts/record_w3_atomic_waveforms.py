"""Full precision, coherent observer frames; no control publishers."""
import argparse
import csv
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from w3_motion_sequence import write_json

KEYS=('q_deg','velocity_deg_s','command_deg','measured_tau_nm','base_free_pred_nm',
      'measured_filtered_nm','corrected_free_pred_nm','compensated_raw_nm',
      'external_filtered_nm','correction_nm','base_external_raw_nm',
      'estimated_velocity_deg_s','estimated_acceleration_deg_s2')
BASE_KEYS=('q_deg','velocity_deg_s','command_deg','measured_tau_nm','base_free_pred_nm',
           'base_external_raw_nm','external_filtered_nm','correction_nm')


def validate_base_frame(p,base_hash):
    if (p['schema']!='w3_frozen_sigma_observer_v1' or p['joint_names']!=[f'right_joint_{j}' for j in range(7)]
            or p['units']!='Nm' or p['base_metadata_sha256']!=base_hash or p['compensation_applied'] is not False):
        raise ValueError('Wrong uncompensated observer identity/units/mode')
    a={k:np.asarray(p['recording_frame'][k],float) for k in BASE_KEYS}
    if any(v.shape!=(7,) or not np.isfinite(v).all() for v in a.values()):raise ValueError('Malformed base frame')
    # Native float32 subtraction is preserved in the packet; perform the same
    # operation here before comparing the serialized full precision values.
    expected=(a['measured_tau_nm'].astype(np.float32)-a['base_free_pred_nm'].astype(np.float32)).astype(float)
    np.testing.assert_allclose(expected,a['base_external_raw_nm'],atol=1e-8,rtol=0)
    np.testing.assert_array_equal(a['correction_nm'],np.zeros(7))
    np.testing.assert_array_equal(p['correction_nm'],np.zeros(7))
    np.testing.assert_array_equal(a['base_external_raw_nm'],p['raw_external_nm'])
    np.testing.assert_array_equal(a['external_filtered_nm'],p['final_external_nm'])
    return a


def validate_frame(p,base_hash):
    if (p['schema']!='w3_kinematic_residual_v1' or p['joint_names']!=[f'right_joint_{j}' for j in range(7)]
            or p['units']!='Nm' or p['base_metadata_sha256']!=base_hash):raise ValueError('Wrong observer identity/units')
    a={k:np.asarray(p['recording_frame'][k],float) for k in KEYS}
    if any(v.shape!=(7,) or not np.isfinite(v).all() for v in a.values()):raise ValueError('Malformed atomic frame')
    for left,right,out in [('measured_tau_nm','base_free_pred_nm','base_external_raw_nm'),
                           ('measured_filtered_nm','corrected_free_pred_nm','compensated_raw_nm')]:
        np.testing.assert_allclose(a[left]-a[right],a[out],atol=1e-8,rtol=0)
    for field,key in [('raw_external_nm','compensated_raw_nm'),('final_external_nm','external_filtered_nm'),
                      ('upstream_external_nm','base_external_raw_nm'),('correction_nm','correction_nm')]:
        np.testing.assert_allclose(a[key],p[field],atol=1e-8,rtol=0)
    return a


def main(stage):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
    from std_msgs.msg import String
    from diagnostic_msgs.msg import DiagnosticArray
    stage=Path(stage).resolve();spec=json.loads((stage/'test_spec.json').read_text())
    base_only=spec.get('recording_mode')=='atomic_base_no_compensation'
    keys=BASE_KEYS if base_only else KEYS
    root=spec['next_topic_root'];rclpy.init();node=rclpy.create_node('right_atomic_waveform_recorder')
    rows=0;last=None;last_stamp=None;last_save=0.;start=time.monotonic();phase='startup';index=0
    f=(stage/'samples.csv').open('x',newline='');pf=(stage/'distribution.jsonl').open('x');df=(stage/'diagnostics.jsonl').open('x')
    fields=['source_timestamp_ns','received_wall_ns','received_mono','elapsed_seconds','phase','step_index',
            'next_state','next_reason','compensation_applied']+[f'{k}_j{j+1}' for k in keys for j in range(7)]
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    def packet(msg):
        nonlocal rows,last,last_stamp
        p=json.loads(msg.data);a=(validate_base_frame if base_only else validate_frame)(p,spec['base_metadata_sha256']);t=int(p['stamp_ns'])
        if last_stamp is not None and t<=last_stamp:raise ValueError('Source clock not increasing')
        now=time.monotonic();row=dict(source_timestamp_ns=t,received_wall_ns=time.time_ns(),received_mono=now,
            elapsed_seconds=now-start,phase=phase,step_index=index,next_state=p['state'],next_reason=p['reason'],
            compensation_applied=int(p['compensation_applied']))
        for k,v in a.items():row.update({f'{k}_j{j+1}':float(x) for j,x in enumerate(v)})
        writer.writerow(row);pf.write(json.dumps(dict(received_wall_ns=time.time_ns(),packet=p))+'\n')
        rows+=1;last=now;last_stamp=t
    def diag(topic,msg):
        statuses=[]
        for st in msg.status:
            level=int.from_bytes(st.level,'little') if isinstance(st.level,(bytes,bytearray)) else int(st.level)
            statuses.append(dict(name=st.name,level=level,message=st.message,values={v.key:json.loads(v.value) for v in st.values}))
        df.write(json.dumps(dict(topic=topic,received_wall_ns=time.time_ns(),status=statuses))+'\n')
    node.create_subscription(String,root+'/distribution',packet,QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE))
    for topic in (root+'/status','/next/right/frozen_sigma/status','/factr2/right/adapter_status','/factr2/w3_health'):
        node.create_subscription(DiagnosticArray,topic,lambda m,t=topic:diag(t,m),qos_profile_sensor_data)
    def state(status,error=None):write_json(stage/'recorder_state.json',dict(status=status,error=error,rows=rows,
        updated_mono=time.monotonic(),last_join_mono=last,source_rate_hz=rows/max(.01,time.monotonic()-start),
        recording_mode='atomic_full_precision',pid=os.getpid()))
    def stop(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:
        state('starting')
        while not (stage/'STOP_RECORDER').exists():
            rclpy.spin_once(node,timeout_sec=.005);now=time.monotonic()
            if now-last_save>=.1:
                path=stage/'test_state.json'
                if path.exists():s=json.loads(path.read_text());phase=s.get('phase','startup');index=s.get('step_index',0)
                state('ready' if rows>=100 and last is not None and 0<=now-last<.25 else 'starting')
                f.flush();pf.flush();df.flush();last_save=now
        state('closed')
    except KeyboardInterrupt:state('closed')
    except BaseException as e:state('failed',str(e));raise
    finally:f.close();pf.close();df.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
