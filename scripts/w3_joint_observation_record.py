"""Exact-stamp joint/NEXT recorder for a frozen live observation; no commands."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray
from message_filters import Subscriber,TimeSynchronizer
from w3_motion_sequence import write_json


def diagnostic_level(value):
    # ROS Humble represents uint8 as a one-byte bytes object on this host.
    return int.from_bytes(value,'little') if isinstance(value,(bytes,bytearray)) else int(value)


def main(stage):
    assert not (stage/'samples.csv').exists(),'Never overwrite observation samples'
    spec=json.loads((stage/'test_spec.json').read_text());assert spec['expected_units'].startswith('joint external torque')
    names=[f'right_joint_{j}' for j in range(7)]
    rclpy.init();node=Node('long_joint_observation_recorder_right')
    keys=('q_deg','velocity_deg_s','command_deg','measured_tau_nm','free_pred_tau_nm','external_raw_nm','external_filtered_nm')
    fields=['source_timestamp_ns','received_wall_ns','elapsed_seconds','phase','step_index','next_state']
    fields += [f'{key}_j{j+1}' for key in keys for j in range(7)]
    f=(stage/'samples.csv').open('w',newline='');writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    diag=(stage/'diagnostics.jsonl').open('w');start=time.monotonic();rows=0;last_join=None;last_stamp=None
    phase='startup';step_index=0;next_state='missing';next_at=0.;last_save=0.
    def stamp(m):return m.header.stamp.sec*10**9+m.header.stamp.nanosec
    def vector(m,field):
        values=getattr(m,field)
        if len(m.name)!=len(values) or len(set(m.name))!=len(m.name):raise ValueError('Malformed joint data')
        a=np.asarray([values[m.name.index(n)] for n in names],float)
        if not np.isfinite(a).all():raise ValueError('Nonfinite observation')
        return a
    def joined(pos,vel,cmd,effort,pred,raw,filtered):
        nonlocal rows,last_join,last_stamp
        now=time.monotonic();t=stamp(pos)
        if len({stamp(x) for x in (pos,vel,cmd,effort,pred,raw,filtered)})!=1:raise ValueError('Joined source stamps differ')
        if last_stamp is not None and t<=last_stamp:raise ValueError('Nonmonotonic observations')
        a=[np.rad2deg(vector(pos,'position')),np.rad2deg(vector(vel,'velocity')),np.rad2deg(vector(cmd,'position')),
            vector(effort,'effort'),vector(pred,'position'),vector(raw,'position'),vector(filtered,'position')]
        np.testing.assert_allclose(a[3]-a[4],a[5],atol=1e-5)
        row=dict(source_timestamp_ns=t,received_wall_ns=time.time_ns(),elapsed_seconds=now-start,phase=phase,
            step_index=step_index,next_state=next_state if now-next_at<=.25 else 'stale')
        for key,array in zip(keys,a):
            for j,value in enumerate(array):row[f'{key}_j{j+1}']=float(value)
        writer.writerow(row);rows+=1;last_join=now;last_stamp=t
    def diagnostic(topic,msg):
        nonlocal next_state,next_at
        statuses=[]
        for s in msg.status:
            values={v.key:json.loads(v.value) for v in s.values}
            level=diagnostic_level(s.level)
            statuses.append(dict(name=s.name,message=s.message,level=level,values=values))
            if s.name=='factr2/next/right':next_state=values.get('state','unknown');next_at=time.monotonic()
        diag.write(json.dumps(dict(source_timestamp_ns=stamp(msg),received_wall_ns=time.time_ns(),topic=topic,status=statuses),ensure_ascii=False)+'\n')
    reliable=QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE)
    topics=['/factr2/right/joint_pos','/factr2/right/joint_vel','/factr2/right/joint_cmd','/factr2/right/joint_effort',
        '/next/right/free_joint_torque_pred','/next/right/external_joint_torque/raw','/next/right/external_joint_torque']
    subscribers=[Subscriber(node,JointState,t,qos_profile=reliable if i<4 else qos_profile_sensor_data) for i,t in enumerate(topics)]
    sync=TimeSynchronizer(subscribers,50);sync.registerCallback(joined)
    for topic in ('/next/right/status','/factr2/right/adapter_status'):
        node.create_subscription(DiagnosticArray,topic,lambda m,t=topic:diagnostic(t,m),qos_profile_sensor_data)
    def state(status):
        write_json(stage/'recorder_state.json',dict(status=status,rows=rows,updated_mono=time.monotonic(),last_join_mono=last_join,
            source_rate_hz=rows/max(.01,time.monotonic()-start),pid=os.getpid()))
    try:
        state('starting')
        while not (stage/'STOP_RECORDER').exists():
            rclpy.spin_once(node,timeout_sec=.005);now=time.monotonic()
            if now-last_save>=.1:
                if (stage/'test_state.json').exists():
                    current=json.loads((stage/'test_state.json').read_text());phase=current.get('phase','startup');step_index=current.get('step_index',0)
                state('ready' if rows>=100 and last_join is not None and now-last_join<.25 and next_state=='valid' and now-next_at<=.25 else 'starting')
                f.flush();diag.flush();last_save=now
        state('closed')
    except BaseException as e:
        write_json(stage/'recorder_state.json',dict(status='failed',rows=rows,error=str(e),updated_mono=time.monotonic()));raise
    finally:f.close();diag.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);main(p.parse_args().stage.resolve())
