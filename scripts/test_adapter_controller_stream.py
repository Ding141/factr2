#!/usr/bin/env python3
"""60s actual C++ controller/mock → separate NEXT adapter → DDS subscriber."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from ros_test_processes import ROOT, W3, configure_dds, stop, w3_process

parser=argparse.ArgumentParser()
parser.add_argument('--side',choices=['left','right'],default='left')
parser.add_argument('--seconds',type=float,default=60.0)
parser.add_argument('--domain',type=int,default=78)
a=parser.parse_args()
configure_dds(a.domain)
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray
assert not any('w3_dual_arm_ws' in p for p in sys.path)
assert 'torch' not in sys.modules
out=ROOT/'reports/generated/3'
out.mkdir(parents=True,exist_ok=True)
logs=[]; processes=[]; node=None
try:
    for label in ('controller','adapter'):
        logs.append(open(out/f'stream_{a.side}_{label}.log','w'))
    processes.append(w3_process([W3/'build/ieir_controllers/command_state_mock',a.side,a.seconds+8],logs[0]))
    processes.append(subprocess.Popen([str(ROOT/'install/factr2_w3_adapter/lib/factr2_w3_adapter/w3_next_adapter'),
        '--ros-args','--params-file',str(ROOT/f'factr2_w3_adapter/config/{a.side}_mock.yaml'),
        '-r',f'__node:=w3_next_adapter_{a.side}'],stdout=logs[1],stderr=subprocess.STDOUT))
    rclpy.init()
    node=rclpy.create_node(f'adapter_stream_probe_{a.side}')
    streams={key:[] for key in ('joint_pos','joint_vel','joint_cmd','joint_effort')}
    subscriptions=[node.create_subscription(JointState,f'/factr2/{a.side}/{key}',
        lambda m,k=key:streams[k].append(m),qos_profile_sensor_data) for key in streams]
    statuses=[]
    subscriptions.append(node.create_subscription(DiagnosticArray,f'/factr2/{a.side}/adapter_status',statuses.append,qos_profile_sensor_data))
    deadline=time.monotonic()+5
    while min(map(len,streams.values()))<10 and time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=0.01)
    assert min(map(len,streams.values()))>=10, 'no complete startup samples'
    start_ns=node.get_clock().now().nanoseconds
    started=time.monotonic()
    while time.monotonic()-started<a.seconds:
        rclpy.spin_once(node,timeout_sec=0.01)
        assert all(p.poll() is None for p in processes), 'fixture/adapter exited early'
    duration=time.monotonic()-started
    end_ns=node.get_clock().now().nanoseconds
    deadline=time.monotonic()+0.15
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=0.005)
    summary={'side':a.side,'domain':a.domain,'duration_seconds':duration,'input_hz':300,
             'source':'real JointPositionController with double mock interfaces; no hardware',
             'controller_commit':'d32262d','streams':{},'endpoints':{}}
    expected_names=[f'{a.side}_joint_{i}' for i in range(7)]
    stamps={}
    for key,field in [('joint_pos','position'),('joint_vel','velocity'),('joint_cmd','position'),('joint_effort','effort')]:
        selected=[m for m in streams[key] if start_ns<=m.header.stamp.sec*10**9+m.header.stamp.nanosec<end_ns]
        stamps[key]=[m.header.stamp.sec*10**9+m.header.stamp.nanosec for m in selected]
        rate=len(selected)/duration
        assert 45<=rate<=55, f'{key} frequency {rate}'
        assert all(b>x for x,b in zip(stamps[key],stamps[key][1:])), 'non-increasing stamp'
        for msg in selected:
            assert msg.name==expected_names and len(getattr(msg,field))==7
            assert all(not getattr(msg,f) for f in ('position','velocity','effort') if f!=field)
            expected=[1.0]*7 if key=='joint_effort' else [0.0]*7 if key=='joint_vel' else [0.01*i for i in range(7)]
            assert all(abs(x-y)<1e-12 for x,y in zip(getattr(msg,field),expected))
        gaps=[(b-x)*1e-9 for x,b in zip(stamps[key],stamps[key][1:])]
        summary['streams'][key]={'rows':len(selected),'hz':rate,'max_gap_seconds':max(gaps),'first_stamp_ns':stamps[key][0],'last_stamp_ns':stamps[key][-1]}
        info=node.get_publishers_info_by_topic(f'/factr2/{a.side}/{key}')
        assert len(info)==1
        summary['endpoints'][key]=str(info[0].qos_profile)
    assert all(v==stamps['joint_pos'] for v in stamps.values()), 'four stream stamp sequences differ'
    summary['publishers']=node.get_publisher_names_and_types_by_node(f'w3_next_adapter_{a.side}','/')
    summary['clients']=node.get_client_names_and_types_by_node(f'w3_next_adapter_{a.side}','/')
    assert not summary['clients']
    allowed={f'/factr2/{a.side}/{k}' for k in streams}|{f'/factr2/{a.side}/adapter_status','/rosout','/parameter_events'}
    assert {name for name,_ in summary['publishers']}<=allowed
    summary['diagnostics']={v.key:json.loads(v.value) for v in statuses[-1].status[0].values}
    assert summary['diagnostics']['health_mode']=='mock/no-hardware-health'
    summary['result']='PASS'
    (out/f'stream_{a.side}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['streams']),flush=True)
finally:
    codes=[stop(p) for p in reversed(processes)]
    for f in logs:f.close()
    if node is not None:node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
    print('child exit codes:',codes,flush=True)
