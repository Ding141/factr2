#!/usr/bin/env python3
"""Actual dual read-only launch, dynamic 7th curve, stale UI, port lifecycle."""
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen
import yaml
import numpy as np
from ros_test_processes import ROOT,configure_dds,stop
p=argparse.ArgumentParser();p.add_argument('--hold-seconds',type=float,default=0);a=p.parse_args()
configure_dds(89)
import rclpy
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
from factr2_next.data_collection.quality import write_json
out=ROOT/'reports/generated/6/web';out.mkdir(parents=True,exist_ok=True)
processes=[];logs=[];node=None
try:
    args=[]
    for side,port in [('left',18100),('right',18101)]:
        c=yaml.safe_load((ROOT/f'config/w3/{side}/inference.yaml').read_text())
        c['checkpoint_dir']=json.loads((ROOT/f'reports/generated/5/{side}/summary.json').read_text())['checkpoint']
        file=out/(side+'_infer.yaml');file.write_text(yaml.safe_dump(c));args.append(side+'_inference_config:='+str(file))
        w=yaml.safe_load((ROOT/f'config/w3/{side}/visualize.yaml').read_text());w['web']['port']=port
        file=out/(side+'_web.yaml');file.write_text(yaml.safe_dump(w));args.append(side+'_web_config:='+str(file))
    for label in ('mock','adapter_left','adapter_right','launch','port_conflict','wrong_model'):logs.append(open(out/(label+'.log'),'w'))
    processes.append(subprocess.Popen(['python',str(ROOT/'scripts/w3_mock.py'),'--side','both'],stdout=logs[0],stderr=subprocess.STDOUT))
    for i,side in enumerate(('left','right')):
        processes.append(subprocess.Popen([str(ROOT/'install/factr2_w3_adapter/lib/factr2_w3_adapter/w3_next_adapter'),
            '--ros-args','--params-file',str(ROOT/f'factr2_w3_adapter/config/{side}_mock.yaml'),'-r','__node:=web_adapter_'+side],stdout=logs[1+i],stderr=subprocess.STDOUT))
    processes.append(subprocess.Popen(['ros2','launch','factr2_next','dual_readonly.launch.py']+args,stdout=logs[3],stderr=subprocess.STDOUT,start_new_session=True))
    rclpy.init();node=rclpy.create_node('web_dual_probe');counts={'left':0,'right':0,'source':0}
    def count(m,key):counts[key]+=1
    subs=[node.create_subscription(JointState,f'/next/{s}/free_joint_torque_pred',lambda m,k=s:count(m,k),qos_profile_sensor_data) for s in ('left','right')]
    subs.append(node.create_subscription(JointState,'/joint_states',lambda m:count(m,'source'),qos_profile_sensor_data))
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.003)
    def snapshot(port):return json.load(urlopen(f'http://127.0.0.1:{port}/snapshot',timeout=2))
    deadline=time.monotonic()+12
    while counts['left']<5 or counts['right']<5:
        assert time.monotonic()<deadline and processes[-1].poll() is None,'dual launch failed'
        spin(.05)
    spin(1.5)
    first=snapshot(18100);right=snapshot(18101)
    write_json(out/'initial_left.json',first);write_json(out/'initial_right.json',right)
    write_json(out/'graph.json',{s:node.get_subscriber_names_and_types_by_node('next_web_'+s,'/') for s in ('left','right')})
    assert first['state']=='valid' and right['state']=='valid',(first['state'],right['state'])
    assert len(first['joint_names'])==7 and first['joint_names'][6]=='left_joint_6'
    assert 'ext_j7' in first and 'raw_j7' in first and 'free_j7' in first and 'fb_j1' not in first
    html=urlopen('http://127.0.0.1:18100').read().decode()
    assert 'left_joint_6' in html and 'ext_j7' in html and 'raw_j7' in html and 'Feedback (Nm)' not in html
    # Independent seventh-joint display fixture uses NEXT output topics only.
    # Inference continues publishing its own data; pulse clearly identifies slot 7.
    pulse=node.create_publisher(JointState,'/next/left/external_joint_torque',qos_profile_sensor_data)
    msg=JointState();msg.name=[f'left_joint_{i}' for i in range(7)];msg.position=[0.]*6+[7.123]
    msg.header.stamp=node.get_clock().now().to_msg();pulse.publish(msg);spin(.06)
    got=snapshot(18100)
    assert any(abs(v-7.123)<1e-6 for v in got['ext_j7'] if v is not None)
    assert not any(abs(v-7.123)<1e-6 for v in got['ext_j6'] if v is not None)
    # Port conflict gives actionable startup failure and leaves original server live.
    conflict=subprocess.Popen(['python','-c','from factr2_next.visualization.web_node import main; main()',
        '--ros-args','-p','config_file:='+str(out/'left_web.yaml'),'-r','__node:=web_conflict'],stdout=logs[4],stderr=subprocess.STDOUT)
    assert conflict.wait(timeout=6)!=0
    logs[4].flush();assert 'web_bind_failed' in (out/'port_conflict.log').read_text()
    assert snapshot(18100)['state']=='valid'
    wrong=yaml.safe_load((out/'right_infer.yaml').read_text());wrong['checkpoint_dir']=yaml.safe_load((out/'left_infer.yaml').read_text())['checkpoint_dir']
    wrongfile=out/'wrong.yaml';wrongfile.write_text(yaml.safe_dump(wrong))
    invalid=subprocess.Popen(['python','-c','from factr2_next.inference.inference_node import main; main()',
        '--ros-args','-p','config_file:='+str(wrongfile),'-r','__node:=wrong_checkpoint'],stdout=logs[5],stderr=subprocess.STDOUT)
    assert invalid.wait(timeout=6)!=0
    logs[5].flush();assert 'checkpoint_profile:side' in (out/'wrong_model.log').read_text()
    spin(.2)
    if ('wrong_checkpoint','/') in node.get_node_names_and_namespaces():
        assert not [x for x in node.get_publisher_names_and_types_by_node('wrong_checkpoint','/') if x[0].startswith('/next/')]
    read_only_pubs={s:node.get_publisher_names_and_types_by_node('next_inference_'+s,'/') for s in ('left','right')}
    clients={s:node.get_client_names_and_types_by_node('next_inference_'+s,'/') for s in ('left','right')}
    assert not any(clients.values())
    # Hold valid pages for browser inspection. All holding processes remain read-only.
    write_json(out/'ready.json',dict(left_url='http://127.0.0.1:18100',right_url='http://127.0.0.1:18101',state='valid'))
    if a.hold_seconds:
        print('READY: http://127.0.0.1:18100 and :18101',flush=True);spin(a.hold_seconds)
    # Kill only left inference child; right/source/left HTTP server remain alive.
    launch_pid=processes[-1].pid
    children=Path(f'/proc/{launch_pid}/task/{launch_pid}/children').read_text().split()
    left_pid=None
    for child in children:
        argv=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'next_infer' in argv and 'next_inference_left' in argv:left_pid=int(child)
    assert left_pid,children
    before=counts.copy();Path(out/'left_pid.txt').write_text(str(left_pid))
    import os
    os.kill(left_pid,signal.SIGINT);spin(.5)
    stale=snapshot(18100);right=snapshot(18101)
    assert stale['state']=='stale' and stale['last_valid_age_seconds']>=.25,stale['state']
    assert right['state']=='valid' and counts['right']>before['right']+10 and counts['source']>before['source']+50
    assert processes[0].poll() is None
    write_json(out/'summary.json',dict(result='PASS',left_initial=first['state'],right_initial='valid',seventh_joint_pulse=7.123,
        left_after_stop=stale['state'],right_after_stop=right['state'],source_running=True,
        port_conflict='actionable failure; original server stays live',wrong_checkpoint='failed before output',
        read_only_publishers=read_only_pubs,clients=clients))
    print('PASS: dual launch, seventh curve, stale, side isolation, port conflict, wrong checkpoint')
finally:
    if processes:
        # ros2 launch owns children; SIGINT cleans both webs and remaining inference.
        if processes[-1].poll() is None:processes[-1].send_signal(signal.SIGINT)
    codes=[stop(x) for x in reversed(processes)]
    for f in logs:f.close()
    if node:node.destroy_node()
    if rclpy.ok():rclpy.try_shutdown()
    for port in (18100,18101):
        s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port));s.close()
    print('ports released; exit codes:',codes,flush=True)
