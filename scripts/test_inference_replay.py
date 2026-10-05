#!/usr/bin/env python3
"""Actual checkpoint/InferenceNode: H5 replay equivalence and DDS fault recovery."""
import json
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import yaml
from ros_test_processes import ROOT,configure_dds,stop
configure_dds(88)
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus
from factr2_next.w3_config import SIGNALS
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.offline_eval import read_episode,predict_sequence
from factr2_next.data_collection.quality import write_json
import torch
torch.set_num_threads(1)
out=ROOT/'reports/generated/6/replay';out.mkdir(parents=True,exist_ok=True)
summary=json.loads((ROOT/'reports/generated/5/left/summary.json').read_text())
manifest=json.loads((ROOT/'data/mock/left/split.json').read_text());path=manifest['splits']['test'][0]['path']
loaded=load_checkpoint(summary['checkpoint']);arrays=read_episode(path,'ep_0000');expected,measured=predict_sequence(loaded,arrays)
cfg=yaml.safe_load((ROOT/'config/w3/left/inference.yaml').read_text());cfg['checkpoint_dir']=summary['checkpoint']
config=out/'inference.yaml';config.write_text(yaml.safe_dump(cfg))
processes=[];logs=[];node=None
try:
    for name in ('infer','replay'):logs.append(open(out/(name+'.log'),'w'))
    processes.append(subprocess.Popen(['python','-c','from factr2_next.inference.inference_node import main; main()',
        '--ros-args','-p','config_file:='+str(config)],stdout=logs[0],stderr=subprocess.STDOUT))
    rclpy.init();node=rclpy.create_node('inference_replay_probe')
    received={'free':[],'raw':[],'filtered':[]};statuses=[];inputs=[]
    def ns(m):return m.header.stamp.sec*10**9+m.header.stamp.nanosec
    def take(m,key):
        assert m.name==cfg['joint_names'] and len(m.position)==7 and np.isfinite(m.position).all()
        received[key].append((ns(m),np.array(m.position,dtype=np.float32)))
    def diagnostic(m):
        if m.status:statuses.append((time.monotonic(),{v.key:json.loads(v.value) for v in m.status[0].values}))
    subs=[]
    for key,suffix in [('free','free_joint_torque_pred'),('raw','external_joint_torque/raw'),('filtered','external_joint_torque')]:
        subs.append(node.create_subscription(JointState,'/next/left/'+suffix,lambda m,k=key:take(m,k),qos_profile_sensor_data))
    subs.append(node.create_subscription(DiagnosticArray,'/next/left/status',diagnostic,qos_profile_sensor_data))
    subs.append(node.create_subscription(JointState,'/factr2/left/joint_pos',lambda m:inputs.append(ns(m)),qos_profile_sensor_data))
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.002)
    processes.append(subprocess.Popen(['python',str(ROOT/'scripts/replay_w3_h5.py'),path,'--start-delay','3'],stdout=logs[1],stderr=subprocess.STDOUT))
    deadline=time.monotonic()+22
    while processes[-1].poll() is None and time.monotonic()<deadline:spin(.02)
    assert processes[-1].wait(timeout=2)==0
    spin(.1)
    assert len(inputs)==len(arrays['joint_pos']),len(inputs)
    assert len(received['free'])==len(expected),(len(received['free']),len(expected))
    assert [t for t,_ in received['free']]==inputs[49:]
    np.testing.assert_allclose(np.stack([v for _,v in received['free']]),expected,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(np.stack([v for _,v in received['raw']]),measured-expected,atol=1e-5,rtol=1e-5)
    max_error=float(np.max(abs(np.stack([v for _,v in received['free']])-expected)))
    # Recovery tests use deliberate bad downstream frames; model/node/process unchanged.
    pubs={k:node.create_publisher(JointState,'/factr2/left/'+suffix,qos_profile_sensor_data) for k,(suffix,_) in SIGNALS.items()}
    status_pub=node.create_publisher(DiagnosticArray,'/factr2/left/adapter_status',qos_profile_sensor_data)
    def adapter(ok=True):
        m=DiagnosticArray();m.header.stamp=node.get_clock().now().to_msg()
        s=DiagnosticStatus();s.name='factr2/adapter/left';s.level=DiagnosticStatus.OK if ok else DiagnosticStatus.ERROR;s.message='fixture_ok' if ok else 'fixture_bad';m.status=[s];status_pub.publish(m)
    last_feed_mono=[None]
    def feed(index=0,bad=None):
        adapter();stamp=node.get_clock().now().nanoseconds
        last_feed_mono[0]=time.monotonic()
        if bad=='regression':stamp-=100_000_000
        if bad=='old':stamp-=1_000_000_000
        for k,(_,field) in SIGNALS.items():
            m=JointState();m.name=cfg['joint_names'].copy();m.header.stamp.sec,m.header.stamp.nanosec=divmod(stamp,10**9)
            setattr(m,field,arrays[k][index%len(arrays[k])].astype(float).tolist())
            if k=='measured_joint_torque' and bad=='overflow':m.effort=[2e20]*7
            if k=='joint_pos':
                if bad=='name':m.name.reverse()
                if bad=='shape':m.position.append(0.)
                if bad=='nan':m.position[0]=float('nan')
            pubs[k].publish(m)
        spin(.02)
        return stamp
    spin(.3)
    for _ in range(5):adapter();spin(.02)
    def warm():
        start=len(received['free'])
        for i in range(49):feed(i)
        assert len(received['free'])==start,('early output',len(received['free'])-start)
        feed(49);assert len(received['free'])==start+1,('50th missing',statuses[-1])
    warm()
    last_sent=last_feed_mono[0];start=len(received['free']);status_start=len(statuses)
    # No messages of any kind for .30 s: monotonic watchdog must run independently.
    spin(.30)
    stale=[(t,s) for t,s in statuses[status_start:] if s['state']=='stale']
    assert stale and stale[0][0]-last_sent<=.27,(stale[0][0]-last_sent if stale else 'no stale')
    assert stale[0][1]['history_count']==0 and len(received['free'])==start
    for _ in range(3):adapter();spin(.02)
    warm()
    fault_results={}
    for bad in ('name','shape','nan','regression','old','overflow'):
        count=len(received['free']);feed(0,bad);spin(.04)
        assert len(received['free'])==count
        invalid=[s for _,s in statuses[-5:] if s['state']=='invalid']
        assert invalid,(bad,statuses[-5:])
        fault_results[bad]=invalid[-1]['reason']
        warm()
    count=len(received['free']);spin(.06);feed(0)
    assert len(received['free'])==count
    # First post-gap valid row counts as fresh row 1.
    for i in range(1,49):feed(i)
    assert len(received['free'])==count
    feed(49);assert len(received['free'])==count+1
    adapter(False);spin(.04)
    assert any(s['state']=='invalid' and s['reason'].startswith('adapter:') for _,s in statuses[-5:])
    for _ in range(3):adapter();spin(.02)
    warm()
    write_json(out/'summary.json',dict(result='PASS',h5_path=path,input_rows=len(arrays['joint_pos']),predictions=len(expected),
        max_free_error_nm=max_error,warmup='49 no outputs, 50th one; repeated after every invalid/stale/gap',
        stale_detection_seconds=stale[0][0]-last_sent,fault_reasons=fault_results,last_status=statuses[-1][1]))
    print('PASS: actual H5 replay outputs allclose, 50-frame warmup, stale <=270ms, bad frames/adapter/gap recovery')
finally:
    codes=[stop(p) for p in reversed(processes)]
    for f in logs:f.close()
    if node:node.destroy_node()
    if rclpy.ok():rclpy.try_shutdown()
    print('child exit codes:',codes,flush=True)
