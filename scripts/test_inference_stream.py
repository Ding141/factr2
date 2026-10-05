#!/usr/bin/env python3
"""Real dual-side inference processes, standard mock/adapter, 60s + 10min soak."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import yaml
import numpy as np
from ros_test_processes import ROOT,configure_dds,stop
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=600);p.add_argument('--domain',type=int,default=87)
p.add_argument('--label',choices=['stream','final60'],default='stream')
a=p.parse_args();configure_dds(a.domain)
import rclpy
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import qos_profile_sensor_data
from factr2_next.data_collection.quality import write_json
out=ROOT/'reports/generated/6'/a.label;out.mkdir(parents=True,exist_ok=True)
processes=[];logs=[];inferences={};node=None
streams={s:{'free':[],'raw':[],'filtered':[],'latency_ms':[],'status':[],'rss_mb':[]} for s in ('left','right')}
try:
    for label in ('mock','adapter_left','adapter_right','infer_left','infer_right'):
        logs.append(open(out/(label+'.log'),'w'))
    processes.append(subprocess.Popen(['python',str(ROOT/'scripts/w3_mock.py'),'--side','both'],stdout=logs[0],stderr=subprocess.STDOUT))
    for i,side in enumerate(('left','right')):
        processes.append(subprocess.Popen([str(ROOT/'install/factr2_w3_adapter/lib/factr2_w3_adapter/w3_next_adapter'),
            '--ros-args','--params-file',str(ROOT/f'factr2_w3_adapter/config/{side}_mock.yaml'),'-r','__node:=adapter_'+side],stdout=logs[1+i],stderr=subprocess.STDOUT))
        cfg=yaml.safe_load((ROOT/f'config/w3/{side}/inference.yaml').read_text())
        summary=json.loads((ROOT/f'reports/generated/5/{side}/summary.json').read_text())
        cfg['checkpoint_dir']=summary['checkpoint'];path=out/(side+'.yaml');path.write_text(yaml.safe_dump(cfg))
        infer=subprocess.Popen(['python','-c','from factr2_next.inference.inference_node import main; main()',
            '--ros-args','-p','config_file:='+str(path),'-r','__node:=inference_'+side],stdout=logs[3+i],stderr=subprocess.STDOUT)
        processes.append(infer);inferences[side]=infer
    rclpy.init();node=rclpy.create_node('inference_soak_probe')
    subscriptions=[]
    def receive(m,side,key):
        stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec
        assert m.name==[f'{side}_joint_{i}' for i in range(7)] and len(m.position)==7
        assert not m.velocity and not m.effort and np.isfinite(m.position).all()
        streams[side][key].append(stamp)
        if key=='free':streams[side]['latency_ms'].append((node.get_clock().now().nanoseconds-stamp)*1e-6)
    def status(m,side):
        if m.status:
            v={k.key:json.loads(k.value) for k in m.status[0].values}
            v['received_mono']=time.monotonic();streams[side]['status'].append(v)
    for side in ('left','right'):
        for key,suffix in [('free','free_joint_torque_pred'),('raw','external_joint_torque/raw'),('filtered','external_joint_torque')]:
            subscriptions.append(node.create_subscription(JointState,f'/next/{side}/{suffix}',lambda m,s=side,k=key:receive(m,s,k),qos_profile_sensor_data))
        subscriptions.append(node.create_subscription(DiagnosticArray,f'/next/{side}/status',lambda m,s=side:status(m,s),qos_profile_sensor_data))
    deadline=time.monotonic()+10
    while not all(len(v['free'])>=10 for v in streams.values()) and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.005)
    assert all(len(v['free'])>=10 for v in streams.values()),'no warmed inference outputs'
    for side,v in streams.items():
        pubs=node.get_publisher_names_and_types_by_node('inference_'+side,'/')
        clients=node.get_client_names_and_types_by_node('inference_'+side,'/')
        assert not clients
        assert all(name.startswith('/next/'+side+'/') or name in ('/rosout','/parameter_events') for name,_ in pubs),pubs
        v['publishers']=pubs;v['clients']=clients
        for key in ('free','raw','filtered','latency_ms','status'):v[key].clear()
    start_ros=node.get_clock().now().nanoseconds
    started=time.monotonic();next_rss=0;first60=None
    while time.monotonic()-started<a.seconds:
        rclpy.spin_once(node,timeout_sec=.003)
        assert all(x.poll() is None for x in processes)
        elapsed=time.monotonic()-started
        if elapsed>=next_rss:
            for side,proc in inferences.items():
                status_text=Path(f'/proc/{proc.pid}/status').read_text()
                rss=float(next(x.split()[1] for x in status_text.splitlines() if x.startswith('VmRSS:')))/1024
                streams[side]['rss_mb'].append([elapsed,rss])
            next_rss+=5
        if elapsed>=60 and first60 is None:
            first60={s:{'outputs':len(v['free']),'hz':len(v['free'])/elapsed,
                'source_to_receive_p95_ms':float(np.percentile(v['latency_ms'],95))} for s,v in streams.items()}
            write_json(out/'first60.json',first60)
    elapsed=time.monotonic()-started
    end_ros=node.get_clock().now().nanoseconds
    # Drain final DDS packets after source period; bounded, no backlog chase.
    until=time.monotonic()+.1
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.002)
    report={'result':'PASS','seconds':elapsed,'domain':a.domain,'device':'cpu','cpu_threads':1,
        'visualization_running':False,'source':'300 Hz standard W3 synthetic mock -> actual 50 Hz adapters -> independent InferenceNodes',
        'timing':'model+pre/inverse processing monotonic; latency receive ROS minus final source stamp on one host','first60':first60,'sides':{}}
    for side,v in streams.items():
        mask=[i for i,t in enumerate(v['free']) if start_ros<=t<end_ros]
        v['latency_ms']=[v['latency_ms'][i] for i in mask]
        for key in ('free','raw','filtered'):v[key]=[t for t in v[key] if start_ros<=t<end_ros]
        n=len(v['free']);hz=n/elapsed
        assert hz>=45,(side,hz)
        assert all(b>x for x,b in zip(v['free'],v['free'][1:]))
        assert v['free']==v['raw']==v['filtered'],'three output stamps differ'
        latency=float(np.percentile(v['latency_ms'],95))
        valid=[s for s in v['status'] if s['state']=='valid']
        model_p95=float(np.percentile([s['infer_ms'] for s in valid],95))
        assert latency<=60 and model_p95<20,(latency,model_p95)
        rss=np.array(v['rss_mb']);growth=float(rss[-1,1]-rss[0,1])
        tail_slope=float(np.polyfit(rss[len(rss)//2:,0],rss[len(rss)//2:,1],1)[0])*60 if len(rss)>4 else 0
        assert growth<25 and tail_slope<2,('memory growth',growth,tail_slope)
        report['sides'][side]=dict(outputs=n,hz=hz,source_to_receive_p95_ms=latency,model_p95_ms=model_p95,
            rss_peak_mb=float(rss[:,1].max()),rss_growth_mb=growth,tail_rss_slope_mb_per_minute=tail_slope,
            rss_samples=v['rss_mb'],last_status=valid[-1],publishers=v['publishers'],clients=v['clients'])
    write_json(out/'summary.json',report)
    print(json.dumps({s:{k:v for k,v in r.items() if k not in ('rss_samples','publishers','last_status','clients')} for s,r in report['sides'].items()}),flush=True)
finally:
    codes=[stop(x) for x in reversed(processes)]
    for f in logs:f.close()
    if node:node.destroy_node()
    if rclpy.ok():rclpy.try_shutdown()
    print('child exit codes:',codes,flush=True)
