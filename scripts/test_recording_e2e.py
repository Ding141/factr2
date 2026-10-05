#!/usr/bin/env python3
"""Actual standard mock -> adapter -> RecorderNode, automatic recording only."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import yaml
from ros_test_processes import ROOT, configure_dds, stop
from w3_mock import values

p = argparse.ArgumentParser()
p.add_argument('--side', choices=['left','right'], default='left')
p.add_argument('--domain', type=int, default=81)
p.add_argument('--seconds', type=float, default=60)
p.add_argument('--fault-test', action='store_true')
p.add_argument('--session-label', default='continuous')
a = p.parse_args(); configure_dds(a.domain)
import rclpy
from rclpy.parameter import Parameter
from factr2_next.data_collection.recorder_node import RecorderNode
from factr2_next.data_collection.quality import check, write_json, make_manifest
out = ROOT / 'reports/generated/4' / a.side
out=out/a.session_label
out.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT/f'config/w3/{a.side}/record.yaml').read_text())
cfg.update(output_dir=str(ROOT/f'data/mock/{a.side}'), session_name='synthetic_' + a.side + '_' + a.session_label)
cfg['metadata'] = dict(source='synthetic', tool='mock_tool', gripper='excluded', load='none',
    calibration={'id':'synthetic_v1'}, control={'id':'mock_no_motor_commands'}, trajectory_id='seed0_sine',
    contact={'present':False,'label':'synthetic_free'}, temperature={'source':'unavailable'},
    audit_paths=[], health_gate_enabled=False)
config_path=out/'record.yaml'; config_path.write_text(yaml.safe_dump(cfg))
logs=[]; processes=[]; node=None
faults = [{'at':5,'duration':.35,'kind':'pause'}, {'at':8,'duration':.3,'kind':'nan'},
          {'at':11,'duration':.3,'kind':'rollback'}, {'at':14,'duration':.3,'kind':'skew'}] if a.fault_test else []
try:
    for label in ('mock','adapter'):
        logs.append(open(out/(label+'.log'),'w'))
    processes.append(subprocess.Popen([str(ROOT/'scripts/next.sh'),'python',str(ROOT/'scripts/w3_mock.py'),
        '--side',a.side,'--faults',json.dumps(faults)],stdout=logs[0],stderr=subprocess.STDOUT))
    processes.append(subprocess.Popen([str(ROOT/'install/factr2_w3_adapter/lib/factr2_w3_adapter/w3_next_adapter'),
        '--ros-args','--params-file',str(ROOT/f'factr2_w3_adapter/config/{a.side}_mock.yaml'),
        '-r',f'__node:=w3_next_adapter_{a.side}'],stdout=logs[1],stderr=subprocess.STDOUT))
    rclpy.init()
    node=RecorderNode(parameter_overrides=[Parameter('config_file',value=str(config_path))])
    def spin(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.005)
            assert all(x.poll() is None for x in processes)
    deadline=time.monotonic()+5
    while node.latest_samples is None and time.monotonic()<deadline:
        spin(.02)
    assert node.latest_samples is not None
    starts=[]; episode_starts={}
    for duration in ([a.seconds] if not a.fault_test else [16,1.4,1.4]):
        deadline=time.monotonic()+3
        while (node.latest_sync_time is None or time.monotonic()-node.latest_sync_time>.1) and time.monotonic()<deadline:
            spin(.02)
        start=node.get_clock().now().nanoseconds; starts.append(start)
        assert node._toggle_recording()
        episode_starts[node.writer.episode.name.rsplit("/",1)[-1]]=start
        spin(duration)
        assert node._toggle_recording()
        rows=sum(len(node.writer.file[e]['joint_pos/data']) for e in node.writer.file if 'joint_pos' in node.writer.file[e])
        spin(.3)
        assert rows==sum(len(node.writer.file[e]['joint_pos/data']) for e in node.writer.file if 'joint_pos' in node.writer.file[e])
    path=node.writer.path
    metadata=node.writer.metadata
    node.close()
    accepted_eps=[e for e,v in metadata['episodes'].items() if not v.get('excluded_reason')]
    report=check(path,accepted_eps)
    assert report['accepted'], report['errors']
    total=0
    with h5py.File(path,'r') as h:
        for ep in h:
            if 'joint_pos' not in h[ep]:continue
            ts=h[ep]['joint_pos/timestamps'][:]; total+=len(ts)
            assert ts[0]>episode_starts.get(ep,starts[0])
            for row, stamp in enumerate(ts):
                expected=values(int(stamp),range(7))
                for key,v in zip(('joint_pos','joint_vel','joint_cmd','measured_joint_torque'),expected):
                    np.testing.assert_allclose(h[ep][key]['data'][row],v,atol=2e-7)
    if not a.fault_test:
        assert total>=45*a.seconds
    else:
        assert len(metadata['episodes'])>=7, metadata
        assert any('gap' in v.get('end_reason','') or 'timeout' in v.get('end_reason','') for v in metadata['episodes'].values())
    report.update(result='PASS',side=a.side,seconds=a.seconds,rows=total,episodes_metadata=metadata['episodes'],
                  recording_starts_ns=starts,invalid_frames=node.invalid_frames,source='synthetic standard W3 mock -> actual adapter -> actual recorder')
    write_json(out/('faults.json' if a.fault_test else 'continuous.json'),report)
    print(json.dumps({'path':str(path),'rows':total,'episodes':len(metadata['episodes']),'result':'PASS'}),flush=True)
finally:
    if node is not None:
        node.close();node.destroy_node()
    if rclpy.ok():rclpy.try_shutdown()
    codes=[stop(x) for x in reversed(processes)]
    for f in logs:f.close()
    print('child exit codes:',codes,flush=True)
