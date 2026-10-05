#!/usr/bin/env python3
"""Real recorder recording service, TTY r toggle, SIGINT and closed H5."""
import json
import os
import pty
import signal
import subprocess
import termios
import time
from ros_test_processes import ROOT, configure_dds, stop
configure_dds(84)
import rclpy
from std_srvs.srv import SetBool
from factr2_next.data_collection.quality import check, write_json
out=ROOT/'reports/generated/4/shutdown';out.mkdir(parents=True,exist_ok=True)
# Use the concrete synthetic config produced by the e2e run.
import yaml
cfg=yaml.safe_load((ROOT/'reports/generated/4/left/faults/record.yaml').read_text())
cfg['session_name']='synthetic_shutdown';cfg['output_dir']=str(ROOT/'data/mock/shutdown')
config=out/'record.yaml';config.write_text(yaml.safe_dump(cfg))
master,slave=pty.openpty();before=termios.tcgetattr(slave)
processes=[];logs=[];node=None
try:
    for label in ('mock','adapter','recorder'):logs.append(open(out/(label+'.log'),'w'))
    processes.append(subprocess.Popen(['python',str(ROOT/'scripts/w3_mock.py'),'--side','left'],stdout=logs[0],stderr=subprocess.STDOUT))
    processes.append(subprocess.Popen([str(ROOT/'install/factr2_w3_adapter/lib/factr2_w3_adapter/w3_next_adapter'),
        '--ros-args','--params-file',str(ROOT/'factr2_w3_adapter/config/left_mock.yaml')],stdout=logs[1],stderr=subprocess.STDOUT))
    processes.append(subprocess.Popen(['python','-c','from factr2_next.data_collection.recorder_node import main; main()',
        '--ros-args','-p','config_file:='+str(config)],stdin=slave,stdout=logs[2],stderr=subprocess.STDOUT))
    rclpy.init();node=rclpy.create_node('recorder_shutdown_probe')
    client=node.create_client(SetBool,'/factr2/left/record')
    assert client.wait_for_service(timeout_sec=6)
    def spin(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
    spin(.7)
    assert before!=termios.tcgetattr(slave),'cbreak not entered'
    req=SetBool.Request();req.data=True
    future=client.call_async(req);rclpy.spin_until_future_complete(node,future,timeout_sec=3)
    assert future.result().success
    spin(1.4)
    os.write(master,b'r');spin(.3)  # keyboard stops the service-started recording
    os.write(master,b'r');spin(1.4)
    processes[-1].send_signal(signal.SIGINT)
    assert processes[-1].wait(timeout=5)==0
    assert termios.tcgetattr(slave)==before,'TTY settings not restored'
    files=sorted((ROOT/'data/mock/shutdown').glob('*.h5'));report=check(files[-1])
    assert report['accepted'],report['errors']
    assert len(report['episodes'])==2
    write_json(out/'summary.json',dict(result='PASS',tty_restored=True,ctrl_c_exit=0,closed_h5_report=report))
    print('PASS: service + keyboard, SIGINT 0, TTY restored, closed H5 strict PASS')
finally:
    for p in reversed(processes):stop(p)
    for f in logs:f.close()
    if node:node.destroy_node()
    if rclpy.ok():rclpy.try_shutdown()
    os.close(master);os.close(slave)
