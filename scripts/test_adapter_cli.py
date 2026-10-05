#!/usr/bin/env python3
"""Exercise ROS console/launch registration, startup rejection and SIGINT."""
import json
import subprocess
import time
from ros_test_processes import ROOT, configure_dds, stop
configure_dds(81)
import rclpy
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray

out=ROOT/'reports/generated/3'
result={'domain':81,'startup_rejections':{}}
cases={
    'side_required':[],
    'zero_rate':['--ros-args','-p','side:=left','-p','publish_hz:=0.0'],
    'negative_timeout':['--ros-args','-p','side:=left','-p','input_timeout_seconds:=-1.0'],
    'wrong_root':['--ros-args','-p','side:=left','-p','output_root:=/factr2/right'],
}
for name,args in cases.items():
    completed=subprocess.run(['ros2','run','factr2_w3_adapter','w3_next_adapter',*args],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=8)
    (out/f'cli_{name}.log').write_text(completed.stdout)
    assert completed.returncode!=0 and 'ValueError' in completed.stdout
    result['startup_rejections'][name]={'exit_code':completed.returncode,'result':'PASS'}
rclpy.init()
node=rclpy.create_node('adapter_cli_probe')
received=[]
sub=node.create_subscription(DiagnosticArray,'/factr2/left/adapter_status',received.append,qos_profile_sensor_data)
with open(out/'cli_launch.log','w') as log:
    process=subprocess.Popen(['ros2','launch','factr2_w3_adapter','adapter.launch.py',
                              'side:=left','profile:=mock'],stdout=log,stderr=subprocess.STDOUT)
    try:
        end=time.monotonic()+6
        while not received and time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=0.02)
        assert received and process.poll() is None
        values={v.key:json.loads(v.value) for v in received[-1].status[0].values}
        assert values['health_mode']=='mock/no-hardware-health'
        result['launch']={'diagnostics_received':True,'values':values}
    finally:
        result['launch_exit_code']=stop(process)
        node.destroy_node()
        rclpy.shutdown()
assert result['launch_exit_code']==0
result['result']='PASS'
(out/'cli.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
