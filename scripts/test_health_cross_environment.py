#!/usr/bin/env python3
"""W3 raw mock + health monitor → standard DDS → real-profile NEXT adapters."""
import json
from pathlib import Path
import subprocess
import sys
import time
from ros_test_processes import ROOT, configure_dds, stop, w3_process
configure_dds(80)
import rclpy
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from sensor_msgs.msg import JointState
assert 'torch' not in sys.modules
assert not any('w3_dual_arm_ws' in p for p in sys.path)
out=ROOT/'reports/generated/3'
out.mkdir(parents=True,exist_ok=True)
control=out/'health_mode.txt'
control.write_text('healthy')
processes=[];logs=[];node=None
try:
    for name in ('raw','monitor','left_real','right_real'):
        logs.append(open(out/f'health_{name}.log','w'))
    processes.append(w3_process(['/usr/bin/python3.10',ROOT/'factr2_w3_adapter/test/w3_health_probe.py',control],logs[0]))
    processes.append(w3_process(['/usr/bin/python3.10',ROOT/'factr2_w3_adapter/tools/w3_health_monitor.py'],logs[1]))
    for i,side in enumerate(('left','right')):
        processes.append(subprocess.Popen([str(ROOT/'install/factr2_w3_adapter/lib/factr2_w3_adapter/w3_next_adapter'),
            '--ros-args','--params-file',str(ROOT/f'factr2_w3_adapter/config/{side}_real.yaml'),
            '-r',f'__node:=w3_next_adapter_{side}'],stdout=logs[i+2],stderr=subprocess.STDOUT))
    rclpy.init()
    node=rclpy.create_node('health_cross_environment_probe')
    state_pub=node.create_publisher(JointState,'/joint_states',qos_profile_sensor_data)
    command_pub=node.create_publisher(JointState,'/joint_position_controller/command_state',qos_profile_sensor_data)
    health=[];received={'left':[],'right':[]}
    subs=[node.create_subscription(DiagnosticArray,'/factr2/w3_health',health.append,qos_profile_sensor_data)]
    for side in received:
        subs.append(node.create_subscription(JointState,f'/factr2/{side}/joint_pos',
                    lambda m,s=side:received[s].append(m),qos_profile_sensor_data))

    def drive(seconds):
        end=time.monotonic()+seconds
        next_pub=0
        while time.monotonic()<end:
            if time.monotonic()>=next_pub:
                msg=JointState();msg.header.stamp=node.get_clock().now().to_msg()
                msg.name=[f'{s}_joint_{i}' for s in ('left','right') for i in range(7)]
                msg.position=[0.01*i for s in ('left','right') for i in range(7)]
                msg.velocity=[0.0]*14;msg.effort=[1.0]*14
                state_pub.publish(msg)
                msg.effort=[]
                command_pub.publish(msg)
                next_pub=time.monotonic()+0.004
            rclpy.spin_once(node,timeout_sec=0.001)
            assert all(p.poll() is None for p in processes), 'child exited before test completed'

    def statuses():
        assert health
        return {s.name.rsplit('/',1)[-1]:s for s in health[-1].status}

    drive(1.5)
    assert received['left'] and received['right']
    assert all(s.level==DiagnosticStatus.OK for s in statuses().values())
    results={}
    cases={'missing_left':'missing_slots','duplicate_left':'duplicate_slots','offline_left':'offline_slots',
           'disabled_left':'disabled_slots','fault_left':'fault_slots','nan_left':'nonfinite_slots'}
    for mode,key in cases.items():
        control.write_text(mode)
        drive(0.15)  # Deliver health rejection and drain pre-fault in-flight samples.
        for v in received.values():v.clear()
        drive(0.15)
        status=statuses()
        values={v.key:json.loads(v.value) for v in status['left'].values}
        assert status['left'].level!=DiagnosticStatus.OK and values[key]==[0]
        assert not received['left'] and received['right'], 'fault did not gate only left'
        assert status['right'].level==DiagnosticStatus.OK
        results[mode]={'left_rows':len(received['left']),'right_rows':len(received['right']),
                       'left_values':values}
        control.write_text('healthy')
        drive(0.2)
        assert received['left'] and statuses()['left'].level==DiagnosticStatus.OK
    control.write_text('right_only')
    drive(0.3)
    assert statuses()['right'].level==DiagnosticStatus.OK
    values={v.key:json.loads(v.value) for v in statuses()['right'].values}
    assert values['channel']==1
    results['right_only_channel1']=values
    control.write_text('healthy');drive(0.2)
    control.write_text('stop');drive(0.38)
    for v in received.values():v.clear()
    drive(0.12)
    assert not any(received.values())
    assert all(s.level!=DiagnosticStatus.OK for s in statuses().values())
    results['raw_timeout']={s:{v.key:json.loads(v.value) for v in st.values} for s,st in statuses().items()}
    graph={}
    for name in ('w3_health_monitor','w3_next_adapter_left','w3_next_adapter_right'):
        publishers=node.get_publisher_names_and_types_by_node(name,'/')
        clients=node.get_client_names_and_types_by_node(name,'/')
        assert not clients
        assert not any('commands' in topic or 'trajectory' in topic for topic,_ in publishers)
        graph[name]={'publishers':publishers,'clients':clients,
                     'subscribers':node.get_subscriber_names_and_types_by_node(name,'/')}
    result={'result':'PASS','domain':80,'source':'mock MotorStateArray, not hardware',
            'cases':results,'graph':graph}
    (out/'health_cross_environment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':'PASS','cases':list(results)}),flush=True)
finally:
    codes=[stop(p) for p in reversed(processes)]
    for f in logs:f.close()
    if node is not None:node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
    print('child exit codes:',codes,flush=True)
