"""Read-only clock and stream stability probe; no motion publisher."""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray


def source_timing_ok(source_gaps,source_ages,topics):
    """Receive scheduling jitter is not a missing original source sample."""
    return all(t in source_gaps and t in source_ages and source_gaps[t]<.04
               and 0<=source_ages[t]<=.25 for t in topics)


def main(output,seconds,observer_topic='/next/right/kinematic_compensated/distribution'):
    if not 60<=seconds<=300:raise ValueError('60..300 seconds of static verification')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    service=subprocess.run(['systemctl','is-active','systemd-timesyncd'],capture_output=True,text=True)
    if service.stdout.strip()=='active':raise RuntimeError('Pause automatic clock synchronization before qualifying stability')
    rclpy.init();node=rclpy.create_node('next_static_clock_stability_probe')
    start=time.monotonic();last_offset=time.time_ns()-time.monotonic_ns()
    counts={};last={};gaps={};source_last={};source_gaps={};source_ages={};bad=[];steps=[]
    topics={'/joint_states':JointState,'/joint_position_controller/command_state':JointState,
        '/factr2/right/joint_pos':JointState,'/factr2/right/adapter_status':DiagnosticArray,
        '/factr2/w3_health':DiagnosticArray,observer_topic:String}
    def receive(topic,msg):
        now=time.monotonic();counts[topic]=counts.get(topic,0)+1
        if topic in last and now-start>3:gaps[topic]=max(gaps.get(topic,0),now-last[topic])
        last[topic]=now
        stamp=(int(json.loads(msg.data)['stamp_ns']) if isinstance(msg,String) else
               msg.header.stamp.sec*10**9+msg.header.stamp.nanosec)
        if now-start>3:
            age=(node.get_clock().now().nanoseconds-stamp)*1e-9
            if age<0:bad.append(dict(t=now-start,topic=topic,message='future_source_stamp'))
            source_ages[topic]=max(source_ages.get(topic,0),age)
            if topic in source_last:
                source_gaps[topic]=max(source_gaps.get(topic,0),(stamp-source_last[topic])*1e-9)
                if stamp<source_last[topic]:bad.append(dict(t=now-start,topic=topic,message='source_clock_rollback'))
        source_last[topic]=stamp
        if now-start<=3:return
        if topic=='/factr2/right/adapter_status' and any(s.level!=b'\x00' and s.level!=0 for s in msg.status):bad.append(dict(t=now-start,topic=topic,message=[s.message for s in msg.status]))
        if topic=='/factr2/w3_health':
            if any(s.level!=b'\x00' and s.level!=0 for s in msg.status):bad.append(dict(t=now-start,topic=topic,message=[s.message for s in msg.status]))
    for topic,kind in topics.items():
        # Match the reliable atomic recorder for the coherent model packet;
        # native hardware/health topics retain their advertised sensor QoS.
        qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE) if kind is String else qos_profile_sensor_data
        node.create_subscription(kind,topic,lambda m,t=topic:receive(t,m),qos)
    try:
        while time.monotonic()-start<seconds:
            rclpy.spin_once(node,timeout_sec=.003)
            offset=time.time_ns()-time.monotonic_ns();change=(offset-last_offset)/1e9
            if abs(change)>.002:steps.append(dict(t=time.monotonic()-start,clock_step_seconds=change))
            last_offset=offset
        motion={t:[p.node_name for p in node.get_publishers_info_by_topic(t)] for t in ('/joint_position_command','/left_pose_goal','/right_pose_goal')}
        passed=(not steps and not bad and len(counts)==len(topics) and not any(motion.values())
            and source_timing_ok(source_gaps,source_ages,
                [t for t in topics if not t.endswith('status') and t!='/factr2/w3_health']))
        result=dict(passed=passed,seconds=time.monotonic()-start,time_service_active=False,
            clock_steps=steps,bad_health=bad,counts=counts,max_receive_gap_seconds=gaps,
            max_source_gap_seconds=source_gaps,max_source_age_seconds=source_ages,
            timing_criteria=dict(maximum_source_gap_seconds=.04,maximum_source_age_seconds=.25,
                receive_scheduling_jitter_reported_separately=True),motion_publishers=motion)
        (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        if not passed:raise RuntimeError('Clock or stream stability failed; keep robot stopped')
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=90.)
    p.add_argument('--observer-topic',default='/next/right/kinematic_compensated/distribution')
    a=p.parse_args();main(a.output,a.seconds,a.observer_topic)
