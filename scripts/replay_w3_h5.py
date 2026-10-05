#!/usr/bin/env python3
"""Accepted H5 -> four read-only streams, current ROS stamps, no motor topics."""
import argparse
from pathlib import Path
import time
import h5py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
from factr2_next.data_collection.quality import require
from factr2_next.w3_config import SIGNALS


def main():
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--episode',default='ep_0000')
    p.add_argument('--hz',type=float,default=50);p.add_argument('--loop',action='store_true')
    p.add_argument('--start-delay',type=float,default=3)
    a=p.parse_args();report=require(a.path,[a.episode]);meta=report['metadata'];side=meta['side']
    if a.hz!=50:raise ValueError('W3 replay requires 50 Hz')
    with h5py.File(a.path,'r') as h:arrays={k:h[a.episode][k]['data'][:] for k in SIGNALS}
    rclpy.init();node=Node('w3_h5_replay_'+side)
    pubs={k:node.create_publisher(JointState,f'/factr2/{side}/{suffix}',qos_profile_sensor_data) for k,(suffix,_) in SIGNALS.items()}
    status=node.create_publisher(DiagnosticArray,f'/factr2/{side}/adapter_status',qos_profile_sensor_data)
    index=0;started=time.monotonic();gap_until=0
    def tick():
        nonlocal index,gap_until
        now=node.get_clock().now().to_msg()
        diag=DiagnosticArray();diag.header.stamp=now
        s=DiagnosticStatus();s.name='factr2/adapter/'+side;s.level=DiagnosticStatus.OK;s.message='synthetic_h5_replay';diag.status=[s]
        status.publish(diag)
        if time.monotonic()-started<a.start_delay or time.monotonic()<gap_until:return
        if index>=len(arrays['joint_pos']):
            if not a.loop:return
            index=0;gap_until=time.monotonic()+.3;return  # never join last/first history
        for key,(_,field) in SIGNALS.items():
            m=JointState();m.header.stamp=now;m.name=meta['joint_order'];setattr(m,field,arrays[key][index].astype(float).tolist());pubs[key].publish(m)
        index+=1
    timer=node.create_timer(.02,tick)
    try:
        while rclpy.ok() and (a.loop or index<len(arrays['joint_pos'])):rclpy.spin_once(node,timeout_sec=.02)
        until=time.monotonic()+.2
        while rclpy.ok() and time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
