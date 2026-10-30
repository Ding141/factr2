#!/usr/bin/env python3
"""Independent collection lease monitor; only publishes the bounded-stop request."""
import argparse
import json
import os
from pathlib import Path
import time


class Lease:
    def __init__(self):
        self.armed=False;self.last_heartbeat=None;self.tripped=None
    def heartbeat(self,active,now):
        if self.tripped is not None:return
        self.armed=bool(active);self.last_heartbeat=now
    def check(self,now,supervisor_ok):
        if self.armed and self.tripped is None:
            if not supervisor_ok:self.tripped='recorder_supervisor_missing'
            elif self.last_heartbeat is None or now-self.last_heartbeat>.25:self.tripped='motion_heartbeat_missing'
        return self.tripped


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--digest',required=True);p.add_argument('--supervisor-pid',type=int,required=True);a=p.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.clock import Clock,ClockType
    from std_msgs.msg import Empty,String
    rclpy.init(args=[]);node=Node('targeted_capture_guardian_right');lease=Lease();last_stamp=0;last_status=0.
    stop=node.create_publisher(Empty,'/joint_position_controller/stop',10)
    status=node.create_publisher(String,'/factr2/right/collection_guardian',10)
    def heartbeat(msg):
        nonlocal last_stamp
        try:
            d=json.loads(msg.data);stamp=int(d['stamp_ros_ns'])
            now=node.get_clock().now().nanoseconds
            if (d['motion_sha256']!=a.digest or d['session']!=str(a.session.resolve()) or
                    type(d['active']) is not bool or stamp<=last_stamp or not 0<=now-stamp<=250_000_000):return
            last_stamp=stamp;lease.heartbeat(d['active'],time.monotonic())
        except (ValueError,KeyError,TypeError):return
    node.create_subscription(String,'/factr2/right/collection_heartbeat',heartbeat,10)
    def tick():
        nonlocal last_status
        now=time.monotonic();ok=False
        try:
            os.kill(a.supervisor_pid,0)
            info=json.loads((a.session.parent/'active.json').read_text())
            ok=(info['pid']==a.supervisor_pid and info['motion_sha256']==a.digest and
                info['session']==str(a.session.resolve()) and 0<=now-info['heartbeat_mono']<=.75)
            ok=ok and not any((a.session/name).exists() for name in ('STOP_MOTION','CONTACT_DETECTED'))
        except (OSError,ValueError,KeyError):pass
        was=lease.tripped
        reason=lease.check(now,ok)
        if reason is not None and was is None:
            stop.publish(Empty());node.get_logger().error('Bounded stop: '+reason)
        if now-last_status>=.05:
            s=String();s.data=json.dumps(dict(motion_sha256=a.digest,session=str(a.session.resolve()),
                stamp_ros_ns=node.get_clock().now().nanoseconds,
                last_heartbeat_ros_ns=last_stamp,
                phase='tripped' if reason else ('armed' if lease.armed else ('ready' if ok else 'waiting')),
                reason=reason));status.publish(s);last_status=now
    timer=node.create_timer(.025,tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if lease.armed:
            stop.publish(Empty())
            deadline=time.monotonic()+.1
            while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.005)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
