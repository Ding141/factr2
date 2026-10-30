#!/usr/bin/env python3
"""Execute one frozen targeted session. Only run --send publishes motion."""
import argparse
import fcntl
import json
from pathlib import Path
import threading
import time
import signal

import numpy as np

from w3_motion_sequence import MotionClient, read_active, utc_now
from w3_coverage_motion import CoverageClient
from w3_pose_motion import PoseCoverageClient
from w3_pose_protocol import pose_target
from w3_motion_audit import MotionAuditWriter
from w3_targeted_protocol import load, sample, thermal_declaration


def live_state():
    c=MotionClient('right')
    try:
        c.preflight()
        # Get both arms from the same original JointState, without moving them.
        from sensor_msgs.msg import JointState
        from rclpy.qos import qos_profile_sensor_data
        left={}
        def callback(msg):
            values=dict(zip(msg.name,msg.position))
            if all(f'left_joint_{j}' in values for j in range(7)):
                left['q']=[np.rad2deg(values[f'left_joint_{j}']) for j in range(7)]
                left['received']=time.monotonic()
        c.node.create_subscription(JointState,'/joint_states',callback,qos_profile_sensor_data)
        c.wait_until(lambda:'q' in left and time.monotonic()-left['received']<.25,'left collision pose')
        return c.description,np.rad2deg(c.state['cmd']).tolist(),left['q']
    finally:c.close()


def promotion(events, threshold, required_joints=False):
    if not events or any(e.get('status')!='completed' for e in events):raise RuntimeError('Incomplete profile verification')
    if required_joints and {j for e in events for j in e['active_joints']}!=set(range(7)):
        raise RuntimeError('All seven joints must complete probe verification')
    worst=np.max([e['telemetry']['max_tracking_deg'] for e in events],axis=0)
    if max(worst)>threshold:
        raise RuntimeError(f'Profile promotion refused: tracking {max(worst):.3f}° > {threshold:.3f}°; do not increase gains or retry automatically')
    return dict(accepted=True,max_tracking_deg=worst.tolist(),threshold_deg=threshold)


class CaptureWatchdog:
    """Disk/process checks stay outside the motion loop."""
    def __init__(self, side, digest, identity, session):
        self.side,self.digest,self.identity,self.session=side,digest,identity,session
        self.error=None;self.last_ok=time.monotonic();self.stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        while not self.stop.is_set():
            try:
                info=read_active(self.side,self.digest)
                if any(info.get(k)!=self.identity.get(k) for k in ('pid','session','side','domain','motion_sha256')):
                    raise RuntimeError('Capture identity changed')
                if not 0<=time.monotonic()-info.get('heartbeat_mono',-1e9)<=.75:
                    raise RuntimeError('C recorder supervisor heartbeat expired')
                for name in ('STOP_MOTION','CONTACT_DETECTED'):
                    if (self.session/name).exists():raise RuntimeError('Operator marker: '+name)
                can=json.loads((self.session/'audit/can_record_state.json').read_text())
                if can['status']!='ready' or now_age(time.monotonic(),can['updated_mono'])>.3:
                    raise RuntimeError('Passive CAN timing/temperature recorder is not ready')
                self.last_ok=time.monotonic()
            except BaseException as exc:
                self.error=exc;return
            self.stop.wait(.05)
    def check(self):
        if self.error is not None:raise RuntimeError(str(self.error))
        if time.monotonic()-self.last_ok>.3:raise RuntimeError('Capture watchdog stalled')
    def close(self):self.stop.set();self.thread.join(timeout=1)


def now_age(now,then):
    age=now-then
    if age<0:raise RuntimeError('Monotonic clock regression')
    return age


class TargetedClient(PoseCoverageClient):
    def __init__(self, plan, stationary_left_deg):
        super().__init__(plan,stationary_left_deg)
        from std_msgs.msg import String
        self.labels=self.node.create_publisher(String,'/factr2/right/collection_event',10)
        self.heartbeat_pub=self.node.create_publisher(String,'/factr2/right/collection_heartbeat',10)
        self.node.create_subscription(String,'/factr2/right/collection_guardian',self.guardian_status,10)
        self.guardian=None;self.guardian_received=0.;self.collection_active=False;self.last_heartbeat=0.
        self.telemetry=[];self.previous_q_stamp=None

    def guardian_status(self,msg):
        try:
            d=json.loads(msg.data)
            if d['motion_sha256']==self.capture_digest and d['session']==str(self.capture_session):
                self.guardian=d;self.guardian_received=time.monotonic()
        except (ValueError,KeyError,AttributeError):pass

    def heartbeat(self,force=False):
        from std_msgs.msg import String
        now=time.monotonic()
        if force or now-self.last_heartbeat>=.05:
            msg=String();msg.data=json.dumps(dict(motion_sha256=self.capture_digest,session=str(self.capture_session),
                stamp_ros_ns=self.node.get_clock().now().nanoseconds,active=self.collection_active))
            self.last_heartbeat_stamp_ros_ns=json.loads(msg.data)['stamp_ros_ns']
            self.heartbeat_pub.publish(msg);self.last_heartbeat=now

    def joints(self,msg,key):
        super().joints(msg,key)
        if key=='q':
            if len(msg.velocity)!=len(msg.name) or len(msg.effort)!=len(msg.name):
                self.state.pop('v',None);return
            by_name={n:i for i,n in enumerate(msg.name)}
            if not all(n in by_name for n in self.names):return
            self.state['v']=np.array([msg.velocity[by_name[n]] for n in self.names])
            self.state['tau']=np.array([msg.effort[by_name[n]] for n in self.names])
        elif len(msg.velocity)==len(msg.name):
            by_name={n:i for i,n in enumerate(msg.name)}
            self.state['native_v']=np.array([msg.velocity[by_name[n]] for n in self.names])

    def guard_stationary_left(self):
        super().guard_stationary_left()
        for key in ('left_q','left_cmd'):
            if max(abs(np.rad2deg(self.state[key])-self.stationary_left_deg))>self.plan['config']['left_stationary_tolerance_deg']:
                raise RuntimeError('Left arm differs from frozen collision pose')

    def guard(self):
        super().guard()
        if self.guardian is None or time.monotonic()-self.guardian_received>.25 or self.guardian['phase'] not in ('ready','armed'):
            raise RuntimeError('Independent motion guardian unavailable or tripped')
        self.heartbeat()
        if self.state.get('adapter_hz',0)<90:raise RuntimeError('Require real adapter at >=90Hz')
        if any(key not in self.state or not np.isfinite(self.state[key]).all() for key in ('v','native_v')):raise RuntimeError('Velocity feedback unavailable')

    def preflight(self):
        from rcl_interfaces.srv import GetParameters
        # Parent PoseCoverageClient expects the old slow/fast schema. Use the
        # shared guard preflight and check this protocol's native limits here.
        self.wait_until(lambda:self.ready() and self.state.get('adapter_hz',0)>=90 and self.guardian is not None and self.guardian['phase']=='ready','100Hz adapter and independent guardian')
        initial=CoverageClient.preflight(self)
        if self.description!=self.plan['robot_description']:raise RuntimeError('Live robot model changed')
        request=GetParameters.Request();request.names=['joints','max_velocity','max_acceleration','max_jerk','kp_gains','kd_gains']
        reply=self.call(GetParameters,'/joint_position_controller/get_parameters',request)
        if len(reply.values)!=6 or reply.values[0].type!=9:raise RuntimeError('Missing native controller profile')
        names=list(reply.values[0].string_array_value)
        required=np.max([[s['profile'][k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')]
                         for s in self.plan['steps']],axis=0)
        for parameter,need in zip(reply.values[1:4],np.deg2rad(required)):
            values=np.array(parameter.double_array_value)
            if parameter.type!=8 or len(values)!=len(names) or not np.isfinite(values).all() or np.any(values[[names.index(n) for n in self.names]]+1e-8<need):
                raise RuntimeError('Frozen plan exceeds current native motion limits')
        return initial,reply

    def _native_message(self,step,stamp):
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        msg=JointTrajectory();msg.joint_names=self.names
        msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(stamp,10**9)
        for k in step['knots']:
            point=JointTrajectoryPoint()
            point.positions=np.deg2rad(k['position_deg']).tolist()
            point.velocities=np.deg2rad(k['velocity_deg_s']).tolist()
            point.accelerations=np.deg2rad(k['acceleration_deg_s2']).tolist()
            point.time_from_start.sec,point.time_from_start.nanosec=divmod(round(k['time_seconds']*1e9),10**9)
            msg.points.append(point)
        return msg

    def brake(self):
        from std_msgs.msg import Empty
        self.pending_goal=None;self.expected_deg=None
        self.stop_pub.publish(Empty())
        deadline=time.monotonic()+.2
        while time.monotonic()<deadline:self.ros.spin_once(self.node,timeout_sec=.005)
        self.collection_active=False;self.heartbeat(True)

    def collect_telemetry(self):
        stamp=self.state['q_stamp']
        if stamp==self.previous_q_stamp:return
        self.previous_q_stamp=stamp
        self.telemetry.append([stamp,*np.rad2deg(self.state['q']),*np.rad2deg(self.state['cmd']),
                               *np.rad2deg(self.state['v']),*np.rad2deg(self.state['native_v'])])

    def play_block(self,step,event,capture_guard):
        try:return self._play_block(step,event,capture_guard)
        except BaseException:
            if self.collection_active:self.brake()
            raise

    def _play_block(self,step,event,capture_guard):
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        self.expected_deg=None;self.telemetry=[];self.previous_q_stamp=None
        self.check_mode();self.wait_until(self.ready,'fresh input before block');self.guard();capture_guard()
        if max(abs(np.rad2deg(self.state['cmd'])-step['positions_deg'][0]))>.05:raise RuntimeError('Native start target changed')
        self.wait_until(self.goal_interfaces_ready,'native controller and bounded stop subscribers')
        stationary=bool(step['dwell_seconds'])
        stamp=self.node.get_clock().now().nanoseconds+(0 if stationary else 150_000_000)
        event.update(command_start_ros_ns=stamp,command_end_ros_ns=stamp+round(step['duration_seconds']*1e9))
        label=String();label.data=json.dumps(event,allow_nan=False);self.labels.publish(label)
        native=self._native_message(step,stamp)
        self.collection_active=True;self.heartbeat(True)
        requested=self.last_heartbeat_stamp_ros_ns
        self.wait_until(lambda:self.guardian is not None and self.guardian['phase']=='armed' and
                        self.guardian.get('last_heartbeat_ros_ns',0)>=requested,'guardian armed acknowledgement',seconds=.1)
        capture_guard()
        if not stationary:
            # Explicit endpoint/IK target for rest-to-rest hysteresis and moves;
            # a continuous/probe waveform is one complete native trajectory.
            if len(step['knots'])==2:
                model=__import__('w3_coverage_plan').ArmModel(self.plan['robot_description'],'right',self.plan['config']['tool_frame'])
                target=pose_target(model,step['positions_deg'][-1])
                wrapped=dict(step,command=dict(type='pose_goal',target=target))
                self.bridge_error=None;self.accepted_stamp=None
                self.pending_goal=(wrapped,native,stamp,capture_guard)
                pose=PoseStamped();pose.header=native.header;pose.header.frame_id=target['frame_id']
                pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=target['position_m']
                pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w=target['quaternion_xyzw']
                self.pose_pub.publish(pose)
                self.wait_until(lambda:self.accepted_stamp==stamp or self.bridge_error is not None,'single frozen pose acceptance',seconds=.14)
                if self.bridge_error is not None:raise self.bridge_error
            else:self.publisher.publish(native)
        last=time.monotonic()
        while True:
            self.ros.spin_once(self.node,timeout_sec=.005)
            now=time.monotonic()
            if now-last>.2:raise RuntimeError('Motion monitor stalled >200ms')
            last=now
            elapsed=(self.node.get_clock().now().nanoseconds-stamp)*1e-9
            if stationary:self.expected_deg=np.array(step['positions_deg'][0])
            else:self.expected_deg=sample(step['knots'],[np.clip(elapsed,0,step['duration_seconds'])])[0][0]
            self.guard();capture_guard();self.collect_telemetry()
            if elapsed>=step['duration_seconds']+.05:break
        arrival=self.endpoint(np.deg2rad(step['positions_deg'][-1]),self.plan['config']['arrival_tolerance_deg'])
        rows=np.asarray(self.telemetry,float)
        if len(rows)<2:raise RuntimeError('No independent tracking telemetry')
        tracking=abs(rows[:,1:8]-rows[:,8:15])
        event['telemetry']=dict(samples=len(rows),max_tracking_deg=tracking.max(axis=0).tolist(),
            tracking_p95_deg=np.percentile(tracking,95,axis=0).tolist(),arrival_error_deg=arrival,
            measured_velocity_p95_deg_s=np.percentile(abs(rows[:,15:22]),95,axis=0).tolist(),
            native_velocity_peak_deg_s=abs(rows[:,22:29]).max(axis=0).tolist(),
            actual_end_position_deg=rows[-1,1:8].tolist(),
            feedback_velocity_is_quantized=True)
        self.collection_active=False;self.heartbeat(True)
        return event


def send(path):
    plan,_,digest=load(path);identity=read_active('right',digest);session=Path(identity['session'])
    collision=json.loads((session/'audit/collision.json').read_text())
    if (collision.get('accepted') is not True or collision['plan_sha256']!=digest or
            collision['robot_description_sha256']!=plan['model_sha256'] or
            not np.allclose(collision['left_pose_deg'],plan['left_pose_deg'],atol=1e-8,rtol=0)):
        raise RuntimeError('Frozen collision report mismatch')
    run_path=session/'audit/motion_run.json'
    with (session/'audit/motion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if run_path.exists():raise RuntimeError('This session already attempted motion; never auto-retry')
        report=dict(status='preflight',motion_sha256=digest,motion_name=plan['name'],side='right',
                    family=plan['family'],role=plan['role'],thermal=plan['thermal'],started_utc=utc_now(),events=[],profile_gates=[])
        writer=MotionAuditWriter(run_path);watch=CaptureWatchdog('right',digest,identity,session);client=None
        try:
            writer.submit(report);writer.flush()
            thermal=plan['thermal']
            thermal_declaration(thermal['initial_condition'],thermal['declared_idle_minutes'],thermal['warmup_session'])
            client=TargetedClient(plan,plan['left_pose_deg'])
            client.capture_session=session;client.capture_digest=digest
            initial,reply=client.preflight()
            if max(abs(np.rad2deg(client.state['cmd'])-plan['initial_deg']))>.05 or max(abs(np.rad2deg(initial)-plan['initial_deg']))>1.5:
                raise RuntimeError('Start pose differs from collision-audited plan')
            from quick_capture import parameters
            frozen=parameters((session/'audit/position.yaml').read_text())
            if list(reply.values[0].string_array_value)!=frozen['joints']:raise RuntimeError('Native joint order changed')
            for key,value in zip(('kp_gains','kd_gains'),reply.values[4:]):
                if list(value.double_array_value)!=frozen[key]:raise RuntimeError('Native gains changed after C snapshot')
            def guard():watch.check();writer.check()
            current_profile=None;probe_passed=set()
            report['status']='running'
            for index,step in enumerate(plan['steps']):
                profile=step['profile']['id']
                if current_profile is not None and profile!=current_profile:
                    previous=[e for e in report['events'] if e['speed']==current_profile]
                    gate=promotion(previous,plan['config']['promotion_tracking_deg'])
                    report['profile_gates'].append(dict(gate,from_profile=current_profile,to_profile=profile,kind='promotion'))
                current_profile=profile
                if step['family']==plan['family'] and profile not in probe_passed:
                    probes=[e for e in report['events'] if e['speed']==profile and e['family']=='speed_probe']
                    gate=promotion(probes,plan['config']['promotion_tracking_deg'],True)
                    report['profile_gates'].append(dict(gate,to_profile=profile,kind='seven_joint_probe'))
                    probe_passed.add(profile)
                event={k:step[k] for k in ('name','family','phase','kind','speed','active_joints','pose_index','approach_direction','dwell_seconds','annotations','peak_bounds')}
                event.update(index=index+1,status='moving',utc=utc_now(),thermal_initial_condition=thermal['initial_condition'],
                             elapsed_since_recording_ready_s=time.monotonic()-identity['record_start_mono'])
                report['events'].append(event);writer.submit(report)
                client.play_block(step,event,guard)
                event.update(status='completed',completed_utc=utc_now());writer.submit(report)
            report.update(status='completed',completed_utc=utc_now())
        except BaseException as exc:
            if client is not None and client.collection_active:
                try:client.brake()
                except Exception as stop_error:report['braking_error']=str(stop_error)
            report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),completed_utc=utc_now())
            raise
        finally:
            try:writer.submit(report)
            finally:
                if client is not None:client.close()
                watch.close();writer.close()
    print('Motion finished. Return to C and press Enter to close raw data, then prepare labeled episodes.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--file',type=Path,required=True);p.add_argument('--send',action='store_true');args=p.parse_args()
    plan,_,digest=load(args.file)
    if not args.send:
        print(json.dumps(dict(name=plan['name'],family=plan['family'],role=plan['role'],blocks=len(plan['steps']),sha256=digest,motion_sent=False),indent=2));return
    def terminate(*_):raise KeyboardInterrupt('Termination requested; bounded stop')
    signal.signal(signal.SIGTERM,terminate)
    send(args.file)


if __name__=='__main__':main()
