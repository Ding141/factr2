"""Offline trajectory, no-retuning/label separation and failure-path checks.

No ROS node, CAN socket, motor command or real child process is created.
"""
import copy
import csv
import io
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from w3_targeted_protocol import build,bounds,sample,load,validate,thermal_declaration,coefficients,move
from w3_targeted_capture import TargetedClient,promotion
from w3_targeted_guardian import Lease
from w3_targeted_can_record import feedback
from w3_coverage_plan import load_capture_motion
from prepare_w3_targeted_session import labels,prepare,temperatures
from factr2_next.data_collection.quality import sha256,sidecar_path,write_json
from test_w3_coverage_motion import fixture_description
import quick_capture as capture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plans():
    cfg=yaml.safe_load((ROOT/'config/w3/motions/right_targeted_free_v1.yaml').read_text())
    cfg['anchors_deg']=cfg['anchors_deg'][:1];cfg['dwell_seconds']=[3,5]
    return {family:build(cfg,fixture_description(),cfg['home_deg'],[0]*7,'train',family,
                         thermal_declaration('unclassified'),6.) for family in ('continuous','hysteresis','warmup')}


def dump_plan(tmp_path,plan):
    path=tmp_path/'motion.yaml';path.write_text(yaml.dump(plan,Dumper=yaml.CSafeDumper,sort_keys=False));return path


def test_continuous_preserves_native_interior_motion_and_exact_c2(plans):
    plan=plans['continuous'];selected=[s for s in plan['steps'] if s['family']=='continuous']
    assert len(selected)==9
    for step in selected:
        ks=step['knots'];assert len(ks)<len(step['times_seconds'])/5
        assert any(max(abs(np.array(k['velocity_deg_s'])))>.01 for k in ks[1:-1])
        lo,hi,peak=bounds(ks)
        limits=np.array([step['profile'][k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])[:,None]
        assert np.all(peak<=limits)
        # Check both polynomial sides, not just interpolated samples at ticks.
        for a,b,c in zip(ks,ks[1:],ks[2:]):
            left=coefficients(a,b);right=coefficients(b,c);duration=b['time_seconds']-a['time_seconds']
            for d in range(3):
                import math
                endpoint=sum(left[i]*math.factorial(i)/math.factorial(i-d)/duration**d for i in range(d,6))
                right_value=right[d]*math.factorial(d)/(c['time_seconds']-b['time_seconds'])**d
                np.testing.assert_allclose(endpoint,right_value,atol=1e-8)


def test_hysteresis_contains_matched_low_high_holds_and_dwell_is_labeled(plans):
    steps=plans['hysteresis']['steps']
    for pattern in range(8):
        pair=[s for s in steps if s['kind'].startswith('static_') and f'pattern{pattern}_' in s['kind']]
        assert len(pair)==2 and {s['approach_direction'] for s in pair}=={'from_low','from_high'}
        assert pair[0]['dwell_seconds']==pair[1]['dwell_seconds']
        np.testing.assert_allclose(pair[0]['positions_deg'],pair[1]['positions_deg'])
        assert all(s['annotations'][0]['state']=='stationary_hold' for s in pair)
    assert not any(s['family']=='continuous' for s in steps)


def test_hysteresis_speed_and_acceleration_profiles_change_real_spline():
    cfg=yaml.safe_load((ROOT/'config/w3/motions/right_targeted_free_v1.yaml').read_text())
    q=np.zeros(7);goal=q.copy();goal[0]=4
    selected=cfg['profiles'][:5]
    trajectories=[move(q,goal,p,.01) for p in selected]
    durations=[ks[-1]['time_seconds'] for ks in trajectories]
    assert len(set(durations))==len(selected)  # no identical fixed-duration moves
    low_a,high_a=trajectories[1:3]  # same 10 deg/s cap, different acceleration
    assert high_a[-1]['time_seconds']<low_a[-1]['time_seconds']
    assert bounds(high_a)[2][1,0]>bounds(low_a)[2][1,0]
    for ks,p in zip(trajectories,selected):
        limit=np.array([p[k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])[:,None]
        assert np.all(bounds(ks)[2]<=limit)


@pytest.mark.parametrize('change', ['path','velocity','label','probe','limit'])
def test_tampered_plan_fails_before_motion(plans,change):
    p=copy.deepcopy(plans['continuous']);step=next(s for s in p['steps'] if s['family']=='continuous')
    if change=='path':step['positions_deg'][10][0]+=.01
    elif change=='velocity':step['knots'][2]['velocity_deg_s'][0]+=100
    elif change=='label':step['family']='hysteresis'
    elif change=='probe':p['steps']=[s for s in p['steps'] if s['family']!='speed_probe']
    else:p['config']['tracking_limit_deg']=4
    with pytest.raises(ValueError):validate(p)


def test_dispatch_is_explicit_and_legacy_still_loads(tmp_path,plans):
    path=dump_plan(tmp_path,plans['continuous']);p,raw,digest=load_capture_motion(path)
    assert p['schema']=='w3_targeted_free_motion_v1' and digest==hashlib.sha256(raw).hexdigest()
    old,_,_=load_capture_motion(ROOT/'config/w3/motions/left_all_joints_pilot.yaml')
    assert 'schema' not in old


def test_short_batches_keep_all_preceding_probes(plans):
    cfg=copy.deepcopy(plans['continuous']['config'])
    cfg['collection_selection']=dict(profiles=['v10_a20'],single_joints=[0],anchor_indices=[0],coupled_patterns=1)
    p=build(cfg,plans['continuous']['robot_description'],cfg['home_deg'],[0]*7,'val','continuous',thermal_declaration('unclassified'),10)
    assert {s['speed'] for s in p['steps'] if s['family']=='speed_probe'}=={'v06_a15','v10_a20'}
    assert {s['speed'] for s in p['steps'] if s['family']=='continuous'}=={'v10_a20'}
    assert len([s for s in p['steps'] if s['family']=='continuous'])==2


def test_seven_joint_probe_and_stricter_promotion_gate():
    events=[dict(status='completed',active_joints=[j],telemetry={'max_tracking_deg':[1]*7}) for j in range(7)]
    assert promotion(events,2.5,True)['accepted']
    with pytest.raises(RuntimeError):promotion(events[:6],2.5,True)
    events[-1]['telemetry']['max_tracking_deg'][6]=2.6
    with pytest.raises(RuntimeError):promotion(events,2.5,True)


def test_lease_stops_on_lost_motion_or_supervisor_and_latches():
    lease=Lease();assert lease.check(1,False) is None  # no collection is armed
    lease.heartbeat(True,1);assert lease.check(1.24,True) is None
    assert lease.check(1.26,True)=='motion_heartbeat_missing'
    lease.heartbeat(False,1.27);assert lease.tripped=='motion_heartbeat_missing'
    other=Lease();other.heartbeat(True,2);assert other.check(2.01,False)=='recorder_supervisor_missing'


def test_temperature_declaration_never_invents_a_measurement(plans):
    with pytest.raises(ValueError):thermal_declaration('cold_start',5)
    with pytest.raises(ValueError):thermal_declaration('warm_start')
    d=thermal_declaration('cold_start',30)
    assert not d['temperature_measured'] and not d['steady_temperature_verified']
    assert sum(s['duration_seconds'] for s in plans['warmup']['steps'] if s['family']=='warmup')>=300


def test_passive_can_decodes_temperature_and_ignores_transmit_echo():
    profile={'ranges':{'position':{'min':-12.5,'max':12.5},'velocity':{'min':-45,'max':45},'torque':{'min':-54,'max':54}}}
    frame=bytes([0x11,0x80,0,0x80,0x08,0,35,42])
    d=feedback(0x11,frame,0,{1:profile})
    assert d['temperature_valid'] and d['mos_c']==35 and d['rotor_c']==42
    assert d['position_raw_rad']==pytest.approx(12.5/65535)
    assert feedback(0x11,frame,socket.MSG_DONTROUTE,{1:profile}) is None
    assert feedback(0x20000011,frame,0,{1:profile}) is None


def fake_client(step,plan):
    from std_msgs.msg import String
    client=TargetedClient.__new__(TargetedClient);client.plan=plan;client.names=plan['joint_names']
    clock={'ns':1_000_000_000};published=[]
    client.state={'q':np.deg2rad(step['positions_deg'][0]),'cmd':np.deg2rad(step['positions_deg'][0]),
                  'q_stamp':clock['ns'],'v':np.zeros(7),'native_v':np.zeros(7)}
    client.capture_digest='test';client.capture_session=Path('/tmp/offline-fake');client.collection_active=False;client.last_heartbeat=0
    client.guardian={'phase':'ready','last_heartbeat_ros_ns':0}
    def beat(msg):
        d=json.loads(msg.data);client.guardian={'phase':'armed' if d['active'] else 'ready','last_heartbeat_ros_ns':d['stamp_ros_ns']}
    client.heartbeat_pub=SimpleNamespace(publish=beat)
    client.publisher=SimpleNamespace(publish=published.append);client.labels=SimpleNamespace(publish=lambda _:None)
    client.node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=clock['ns'])))
    client.check_mode=lambda:None;client.ready=lambda:True;client.guard=lambda:None;client.goal_interfaces_ready=lambda:True
    def wait(predicate,*a,**kw):assert predicate()
    client.wait_until=wait
    start=clock['ns']+150_000_000
    def spin(*_,**__):
        clock['ns']+=20_000_000
        q,v,a=sample(step['knots'],[np.clip((clock['ns']-start)*1e-9,0,step['duration_seconds'])])
        client.state.update(q=np.deg2rad(q[0]),cmd=np.deg2rad(q[0]),v=np.deg2rad(v[0]),native_v=np.deg2rad(v[0]),q_stamp=clock['ns'])
    client.ros=SimpleNamespace(spin_once=spin)
    return client,published


def test_one_full_trajectory_is_published_and_internal_velocities_survive(plans):
    step=next(s for s in plans['continuous']['steps'] if s['family']=='continuous')
    c,publications=fake_client(step,plans['continuous']);event={}
    c.play_block(step,event,lambda:None)
    assert len(publications)==1
    assert len(publications[0].points)==len(step['knots'])
    assert any(max(abs(np.array(p.velocities)))>.001 for p in publications[0].points[1:-1])
    assert event['command_end_ros_ns']>event['command_start_ros_ns']
    assert not c.collection_active


def test_capture_failure_after_publish_brakes_once_without_retry(plans):
    step=next(s for s in plans['continuous']['steps'] if s['family']=='continuous')
    c,pubs=fake_client(step,plans['continuous']);brakes=[];calls=[]
    def brake():brakes.append(1);c.collection_active=False
    c.brake=brake
    def guard():
        calls.append(1)
        if len(calls)>=3:raise RuntimeError('recorder failed')
    with pytest.raises(RuntimeError):c.play_block(step,{},guard)
    assert len(pubs)==1 and len(brakes)==1


def test_temperature_alignment_is_causal_and_stale_values_stay_invalid(tmp_path):
    audit=tmp_path/'audit';audit.mkdir()
    (audit/'motor_temperature.csv').write_text('kernel_receive_ns,joint,temperature_valid,mos_c,rotor_c\n1000000000,0,1,35,42\n')
    d=temperatures(tmp_path,np.array([900_000_000,1_100_000_000,1_600_000_000],np.int64))
    assert d['temperature_valid'][:,0].tolist()==[False,True,False]
    assert d['rotor_temperature_c'][1,0]==42 and np.isnan(d['rotor_temperature_c'][0,0])


def test_full_raw_to_labeled_h5_excludes_probes_and_keeps_real_timestamps(tmp_path,plans):
    # Explicit offline fixture: fake trajectories never reach ROS/CAN.
    plan=plans['continuous'];session=tmp_path/'raw';audit=session/'audit';audit.mkdir(parents=True)
    path=dump_plan(audit,plan);digest=sha256(path)
    events=[];timestamp=1_000_000_000;parts=[]
    for index,s in enumerate(plan['steps']):
        start=timestamp;end=start+round(s['duration_seconds']*1e9)
        t=start+np.arange(round(s['duration_seconds']*100),dtype=np.int64)*10_000_000
        q,v,a=sample(s['knots'],(t-start)*1e-9);parts.append((t,q,v))
        events.append({**{k:s[k] for k in ('name','family','speed','kind','active_joints','approach_direction')},
                       'status':'completed','command_start_ros_ns':start,'command_end_ros_ns':end})
        timestamp=end
    write_json(audit/'motion_run.json',dict(status='completed',role='train',family='continuous',motion_sha256=digest,events=events,completed_utc=datetime.now(timezone.utc).isoformat()))
    write_json(audit/'collision.json',dict(accepted=True,plan_sha256=digest))
    write_json(audit/'can_record_state.json',dict(status='closed',socket_dropped_frames=0,kernel_timestamps_missing=0))
    (audit/'raw_can.bin').write_bytes(b'offline fixture')
    (audit/'motor_temperature.csv').write_text('kernel_receive_ns,joint,temperature_valid,mos_c,rotor_c\n1000000000,0,1,35,42\n')
    calibration=tmp_path/'calibration.yaml';calibration.write_text('offsets: []\n')
    cfg=yaml.safe_load((ROOT/'config/w3/right/record.yaml').read_text())
    meta=dict(contract_version='w3_next_v1',side='right',joint_order=plan['joint_names'],sample_hz=100,session_id='offline_fixture',topics=cfg['topics'],
        software_commits={'factr2':'fixture','w3':'fixture'},config_sha256='fixture',tool='bare_attachment_v1',gripper='absent',load='empty_v1',
        calibration={'path':str(calibration),'sha256':sha256(calibration)},control={},trajectory_id='fixture',contact={'present':False},temperature={},audit_paths=[],health_gate_enabled=True,time_source='ROS header ns',source='real',episodes={'ep_0000':{}},targeted_collection={'family':'continuous'})
    raw=session/'fixture.h5'
    t=np.concatenate([x[0] for x in parts]);q=np.concatenate([x[1] for x in parts]);v=np.concatenate([x[2] for x in parts])
    with h5py.File(raw,'w') as h:
        h.attrs.update(schema='factr2_next_h5_v1',session_name='offline_fixture',created_at='fixture');ep=h.create_group('ep_0000')
        for key,value in [('joint_pos',q),('joint_vel',v),('joint_cmd',q),('measured_joint_torque',np.zeros_like(q))]:
            g=ep.create_group(key);g.create_dataset('data',data=np.deg2rad(value).astype(np.float32));g.create_dataset('timestamps',data=t)
    meta['h5_sha256']=sha256(raw);write_json(sidecar_path(raw),meta)
    result=prepare(session,tmp_path/'derived');assert result['accepted']
    side=json.loads((tmp_path/'derived/samples.metadata.json').read_text())
    assert all(e['family']=='continuous' for e in side['episodes'].values())
    with h5py.File(tmp_path/'derived/samples.h5','r') as h:
        for ep in h.values():
            assert set(ep['labels/motion_state'][:]) <= {0,1,2}
            assert not ep['labels/intentional_stationary_hold'][:].any()
            assert np.isin(ep['joint_pos/timestamps'][:],t).all()
    (session/'CONTACT_DETECTED').touch()
    with pytest.raises(ValueError):prepare(session,tmp_path/'must_not_train')


def test_C_reuses_existing_state_nodes_and_supervises_guardian_can_and_recorder(tmp_path,plans,monkeypatch):
    real_root=capture.ROOT;root=tmp_path/'project';root.mkdir()
    config=root/'config/w3/right';config.mkdir(parents=True)
    (config/'record.yaml').write_bytes((real_root/'config/w3/right/record.yaml').read_bytes())
    calibration=root/'cal.yaml';calibration.write_text('offsets: []\n')
    path=dump_plan(tmp_path,plans['continuous'])
    path.with_suffix('.collision.json').write_text(json.dumps(dict(accepted=True,
        plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        robot_description_sha256=plans['continuous']['model_sha256'],left_pose_deg=[0]*7)))
    monkeypatch.setattr(capture,'ROOT',root)
    monkeypatch.setattr(capture,'calibration_from_description',lambda _:calibration)
    monkeypatch.setattr(capture,'check_capture_space',lambda *a,**kw:0)
    monkeypatch.setattr(capture,'summarize',lambda *a:0)
    commands=[];stopped=[];sessions=[]
    def fake_run(command,env,timeout=15):
        if 'list_controllers' in command:
            return '\n'.join(n+' plugin active' for n in ('joint_position_controller','gravity_compensation_controller','joint_state_broadcaster'))
        if 'dump' in command:
            params={}
            if command[-1]=='/robot_state_publisher':params={'robot_description':plans['continuous']['robot_description']}
            elif command[-1]=='/w3_next_adapter_right':
                params=dict(side='right',publish_hz=100.,require_hardware_health=True,state_topic='/joint_states',command_state_topic='/joint_position_controller/command_state',output_reliability='reliable')
            return yaml.safe_dump({'node':{'ros__parameters':params}})
        if 'list' in command:return '/w3_health_monitor\n/w3_next_adapter_right\n'
        if 'call' in command:return 'success=True'
        return ''
    class FakeProcess:
        def __init__(self,command,**kw):
            self.pid=99999999;self.code=None;self.label=Path(kw['stdout'].name).stem;commands.append((self.label,command))
        def poll(self):return self.code
        def wait(self,timeout):self.code=0;stopped.append(self.label);return 0
    def select(*_):
        active=root/'data/quick_capture/right/active.json';info=json.loads(active.read_text())
        assert 'heartbeat_mono' in info and 'record_start_mono' in info
        session=Path(info['session']);sessions.append(session)
        cfg=yaml.safe_load((session/'record.yaml').read_text())
        assert cfg['metadata']['targeted_collection']['family']=='continuous'
        return ([sys.stdin],[],[])
    monkeypatch.setattr(capture,'run',fake_run)
    monkeypatch.setattr(capture.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(capture.os,'killpg',lambda pid,sig:None)
    monkeypatch.setattr(capture.select,'select',select)
    class Input(io.StringIO):
        def isatty(self):return True
    monkeypatch.setattr(sys,'stdin',Input('\n'))
    monkeypatch.setattr(sys,'argv',['quick_capture.py','--side','right','--domain','74','--frequency','100','--motion',str(path),'--reuse-state-components','--w3-workspace',str(tmp_path/'fake_robot')])
    assert capture.main()==0
    assert {name for name,_ in commands}=={'bag','recorder','can_record','guardian'}
    assert stopped==['guardian','can_record','recorder','bag']
    assert not (root/'data/quick_capture/right/active.json').exists()
    bag=next(c for name,c in commands if name=='bag')
    assert '/joint_states' in bag and '/w3_robot_bridge_node/state' in bag and '/factr2/right/collection_guardian' in bag
