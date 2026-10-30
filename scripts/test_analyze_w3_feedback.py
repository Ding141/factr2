import socket
import struct

import numpy as np

from analyze_w3_feedback import raw_can,preceding,canonical,quantized_velocity


def test_only_causal_samples_and_fixed_revolution_are_used():
    np.testing.assert_array_equal(preceding(np.array([10,20,30]),np.array([9,10,19,30,31])),[-1,0,0,2,2])
    np.testing.assert_allclose(canonical(np.array([2*np.pi+.1,-2*np.pi-.2]),np.zeros(2)),[.1,-.2])


def test_CAN_decode_signs_units_and_TX_exclusion(tmp_path):
    profile=dict(ranges=dict(position=dict(min=-12.5,max=12.5),velocity=dict(min=-45.,max=45.),torque=dict(min=-54.,max=54.)),
        signs=dict(position=1.,velocity=1.,torque=1.),position_offset=0.)
    profiles=[profile]*7;calibration=[dict(axis_sign=(-1. if j%2==0 else 1.),zero_offset=0.) for j in range(7)]
    header=struct.Struct('<qqqIHH');path=tmp_path/'raw_can.bin';p,v,t=33000,2048,2049
    with path.open('wb') as f:
        for k in range(110):
            for j in range(7):
                data=bytes([0x10+j+1,p>>8,p&255,v>>4,((v&15)<<4)|(t>>8),t&255,30,31])
                ns=1_000_000_000+k*10_000_000+j*100_000
                f.write(header.pack(ns,ns+1,ns+2,0,8,0));f.write(data)
        f.write(header.pack(ns+100,ns+101,ns+102,1,8,socket.MSG_DONTROUTE));f.write(bytes(8))
    joints,summary=raw_can(path,profiles,calibration,np.zeros(7))
    assert summary['local_tx_echo_frames']==1
    for j in range(7):
        assert len(joints[j]['t'])==110
        sign=calibration[j]['axis_sign']
        np.testing.assert_allclose(joints[j]['v'],np.rad2deg(sign*45/4095))
        np.testing.assert_allclose(joints[j]['q'],np.rad2deg(sign*(-12.5+p*25/65535)))
        np.testing.assert_allclose(joints[j]['lsb'][0],np.rad2deg(25/65535))


def test_nominal_command_quantization_does_not_claim_exact_zero():
    profile=dict(ranges=dict(velocity=dict(min=-45.,max=45.)),signs=dict(velocity=1.))
    expected=np.rad2deg(45/4095)
    np.testing.assert_allclose(quantized_velocity([0],profile,1),[expected])
    np.testing.assert_allclose(quantized_velocity([0],profile,-1),[-expected])


def test_bag_layers_preserve_headers_units_and_joint_mapping(tmp_path):
    import pytest,sqlite3
    pytest.importorskip('w3_robot_bridge.msg')
    from w3_robot_bridge.msg import MotorCommandArray,MotorCommand,MotorStateArray,MotorState
    from sensor_msgs.msg import JointState
    from control_msgs.msg import DynamicJointState,InterfaceValue
    from rclpy.serialization import serialize_message
    from analyze_w3_feedback import read_layers
    con=sqlite3.connect(tmp_path/'test.db3')
    con.execute('create table topics (id integer,name text)')
    con.execute('create table messages (timestamp integer,topic_id integer,data blob)')
    con.executemany('insert into topics values (?,?)',[(1,'/joint_states'),(2,'/w3_robot_bridge_node/commands'),(3,'/w3_robot_bridge_node/state'),(4,'/dm_hardware_interface/feedback_state'),(5,'/dynamic_joint_states')])
    initial=np.deg2rad([0,50,0,-55,0,0,0.]);calibration=[dict(axis_sign=(-1. if j%2==0 else 1.),zero_offset=.1*j) for j in range(7)]
    for k in range(110):
        m=JointState();m.header.stamp.sec=2;m.header.stamp.nanosec=k*1_000_000
        m.name=[f'right_joint_{j}' for j in range(7)];m.position=initial.tolist();m.velocity=[.01]*7
        con.execute('insert into messages values (?,?,?)',(2_000_000_000+k*1_000_000+100,1,serialize_message(m)))
        con.execute('insert into messages values (?,?,?)',(2_000_000_000+k*1_000_000+300,4,serialize_message(m)))
        dynamic=DynamicJointState();dynamic.header=m.header;dynamic.joint_names=m.name
        dynamic.interface_values=[InterfaceValue(interface_names=['feedback_source_age','feedback_receive_age','feedback_valid'],values=[.002,.001,0. if k==0 else 1.]) for _ in range(7)]
        con.execute('insert into messages values (?,?,?)',(2_000_000_000+k*1_000_000+400,5,serialize_message(dynamic)))
        b=MotorCommandArray();b.header=m.header;s=MotorStateArray();s.header=m.header
        for j,c in enumerate(calibration):
            v=MotorCommand();v.channel=1;v.motor_index=j;v.mode=3
            v.position=float(c['axis_sign']*initial[j]+c['zero_offset']);v.velocity=c['axis_sign']*.01;v.kp=10.;v.kd=1.
            b.commands.append(v)
            state=MotorState();state.channel=1;state.motor_index=j;state.position=v.position;state.velocity=v.velocity;s.motors.append(state)
        con.execute('insert into messages values (?,?,?)',(2_000_000_000+k*1_000_000+200,2,serialize_message(b)))
        con.execute('insert into messages values (?,?,?)',(2_000_000_000+k*1_000_000+150,3,serialize_message(s)))
    con.commit();con.close();metadata={};layers=read_layers(tmp_path,calibration,initial,metadata)
    for name in ('state','bridge','bridge_state','source_state'):
        assert layers[name]['t'][0]==2_000_000_000
        np.testing.assert_allclose(layers[name]['q'][0],np.rad2deg(initial),atol=1e-5)
        np.testing.assert_allclose(layers[name]['v'][0],np.rad2deg(.01),atol=1e-6)
    assert layers['source_state']['received'][0]-layers['source_state']['t'][0]==300
    assert metadata['J1']['recorded_rows']==110 and metadata['J1']['invalid_rows']==1
    assert metadata['J7']['source_age_ms']['p95']==2.
    # Legacy bags remain usable without inventing source timestamps.
    con=sqlite3.connect(tmp_path/'test.db3');con.execute('delete from messages where topic_id=4');con.commit();con.close()
    assert 'source_state' not in read_layers(tmp_path,calibration,initial)
