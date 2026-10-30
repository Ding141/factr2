#!/usr/bin/env python3
"""Passive SocketCAN recorder: never sends CAN frames or changes the interface.

Variable binary records: <qqqIHH followed by DLC bytes: kernel software receive
ns (0 if unavailable), user wall ns, monotonic ns, raw CAN id, DLC, recvmsg flags.
Motor acquisition time is NOT available. Local transmit echoes are not decoded
as motor feedback. Temperature bytes are nominal Celsius, not calibrated probes.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import signal
import socket
import struct
import time
import queue
import threading

import yaml
from w3_motion_sequence import write_json

HEADER=struct.Struct('<qqqIHH')


def feedback(can_id,data,flags,profiles):
    if len(data)<8 or can_id & (0x20000000|0x40000000) or flags & socket.MSG_DONTROUTE:return None
    identifier=can_id & (0x1fffffff if can_id & 0x80000000 else 0x7ff)
    motor=data[0]&15
    if motor not in profiles:return None
    # Match the driver's supported dedicated/shared master-ID feedback routing.
    if identifier>=0x7ff:return None
    p=profiles[motor];ranges=p['ranges']
    codes=[(data[1]<<8)|data[2],(data[3]<<4)|(data[4]>>4),((data[4]&15)<<8)|data[5]]
    values=[]
    for code,key,bits in zip(codes,('position','velocity','torque'),(16,12,12)):
        r=ranges[key];values.append(r['min']+code*(r['max']-r['min'])/(2**bits-1))
    mos,rotor=data[6],data[7]
    return dict(motor=motor,temperature_valid=0<mos<150 and 0<rotor<150,
                mos_c=mos,rotor_c=rotor,error=data[0]>>4,position_raw_rad=values[0],velocity_raw_rad_s=values[1],torque_raw_nm=values[2])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interface',default='can1')
    p.add_argument('--session',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]{1,15}',a.interface):p.error('Invalid interface name')
    folder=a.session/'audit';profiles={}
    for j in range(1,8):
        profile=yaml.safe_load((folder/f'motor_config/can1_id{j}.yaml').read_text())
        if profile['can_id']!=j:raise ValueError('This collector expects right-arm motor IDs 1..7')
        profiles[j]=profile
    if (folder/'raw_can.bin').exists():raise ValueError('Raw CAN log already exists')
    stop=False
    def terminate(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,terminate);signal.signal(signal.SIGTERM,terminate)
    state=dict(status='starting',interface=a.interface,frames=0,kernel_timestamps_missing=0,socket_dropped_frames=0,
               last_received_mono={},last_temperature_c={},temperature_samples_by_motor={str(j):0 for j in range(1,8)},
               timestamp_kind='Linux kernel software receive; not device acquisition',raw_format='<qqqIHH + DLC bytes')
    write_json(folder/'can_record_state.json',state)
    sock=None;reader=None;received=queue.Queue(maxsize=16384);reader_errors=[]
    try:
        sock=socket.socket(socket.PF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,2**20)
        timestamp_type=getattr(socket,'SO_TIMESTAMPNS',35)
        sock.setsockopt(socket.SOL_SOCKET,timestamp_type,1)
        sock.setsockopt(socket.SOL_SOCKET,40,1)  # SO_RXQ_OVFL: count kernel receive drops.
        sock.setsockopt(socket.SOL_CAN_RAW,5,1)  # CAN_RAW_FD_FRAMES; receive only.
        sock.bind((a.interface,));sock.settimeout(.05)
        state['socket_receive_buffer_bytes']=sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
        state['userspace_queue_capacity_frames']=received.maxsize
        state['userspace_queue_peak_frames']=0
        state['maximum_writer_lag_seconds']=0.
        def receive():
            # Disk flushes and JSON/temperature formatting must not block the
            # small kernel CAN receive queue. Keep every original ancillary
            # timestamp and overflow counter; no filtering or loss concealment.
            try:
                while not stop:
                    try:frame,ancillary,flags,_=sock.recvmsg(72,socket.CMSG_SPACE(16)+socket.CMSG_SPACE(4))
                    except socket.timeout:continue
                    received.put_nowait((frame,ancillary,flags,time.time_ns(),time.monotonic_ns()))
            except BaseException as exc:reader_errors.append(exc)
        reader=threading.Thread(target=receive,name='passive-can-receive',daemon=True);reader.start()
        last_save=0.;last_temperature={}
        with (folder/'raw_can.bin').open('xb',buffering=2**20) as raw,(folder/'motor_temperature.csv').open('x',newline='') as temp:
            writer=csv.writer(temp);writer.writerow(['kernel_receive_ns','received_wall_ns','received_mono_ns','joint','motor_id','mos_c','rotor_c','temperature_valid','error','position_raw_rad','velocity_raw_rad_s','torque_raw_nm'])
            while not stop or reader.is_alive() or not received.empty():
                if reader_errors:raise RuntimeError('Passive CAN receive thread failed: '+str(reader_errors[0]))
                try:
                    frame,ancillary,flags,wall,mono=received.get(timeout=.05)
                except queue.Empty:frame=None
                now=time.monotonic()
                if frame:
                    state['userspace_queue_peak_frames']=max(state['userspace_queue_peak_frames'],received.qsize()+1)
                    state['maximum_writer_lag_seconds']=max(state['maximum_writer_lag_seconds'],now-mono*1e-9)
                    can_id,length=struct.unpack_from('=IB',frame);payload=frame[8:8+length]
                    if length>64 or len(payload)!=length:raise ValueError('Malformed CAN record')
                    kernel=0
                    for level,kind,value in ancillary:
                        if level==socket.SOL_SOCKET and kind==timestamp_type:
                            sec,ns=struct.unpack('=qq',value[:16]);kernel=sec*10**9+ns
                        elif level==socket.SOL_SOCKET and kind==40:state['socket_dropped_frames']=struct.unpack('=I',value[:4])[0]
                    raw.write(HEADER.pack(kernel,wall,mono,can_id,length,flags&0xffff));raw.write(payload)
                    state['frames']+=1;state['kernel_timestamps_missing']+=int(kernel==0)
                    value=feedback(can_id,payload,flags,profiles)
                    if value:
                        j=str(value['motor']);received_at=mono*1e-9;state['last_received_mono'][j]=received_at
                        if received_at-last_temperature.get(j,-1e9)>=.1:
                            writer.writerow([kernel,wall,mono,int(j)-1,j,value['mos_c'],value['rotor_c'],int(value['temperature_valid']),value['error'],value['position_raw_rad'],value['velocity_raw_rad_s'],value['torque_raw_nm']])
                            last_temperature[j]=received_at
                            if value['temperature_valid']:
                                state['temperature_samples_by_motor'][j]+=1
                                state['last_temperature_c'][j]=dict(mos=value['mos_c'],rotor=value['rotor_c'])
                if now-last_save>=.1:
                    fresh=all(0<=now-state['last_received_mono'].get(str(j),-1e9)<=.25 for j in range(1,8))
                    if state['socket_dropped_frames'] or state['kernel_timestamps_missing']:raise RuntimeError('Incomplete kernel CAN timing; stop collection')
                    state.update(status='ready' if fresh else 'waiting_feedback',updated_mono=now)
                    write_json(folder/'can_record_state.json',state);raw.flush();temp.flush();last_save=now
            state.update(status='closed',updated_mono=time.monotonic());write_json(folder/'can_record_state.json',state)
    except BaseException as exc:
        state.update(status='failed',error=str(exc),updated_mono=time.monotonic());write_json(folder/'can_record_state.json',state);raise
    finally:
        stop=True
        if reader is not None:reader.join(timeout=.2)
        if sock is not None:sock.close()


if __name__=='__main__':main()
