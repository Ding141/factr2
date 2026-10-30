"""Exercise the complete passive receive/write loop using a fake socket only."""
import json
import socket
import struct
import sys

import pytest
import yaml

import w3_targeted_can_record as recorder


@pytest.mark.parametrize('dropped',[0,321])
def test_receiver_drains_all_queued_frames_and_never_hides_kernel_overflow(tmp_path,monkeypatch,dropped):
    audit=tmp_path/'audit';configs=audit/'motor_config';configs.mkdir(parents=True)
    ranges={k:dict(min=-10,max=10) for k in ('position','velocity','torque')}
    for j in range(1,8):(configs/f'can1_id{j}.yaml').write_text(yaml.safe_dump(dict(can_id=j,ranges=ranges)))
    handlers={}
    monkeypatch.setattr(recorder.signal,'signal',lambda sig,fn:handlers.setdefault(sig,fn))
    class FakeSocket:
        def __init__(self):self.n=0;self.closed=False
        def setsockopt(self,*a):pass
        def getsockopt(self,*a):return 425984
        def bind(self,*a):pass
        def settimeout(self,*a):pass
        def close(self):self.closed=True
        def recvmsg(self,*a):
            self.n+=1;j=(self.n-1)%7+1
            payload=bytes([0x10+j,128,0,128,8,0,35,42])
            frame=struct.pack('=IB3x',0x10+j,8)+payload
            ancillary=[(socket.SOL_SOCKET,35,struct.pack('=qq',100,self.n))]
            if dropped:ancillary.append((socket.SOL_SOCKET,40,struct.pack('=I',dropped)))
            if self.n==700:handlers[recorder.signal.SIGTERM](None,None)
            return frame,ancillary,0,None
    fake=FakeSocket();monkeypatch.setattr(recorder.socket,'socket',lambda *a:fake)
    monkeypatch.setattr(sys,'argv',['record','--session',str(tmp_path)])
    if dropped:
        with pytest.raises(RuntimeError,match='Incomplete kernel CAN'):recorder.main()
    else:recorder.main()
    state=json.loads((audit/'can_record_state.json').read_text())
    assert fake.closed and state['socket_dropped_frames']==dropped
    if not dropped:
        assert state['status']=='closed' and state['frames']==700
        assert (audit/'raw_can.bin').stat().st_size==700*(recorder.HEADER.size+8)
        assert state['userspace_queue_peak_frames']>1
    else:assert state['status']=='failed'
