"""Dropped transport messages must not misalign every subsequent W3 frame."""
from message_filters import SimpleFilter, ApproximateTimeSynchronizer
from sensor_msgs.msg import JointState
from factr2_next.data_collection.recorder_node import make_synchronizer
from factr2_next.w3_samples import frame
from factr2_next.w3_config import SIGNALS
from factr2_next.data_collection.recorder_node import subscription_qos
from rclpy.qos import ReliabilityPolicy
import pytest


def playback(factory):
    inputs=[SimpleFilter() for _ in range(4)]
    synchronizer=factory(inputs)
    accepted=[];rejected=[]
    names=[f'right_joint_{i}' for i in range(7)]
    keys=list(SIGNALS);specs={k:dict(field=field) for k,(_,field) in SIGNALS.items()}
    def receive(*messages):
        now=max(m.header.stamp.sec*10**9+m.header.stamp.nanosec for m in messages)+1_000_000
        try:accepted.append(frame(messages,keys,specs,names,now)[0])
        except ValueError as exc:rejected.append(str(exc))
    synchronizer.registerCallback(receive)
    for tick in range(300):
        for index,(_,field) in enumerate(SIGNALS.values()):
            if tick==20 and index==1:continue
            message=JointState();message.name=names
            message.header.stamp.sec=10+tick//100
            message.header.stamp.nanosec=(tick%100)*10_000_000
            setattr(message,field,[float(tick)]*7)
            inputs[index].signalMessage(message)
    return accepted,rejected


def test_exact_w3_sync_recovers_after_one_lost_stream_message():
    accepted,rejected=playback(lambda inputs:make_synchronizer(inputs,{'queue_size':20},True))
    assert len(accepted)==299 and not rejected
    assert 10_200_000_000 not in accepted
    assert accepted[-1]==12_990_000_000
    # Reproduce the original failure: approximate matching consumes vectors
    # from different frames, so the strict contract then rejects good traffic.
    old_accepted,old_rejected=playback(lambda inputs:ApproximateTimeSynchronizer(inputs,20,.03))
    assert len(old_accepted)<len(accepted) and 'frame_stamp' in old_rejected


def test_capture_qos_keeps_legacy_and_non_w3_compatibility():
    assert subscription_qos({}, True).reliability == ReliabilityPolicy.BEST_EFFORT
    qos = subscription_qos({'reliability':'reliable','queue_size':20}, True)
    assert qos.reliability == ReliabilityPolicy.RELIABLE and qos.depth == 20
    assert subscription_qos({'reliability':'reliable'}, False).reliability == ReliabilityPolicy.BEST_EFFORT
    with pytest.raises(ValueError, match='sync.reliability'):
        subscription_qos({'reliability':'typo'}, True)
