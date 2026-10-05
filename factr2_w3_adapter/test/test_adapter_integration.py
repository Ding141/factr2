"""Real rclpy pub/sub tests of failure, recovery, fields and endpoint ownership."""
import json
import time
from pathlib import Path
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rosgraph_msgs.msg import Clock
from factr2_w3_adapter.adapter_node import AdapterNode


class Harness:
    def __init__(self, real=False, sim=False):
        self.probe = rclpy.create_node('adapter_integration_probe')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.probe)
        self.nodes = {}
        self.received = {s: {k: [] for k in ('joint_pos','joint_vel','joint_cmd','joint_effort')} for s in ('left','right')}
        self.status = {s: [] for s in ('left','right')}
        self.subscriptions = []
        for side in ('left','right'):
            node = AdapterNode(namespace=f'/contract_test/{side}', parameter_overrides=[
                Parameter('side', value=side), Parameter('require_hardware_health', value=real),
                Parameter('use_sim_time', value=sim)])
            self.nodes[side] = node
            self.executor.add_node(node)
            for key in self.received[side]:
                self.subscriptions.append(self.probe.create_subscription(JointState, f'/factr2/{side}/{key}',
                    lambda m, s=side, k=key: self.received[s][k].append(m), qos_profile_sensor_data))
            self.subscriptions.append(self.probe.create_subscription(DiagnosticArray, f'/factr2/{side}/adapter_status',
                lambda m, s=side: self.status[s].append(m), qos_profile_sensor_data))
        self.state_pub = self.probe.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.command_pub = self.probe.create_publisher(JointState, '/joint_position_controller/command_state', qos_profile_sensor_data)
        self.health_pub = self.probe.create_publisher(DiagnosticArray, '/factr2/w3_health', qos_profile_sensor_data)
        self.clock_pub = self.probe.create_publisher(Clock, '/clock', 10)
        self.drive(0.2, state=False, command=False)

    def message(self, command=False, stamp=None):
        msg = JointState()
        msg.header.stamp = self.probe.get_clock().now().to_msg() if stamp is None else stamp
        names = [f'{s}_joint_{i}' for s in ('left','right') for i in range(7)] + ['gripper']
        msg.name = list(reversed(names)) if command else names[5:] + names[:5]
        msg.position = [float(int(n[-1]) + (100 if n.startswith('right') else 0)) if n != 'gripper' else 999.0 for n in msg.name]
        if command:
            msg.position = [x + 0.1 for x in msg.position]
        msg.velocity = [x+10 for x in msg.position]
        msg.effort = [] if command else [x+20 for x in msg.position]
        return msg

    def drive(self, seconds, state=True, command=True, damage=None, health=None, stamp=None):
        end = time.monotonic() + seconds
        next_pub = 0.0
        while time.monotonic() < end:
            if time.monotonic() >= next_pub:
                sm, cm = self.message(stamp=stamp), self.message(command=True, stamp=stamp)
                if damage:
                    damage(sm, cm)
                if state: self.state_pub.publish(sm)
                if command: self.command_pub.publish(cm)
                if health is not None:
                    msg = DiagnosticArray()
                    msg.header.stamp = sm.header.stamp
                    msg.status = [DiagnosticStatus(name=f'factr2/w3/{s}', level=bytes([level])) for s, level in health.items()]
                    self.health_pub.publish(msg)
                next_pub = time.monotonic() + (0.03 if stamp is not None else 0.006)  # Above the 50 Hz timer rate.
            self.executor.spin_once(timeout_sec=0.001)

    def clear(self):
        for side in self.received:
            for data in self.received[side].values(): data.clear()

    def close(self):
        for node in [*self.nodes.values(), self.probe]:
            self.executor.remove_node(node)
            node.destroy_node()
        self.executor.shutdown()


@pytest.fixture
def harness():
    rclpy.init()
    h = Harness()
    try: yield h
    finally:
        h.close()
        rclpy.shutdown()


def test_ADP01_02_fields_names_static_and_qos(harness):
    h = harness
    h.drive(0.35)
    endpoints = {}
    for topic in ('/joint_states', '/joint_position_controller/command_state', '/factr2/w3_health'):
        infos = h.probe.get_subscriptions_info_by_topic(topic)
        adapters = [info for info in infos if info.node_name == 'w3_next_adapter']
        assert len(adapters) == 2
        for info in adapters:
            assert info.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
            assert info.qos_profile.durability == DurabilityPolicy.VOLATILE
        endpoints[topic + ' inputs'] = [str(info.qos_profile) for info in adapters]
    for side in ('left','right'):
        expected = [float(i + (100 if side == 'right' else 0)) for i in range(7)]
        for key, field, offset in [('joint_pos','position',0),('joint_cmd','position',0.1),
                                    ('joint_vel','velocity',10),('joint_effort','effort',20)]:
            messages = h.received[side][key]
            assert len(messages) >= 8
            for msg in messages:
                assert msg.name == [f'{side}_joint_{i}' for i in range(7)]
                assert list(getattr(msg,field)) == pytest.approx([v+offset for v in expected])
                assert all(not getattr(msg,f) for f in ('position','velocity','effort') if f != field)
            info = h.probe.get_publishers_info_by_topic(f'/factr2/{side}/{key}')
            assert len(info) == 1
            assert info[0].qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
            assert info[0].qos_profile.durability == DurabilityPolicy.VOLATILE
            endpoints[f'/factr2/{side}/{key}'] = str(info[0].qos_profile)
        stamps = [[m.header.stamp.sec*10**9+m.header.stamp.nanosec for m in h.received[side][k]] for k in h.received[side]]
        assert all(a==stamps[0] for a in stamps)
        assert all(b>a for a,b in zip(stamps[0],stamps[0][1:]))
        names = h.probe.get_publisher_names_and_types_by_node('w3_next_adapter', f'/contract_test/{side}')
        assert {n for n,_ in names} <= {f'/factr2/{side}/{k}' for k in h.received[side]} | {f'/factr2/{side}/adapter_status','/rosout','/parameter_events'}
        assert not h.probe.get_client_names_and_types_by_node('w3_next_adapter', f'/contract_test/{side}')
        result = h.nodes[side].set_parameters([Parameter('publish_hz',value=25.0)])
        assert not result[0].successful
    # Report endpoint inspection without ROS CLI daemon or external files.
    path = Path('reports/generated/3')
    if path.is_dir(): (path/'integration_qos.json').write_text(json.dumps(endpoints,indent=2))


@pytest.mark.parametrize('damage', [
    lambda s,c: s.name.__setitem__(0,s.name[1]),
    lambda s,c: s.position.pop(),
    lambda s,c: s.effort.__setitem__(s.name.index('left_joint_0'),float('nan')),
    lambda s,c: c.velocity.pop(),
])
def test_ADP03_bad_messages_stop_and_fresh_pair_recovers(harness, damage):
    h=harness
    h.drive(0.12)
    h.drive(0.08,damage=damage)  # Drain frames already queued before the bad input.
    h.clear()
    h.drive(0.1,damage=damage)
    assert not h.received['left']['joint_pos']
    assert h.nodes['left'].gate.counts['invalid'] > 0
    assert h.nodes['left'].gate.last_rejection
    h.drive(0.15)
    assert h.received['left']['joint_pos']


def test_ADP04_05_07_timeout_skew_future_and_no_old_resampling(harness):
    h=harness
    h.drive(0.15)
    h.drive(0.06,state=False)  # State stops; command continues.
    h.clear()
    h.drive(0.28,state=False)
    assert not h.received['left']['joint_pos']
    assert h.nodes['left'].gate.counts['duplicate'] + h.nodes['left'].gate.counts['skew'] > 0
    h.drive(0.12)
    h.drive(0.28,state=False,command=False)
    assert h.nodes['left'].gate.counts['stale'] > 0
    h.drive(0.12)
    h.drive(0.06,damage=lambda s,c:setattr(c.header.stamp,'sec',c.header.stamp.sec+1))
    h.clear()
    h.drive(0.08,damage=lambda s,c:setattr(c.header.stamp,'sec',c.header.stamp.sec+1))
    assert not h.received['left']['joint_pos']
    h.drive(0.12)
    def skew(s,c):
        ns=c.header.stamp.sec*10**9+c.header.stamp.nanosec-40_000_000
        c.header.stamp.sec,c.header.stamp.nanosec=divmod(ns,10**9)
    h.drive(0.08,damage=skew)
    h.clear()
    h.drive(0.08,damage=skew)
    assert not h.received['left']['joint_pos']
    assert h.nodes['left'].gate.counts['skew'] > 0
    h.drive(0.15)
    assert h.received['left']['joint_pos']


def test_ADP08_health_real_side_isolation_and_recovery():
    rclpy.init()
    h=Harness(real=True)
    try:
        h.drive(0.1)
        assert not h.received['left']['joint_pos']
        h.drive(0.2,health={'left':0,'right':0})
        assert h.received['left']['joint_pos'] and h.received['right']['joint_pos']
        h.drive(0.07,health={'left':2,'right':0})
        h.clear()
        h.drive(0.15,health={'left':2,'right':0})
        assert not h.received['left']['joint_pos'] and h.received['right']['joint_pos']
        h.drive(0.2,health={'left':0,'right':0})
        assert h.received['left']['joint_pos']
        h.drive(0.35)
        h.clear()
        h.drive(0.08)
        assert not h.received['left']['joint_pos'] and not h.received['right']['joint_pos']
    finally:
        h.close(); rclpy.shutdown()


def test_ADP05_ROS_clock_rollback_and_paused_clock_watchdog():
    rclpy.init()
    h=Harness(sim=True)
    try:
        from builtin_interfaces.msg import Time
        def set_clock(seconds):
            c=Clock(); c.clock=Time(sec=seconds)
            h.clock_pub.publish(c)
            h.drive(0.08,state=False,command=False)
        set_clock(10)
        h.drive(0.004,stamp=Time(sec=10))
        h.drive(0.04,state=False,command=False)
        assert h.received['left']['joint_pos']
        set_clock(5)
        h.clear()
        assert h.nodes['left'].gate.counts['clock_reset'] >= 1
        h.drive(0.004,stamp=Time(sec=5))
        h.drive(0.04,state=False,command=False)
        assert h.received['left']['joint_pos']
        h.drive(0.35,state=False,command=False)
        assert h.nodes['left'].gate.counts['stale'] >= 1
    finally:
        h.close(); rclpy.shutdown()


def test_NEXT_message_filters_four_stream_compatibility(harness):
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    h=harness
    filters=[Subscriber(h.probe,JointState,f'/factr2/left/{key}',qos_profile=qos_profile_sensor_data)
             for key in ('joint_pos','joint_vel','joint_cmd','joint_effort')]
    synchronizer=ApproximateTimeSynchronizer(filters,queue_size=20,slop=0.03)
    samples=[]
    synchronizer.registerCallback(lambda *messages:samples.append(messages))
    h.drive(0.3)
    assert len(samples)>=8
    for messages in samples:
        assert all(m.header.stamp==messages[0].header.stamp for m in messages)
        assert list(messages[1].velocity)==[10.0+i for i in range(7)]
        assert list(messages[3].effort)==[20.0+i for i in range(7)]


def test_ament_package_registration_and_installed_profiles():
    from ament_index_python.packages import get_package_share_directory
    directory=Path(get_package_share_directory('factr2_w3_adapter'))
    assert (directory/'launch/adapter.launch.py').is_file()
    for side in ('left','right'):
        for mode in ('mock','real'):
            assert (directory/'config'/f'{side}_{mode}.yaml').is_file()
