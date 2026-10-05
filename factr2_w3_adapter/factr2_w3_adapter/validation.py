"""Clock-explicit validation and sampling, independent of ROS and NumPy."""
from collections import Counter, deque
from dataclasses import dataclass
import math
import re


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def topic_valid(topic):
    return bool(re.fullmatch(r'/[A-Za-z_][A-Za-z_0-9]*(/[A-Za-z_][A-Za-z_0-9]*)*', topic))


@dataclass(frozen=True)
class AdapterConfig:
    side: str
    state_topic: str = '/joint_states'
    command_state_topic: str = '/joint_position_controller/command_state'
    output_root: str = ''
    publish_hz: float = 50.0
    input_timeout_seconds: float = 0.25
    max_source_skew_seconds: float = 0.03
    health_topic: str = '/factr2/w3_health'
    require_hardware_health: bool = False
    hardware_health_timeout_seconds: float = 0.25

    def __post_init__(self):
        if self.side not in ('left', 'right'):
            raise ValueError('side must explicitly be left or right')
        if not self.output_root:
            object.__setattr__(self, 'output_root', f'/factr2/{self.side}')
        if self.output_root != f'/factr2/{self.side}':
            raise ValueError('w3_next_v1 output_root must be /factr2/{side}')
        for key in ('state_topic', 'command_state_topic', 'health_topic', 'output_root'):
            if not topic_valid(getattr(self, key)):
                raise ValueError(f'Invalid absolute topic: {key}')
        for key in ('publish_hz', 'input_timeout_seconds', 'hardware_health_timeout_seconds'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.max_source_skew_seconds) or self.max_source_skew_seconds < 0:
            raise ValueError('max_source_skew_seconds must be finite and nonnegative')
        if not isinstance(self.require_hardware_health, bool):
            raise ValueError('require_hardware_health must be bool')

    @property
    def names(self):
        return tuple(f'{self.side}_joint_{i}' for i in range(7))


@dataclass(frozen=True)
class Source:
    stamp: int
    received: float
    vectors: dict


def joint_vectors(msg, names, command=False):
    fields = ('position', 'velocity') if command else ('position', 'velocity', 'effort')
    if len(set(msg.name)) != len(msg.name):
        raise ValueError('duplicate_names')
    if any(not name for name in msg.name):
        raise ValueError('empty_name')
    if any(len(getattr(msg, field)) != len(msg.name) for field in fields):
        raise ValueError('array_length')
    if command and len(msg.effort):
        raise ValueError('command_effort_not_empty')
    indices = {name: i for i, name in enumerate(msg.name)}
    if any(name not in indices for name in names):
        raise ValueError('missing_joints')
    vectors = {field: tuple(float(getattr(msg, field)[indices[name]]) for name in names)
               for field in fields}
    if any(not math.isfinite(v) for values in vectors.values() for v in values):
        raise ValueError('nonfinite')
    return vectors


class AdapterGate:
    """All decisions use supplied ROS nanoseconds and monotonic seconds.

    Invalid messages invalidate *both* input caches. Stamp high-water marks are
    retained until a ROS clock rollback so recovery cannot reuse earlier frames.
    """
    def __init__(self, config):
        self.cfg = config
        self.sources = {'state': None, 'command': None}
        self.last_stamp = {'state': None, 'command': None}
        self.last_output_stamp = None
        self.last_ros = None
        self.health = None
        self.health_ready = not config.require_hardware_health
        self.counts = Counter(invalid=0, stale=0, duplicate=0, skew=0, health_rejected=0,
                              clock_reset=0, outputs=0)
        self.reason = 'waiting_inputs'
        self.last_rejection = ''
        self.output_times = deque()
        self.metrics = {}

    def clear_sources(self):
        self.sources = {'state': None, 'command': None}

    def observe_clock(self, now_ros):
        if self.last_ros is not None and now_ros < self.last_ros:
            self.clear_sources()
            self.last_stamp = {'state': None, 'command': None}
            self.last_output_stamp = None
            self.health = None
            self.health_ready = not self.cfg.require_hardware_health
            self.counts['clock_reset'] += 1
            self.reason = 'clock_rollback'
        self.last_ros = now_ros

    def reject(self, reason, counter='invalid', clear=True):
        self.reason = reason
        self.last_rejection = reason
        self.counts[counter] += 1
        if clear:
            self.clear_sources()
        return False

    def ingest(self, kind, msg, now_ros, now_mono):
        self.observe_clock(now_ros)
        stamp = stamp_ns(msg.header.stamp)
        if stamp <= 0 or now_ros <= 0:
            return self.reject(kind + '_zero_stamp')
        if stamp > now_ros:
            return self.reject(kind + '_future_stamp')
        if (now_ros - stamp) * 1e-9 > self.cfg.input_timeout_seconds:
            return self.reject(kind + '_source_stale', 'stale')
        previous = self.last_stamp[kind]
        if previous is not None and stamp <= previous:
            return self.reject(kind + ('_duplicate_stamp' if stamp == previous else '_stamp_rollback'),
                               'duplicate')
        try:
            values = joint_vectors(msg, self.cfg.names, command=kind == 'command')
        except (ValueError, TypeError, OverflowError) as e:
            return self.reject(kind + '_' + str(e))
        self.last_stamp[kind] = stamp
        self.sources[kind] = Source(stamp, now_mono, values)
        return True

    def ingest_health(self, msg, now_ros, now_mono):
        self.observe_clock(now_ros)
        entries = [s for s in msg.status if s.name == f'factr2/w3/{self.cfg.side}']
        stamp = stamp_ns(msg.header.stamp)
        # A repeated diagnostic must not refresh the receive watchdog.
        valid = (len(entries) == 1 and entries[0].level in (0, b'\x00') and stamp > 0 and stamp <= now_ros)
        if self.health is not None and stamp <= self.health[0]:
            valid = False
        self.health = (stamp, now_mono, valid)
        if self.cfg.require_hardware_health and not valid:
            self.health_ready = False
            self.reject('health_invalid', 'health_rejected')
        return valid

    def check_health(self, now_ros, now_mono):
        if not self.cfg.require_hardware_health:
            self.metrics['health_mode'] = 'mock/no-hardware-health'
            return True
        self.metrics['health_mode'] = 'required'
        h = self.health
        valid = h is not None and h[2] and 0 <= (now_ros - h[0]) * 1e-9 <= self.cfg.hardware_health_timeout_seconds
        if h is not None:
            self.metrics['health_receive_age_seconds'] = now_mono - h[1]
            self.metrics['health_source_age_seconds'] = (now_ros - h[0]) * 1e-9
            self.metrics['health_source_stamp_ns'] = h[0]
            valid = valid and 0 <= now_mono - h[1] <= self.cfg.hardware_health_timeout_seconds
        if not valid:
            self.health_ready = False
            return self.reject('health_missing_stale_or_error', 'health_rejected')
        if not self.health_ready:
            # Recovery requires state AND command received after this health OK.
            self.clear_sources()
            self.health_ready = True
            self.reason = 'waiting_inputs_after_health_recovery'
            return False
        return True

    def sample(self, now_ros, now_mono):
        self.observe_clock(now_ros)
        self.metrics = {}
        for kind, source in self.sources.items():
            self.metrics[kind + '_receive_age_seconds'] = None if source is None else now_mono - source.received
            self.metrics[kind + '_source_age_seconds'] = None if source is None else (now_ros - source.stamp) * 1e-9
        while self.output_times and self.output_times[0] < now_mono - 1:
            self.output_times.popleft()
        self.metrics['output_hz_last_second'] = len(self.output_times)
        if not self.check_health(now_ros, now_mono):
            return None
        if any(source is None for source in self.sources.values()):
            self.reason = 'waiting_inputs'
            return None
        state, command = self.sources['state'], self.sources['command']
        for kind in ('state', 'command'):
            ages = (self.metrics[kind + '_receive_age_seconds'], self.metrics[kind + '_source_age_seconds'])
            if any(age < 0 or age > self.cfg.input_timeout_seconds for age in ages):
                self.reject(kind + '_stale', 'stale')
                return None
        skew = abs(state.stamp - command.stamp) * 1e-9
        self.metrics['source_skew_seconds'] = skew
        if skew > self.cfg.max_source_skew_seconds:
            self.reject('source_skew', 'skew')
            return None
        if self.last_output_stamp is not None and state.stamp <= self.last_output_stamp:
            self.reject('state_already_sampled', 'duplicate', clear=False)
            return None
        self.last_output_stamp = state.stamp
        self.counts['outputs'] += 1
        self.output_times.append(now_mono)
        self.reason = 'ok'
        return state.stamp, {'joint_pos': ('position', state.vectors['position']),
                             'joint_vel': ('velocity', state.vectors['velocity']),
                             'joint_cmd': ('position', command.vectors['position']),
                             'joint_effort': ('effort', state.vectors['effort'])}
