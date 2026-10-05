"""Bridge-reported motor health; does not infer physical receive timestamps."""
from collections import defaultdict
import math
from .validation import stamp_ns


def motor_health(motors, side):
    channel = 0 if side == 'left' else 1
    slots = defaultdict(list)
    for motor in motors:
        if motor.channel == channel and 0 <= motor.motor_index < 7:
            slots[motor.motor_index].append(motor)
    result = dict(side=side, channel=channel, missing_slots=[], duplicate_slots=[],
                  offline_slots=[], disabled_slots=[], fault_slots=[], nonfinite_slots=[])
    for slot in range(7):
        entries = slots[slot]
        if not entries:
            result['missing_slots'].append(slot)
        if len(entries) > 1:
            result['duplicate_slots'].append(slot)
        for m in entries:
            if not m.online:
                result['offline_slots'].append(slot)
            if not m.enabled:
                result['disabled_slots'].append(slot)
            if m.error_flags != 1:
                result['fault_slots'].append(slot)
            if not all(math.isfinite(x) for x in (m.position, m.velocity, m.torque)):
                result['nonfinite_slots'].append(slot)
    for key in result:
        if key.endswith('_slots'):
            result[key] = sorted(set(result[key]))
    result['healthy'] = not any(v for k, v in result.items() if k.endswith('_slots'))
    return result


class HealthGate:
    def __init__(self, timeout_seconds=0.25, sides=('left', 'right')):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be finite and positive')
        if not sides or len(set(sides)) != len(sides) or any(s not in ('left', 'right') for s in sides):
            raise ValueError('sides must be a unique nonempty subset of left/right')
        self.timeout = timeout_seconds
        self.sides = tuple(sides)
        self.raw = None
        self.last_stamp = None
        self.last_ros = None
        self.reason = 'missing_raw'

    def observe_clock(self, now_ros):
        if self.last_ros is not None and now_ros < self.last_ros:
            self.raw = None
            self.last_stamp = None
            self.reason = 'clock_rollback'
        self.last_ros = now_ros

    def ingest(self, msg, now_ros, now_mono):
        self.observe_clock(now_ros)
        stamp = stamp_ns(msg.header.stamp)
        if (stamp <= 0 or stamp > now_ros or (now_ros - stamp) * 1e-9 > self.timeout
                or (self.last_stamp is not None and stamp <= self.last_stamp)):
            self.raw = None
            self.reason = 'invalid_raw_stamp'
            return False
        self.raw = (msg, stamp, now_mono)
        self.last_stamp = stamp
        self.reason = 'ok'
        return True

    def assess(self, now_ros, now_mono):
        self.observe_clock(now_ros)
        result = {}
        for side in self.sides:
            if self.raw is None:
                values = motor_health([], side)
                values.update(raw_source_stamp_ns=0, raw_receive_age_seconds=None,
                              raw_source_age_seconds=None, healthy=False, reason=self.reason)
            else:
                msg, stamp, received = self.raw
                values = motor_health(msg.motors, side)
                receive_age = now_mono - received
                source_age = (now_ros - stamp) * 1e-9
                fresh = 0 <= receive_age <= self.timeout and 0 <= source_age <= self.timeout
                values.update(raw_source_stamp_ns=stamp, raw_receive_age_seconds=receive_age,
                              raw_source_age_seconds=source_age,
                              healthy=values['healthy'] and fresh,
                              reason='ok' if values['healthy'] and fresh else ('raw_stale' if not fresh else 'motor_unhealthy'))
            result[side] = values
        return result
