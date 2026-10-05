"""Shared W3 frame validation. Times are integer nanoseconds, values SI units."""
import numpy as np
from factr2_next.w3_config import SIGNALS


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def frame(msgs, keys, specs, names, now_ns, max_age=.25):
    if len(msgs) != 4:
        raise ValueError('frame_count')
    stamps = [stamp_ns(m) for m in msgs]
    if len(set(stamps)) != 1 or stamps[0] <= 0:
        raise ValueError('frame_stamp')
    age = (now_ns - stamps[0]) * 1e-9
    if not 0 <= age <= max_age:
        raise ValueError('frame_age')
    samples = {}
    for key, msg in zip(keys, msgs):
        if list(msg.name) != list(names):
            raise ValueError('frame_names')
        field = specs[key]['field']
        a = np.asarray(getattr(msg, field), dtype=np.float32)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('frame_values')
        if any(getattr(msg, f) for f in ('position', 'velocity', 'effort') if f != field):
            raise ValueError('frame_unused_field')
        samples[key] = a
    if set(samples) != set(SIGNALS):
        raise ValueError('frame_keys')
    return stamps[0], samples


def features(pos, vel, cmd):
    return np.concatenate((pos, vel, np.asarray(cmd) - pos), axis=-1).astype(np.float32)
