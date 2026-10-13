#!/usr/bin/python3.10
"""Offline audit of the control-cycle targets frozen in a capture rosbag.

Run from the W3 ROS environment. Does not initialize ROS or publish commands.
The finite differences are target continuity checks, not hardware jerk accuracy.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


def command_blobs(bag):
    metadata = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    if str(metadata.get('compression_mode', '')).lower() in ('file', 'message'):
        # The installed rosbag2 compression reader handles chunk decompression;
        # do not feed a compressed file to SQLite or skip its samples.
        import rosbag2_py
        reader = rosbag2_py.SequentialCompressionReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=metadata['storage_identifier']),
                    rosbag2_py.ConverterOptions('', ''))
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/joint_position_controller/command_state']))
        while reader.has_next():
            topic, blob, _ = reader.read_next()
            if topic == '/joint_position_controller/command_state':
                yield blob
        return
    for path in sorted(bag.glob('*.db3')):
        with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro', uri=True) as db:
            for blob, in db.execute("SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
                    "WHERE t.name='/joint_position_controller/command_state' ORDER BY m.timestamp"):
                yield blob


def audit(session):
    config = yaml.safe_load((session/'audit/position.yaml').read_text())
    params = next(iter(config.values()))['ros__parameters']
    names = params['joints']
    rows = []
    for blob in command_blobs(session/'audit/bag'):
        m = deserialize_message(blob, JointState)
        ix = [m.name.index(name) for name in names]
        rows.append((m.header.stamp.sec*10**9+m.header.stamp.nanosec,
            [m.position[k] for k in ix], [m.velocity[k] for k in ix]))
    if len(rows) < 500:
        raise ValueError('Too few controller-cycle samples for continuity audit')
    rows.sort(key=lambda item: item[0])
    t=np.array([r[0] for r in rows],dtype=np.int64)
    q=np.array([r[1] for r in rows]);v=np.array([r[2] for r in rows])
    dt=np.diff(t)*1e-9
    if np.any(dt <= 0) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ValueError('Nonfinite state or repeated command timestamps')
    local=dt < .02
    a=np.diff(v,axis=0)/dt[:,None]
    error=np.diff(q,axis=0)-.5*(v[1:]+v[:-1])*dt[:,None]
    speed=np.max(abs(v),axis=0);acc=np.max(abs(a[local]),axis=0)
    residual=np.max(abs(error[local]),axis=0)
    def caps(key, default):
        value=params.get(key,[])
        return np.array(value if value else [default]*len(names))
    accepted=bool(np.all(speed <= caps('max_velocity',.4)+1e-5) and
        np.all(acc <= caps('max_acceleration',1.75)+1e-3) and np.all(residual <= 5e-5))
    report=dict(accepted=accepted,samples=len(t),joints=names,
        median_period_s=float(np.median(dt)),max_gap_s=float(max(dt)),
        gaps_over_20ms=int(np.sum(~local)),max_speed_deg_s=np.rad2deg(speed).tolist(),
        max_acceleration_deg_s2=np.rad2deg(acc).tolist(),
        max_step_vs_integrated_velocity_error_deg=np.rad2deg(residual).tolist(),
        limitation='Checks published target continuity and configured caps. Bag gaps cannot prove all unrecorded control cycles; actual motor smoothness requires the separate capture diagnostic.')
    out=session/'analysis';out.mkdir(exist_ok=True)
    (out/'controller_smoothness.json').write_text(json.dumps(report,indent=2))
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args();result=audit(args.session)
    print(json.dumps(result,indent=2))
    if not result['accepted']:
        raise SystemExit('Control target continuity audit failed; training blocked')
