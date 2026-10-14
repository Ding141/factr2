#!/usr/bin/env python3
"""Compile contact-free endpoint goals. Dense samples are for audit, never streaming.

Joint excitation uses a single complete sparse trajectory per sweep. Task motion
uses one PoseStamped per target, with its frozen IK/null-space solution executed
by w3_pose_motion. The native controller performs the continuous quintic motion.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from w3_baseline_protocol import task_path_ok
from w3_coverage_plan import ArmModel, SCHEMA, summary, validate_config, validate_plan


def sample_knots(knots, dt):
    """Exactly the rest-to-rest quintics used by the native joint controller."""
    times, positions = [], []
    for index, (a, b) in enumerate(zip(knots, knots[1:])):
        start, end = a['time_seconds'], b['time_seconds']
        n = int(round((end-start)/dt))
        if n < 1 or abs(n*dt-(end-start)) > 1e-7:
            raise ValueError('Knots must lie on the sample grid')
        u = np.arange(n+1)/n
        s = u**3*(10-15*u+6*u*u)
        q = np.asarray(a['position_deg']) + s[:, None]*(
            np.asarray(b['position_deg'])-a['position_deg'])
        times.extend((start+np.arange(n+1)*dt)[1 if index else 0:])
        positions.extend(q[1 if index else 0:])
    return np.asarray(times), np.asarray(positions)


def compile_knots(points, limits, dt, jerk_limit=120.):
    # Exact maxima: quintic v=1.875*D/T, a=10/sqrt(3)*D/T^2,
    # jerk=60*D/T^3. Reserve 5% for floating point/native Bernstein checks.
    knots = [dict(time_seconds=0., position_deg=np.asarray(points[0]).tolist())]
    for start, end in zip(points, points[1:]):
        delta = abs(np.asarray(end)-start)
        duration = max(.5, float(np.max(1.875*delta/limits['velocity_deg_s'])),
            float(np.sqrt(np.max(10/math.sqrt(3)*delta/limits['acceleration_deg_s2']))),
            float(np.cbrt(np.max(60*delta/jerk_limit))))*1.05
        duration = math.ceil(duration/dt)*dt
        knots.append(dict(time_seconds=knots[-1]['time_seconds']+duration,
                          position_deg=np.asarray(end).tolist()))
    t, q = sample_knots(knots, dt)
    return knots, t, q


def pose_target(model, q):
    transform = model.fk(np.deg2rad(q))
    return dict(frame_id=model.root_frame, position_m=transform[:3, 3].tolist(),
                quaternion_xyzw=Rotation.from_matrix(transform[:3, :3]).as_quat().tolist())


def build_pose_protocol(recipe, description, initial_deg, role):
    if role not in ('train', 'val', 'test'):
        raise ValueError('Require independent train/val/test role')
    cfg = copy.deepcopy(recipe['coverage'])
    model = ArmModel(description, cfg['side'], cfg['tool_frame'])
    home, bounds = validate_config(cfg, model)
    spec, tasks = recipe['protocol'], recipe['task_coverage']
    if (set(spec) != {'seeds', 'max_condition', 'dwell_seconds', 'task_pose_count',
                     'task_pose_bounds_deg', 'return_offset_deg'}
            or set(spec['seeds']) != {'train', 'val', 'test'}
            or any(type(s) is not int for s in spec['seeds'].values())
            or set(tasks) != {'shared_pose_seed', 'coupled_amplitude_deg', 'coupled_cycles', 'cartesian_repeats'}
            or type(tasks['shared_pose_seed']) is not int
            or tasks['coupled_cycles'] != 1 or tasks['cartesian_repeats'] != 1):
        raise ValueError('Pose protocol requires one cycle/repeat and explicit fixed seeds')
    if len(set(spec['seeds'].values())) != 3:
        raise ValueError('Require three distinct role seeds')
    maximum, dwell = float(spec['max_condition']), float(spec['dwell_seconds'])
    count = spec['task_pose_count']
    if not 1 < maximum <= 20 or not 5 <= dwell <= 30 or type(count) is not int or not 1 <= count <= 12:
        raise ValueError('Invalid pose protocol limits')
    pose_bounds = np.asarray(spec['task_pose_bounds_deg'], float)
    probe = np.asarray(spec['return_offset_deg'], float)
    amplitude = np.asarray(tasks['coupled_amplitude_deg'], float)
    if (pose_bounds.shape != (7, 2) or probe.shape != (7,) or amplitude.shape != (7,)
            or not np.isfinite(np.r_[pose_bounds.ravel(), probe, amplitude]).all()
            or np.any(pose_bounds[:, 0] >= pose_bounds[:, 1])
            or np.any(pose_bounds[:, 0] < bounds[:, 0]) or np.any(pose_bounds[:, 1] > bounds[:, 1])
            or np.any(amplitude <= 0) or np.max(abs(probe)) > 10 or not np.any(probe)):
        raise ValueError('Invalid task bounds/probes/amplitudes')
    if not task_path_ok(model, [home], maximum):
        raise ValueError('Home must be well conditioned')
    dt = cfg['sample_period_seconds']
    current = np.asarray(initial_deg, float)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError('Invalid initial command')
    rng = np.random.default_rng(spec['seeds'][role])
    anchors = [home.copy()]
    shared_rng = np.random.default_rng(tasks['shared_pose_seed'])
    for _ in range(2000):
        if len(anchors) == count:
            break
        p = shared_rng.uniform(pose_bounds[:, 0], pose_bounds[:, 1])
        if np.min(np.max(abs(np.asarray(anchors)-p), axis=1)) < 4:
            continue
        path = np.vstack([np.linspace(anchors[-1], p, 101), np.linspace(p-probe, p+probe, 101)])
        if (np.any(path < bounds[:, 0]) or np.any(path > bounds[:, 1])
                or not task_path_ok(model, path, maximum)):
            continue
        anchors.append(p)
    if len(anchors) != count:
        raise ValueError('Cannot find shared conditioned anchors')
    steps = []

    def add(points, phase, kind, speed='slow', joint=None, command_type='pose_goal', conditioned=True):
        nonlocal current
        limits = cfg['speeds'][speed]
        knots, t, q = compile_knots(points, limits, dt)
        if conditioned and not task_path_ok(model, q, maximum):
            raise ValueError('Ill-conditioned complete path: '+kind)
        command = dict(type=command_type, knots=knots)
        if command_type == 'pose_goal':
            command['target'] = pose_target(model, q[-1])
        steps.append(dict(name=f'{len(steps)+1:03d}_{kind}_{speed}', phase=phase, kind=kind,
            speed=speed, repetition=1, joint=joint, duration_seconds=float(t[-1]), hold_seconds=0.,
            times_seconds=t.tolist(), positions_deg=q.tolist(), command=command))
        current = q[-1].copy()

    def move(goal, kind, speed='slow', conditioned=True):
        if np.max(abs(np.asarray(goal)-current)) > 1e-8:
            add([current, goal], 'transition', kind, speed, conditioned=conditioned)

    def hold(kind):
        n = int(math.ceil(dwell/dt))
        steps.append(dict(name=f'{len(steps)+1:03d}_{kind}', phase='cartesian', kind=kind,
            speed='slow', repetition=1, joint=None, duration_seconds=n*dt, hold_seconds=0.,
            times_seconds=(np.arange(n+1)*dt).tolist(),
            positions_deg=np.repeat(current[None], n+1, axis=0).tolist(), command=dict(type='hold')))

    move(home, 'approach', conditioned=False)
    hold('static_home_before')
    # NEXT Appendix A.3: independent joints first, then multi-joint task motion.
    for speed in ('slow', 'fast'):
        for joint in range(7):
            low, high = home.copy(), home.copy()
            low[joint], high[joint] = bounds[joint]
            endpoints = [low, high] if rng.integers(2) else [high, low]
            add([home, *endpoints, home], 'single_joint', f'sweep_j{joint+1}', speed, joint,
                command_type='joint_trajectory', conditioned=False)
    hold('static_home_after_excitation')
    for index, anchor in enumerate(anchors):
        move(anchor, f'pose{index}_approach')
        for direction, offset in (('low', -probe), ('high', probe)):
            move(anchor+offset, f'pose{index}_condition_{direction}')
            move(anchor, f'pose{index}_return_{direction}')
            hold(f'static_pose{index}_from_{direction}')
        for speed in ('slow', 'fast'):
            # Different full postures excite all joints, with bounded endpoint
            # goals rather than hundreds of interpolated sender windows.
            for pattern in range(2):
                offset = rng.choice([-1., 1.], 7)*np.minimum(amplitude,
                    np.minimum(anchor-bounds[:, 0], bounds[:, 1]-anchor)*.8)
                for _ in range(15):
                    targets = [anchor+offset, anchor-offset, anchor]
                    if task_path_ok(model, np.vstack([np.linspace(a, b, 101) for a, b in
                            zip([anchor, *targets[:-1]], targets)]), maximum):
                        break
                    offset *= .8
                else:
                    raise ValueError('No conditioned coupled goals')
                for leg, goal in enumerate(targets):
                    add([current, goal], 'cartesian', f'pose{index}_coupled{pattern}_leg{leg}', speed)
            center = model.fk(np.deg2rad(anchor))[:3, 3]
            for axis in range(3):
                for leg, distance in enumerate((-cfg['cartesian']['line_half_extent_m'],
                                                 cfg['cartesian']['line_half_extent_m'], 0.)):
                    xyz = center.copy(); xyz[axis] += distance
                    # Offline inverse solution includes the redundancy choice;
                    # the bridge accepts only this collision-audited solution.
                    goal = (anchor if distance == 0 else np.rad2deg(model.ik_position(
                        xyz, np.deg2rad(current), np.deg2rad(bounds))))
                    add([current, goal], 'cartesian', f'pose{index}_line_{"xyz"[axis]}_leg{leg}', speed)
            # Four complete Cartesian endpoint goals describe an XY circuit.
            # The native joint interpolation need not trace an exact circle.
            radius = cfg['cartesian']['circle_radius_m']
            for leg, angle in enumerate((np.pi/2, np.pi, 3*np.pi/2, 2*np.pi)):
                xyz = center+np.array([radius*(np.cos(angle)-1), radius*np.sin(angle), 0.])
                goal = (anchor if leg == 3 else np.rad2deg(model.ik_position(
                    xyz, np.deg2rad(current), np.deg2rad(bounds))))
                add([current, goal], 'cartesian', f'pose{index}_circle_xy_leg{leg}', speed)
        hold(f'static_pose{index}_after_motion')
    move(home, 'finish_home'); hold('static_home_finish')
    cfg.update(execution_mode='direct_pose_goals', continuous_repetitions=False)
    cfg['baseline_protocol'] = dict(version=2, role=role, seed=spec['seeds'][role],
        max_condition=maximum, dwell_seconds=dwell, shared_pose_seed=tasks['shared_pose_seed'],
        task_poses_deg=np.asarray(anchors).tolist(),
        recipe_sha256=hashlib.sha256(yaml.safe_dump(recipe, sort_keys=True).encode()).hexdigest())
    plan = dict(schema=SCHEMA, name=cfg['name'], side=cfg['side'], config=cfg,
        robot_description=description, model_sha256=model.sha256, joint_names=model.names,
        initial_deg=list(initial_deg), steps=steps)
    validate_plan(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--urdf', required=True, type=Path)
    parser.add_argument('--initial-deg', required=True, nargs=7, type=float)
    parser.add_argument('--role', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    plan = build_pose_protocol(yaml.safe_load(args.config.read_text()), args.urdf.read_text(),
                               args.initial_deg, args.role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.dump(plan, Dumper=getattr(yaml, 'CSafeDumper', yaml.SafeDumper), sort_keys=False))
    print(json.dumps(summary(plan), indent=2))


if __name__ == '__main__':
    main()
