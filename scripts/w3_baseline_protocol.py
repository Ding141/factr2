#!/usr/bin/env python3
"""Offline NEXT protocol: joint excitation, conditioned task poses, matched returns.

No ROS imports, publishers or hardware actions. The resulting plan uses the
existing C2 window sender; dwell periods are explicit constant-position samples.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from w3_coverage_plan import ArmModel, SCHEMA, validate_config, validate_plan, summary
from w3_smooth_coverage import retime


def conditions(model, positions, length=.4):
    from factr2_next.inference.endpoint_wrench import geometry_quality
    return [geometry_quality(model.fk(np.deg2rad(q), True)[1], length)['condition_number']
            for q in positions]


def task_path_ok(model, positions, maximum):
    # Check every command sample, not only endpoints or unretimed IK waypoints.
    return all(c is not None and c <= maximum for c in conditions(model, positions))


def build_protocol(recipe, description, initial_deg, role):
    if role not in ('train', 'val', 'test'):
        raise ValueError('Independent session role must be train/val/test')
    cfg = copy.deepcopy(recipe['coverage'])
    model = ArmModel(description, cfg['side'], cfg['tool_frame'])
    home, bounds = validate_config(cfg, model)
    spec = recipe['protocol']
    if (set(spec) != {'seeds', 'max_condition', 'dwell_seconds', 'task_pose_count',
                     'task_pose_bounds_deg', 'return_offset_deg'}
            or set(spec['seeds']) != {'train', 'val', 'test'}
            or any(type(s) is not int for s in spec['seeds'].values())):
        raise ValueError('Invalid protocol fields or session seeds')
    seed = int(spec['seeds'][role])
    if len(set(spec['seeds'].values())) != 3:
        raise ValueError('Use different fixed seeds for the three sessions')
    maximum = float(spec['max_condition'])
    dwell = float(spec['dwell_seconds'])
    if not 1 < maximum <= 100 or not 3 <= dwell <= 30:
        raise ValueError('Invalid geometry/dwell limits')
    count = spec['task_pose_count']
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError('task_pose_count must be 1..12')
    pose_bounds = np.asarray(spec['task_pose_bounds_deg'], dtype=float)
    if (pose_bounds.shape != (7, 2) or not np.isfinite(pose_bounds).all()
            or np.any(pose_bounds[:, 0] >= pose_bounds[:, 1])
            or np.any(pose_bounds[:, 0] < bounds[:, 0])
            or np.any(pose_bounds[:, 1] > bounds[:, 1])):
        raise ValueError('Task pose bounds must be inside working bounds')
    probe = np.asarray(spec['return_offset_deg'], dtype=float)
    if probe.shape != (7,) or not np.isfinite(probe).all() or np.max(abs(probe)) > 10 or not np.any(probe):
        raise ValueError('Invalid return probe offset')
    if (np.any(home-probe < bounds[:,0]) or np.any(home+probe > bounds[:,1])
            or np.any(home < pose_bounds[:,0]) or np.any(home > pose_bounds[:,1])):
        raise ValueError('Home and return probes must fit task/working bounds')
    rng = np.random.default_rng(seed)
    steps = []
    current = np.asarray(initial_deg, dtype=float)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError('Invalid initial target')
    dt = cfg['sample_period_seconds']

    def append(curve, phase, kind, speed='slow', joint=None, conditioned=False):
        nonlocal current
        limits = copy.deepcopy(cfg['speeds'][speed])
        if phase in ('single_joint', 'cartesian'):
            # Independent sessions vary timing as well as task poses, within
            # the frozen upper limits. This avoids testing only one replay speed.
            factor = rng.uniform(.85, 1.)
            limits['velocity_deg_s'] = (np.asarray(limits['velocity_deg_s'])*factor).tolist()
        t, q = retime(curve, limits, dt)
        if conditioned and not task_path_ok(model, q, maximum):
            raise ValueError('Task path approaches singularity: ' + kind)
        steps.append(dict(name=f'{len(steps)+1:03d}_{kind}_{speed}', phase=phase,
            kind=kind, speed=speed, repetition=1, joint=joint,
            duration_seconds=float(t[-1]), hold_seconds=0.,
            times_seconds=t.tolist(), positions_deg=q.tolist()))
        current = q[-1].copy()

    def move(goal, kind, conditioned=False, speed='slow'):
        start = current.copy()
        goal = np.asarray(goal)
        if np.max(abs(goal-start)) < 1e-8:
            return
        append(lambda s: start[None, :] + s[:, None]*(goal-start),
               'transition', kind, speed, conditioned=conditioned)

    def hold(kind):
        n = int(np.ceil(dwell/dt))
        q = np.repeat(current[None, :], n+1, axis=0)
        if not task_path_ok(model, [current], maximum):
            raise ValueError('Unconditioned static task pose')
        steps.append(dict(name=f'{len(steps)+1:03d}_{kind}', phase='cartesian',
            kind=kind, speed='slow', repetition=1, joint=None,
            duration_seconds=n*dt, hold_seconds=0.,
            times_seconds=(np.arange(n+1)*dt).tolist(), positions_deg=q.tolist()))

    if not task_path_ok(model, [home], maximum):
        raise ValueError('Choose a well-conditioned home; never widen the runtime threshold')
    # A singular initial pose is allowed only for the approach transition.
    move(home, 'approach')
    hold('static_home_before')
    for speed in ('slow', 'fast'):
        for joint in range(7):
            center = np.mean(bounds[joint]); amp = np.ptp(bounds[joint])/2
            phase = np.arcsin((home[joint]-center)/amp)
            direction = int(rng.choice([-1, 1]))
            def curve(s, joint=joint, center=center, amp=amp, phase=phase, direction=direction):
                q = np.repeat(home[None, :], len(s), axis=0)
                q[:, joint] = center + amp*np.sin(phase+direction*2*np.pi*s)
                return q
            append(curve, 'single_joint', f'sine_j{joint+1}', speed, joint)
    hold('static_home_after_excitation')

    # Select poses AND all connecting/return segments by the same scaled
    # Jacobian criterion as deployment. Collision checks remain mandatory.
    poses = [home.copy()]
    attempts = 0
    while len(poses) < count:
        attempts += 1
        if attempts > 2000:
            raise ValueError('No conditioned pose set; reduce extent or change task bounds')
        p = rng.uniform(pose_bounds[:, 0], pose_bounds[:, 1])
        samples = np.vstack([np.linspace(poses[-1], p, 101),
                             np.linspace(p-probe, p+probe, 101)])
        if np.any(samples < bounds[:, 0]) or np.any(samples > bounds[:, 1]):
            continue
        if task_path_ok(model, samples, maximum):
            poses.append(p)
    for i, pose in enumerate(poses):
        move(pose, f'pose{i}_approach', conditioned=True)
        for direction, offset in (('low', -probe), ('high', probe)):
            move(pose+offset, f'pose{i}_condition_{direction}', conditioned=True)
            move(pose, f'pose{i}_return_{direction}', conditioned=True)
            hold(f'static_pose{i}_from_{direction}')
        center = model.fk(np.deg2rad(pose))[:3, 3]
        u = np.linspace(0., 1., 129)
        for kind in ('line_x', 'line_y', 'line_z', 'circle_xy'):
            xyz = np.repeat(center[None, :], len(u), axis=0)
            angle = 2*np.pi*u
            if kind.startswith('line_'):
                xyz[:, 'xyz'.index(kind[-1])] += cfg['cartesian']['line_half_extent_m']*np.sin(angle)
            else:
                radius = cfg['cartesian']['circle_radius_m']
                xyz[:, 0] += radius*(np.cos(angle)-1)
                xyz[:, 1] += radius*np.sin(angle)
            seed_q = np.deg2rad(pose); q = [pose.copy()]
            for point in xyz[1:]:
                seed_q = model.ik_position(point, seed_q, np.deg2rad(bounds))
                q.append(np.rad2deg(seed_q))
            curve = CubicSpline(u, q, axis=0)
            for speed in ('slow', 'fast'):
                move(pose, f'pose{i}_reset_{kind}_{speed}', conditioned=True)
                append(curve, 'cartesian', f'pose{i}_{kind}', speed, conditioned=True)
                move(pose, f'pose{i}_reset_after_{kind}_{speed}', conditioned=True)
        hold(f'static_pose{i}_after_motion')
    # Validate the actual final transition too; do not silently return to zero.
    move(home, 'finish_home', conditioned=True)
    hold('static_home_finish')
    cfg['baseline_protocol'] = dict(version=1, role=role, seed=seed,
                                   max_condition=maximum, dwell_seconds=dwell)
    plan = dict(schema=SCHEMA, name=cfg['name'], side=cfg['side'], config=cfg,
        robot_description=description, model_sha256=model.sha256,
        joint_names=model.names, initial_deg=list(initial_deg), steps=steps)
    validate_plan(plan)
    return plan, dict(role=role, seed=seed, task_poses_deg=np.asarray(poses).tolist(),
        task_max_condition=maximum, approach_may_be_singular=True,
        joint_sweeps_may_be_singular=True, automatic_rezero=False,
        limitations=['Geometry is not collision clearance',
                     'No-contact returns are friction probes, not external-load release validation'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--initial-deg', type=float, nargs=7, required=True)
    p.add_argument('--role', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; keep frozen plans immutable')
    plan, report = build_protocol(yaml.safe_load(args.config.read_text()),
                                 args.urdf.read_text(), args.initial_deg, args.role)
    raw = yaml.safe_dump(plan, sort_keys=False).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    report.update(plan_sha256=hashlib.sha256(raw).hexdigest(), **summary(plan))
    args.output.with_suffix('.protocol.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'task_poses_deg'}, indent=2))


if __name__ == '__main__':
    main()
