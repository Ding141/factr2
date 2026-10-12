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
    if not np.isfinite(length) or length <= 0:
        raise ValueError('Invalid characteristic length')
    positions = np.asarray(positions, dtype=float).reshape(-1, 7)
    result = []
    for start in range(0, len(positions), 2048):
        j = model.jacobians(np.deg2rad(positions[start:start+2048]))
        j[:, :3] /= length
        s = np.linalg.svd(j.transpose(0, 2, 1), compute_uv=False)
        result.extend(float(a/b) if b > 1e-12 else None for a, b in zip(s[:, 0], s[:, -1]))
    return result


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
    revision = recipe.get('task_coverage')
    if revision is not None:
        if (set(revision) != {'shared_pose_seed', 'coupled_amplitude_deg',
                             'coupled_cycles', 'cartesian_repeats'}
                or type(revision['shared_pose_seed']) is not int
                or type(revision['coupled_cycles']) is not int
                or not 1 <= revision['coupled_cycles'] <= 10
                or type(revision['cartesian_repeats']) is not int
                or not 1 <= revision['cartesian_repeats'] <= cfg['repeats']):
            raise ValueError('Invalid task coverage recipe')
        amplitude = np.asarray(revision['coupled_amplitude_deg'], dtype=float)
        if amplitude.shape != (7,) or not np.isfinite(amplitude).all() or np.any(amplitude <= 0):
            raise ValueError('Invalid coupled amplitude')
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

    def append(curve, phase, kind, speed='slow', joint=None, conditioned=False, repetition=1):
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
            kind=kind, speed=speed, repetition=repetition, joint=joint,
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
    # Same deployment anchors, freshly recorded in independent sessions. Role
    # seeds still vary speed, direction and coupled trajectories, not coverage.
    pose_rng = np.random.default_rng(revision['shared_pose_seed']) if revision else rng
    attempts = 0
    while len(poses) < count:
        attempts += 1
        if attempts > 2000:
            raise ValueError('No conditioned pose set; reduce extent or change task bounds')
        p = pose_rng.uniform(pose_bounds[:, 0], pose_bounds[:, 1])
        if revision and np.min(np.max(abs(np.asarray(poses)-p), axis=1)) < 4.:
            continue
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
        if revision:
            # Concurrent joint excitation spans tracking errors and reversals
            # at combination poses; a single-home axis sweep cannot supply it.
            amp = np.minimum(amplitude, np.minimum(pose-bounds[:, 0], bounds[:, 1]-pose)*.8)
            for pattern in range(2):
                phase = rng.uniform(-np.pi, np.pi, 7)
                frequency = rng.integers(1, 3, 7)*revision['coupled_cycles']
                def coupled(s, amp=amp, phase=phase, frequency=frequency):
                    envelope = np.sin(np.pi*s)**2
                    return (pose[None, :] + envelope[:, None]*amp
                            * np.sin(2*np.pi*s[:, None]*frequency+phase))
                # Shrink excursions, never raise the conditioning threshold.
                for _ in range(12):
                    if task_path_ok(model, coupled(np.linspace(0, 1, 1001)), maximum):
                        break
                    amp *= .8
                else:
                    raise ValueError('No conditioned coupled path at task pose')
                if np.min(amp) < .5:
                    raise ValueError('Coupled path too small; change task anchors')
                for speed in ('slow', 'fast'):
                    append(coupled, 'cartesian', f'pose{i}_coupled_sine{pattern}', speed, conditioned=True)
                hold(f'static_pose{i}_after_coupled{pattern}')
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
            for repeat in range(1, (revision['cartesian_repeats'] if revision else 1)+1):
                for speed in ('slow', 'fast'):
                    suffix = f'_r{repeat}' if revision else ''
                    move(pose, f'pose{i}_reset_{kind}_{speed}{suffix}', conditioned=True)
                    append(curve, 'cartesian', f'pose{i}_{kind}{suffix}', speed,
                           conditioned=True, repetition=repeat)
                    move(pose, f'pose{i}_reset_after_{kind}_{speed}{suffix}', conditioned=True)
        hold(f'static_pose{i}_after_motion')
    # Validate the actual final transition too; do not silently return to zero.
    move(home, 'finish_home', conditioned=True)
    hold('static_home_finish')
    cfg['baseline_protocol'] = dict(version=1, role=role, seed=seed,
                                   max_condition=maximum, dwell_seconds=dwell)
    if revision:
        cfg['baseline_protocol'].update(version=2, shared_pose_seed=revision['shared_pose_seed'],
            task_poses_deg=np.asarray(poses).tolist(),
            recipe_sha256=hashlib.sha256(yaml.safe_dump(recipe, sort_keys=True).encode()).hexdigest())
    plan = dict(schema=SCHEMA, name=cfg['name'], side=cfg['side'], config=cfg,
        robot_description=description, model_sha256=model.sha256,
        joint_names=model.names, initial_deg=list(initial_deg), steps=steps)
    validate_plan(plan)
    durations = {'single_joint': 0., 'multi_joint': 0., 'static': 0., 'transition': 0.}
    for step in steps:
        group = ('static' if step['kind'].startswith('static_') else
                 'multi_joint' if step['phase'] == 'cartesian' else step['phase'])
        durations[group] += step['duration_seconds']
    return plan, dict(role=role, seed=seed, task_poses_deg=np.asarray(poses).tolist(),
        task_max_condition=maximum, approach_may_be_singular=True,
        joint_sweeps_may_be_singular=True, automatic_rezero=False,
        duration_by_group_seconds=durations,
        fraction_by_group={k:v/sum(durations.values()) for k,v in durations.items()},
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
