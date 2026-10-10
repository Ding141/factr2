"""Offline URDF kinematics and reproducible free-space coverage planning (no ROS)."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

SCHEMA = 'w3_coverage_motion_v1'


def vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{label}: expected {length} numbers')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f'{label}: expected finite numbers')
    return np.asarray(value, dtype=float)


def scalar(value, low, high, label):
    result = vector([value], 1, label)[0]
    if not low <= result <= high:
        raise ValueError(f'{label}: expected {low}..{high}')
    return float(result)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


class ArmModel:
    def __init__(self, description, side, tool_frame):
        self.description = description
        self.sha256 = hashlib.sha256(description.encode()).hexdigest()
        self.names = [f'{side}_joint_{i}' for i in range(7)]
        root = ET.fromstring(description)
        joints = root.findall('joint')
        by_child = {j.find('child').get('link'): j for j in joints}
        if len(by_child) != len(joints):
            raise ValueError('URDF has duplicate child links')
        chain, visited, child = [], set(), tool_frame
        if child not in {link.get('name') for link in root.findall('link')}:
            raise ValueError(f'No tool frame {child}')
        while child in by_child:
            if child in visited:
                raise ValueError('Cyclic URDF chain')
            visited.add(child)
            joint = by_child[child]
            chain.append(joint)
            child = joint.find('parent').get('link')
        self.chain = []
        movable = []
        for joint in reversed(chain):
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite(np.r_[xyz, rpy]).all():
                raise ValueError('Invalid URDF origin')
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
            index, axis = None, None
            if joint.get('type') != 'fixed':
                if joint.get('name') not in self.names or joint.get('type') != 'revolute':
                    raise ValueError('Tool chain must contain exactly the selected seven revolute joints')
                index = self.names.index(joint.get('name'))
                movable.append(index)
                element = joint.find('axis')
                axis = np.fromstring(element.get('xyz', '1 0 0'), sep=' ') if element is not None else np.array([1., 0, 0])
                if len(axis) != 3 or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
                    raise ValueError('Invalid URDF axis')
                axis = axis / np.linalg.norm(axis)
            self.chain.append((transform, index, axis))
        if sorted(movable) != list(range(7)):
            raise ValueError('Tool chain does not contain seven unique arm joints')
        limits = []
        for name in self.names:
            hardware = root.findall(f'ros2_control/joint[@name="{name}"]')
            if len(hardware) != 1:
                raise ValueError(f'{name}: require expanded live URDF with ros2_control limits')
            params = {p.get('name'): p.text for p in hardware[0].findall('param')}
            bounds = [float(params[k]) for k in ('urdf_lower', 'urdf_upper')]
            geometric = next(j for j in joints if j.get('name') == name).find('limit')
            if geometric is not None:
                bounds = [max(bounds[0], float(geometric.get('lower'))), min(bounds[1], float(geometric.get('upper')))]
            if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f'{name}: invalid limits')
            limits.append(bounds)
        self.limits_deg = np.rad2deg(limits)

    def fk(self, q, jacobian=False):
        q = np.asarray(q, dtype=float)
        transform, origins, axes = np.eye(4), {}, {}
        for origin, index, axis in self.chain:
            transform = transform @ origin
            if index is not None:
                origins[index] = transform[:3, 3].copy()
                axes[index] = transform[:3, :3] @ axis
                rot = np.eye(4)
                rot[:3, :3] = rotation(axis, q[index])
                transform = transform @ rot
        if not jacobian:
            return transform
        j = np.zeros((6, 7))
        for i in range(7):
            j[:3, i] = np.cross(axes[i], transform[:3, 3] - origins[i])
            j[3:, i] = axes[i]
        return transform, j

    def ik_position(self, position, seed, bounds):
        # Position-only IK: orientation is allowed to vary, as in Cartesian-like motion.
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(150):
            pose, jacobian = self.fk(q, True)
            error = position - pose[:3, 3]
            if np.linalg.norm(error) < .0000001:
                return q
            j = jacobian[:3]
            delta = j.T @ np.linalg.solve(j @ j.T + .002**2 * np.eye(3), error)
            delta *= min(1., .025 / max(np.max(np.abs(delta)), 1e-12))
            improved = False
            for scale in (1., .5, .25, .1):
                candidate = np.clip(q + scale * delta, bounds[:, 0], bounds[:, 1])
                if np.linalg.norm(position - self.fk(candidate)[:3, 3]) < np.linalg.norm(error):
                    q, improved = candidate, True
                    break
            if not improved:
                break
        raise ValueError('Cartesian path unreachable inside working bounds; reduce extent or change home/bounds')


def validate_config(config, model):
    required = {'version', 'name', 'side', 'tool_frame', 'home_deg', 'working_bounds_deg',
                'repeats', 'sample_period_seconds', 'hold_seconds', 'hardware_margin_deg',
                'arrival_tolerance_deg', 'start_tolerance_deg', 'tracking_limit_deg',
                'speeds', 'cartesian'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'trajectory_profile', 'finish_at_home', 'continuous_repetitions'} or type(config['version']) is not int or config['version'] != 1:
        raise ValueError('Invalid coverage configuration keys/version')
    if type(config.get('continuous_repetitions',False)) is not bool:
        raise ValueError('continuous_repetitions must be boolean')
    if type(config.get('finish_at_home',False)) is not bool:
        raise ValueError('finish_at_home must be boolean')
    if config['side'] not in ('left', 'right') or not isinstance(config['name'], str) or not config['name'].strip():
        raise ValueError('Invalid side/name')
    home = vector(config['home_deg'], 7, 'home')
    if not isinstance(config['working_bounds_deg'], list) or len(config['working_bounds_deg']) != 7:
        raise ValueError('working_bounds_deg requires seven pairs')
    bounds = np.array([vector(row, 2, 'working bounds') for row in config['working_bounds_deg']])
    margin = scalar(config['hardware_margin_deg'], .5, 15, 'hardware margin')
    if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(home < bounds[:, 0]) or np.any(home > bounds[:, 1]):
        raise ValueError('Home must lie within nonempty working bounds')
    if np.any(bounds[:, 0] < model.limits_deg[:, 0] + margin) or np.any(bounds[:, 1] > model.limits_deg[:, 1] - margin):
        raise ValueError('Working bounds exceed live hardware/model limits with margin')
    if type(config['repeats']) is not int or not 1 <= config['repeats'] <= 20:
        raise ValueError('repeats must be integer 1..20')
    profile = config.get('trajectory_profile', 'legacy')
    if profile not in ('legacy', 'continuous_sine'):
        raise ValueError('Unknown trajectory profile')
    for key, low, high in [('sample_period_seconds', .01, .05), ('hold_seconds', 0 if profile == 'continuous_sine' else .5, 10),
                          ('arrival_tolerance_deg', .05, 2), ('start_tolerance_deg', .05, 2),
                          ('tracking_limit_deg', .5, 5)]:
        scalar(config[key], low, high, key)
    if not isinstance(config['speeds'], dict) or set(config['speeds']) != {'slow', 'fast'}:
        raise ValueError('speeds requires slow and fast')
    for speed in config['speeds'].values():
        if not isinstance(speed, dict) or set(speed) != {'velocity_deg_s', 'acceleration_deg_s2'}:
            raise ValueError('Each speed requires velocity and acceleration')
        vector(speed['velocity_deg_s'], 7, 'velocity')
        vector(speed['acceleration_deg_s2'], 7, 'acceleration')
        for key, maximum in [('velocity_deg_s', 30), ('acceleration_deg_s2', 100)]:
            if any(not .1 <= value <= maximum for value in speed[key]):
                raise ValueError(f'{key} exceeds conservative tool bounds .1..{maximum}')
    if np.any(np.array(config['speeds']['fast']['velocity_deg_s']) <= config['speeds']['slow']['velocity_deg_s']):
        raise ValueError('Every fast velocity limit must exceed slow')
    cart = config['cartesian']
    if not isinstance(cart, dict) or set(cart) != {'line_half_extent_m', 'circle_radius_m', 'waypoint_spacing_m'}:
        raise ValueError('Invalid Cartesian configuration')
    scalar(cart['line_half_extent_m'], .001, .05, 'line extent')
    scalar(cart['circle_radius_m'], .001, .05, 'circle radius')
    scalar(cart['waypoint_spacing_m'], .0002, .002, 'Cartesian waypoint spacing')
    return home, bounds


def smooth_path(path, speed, dt):
    """Quintic time scaling and discrete limits matching controller linear interpolation."""
    path = np.asarray(path)
    velocity = np.asarray(speed['velocity_deg_s'])
    acceleration = np.asarray(speed['acceleration_deg_s2'])
    distance = np.sum(np.abs(np.diff(path, axis=0)), axis=0)
    duration = max(.5, np.max(1.875 * distance / velocity), np.sqrt(np.max(6 * distance / acceleration)))
    for _ in range(20):
        count = int(math.ceil(duration / dt))
        duration = count * dt
        t = np.arange(count + 1) * dt
        u = t / duration
        s = 10*u**3 - 15*u**4 + 6*u**5
        x = np.linspace(0, 1, len(path))
        q = np.column_stack([np.interp(s, x, path[:, i]) for i in range(7)])
        v = np.diff(q, axis=0) / dt
        a = np.diff(np.vstack([np.zeros(7), v, np.zeros(7)]), axis=0) / dt
        ratio = max(np.max(np.abs(v) / velocity), np.sqrt(np.max(np.abs(a) / acceleration)))
        if ratio <= 1.000001:
            return t, q
        duration *= ratio * 1.03
    raise ValueError('Cannot retime Cartesian path to velocity/acceleration limits')


def build_plan(config, description, initial_deg):
    if config.get('trajectory_profile') == 'continuous_sine':
        from w3_smooth_coverage import build_smooth_plan
        return build_smooth_plan(config, description, initial_deg)
    model = ArmModel(description, config['side'], config['tool_frame'])
    home, bounds = validate_config(config, model)
    initial = vector(initial_deg, 7, 'initial command')
    margin = config['hardware_margin_deg']
    if np.any(initial < model.limits_deg[:, 0] + margin) or np.any(initial > model.limits_deg[:, 1] - margin):
        raise ValueError('Initial command outside hardware margin')
    steps = []
    current = initial

    def append(path, phase, kind, speed, repetition, joint=None):
        nonlocal current
        t, q = smooth_path(path, config['speeds'][speed], config['sample_period_seconds'])
        step = {'name': f'{len(steps)+1:03d}_{phase}_{kind}_{speed}_{repetition}', 'phase': phase,
                'kind': kind, 'speed': speed, 'repetition': repetition, 'joint': joint,
                'duration_seconds': float(t[-1]), 'hold_seconds': config['hold_seconds'],
                'times_seconds': t.tolist(), 'positions_deg': q.tolist()}
        steps.append(step)
        current = q[-1]

    append([initial, home], 'transition', 'approach', 'slow', 0)
    # All single-joint sweeps precede all multi-joint paths; each speed covers both directions.
    for speed in ('slow', 'fast'):
        for repeat in range(1, config['repeats'] + 1):
            for joint in range(7):
                for endpoint, label in [(bounds[joint, 0], 'low'), (bounds[joint, 1], 'high'), (home[joint], 'home')]:
                    goal = home.copy()
                    goal[joint] = endpoint
                    append([current, goal], 'single_joint', label, speed, repeat, joint)
    center = model.fk(np.deg2rad(home))[:3, 3]
    cart = config['cartesian']
    paths = []
    for axis in range(3):
        n = int(math.ceil(2 * cart['line_half_extent_m'] / cart['waypoint_spacing_m'])) + 1
        legs = []
        for start, end in [(0, -cart['line_half_extent_m']),
                           (-cart['line_half_extent_m'], cart['line_half_extent_m']),
                           (cart['line_half_extent_m'], 0)]:
            points = np.repeat(center[None, :], n, axis=0)
            points[:, axis] += np.linspace(start, end, n)
            legs.append(points)
        paths.append((f'line_{"xyz"[axis]}', legs))
    radius = cart['circle_radius_m']
    # Circle in world XY with its initial/final point at home (center shifted by -radius).
    angles = np.linspace(0, 2 * np.pi, int(math.ceil(2*np.pi*radius/cart['waypoint_spacing_m'])) + 1)
    points = np.repeat(center[None, :], len(angles), axis=0)
    points[:, 0] += radius * (np.cos(angles) - 1)
    points[:, 1] += radius * np.sin(angles)
    paths.append(('circle_xy', [points]))
    solutions = []
    for kind, legs in paths:
        seed = np.deg2rad(home)
        resolved_legs = []
        for points in legs:
            q = [np.rad2deg(seed)]
            for position in points[1:]:
                seed = model.ik_position(position, seed, np.deg2rad(bounds))
                q.append(np.rad2deg(seed))
            resolved_legs.append(np.asarray(q))
        solutions.append((kind, resolved_legs))
    for speed in ('slow', 'fast'):
        for repeat in range(1, config['repeats'] + 1):
            for kind, legs in solutions:
                for leg, path in enumerate(legs, 1):
                    append(path, 'cartesian', f'{kind}_leg{leg}', speed, repeat)
                # Return to identical joint home, including the IK null-space drift.
                append([current, home], 'transition', 'reset_home', speed, repeat)
    append([current, initial], 'transition', 'return_initial', 'slow', 0)
    plan = {'schema': SCHEMA, 'name': config['name'], 'side': config['side'],
            'config': config, 'robot_description': description, 'model_sha256': model.sha256,
            'joint_names': model.names, 'initial_deg': initial.tolist(), 'steps': steps}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'schema', 'name', 'side', 'config', 'robot_description',
                                                'model_sha256', 'joint_names', 'initial_deg', 'steps'} or plan['schema'] != SCHEMA:
        raise ValueError('Invalid coverage plan schema/keys')
    config = plan['config']
    model = ArmModel(plan['robot_description'], plan['side'], config['tool_frame'])
    home, bounds = validate_config(config, model)
    if plan['model_sha256'] != model.sha256 or plan['joint_names'] != model.names or plan['side'] != config['side'] or plan['name'] != config['name']:
        raise ValueError('Model hash, names, side or name mismatch')
    previous = vector(plan['initial_deg'], 7, 'initial')
    steps = plan['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 1000:
        raise ValueError('Invalid number of steps')
    seen = set()
    continuous = config.get('trajectory_profile') == 'continuous_sine'
    previous_velocity = np.zeros(7)
    previous_acceleration = np.zeros(7)
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != {'name', 'phase', 'kind', 'speed', 'repetition', 'joint', 'duration_seconds', 'hold_seconds', 'times_seconds', 'positions_deg'}:
            raise ValueError('Invalid step fields')
        if not isinstance(step['name'], str) or not step['name'] or step['name'] in seen:
            raise ValueError('Duplicate/invalid step name')
        seen.add(step['name'])
        if step['phase'] not in ('transition', 'single_joint', 'cartesian') or step['speed'] not in config['speeds']:
            raise ValueError('Invalid phase/speed')
        if type(step['repetition']) is not int or not 0 <= step['repetition'] <= config['repeats']:
            raise ValueError('Invalid repetition')
        if step['joint'] is not None and (type(step['joint']) is not int or not 0 <= step['joint'] <= 6):
            raise ValueError('Invalid joint index')
        scalar(step['hold_seconds'], 0 if config.get('trajectory_profile') == 'continuous_sine' else .5, 10, 'step hold')
        scalar(step['duration_seconds'], .1, 600, 'step duration')
        if not isinstance(step['times_seconds'], list) or not 2 <= len(step['times_seconds']) <= 60001:
            raise ValueError('Invalid time samples')
        t = vector(step['times_seconds'], len(step['times_seconds']), 'times')
        if not isinstance(step['positions_deg'], list) or len(step['positions_deg']) != len(t):
            raise ValueError('Position/time count mismatch')
        q = np.array([vector(row, 7, 'position') for row in step['positions_deg']])
        dt = np.diff(t)
        if abs(t[0]) > 1e-9 or abs(t[-1] - step['duration_seconds']) > 1e-7 or np.any(np.abs(dt - config['sample_period_seconds']) > 1e-7):
            raise ValueError('Invalid trajectory time grid/duration')
        if np.max(np.abs(q[0] - previous)) > 1e-6:
            raise ValueError('Discontinuous step boundary')
        margin = config['hardware_margin_deg']
        if np.any(q < model.limits_deg[:, 0] + margin) or np.any(q > model.limits_deg[:, 1] - margin):
            raise ValueError('Trajectory exceeds hardware/model margin')
        if step['phase'] != 'transition' and (np.any(q < bounds[:, 0] - 1e-8) or np.any(q > bounds[:, 1] + 1e-8)):
            raise ValueError('Trajectory exceeds working bounds')
        v = np.diff(q, axis=0) / dt[:, None]
        # Logical block boundaries need not be stationary in a continuous recording.
        if continuous:
            if step_index+1<len(steps):
                next_q=np.asarray(steps[step_index+1]['positions_deg'][:2])
                next_velocity=(next_q[1]-next_q[0])/config['sample_period_seconds']
            else:next_velocity=np.zeros(7)
            a=np.diff(np.vstack([previous_velocity,v,next_velocity]),axis=0)/config['sample_period_seconds']
        else:
            a=np.diff(np.vstack([np.zeros(7),v,np.zeros(7)]),axis=0)/config['sample_period_seconds']
        speed = config['speeds'][step['speed']]
        if np.any(np.abs(v) > np.array(speed['velocity_deg_s']) + 1e-5) or np.any(np.abs(a) > np.array(speed['acceleration_deg_s2']) + 1e-5):
            raise ValueError('Trajectory exceeds velocity/acceleration limit')
        if config.get('trajectory_profile') == 'continuous_sine':
            jerk=np.diff(np.vstack([previous_acceleration,a]),axis=0)/config['sample_period_seconds']
            previous_acceleration=a[-2] if len(a)>1 else a[-1]
            previous_velocity=v[-1]
            if np.max(np.abs(jerk))>120.0001:
                raise ValueError('Smooth trajectory exceeds discrete jerk limit')
        if step['phase'] == 'single_joint':
            if step['joint'] is None or np.max(np.abs(np.delete(q - home, step['joint'], axis=1))) > 1e-7:
                raise ValueError('Single-joint sweep moves other joints')
        previous = q[-1]
    return model


def load_coverage(path):
    raw = Path(path).read_bytes()
    plan = yaml.load(raw, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    validate_plan(plan)
    return plan, raw, hashlib.sha256(raw).hexdigest()


def load_capture_motion(path):
    # Keep old small-motion limits unchanged; dispatch solely by explicit schema.
    raw = Path(path).read_bytes()
    parsed = yaml.load(raw, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    if isinstance(parsed, dict) and parsed.get('schema') == SCHEMA:
        return load_coverage(path)
    from w3_motion_sequence import load_motion
    return load_motion(path)


def summary(plan):
    positions = np.vstack([s['positions_deg'] for s in plan['steps']])
    return {'name': plan['name'], 'side': plan['side'], 'steps': len(plan['steps']),
            'duration_seconds': sum(s['duration_seconds'] + s['hold_seconds'] for s in plan['steps']),
            'planned_min_deg': positions.min(axis=0).tolist(), 'planned_max_deg': positions.max(axis=0).tolist(),
            'working_bounds_deg': plan['config']['working_bounds_deg'],
            'model_sha256': plan['model_sha256']}
