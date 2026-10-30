"""Audit measured training coverage before fitting the unchanged NEXT model.

Only train and validation enter this diagnostic. Near-identical static inputs
with different torque are an observability warning, not evidence of contact.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def group_name(info):
    if info.get('kind', '').startswith('static_'):
        return 'static'
    if info.get('family') in ('continuous', 'hysteresis'):
        count = len(info.get('active_joints', []))
        return 'single_joint' if count == 1 else 'multi_joint' if count > 1 else 'unlabelled'
    if info.get('phase') == 'cartesian':
        return 'multi_joint'
    return info.get('phase', 'unlabelled')


def audit_coverage(train_ds, val_ds, requirements):
    fractions = {key: float(requirements[key]) for key in (
        'minimum_static_fraction', 'minimum_multi_joint_fraction', 'maximum_single_joint_fraction')}
    distance_limit = float(requirements['max_validation_pose_distance_deg'])
    if (not np.isfinite(list(fractions.values())+[distance_limit]).all()
            or any(not 0 <= v <= 1 for v in fractions.values()) or distance_limit <= 0):
        raise ValueError('Invalid measured coverage requirements')
    counts = {}
    static = {'train': [], 'val': []}
    training_positions = []
    fast_velocity = {'train': [], 'val': []}
    fast_counts = {'train': 0, 'val': 0}
    for role, ds in (('train', train_ds), ('val', val_ds)):
        counts[role] = {}
        for (x, y), info in zip(ds.episodes, ds.episode_info):
            group = group_name(info)
            counts[role][group] = counts[role].get(group, 0)+len(x)-ds.history+1
            if info.get('speed') == 'fast' and group in ('single_joint', 'multi_joint'):
                fast_velocity[role].append(np.rad2deg(x[ds.history-1:, 7:14]))
                fast_counts[role] += len(x)-ds.history+1
            if role == 'train':
                training_positions.append(np.rad2deg(x[ds.history-1::25, :7]))
            if group == 'static':
                # Use the settled tail; a motion-to-hold transition must not
                # supply artificial directional memory for a long static hold.
                tail = x[max(ds.history-1, len(x)-500):].astype(float)
                static[role].append(dict(episode=info['episode'], path=info['path'], kind=info['kind'],
                    pose_deg=np.rad2deg(tail[:, :7].mean(0)).tolist(),
                    input_mean=tail.mean(0).tolist(), input_span=np.ptp(tail, axis=0).tolist(),
                    torque_mean_nm=y[-len(tail):].astype(float).mean(0).tolist()))
    ratios = {role:{k:v/sum(c.values()) for k,v in c.items()} for role,c in counts.items()}
    reasons = []
    speed_coverage = {}
    minimum_fast = float(requirements.get('minimum_fast_fraction', 0.))
    minimum_speed = float(requirements.get('minimum_fast_speed_p95_deg_s', 0.))
    minimum_direction_samples = requirements.get('minimum_fast_direction_samples', 0)
    if (not np.isfinite([minimum_fast, minimum_speed]).all()
            or not 0 <= minimum_fast <= 1 or minimum_speed < 0
            or type(minimum_direction_samples) is not int or minimum_direction_samples < 0):
        raise ValueError('Invalid measured fast coverage requirements')
    for role in ('train', 'val'):
        r = ratios[role]
        if r.get('static', 0) < fractions['minimum_static_fraction']:
            reasons.append(role+':insufficient_static')
        if r.get('multi_joint', 0) < fractions['minimum_multi_joint_fraction']:
            reasons.append(role+':insufficient_multi_joint')
        if r.get('single_joint', 0) > fractions['maximum_single_joint_fraction']:
            reasons.append(role+':single_joint_dominates')
        if r.get('unlabelled', 0):
            reasons.append(role+':missing_protocol_labels')
        velocities = np.vstack(fast_velocity[role]) if fast_velocity[role] else np.empty((0, 7))
        positive, negative, direction_samples = [], [], []
        for joint in range(7):
            values = velocities[:, joint]
            for sign, collection in ((1., positive), (-1., negative)):
                moving = sign*values[sign*values > .5]
                collection.append(float(np.percentile(moving, 95)) if len(moving) else 0.)
                direction_samples.append(int(np.sum(sign*values >= max(minimum_speed, .5))))
        fast_fraction = fast_counts[role]/sum(counts[role].values())
        speed_coverage[role] = dict(fast_window_fraction=fast_fraction,
            positive_speed_p95_deg_s=positive, negative_speed_p95_deg_s=negative,
            samples_at_required_speed_by_joint_and_direction=np.asarray(direction_samples).reshape(7, 2).tolist())
        if fast_fraction < minimum_fast:
            reasons.append(role+':insufficient_fast_windows')
        if min(positive+negative) < minimum_speed:
            reasons.append(role+':insufficient_measured_bidirectional_fast_speed')
        if min(direction_samples) < minimum_direction_samples:
            reasons.append(role+':insufficient_bidirectional_fast_duration')
    tree = cKDTree(np.vstack(training_positions))
    for info in static['val']:
        distance = float(tree.query(info['pose_deg'], p=np.inf)[0])
        info['nearest_train_max_joint_distance_deg'] = distance
        if distance > distance_limit:
            reasons.append('val:uncovered_static_pose:'+info['episode'])
    if not static['train'] or not static['val']:
        reasons.append('missing_static_blocks')
    # Tight physical tolerances, applied to all 21 input features and spans.
    # This is a diagnostic of potential ambiguity, not a proof of an error floor.
    tolerance = np.r_[np.full(7, np.deg2rad(.1)), np.full(7, .005),
                      np.full(7, np.deg2rad(.1))]
    ambiguous = []
    records = [(role, item) for role in ('train', 'val') for item in static[role]]
    for (role_a, a), (role_b, b) in itertools.combinations(records, 2):
        span = np.maximum(a['input_span'], b['input_span'])/tolerance
        difference = abs(np.asarray(a['input_mean'])-b['input_mean'])/tolerance
        gap = abs(np.asarray(a['torque_mean_nm'])-b['torque_mean_nm'])
        if np.max(span) <= 1 and np.max(difference) <= 1 and np.max(gap) > .2:
            ambiguous.append(dict(a=[role_a, a['path'], a['episode']],
                                  b=[role_b, b['path'], b['episode']], torque_gap_nm=gap.tolist()))
    return dict(passed=not reasons, reasons=reasons, window_counts=counts, window_fractions=ratios,
        validation_static_coverage=static['val'], potential_static_ambiguity=ambiguous,
        measured_speed_coverage=speed_coverage,
        requirements=requirements, test_used=False, automatic_rezero=False,
        interpretation='Approximate static input matches are warnings; do not relabel torque or subtract bias')
