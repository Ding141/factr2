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
    for role, ds in (('train', train_ds), ('val', val_ds)):
        counts[role] = {}
        for (x, y), info in zip(ds.episodes, ds.episode_info):
            group = group_name(info)
            counts[role][group] = counts[role].get(group, 0)+len(x)-ds.history+1
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
        requirements=requirements, test_used=False, automatic_rezero=False,
        interpretation='Approximate static input matches are warnings; do not relabel torque or subtract bias')
