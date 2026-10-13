#!/usr/bin/env python3
"""Build an audited training continuation; no ROS or motion publication."""
import copy
import json
from pathlib import Path

from factr2_next.data_collection.quality import sha256
from prepare_w3_baseline_campaign import collect_source, event_blocks
from w3_baseline_protocol import build_protocol
from w3_coverage_plan import load_coverage, validate_plan


def resume_protocol_groups(reference_path, parent_session, description, initial_deg, finish_home=True):
    """Recover complete frozen groups using data continuity, never model scores.

    Full poses are repeated when a member was interrupted, preserving the
    conditioning/return histories. The final home group is always repeated.
    Mechanical geometry must match; prepare additionally verifies actual arm
    gains/interfaces/calibration using its explicit arm_v2 profile scope.
    """
    import re
    import numpy as np
    from prepare_w3_baseline_campaign import event_streams
    from factr2_next.data_collection.campaign import geometry_sha256, arm_control_profile
    from w3_coverage_plan import ArmModel
    from w3_baseline_protocol import task_path_ok
    from w3_smooth_coverage import retime
    reference, _, digest = load_coverage(reference_path)
    role = reference['config']['baseline_protocol']['role']
    if geometry_sha256(description) != geometry_sha256(reference['robot_description']):
        raise ValueError('Recovery mechanical model changed')
    parents = parent_session if isinstance(parent_session, (list,tuple)) else [parent_session]
    candidates = []; excluded = {}; profiles = set()
    reference_steps = {s['name']:s for s in reference['steps']}
    for folder in parents:
        parent = Path(folder).resolve()
        original, _, _ = load_coverage(parent/'audit/motion.yaml')
        for field in ('home_deg','working_bounds_deg','tool_frame','baseline_protocol','speeds'):
            if original['config'][field] != reference['config'][field]:
                raise ValueError('Recovery reference config changed: '+field)
        if geometry_sha256(original['robot_description']) != geometry_sha256(description):
            raise ValueError('Recovery source mechanical model changed')
        for step in original['steps']:
            if step['phase'] != 'transition' and step != reference_steps.get(step['name']):
                raise ValueError('Recovery source critical event changed: '+step['name'])
        audit = json.loads((parent/'audit/motion_run.json').read_text())
        completed = [e['name'] for e in audit['events'] if e.get('status') == 'completed']
        source, meta, quality, run, _ = collect_source(parent, role, completed,
                                                      allow_evaluation_resume=role in ('val','test'))
        profiles.add(arm_control_profile(meta))
        streams = event_streams(source); usable = set()
        for event in run['events']:
            if event['phase'] == 'transition':
                continue
            try:
                pieces = event_blocks(source, meta, dict(events=[event]), original, streams=streams)
                if len(pieces) != 1:
                    raise ValueError('Event crosses a recording interruption')
                usable.add(event['name'])
            except ValueError as exc:
                excluded[str(parent)+'/'+event['name']] = str(exc)
        candidates.append(dict(parent_session=str(parent),parent_h5_sha256=quality['h5_sha256'],
                               usable=usable,events=[]))
    if len(profiles) != 1:
        raise ValueError('Recovery sources have different physical arm profiles')
    groups = {}
    for step in reference['steps']:
        match = re.match(r'(?:static_)?pose(\d+)_', step['kind'])
        group = ('pose'+match.group(1) if match else
                 'home_before' if step['kind'] in ('approach','static_home_before') else
                 'home_finish' if step['kind'] in ('finish_home','static_home_finish') else
                 step['kind']+'_'+step['speed'])
        groups.setdefault(group, []).append(step)
    model = ArmModel(description, reference['side'], reference['config']['tool_frame'])
    plan = copy.deepcopy(reference)
    plan.update(robot_description=description, model_sha256=model.sha256,
                initial_deg=list(initial_deg), steps=[])
    current = np.asarray(initial_deg, float); retained = []; recaptured = []
    cfg = plan['config']
    for group, steps in groups.items():
        critical = [s for s in steps if s['phase'] != 'transition']
        chosen = next((c for c in candidates if critical and all(s['name'] in c['usable'] for s in critical)),None)
        if chosen is not None and (group != 'home_finish' or not finish_home):
            retained.extend(s['name'] for s in critical)
            chosen['events'].extend(s['name'] for s in critical)
            continue
        recaptured.append(group)
        # Regenerate the original start-up approach from the actual held target.
        selected = steps[1:] if group == 'home_before' and steps[0]['kind'] == 'approach' else steps
        goal = np.asarray(selected[0]['positions_deg'][0])
        if np.max(np.abs(goal-current)) > 1e-8:
            start = current.copy()
            t, q = retime(lambda s:start[None,:]+s[:,None]*(goal-start),
                          cfg['speeds']['slow'], cfg['sample_period_seconds'])
            # Joint-space start-up may depart the singular zero pose. Task
            # connections thereafter retain the frozen conditioning limit.
            if plan['steps'] and not task_path_ok(model,q,cfg['baseline_protocol']['max_condition']):
                raise ValueError('Recovery task connection approaches singularity')
            label = 'resume_approach_'+group
            plan['steps'].append(dict(name=label,phase='transition',kind=label,
                speed='slow',repetition=0,joint=None,duration_seconds=float(t[-1]),
                hold_seconds=0.,times_seconds=t.tolist(),positions_deg=q.tolist()))
        plan['steps'].extend(copy.deepcopy(selected))
        current = np.asarray(selected[-1]['positions_deg'][-1])
    if plan['steps']:
        validate_plan(plan)
    return plan, dict(role=role, reference=str(Path(reference_path).resolve()),
        reference_sha256=digest,
        retained_sources=[{k:v for k,v in c.items() if k!='usable'} for c in candidates if c['events']],
        control_profile_sha256=next(iter(profiles)),
        events=retained,excluded_events=excluded,recaptured_groups=recaptured,
        geometry_sha256=geometry_sha256(description),raw_files_modified=False,
        evaluation_scores_inspected=False,control_profile_scope_required='arm_v2')


def resume_training_plan(recipe, description, initial_deg, parent_session, completed_speeds=('slow',)):
    parents = parent_session if isinstance(parent_session, (list, tuple)) else [parent_session]
    if not completed_speeds or set(completed_speeds)-{'slow', 'fast'}:
        raise ValueError('Invalid completed sweep speeds')
    full, report = build_protocol(recipe, description, initial_deg, 'train')
    reference = copy.deepcopy(full)
    signatures = set(); retained = []; excluded = {}; frozen_profile = None
    required = {(s['kind'], s['phase'], s['speed']) for s in full['steps'] if s['phase'] == 'single_joint'}
    for parent in parents:
        parent = Path(parent).resolve()
        original, _, _ = load_coverage(parent/'audit/motion.yaml')
        audit = json.loads((parent/'audit/motion_run.json').read_text())
        completed = [e['name'] for e in audit['events'] if e.get('status') == 'completed'
                     and e['phase'] == 'single_joint' and e['speed'] in completed_speeds]
        if not completed:
            continue
        source, meta, quality, selected_run, profile = collect_source(parent, 'train', completed)
        for field in ('home_deg', 'working_bounds_deg', 'tool_frame', 'baseline_protocol'):
            if original['config'][field] != full['config'][field]:
                raise ValueError('Resume config/profile mismatch: ' + field)
        if original['model_sha256'] != full['model_sha256']:
            raise ValueError('Resume robot profile mismatch')
        if frozen_profile is not None and profile != frozen_profile:
            raise ValueError('Retained sources have different control profiles')
        reusable = []
        for event in selected_run['events']:
            signature = (event['kind'], event['phase'], event['speed'])
            if original['config']['speeds'][event['speed']] != full['config']['speeds'][event['speed']]:
                raise ValueError('Resume selected speed profile mismatch: '+event['speed'])
            if signature not in required:
                raise ValueError('Retained sweep not present in full reference')
            if signature in signatures:
                continue
            try:
                pieces = event_blocks(source, meta, dict(events=[event]), original)
                if len(pieces) != 1:
                    raise ValueError('Sweep crosses a raw feedback interruption')
            except ValueError as exc:
                excluded[str(parent)+'/'+event['name']] = str(exc)
                continue
            signatures.add(signature); reusable.append(event['name'])
        if reusable:
            frozen_profile = profile
            retained.append(dict(parent_session=str(parent), parent_h5_sha256=quality['h5_sha256'],
                                 events=reusable))
    if not retained:
        raise ValueError('No complete continuous sweeps to reuse')
    full['steps'] = [s for s in full['steps'] if (s['kind'], s['phase'], s['speed']) not in signatures]
    validate_plan(full)  # Static/task blocks remain whole; validate their actual joins.
    report.update(retained_sources=retained, excluded_events=excluded,
        selected_completed_events=[name for source in retained for name in source['events']],
        raw_files_modified=False, failed_fast_block_reused=False,
        remaining_steps=len(full['steps']), reference_steps=len(reference['steps']))
    if len(retained) == 1:
        report.update(parent_session=retained[0]['parent_session'],parent_h5_sha256=retained[0]['parent_h5_sha256'])
    return full, reference, report


def resume_evaluation_plan(reference_path, parent_session, description, initial_deg):
    """Keep complete protocol groups; recollect interrupted groups without score selection."""
    import re
    import numpy as np
    from prepare_w3_baseline_campaign import event_streams
    from w3_baseline_protocol import task_path_ok
    from w3_coverage_plan import ArmModel
    from w3_smooth_coverage import retime
    reference, _, digest = load_coverage(reference_path)
    role = reference['config']['baseline_protocol']['role']
    if role not in ('val', 'test'):
        raise ValueError('Evaluation recovery requires frozen val/test reference')
    parent = Path(parent_session).resolve()
    if sha256(parent/'audit/motion.yaml') != digest:
        raise ValueError('Evaluation source must match the exact frozen reference')
    model = ArmModel(description, reference['side'], reference['config']['tool_frame'])
    if model.sha256 != reference['model_sha256']:
        raise ValueError('Evaluation recovery robot model changed')
    audit = json.loads((parent/'audit/motion_run.json').read_text())
    completed = [e['name'] for e in audit['events'] if e.get('status') == 'completed']
    source, meta, quality, run, _ = collect_source(parent,role,completed,allow_evaluation_resume=True)
    streams = event_streams(source); usable = set(); excluded = {}
    for event in run['events']:
        try:
            pieces = event_blocks(source,meta,dict(events=[event]),reference,streams=streams)
            if len(pieces) != 1:
                raise ValueError('Event crosses a raw feedback interruption')
            usable.add(event['name'])
        except ValueError as exc:
            excluded[event['name']] = str(exc)
    groups = {}
    for step in reference['steps']:
        kind = step['kind']; match = re.match(r'(?:static_)?pose(\d+)_',kind)
        if match:
            group = 'pose'+match.group(1)
        elif kind in ('approach','static_home_before'):
            group = 'home_before'
        elif kind in ('finish_home','static_home_finish'):
            group = 'home_finish'
        else:
            group = kind+'_'+step['speed']
        groups.setdefault(group,[]).append(step)
    plan = copy.deepcopy(reference); plan['initial_deg'] = list(initial_deg); plan['steps'] = []
    current = np.asarray(initial_deg,float); selected = []; recaptured = []
    cfg = plan['config']
    for group, steps in groups.items():
        critical = [s for s in steps if s['phase'] != 'transition']
        if critical and all(s['name'] in usable for s in critical):
            selected.extend(s['name'] for s in critical)
            continue
        recaptured.append(group)
        goal = np.asarray(steps[0]['positions_deg'][0])
        if np.max(np.abs(goal-current)) > 1e-8:
            start = current.copy()
            t,q = retime(lambda s:start[None,:]+s[:,None]*(goal-start),cfg['speeds']['slow'],cfg['sample_period_seconds'])
            if not task_path_ok(model,q,cfg['baseline_protocol']['max_condition']):
                raise ValueError('Recovery connection approaches singularity')
            plan['steps'].append(dict(name='resume_approach_'+group,phase='transition',
                kind='resume_approach_'+group,speed='slow',repetition=0,joint=None,
                duration_seconds=float(t[-1]),hold_seconds=0.,times_seconds=t.tolist(),positions_deg=q.tolist()))
        plan['steps'].extend(copy.deepcopy(steps))
        current = np.asarray(steps[-1]['positions_deg'][-1])
    if not selected or not plan['steps']:
        raise ValueError('No partial evaluation continuation is needed or reusable')
    validate_plan(plan)
    receipt = dict(role=role,reference=str(Path(reference_path).resolve()),reference_sha256=digest,
        parent_session=str(parent),parent_h5_sha256=quality['h5_sha256'],events=selected,
        excluded_events=excluded,recaptured_groups=recaptured,
        raw_files_modified=False,evaluation_scores_inspected=False)
    return plan, receipt
