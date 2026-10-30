"""Fit a frozen, causal free-motion correction with recording/cycle holdouts.

No ROS and no control APIs. No random overlapping-window splits. Explicit block
roles can be frozen before new physical collection. Historical tests previously
inspected remain retrospective, regardless of the new observation's split.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from factr2_next.data_collection.quality import write_json
from factr2_next.inference.kinematic_compensation import (
    SCHEMA, DEFAULT_SIGNAL, SignalProcessor, ResidualModel, LowPass, features, feature_names, digest,
    POSE_MODES, augment_position_features,
)

ROOT = Path(__file__).resolve().parents[1]


def series(stamps, x, measured, free, group, labels=None):
    processor = SignalProcessor()
    result = dict(t=np.asarray(stamps, np.int64), x=np.asarray(x), measured=measured, free=free,
                  groups=np.asarray(group), labels=np.asarray(labels) if labels is not None else np.full(len(x), 'historical'))
    full, target, eligible, position_filtered = [], [], [], []
    for t, xx, tau, mu in zip(stamps, x, measured, free):
        f = processor.update(int(t), xx[:7], xx[7:14], xx[:7]+xx[14:], tau, mu)
        full.append(features(f['actual'], f['command'], 'velocity_acceleration'))
        position_filtered.append(f['actual'][0])
        target.append(f['measured']-f['free']); eligible.append(f['ready'])
    result.update(features=np.asarray(full), target=np.asarray(target), eligible=np.asarray(eligible),
                  position_filtered=np.asarray(position_filtered))
    return result


def historical(parent, source, output):
    import run_w3_next_u_ablation as core
    cfg, manifest = core.verify(parent)
    mean_source = source/'mean_checkpoint'
    if digest(mean_source/'model.pt') != digest(parent/'results/seed_0/NEXT_MSE/model.pt'):
        raise ValueError('cached mean is not deployed mean')
    with np.load(mean_source/'normalization.npz') as f:
        y_std = f['y_std']
    result = {}
    for role in ('train', 'val', 'test'):
        ds = core.dataset(manifest, cfg, role)
        with np.load(ROOT/'data/right/next_u_frozen_sigma_20261023_v1/cache'/f'{role}.npz') as h:
            mean = h['mean_free_nm']; tau = mean+h['error_z']*y_std
        offset = 0; items = []
        import h5py
        for (x, _), info in zip(ds.episodes, ds.episode_info):
            x = x[49:]; n = len(x)
            with h5py.File(info['path']) as h:
                ts = h[info['episode']+'/joint_pos/timestamps'][49:]
            item = series(ts, x, tau[offset:offset+n], mean[offset:offset+n],
                          np.full(n, info['path']))
            offset += n; items.append(item)
        assert offset == len(mean)
        result[role] = items
        print(f'historical {role}: {offset} frames / {len(items)} episodes', flush=True)
    write_json(output/'historical_sources.json', manifest)
    return result


def recent(path, block_roles=None):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    def array(key):
        return np.array([[float(r[f'{key}_j{j}']) for j in range(1, 8)] for r in rows])
    q, v, cmd = [np.deg2rad(array(k)) for k in ('q_deg', 'velocity_deg_s', 'command_deg')]
    x = np.concatenate([q, v, cmd-q], axis=1)
    item = series([int(r['source_timestamp_ns']) for r in rows], x, array('measured_tau_nm'),
                  array('free_pred_tau_nm'), np.full(len(rows), str(path)), [r['exact_event'] for r in rows])
    t = item['t']; lab = item['labels']; valid = np.array([r['next_state']=='valid' for r in rows])
    role = np.full(len(rows), 'unused', dtype='<U8')
    for name in np.unique(lab):
        mask = lab == name
        # A full second on either side is excluded from fitting/selection/test.
        safe = mask & (t >= t[mask].min()+1_000_000_000) & (t <= t[mask].max()-1_000_000_000)
        if block_roles is not None:
            if name not in block_roles and name != 'unclassified':
                raise ValueError('observation event missing from frozen split: '+str(name))
            assignment = block_roles.get(name, 'unused')
            if assignment not in ('train', 'val', 'test', 'unused'):
                raise ValueError('invalid frozen split role: '+str(assignment))
            role[safe] = assignment
        elif name.startswith('cycle1') or name in ('baseline_before', 'pilot_out', 'pilot_return'):
            role[safe] = 'train'
        elif name.startswith('cycle2') or name == 'pilot_settle':
            role[safe] = 'val'
        elif name.startswith('cycle3') or name == 'baseline_after':
            role[safe] = 'test'
    role[~valid] = 'unused'; item['role'] = role
    if any(np.count_nonzero(role == split) < 100 for split in ('train', 'val', 'test')):
        raise ValueError('need at least 100 valid, purged frames in each observation split')
    return item


def smooth(values, t, hz, initial=None):
    f = LowPass(hz)
    result = []
    last = initial
    for value, stamp in zip(values, t):
        if last is not None and stamp-last > 40_000_000:
            f = LowPass(hz)
        dt = .01 if last is None else (int(stamp)-int(last))*1e-9
        result.append(f.update(value, dt)); last = stamp
    return np.asarray(result)


def predict(item, artifact):
    model = ResidualModel(artifact)
    n = len(model.scale)
    xx = (augment_position_features(item['features'][:, :70], item['position_filtered'], model.mode)
          if model.mode in POSE_MODES else item['features'][:, :n])
    q = item['position_filtered']
    return model.predict_features(xx,q,item['eligible'])


def train_artifact(items, mode, lam, source_hash, scope, acceleration_floor=.05):
    n = len(feature_names(mode))
    xx, yy, qq, groups, stationary = [], [], [], [], []
    for item, mask in items:
        mask = mask & item['eligible']
        xx.append(item['features'][mask, :n]); yy.append(item['target'][mask]); qq.append(item['x'][mask, :7]); groups.append(item['groups'][mask])
        static = mask & (item['labels']=='baseline_before')
        if static.any():stationary.append(item['features'][static, :n])
    x, y, q, groups = [np.concatenate(a) for a in (xx, yy, qq, groups)]
    # Noise floors prevent unexcited joints acquiring arbitrarily large weights.
    floors = np.repeat([.005, .005, .1, .1, .1, .1, .005, .005]+
                       ([acceleration_floor, acceleration_floor] if n==70 else []), 7)
    if n==70 and stationary:
        static = np.concatenate(stationary)[:, 56:]
        # Train-only stationary MAD raises the acceleration noise floor. Lower
        # physical excitation must not create a learned encoder-noise correction.
        sigma = 1.4826*np.median(abs(static-np.median(static,axis=0)),axis=0)
        floors[56:] = np.maximum(floors[56:], 3*sigma)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), floors)
    weights = np.zeros(len(x))
    for g in np.unique(groups):
        mask = groups == g; weights[mask] = 1/(len(np.unique(groups))*mask.sum())
    z = x/scale
    active = np.ptp(x, axis=0) >= 2*floors
    z[:, ~active] = 0.
    coef = np.linalg.solve((z*weights[:, None]).T@z + lam*np.eye(n), (z*weights[:, None]).T@y)
    caps = np.maximum(.05, 2*np.percentile(abs(y), 99, axis=0))
    return dict(schema=SCHEMA, feature_mode=mode, feature_names=feature_names(mode),
                signal=DEFAULT_SIGNAL, weights=coef.tolist(), feature_scale=scale.tolist(),
                feature_min=x.min(0).tolist(), feature_max=x.max(0).tolist(),
                position_bounds_rad=np.stack([q.min(0)-np.deg2rad(1), q.max(0)+np.deg2rad(1)], axis=1).tolist(),
                correction_cap_nm=caps.tolist(), enabled_joints=[True]*7,
                active_features=active.tolist(),
                excitation_floor=floors.tolist(),
                acceleration_floor_rule='max(configured floor, 3 * train-only stationary MAD sigma)',
                base_metadata_sha256=source_hash, scope=scope, ridge=lam,
                uses_torque_as_feature=False, online_learning=False, intercept=False,
                deployment_ready=False, uncertainty='not calibrated for compensated output')


def mse(items, artifact=None):
    sums = np.zeros(7); counts = 0
    for item, mask in items:
        mask = mask & item['eligible']
        error = item['target'] if artifact is None else item['target']-predict(item, artifact)[0]
        sums += np.sum(error[mask]**2, axis=0); counts += int(mask.sum())
    if not counts:
        raise ValueError('empty evaluation split')
    return sums/counts


def metrics(item, mask, artifact):
    mask = mask & item['eligible']; correction, scope = predict(item, artifact)
    original = item['measured']-item['free']
    output = smooth(item['target']-correction, item['t'], DEFAULT_SIGNAL['output_hz'])
    variants = dict(original=original, input_filtered=item['target'],
                    filter_only=smooth(item['target'], item['t'], DEFAULT_SIGNAL['output_hz']),
                    compensated_unfiltered_output=item['target']-correction, final=output)
    report = dict(rows=int(mask.sum()), compensation_applied_fraction=float(np.mean(scope[mask])), variants={})
    for name, values in variants.items():
        values = values[mask]
        report['variants'][name] = dict(rms_nm=np.sqrt(np.mean(values**2, axis=0)).tolist(),
            bias_nm=values.mean(0).tolist(), absolute_p95_nm=np.percentile(abs(values),95,axis=0).tolist(),
            absolute_peak_nm=abs(values).max(0).tolist())
    return report, variants, correction, scope


def contact_transfer():
    t = np.arange(500)*10_000_000+1
    pulse = np.zeros((500,7)); pulse[100:300] = .3
    response = smooth(smooth(pulse,t,DEFAULT_SIGNAL['input_torque_hz']),t,DEFAULT_SIGNAL['output_hz'])
    rise = response[100:300,0]/.3
    return dict(injection_nm=.3, fixed_kinematics=True, physical_contact_validation=False,
                first_10pct_ms=int(np.flatnonzero(rise>=.1)[0])*10,
                rise_90pct_ms=int(np.flatnonzero(rise>=.9)[0])*10,
                settled_gain=float(response[299,0]/.3),
                response_at_20ms=float(response[102,0]/.3))


def run(args):
    output = args.output.resolve(); output.mkdir(exist_ok=False,parents=True)
    started = time.monotonic(); source_hash = digest(args.base/'metadata.json')
    split_manifest = getattr(args, 'split_manifest', None)
    split = json.loads(split_manifest.read_text()) if split_manifest else None
    if split is not None and not isinstance(split.get('block_roles'), dict):
        raise ValueError('split manifest must contain a frozen block_roles mapping')
    if split is not None:
        checked = json.loads((args.observation/'label_report.json').read_text())
        if (checked['plan_sha256'] != split['plan_sha256']
                or checked['split_manifest_sha256'] != digest(split_manifest)
                or checked['labeled_sha256'] != digest(args.observation/'labeled_samples.csv')):
            raise ValueError('labeled observation and frozen split identity differ')
    new = recent(args.observation/'labeled_samples.csv', split['block_roles'] if split else None)
    hist = historical(args.parent, args.base, output)
    mask = lambda role: new['role']==role
    frozen = dict(schema=SCHEMA, base=str(args.base.resolve()), base_metadata_sha256=source_hash,
        observation=str(args.observation.resolve()), observation_sha256=digest(args.observation/'labeled_samples.csv'),
        split=('recording split retained for historical data; predeclared whole physical block roles; 1s boundary purge'
               if split else 'recording split retained for historical data; cycle1 train / cycle2 validation / cycle3 retrospective test; 1s boundary purge'),
        signal=DEFAULT_SIGNAL, recipe='no intercept, no measured torque/residual as feature, no online adaptation',
        candidate_ridge=[.0001,.001,.01,.1,1.], scopes=['pooled','local_pose'],
        test_role=('new physical test blocks reserved before collection; historical tests retrospective; no contact ground truth'
                   if split else 'previously inspected; not a blind contact or generalization test'),
        automatic_robot_motion=False)
    acceleration_floor = getattr(args, 'acceleration_floor', .05)
    if not np.isfinite(acceleration_floor) or not .001<=acceleration_floor<=.5:
        raise ValueError('acceleration feature floor must be 0.001..0.5 rad/s²')
    frozen['acceleration_feature_floor_rad_s2'] = acceleration_floor
    if split_manifest:
        frozen['split_manifest'] = str(split_manifest.resolve())
        frozen['split_manifest_sha256'] = digest(split_manifest)
        frozen['plan_sha256'] = split.get('plan_sha256')
        frozen['block_roles'] = split['block_roles']
        write_json(output/'frozen_split.json', split)
    write_json(output/'recipe.json',frozen)
    val_local=[(new,mask('val'))]; val_old=[(i,np.ones(len(i['t']),bool)) for i in hist['val']]
    base_local=mse(val_local); base_old=mse(val_old)
    candidates=[]; modes={}
    for mode in ('velocity','velocity_acceleration'):
        for scope in ('pooled','local_pose'):
            train=[(new,mask('train'))]
            if scope=='pooled':train += [(i,np.ones(len(i['t']),bool)) for i in hist['train']]
            for lam in frozen['candidate_ridge']:
                artifact=train_artifact(train,mode,lam,source_hash,scope,acceleration_floor)
                local=mse(val_local,artifact); old=mse(val_old,artifact)
                enabled=(local<=.98*base_local)&(old<=1.05*base_old)
                artifact['enabled_joints']=enabled.tolist()
                score=float(np.mean(np.where(enabled,local/np.maximum(base_local,1e-8),1.)))
                candidate=dict(feature_mode=mode,scope=scope,ridge=lam,score=score,enabled=enabled.tolist(),
                               validation_local_rms=np.sqrt(local).tolist(),validation_historical_rms=np.sqrt(old).tolist())
                candidates.append((score,artifact,candidate))
                print('candidate',mode,scope,lam,'score',round(score,4),'enabled',enabled.tolist(),flush=True)
        choices=[v for v in candidates if v[1]['feature_mode']==mode]
        modes[mode]=min(choices,key=lambda v:v[0])[1]
    score,artifact,selected=min(candidates,key=lambda v:v[0])
    if not any(artifact['enabled_joints']):
        raise RuntimeError('No correction passed held-out validation; do not activate learned correction')
    artifact['selection']=selected; artifact['validation_only_selection']=True
    write_json(output/'model.json',artifact)
    for mode,a in modes.items():write_json(output/(mode+'.json'),a)
    comparison=dict(recipe=frozen,selected=selected,candidates=[v[2] for v in candidates],
                    contact_transfer=contact_transfer(),splits={},elapsed_seconds=time.monotonic()-started)
    for role in ('train','val','test'):
        r,variants,corr,scope=metrics(new,mask(role),artifact)
        r['velocity_only']=metrics(new,mask(role),modes['velocity'])[0]['variants']
        r['velocity_acceleration']=metrics(new,mask(role),modes['velocity_acceleration'])[0]['variants']
        comparison['splits']['recent_'+role]=r
        hs=[(i,np.ones(len(i['t']),bool)) for i in hist[role]]
        comparison['splits']['historical_'+role]=dict(input_filtered_rms_nm=np.sqrt(mse(hs)).tolist(),
            compensated_rms_nm=np.sqrt(mse(hs,artifact)).tolist(),recording_count=len({str(i['groups'][0]) for i in hist[role]}))
    r,variants,corr,scope=metrics(new,np.ones(len(new['t']),bool),artifact)
    np.savez_compressed(output/'replay.npz',timestamp_ns=new['t'],roles=new['role'],labels=new['labels'],
                        correction_nm=corr,supported=scope,eligible=new['eligible'],**variants)
    write_json(output/'comparison.json',comparison)
    files = ['model.json','recipe.json','comparison.json','replay.npz','velocity.json','velocity_acceleration.json']
    if split_manifest:files.append('frozen_split.json')
    write_json(output/'manifest.json',dict(schema=SCHEMA,files={n:digest(output/n) for n in files},
               base_metadata_sha256=source_hash,deployment_ready=False))
    print(json.dumps(dict(output=str(output),selected=selected,elapsed_seconds=comparison['elapsed_seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,default=ROOT/'log/reciprocal_frozen_sigma_20261023_v3')
    p.add_argument('--parent',type=Path,default=ROOT/'data/right/next_u_ablation_20261023_v1')
    p.add_argument('--base',type=Path,default=ROOT/'models/w3/right/frozen_sigma_observer_20261023')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-manifest',type=Path,
                   help='Predeclared capture test_spec.json with exact event-to-role mapping')
    p.add_argument('--acceleration-floor',type=float,default=.05,
                   help='Minimum acceleration feature noise floor in rad/s²; stationary training noise may raise it')
    run(p.parse_args())
