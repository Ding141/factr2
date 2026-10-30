#!/usr/bin/env python3
"""Prepare and run an offline NEXT versus NEXT+MD controlled comparison."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import time

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from factr2_next.data_collection.quality import validate_manifest, sha256, write_json
from factr2_next.data_collection.campaign import validate_independent_manifest
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.models import build_model
from factr2_next.training.motion_discriminator import stationary_floor, parent_features, MDDataset
from factr2_next.training.train import set_seed, run_epoch
from factr2_next.training.validation_metrics import validation_epoch


def check_parent_isolation(reports):
    parents, hashes = {}, {}
    for role, items in reports.items():
        for item in items:
            derivation = item['metadata']['derivation']
            for registry, key in ((parents, 'parent_session_id'), (hashes, 'parent_h5_sha256')):
                identity = derivation[key]
                if identity in registry:
                    raise ValueError('md_duplicate_parent:'+identity)
                registry[identity] = role
    return dict(parent_session_roles=parents, parent_hash_roles=hashes)


def paired_initial_model(model_cfg):
    """Match all shared initial weights and the baseline's shuffle/dropout RNG.

    The historical baseline used the global seed for initialization and its
    DataLoader. Extra input weights must not silently change its batch ordering.
    """
    original = build_model(model_cfg,21,7,50)
    rng_after_original = torch.get_rng_state()
    augmented = build_model(model_cfg,126,7,50)
    target = augmented.state_dict()
    for name, value in original.state_dict().items():
        if name == 'lstm.weight_ih_l0':
            target[name][:,:21].copy_(value)
        else:
            if target[name].shape != value.shape:raise ValueError('md_common_weight_shape:'+name)
            target[name].copy_(value)
    augmented.load_state_dict(target)
    torch.set_rng_state(rng_after_original)
    return augmented


def prepare(manifest_path, baseline_run, output):
    manifest_path, baseline_run, output = [Path(p).resolve() for p in (manifest_path, baseline_run, output)]
    if output.exists():
        raise ValueError('Never overwrite an MD experiment')
    baseline = load_checkpoint(baseline_run, expected={'side':'right', 'history':50, 'sample_hz':100})
    if baseline.metadata['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('md_baseline_manifest_differs')
    cfg = copy.deepcopy(baseline.config)
    if cfg['train']['seed'] != 0 or cfg['data']['history'] != 50:
        raise ValueError('This paired comparison is fixed to baseline seed 0 and H=50')
    manifest, reports = validate_manifest(str(manifest_path), dict(side='right',
        joint_order=cfg['joint_names'], contract_version=cfg['contract_version'], sample_hz=100))
    validate_independent_manifest(manifest, reports)
    parents = check_parent_isolation(reports)
    package = Path(__import__('factr2_next').__file__).resolve().parent
    # Reusing an already completed baseline is permitted only with unchanged
    # shared model, data, optimizer/loss and diagnostic implementations.
    shared = {}
    for name in ('models.py', 'dataset.py', 'train.py', 'validation_metrics.py'):
        actual = sha256(package/'training'/name)
        if actual != baseline.metadata['training_source_files_sha256'][name]:
            raise ValueError('md_baseline_shared_source_changed:'+name)
        shared[name] = actual
    output.mkdir(parents=True)
    shutil.copy2(manifest_path, output/'split.json')
    # No validation/test outcomes participate in this calibration.
    recipe = stationary_floor(manifest['splits']['train'])
    write_json(output/'md_recipe.json', recipe)
    frozen_recipe_hash = sha256(output/'md_recipe.json')
    cache_path = output/'md_features.h5'
    provenance = {}
    with h5py.File(cache_path, 'w') as cache:
        cache.attrs.update(schema='next_md_cache_v1', manifest_sha256=sha256(manifest_path),
                           recipe_sha256=frozen_recipe_hash)
        for role in ('train', 'val', 'test'):
            for entry in manifest['splits'][role]:
                values, info = parent_features(entry, recipe['thresholds_rad_s'], recipe['max_gap_ns'])
                group = cache.create_group(entry['h5_sha256'])
                for ep, value in values.items():
                    group.create_dataset(ep, data=value, compression='lzf')
                provenance[entry['path']] = dict(role=role, **info)
    cfg['data']['manifest'] = str(output/'split.json')
    cfg['train']['stop_file'] = str(output/'STOP_TRAINING')
    cfg['comparison'] = dict(schema='next_md_ablation_v1', baseline_run=str(baseline_run),
        baseline_model_sha256=sha256(baseline_run/'model.pt'), baseline_reused=True,
        feature_cache=str(cache_path), recipe=str(output/'md_recipe.json'),
        feature_change='append 15 threshold-major MD velocity memories (105 values) to original 21 inputs',
        no_velocity_filtering=True, no_signal_time_shift=True, no_torque_bias_subtraction=True,
        model_size_change='input projection grows from 21 to 126; hidden/head widths unchanged',
        initialization='shared initial weights and global DataLoader/dropout RNG match baseline seed 0',
        single_seed_pilot=True, automatic_deployment=False)
    (output/'experiment.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    base_ds = ManifestDataset(manifest['splits']['train'], cfg['data']['keys'])
    md_ds = MDDataset(manifest['splits']['train'], cfg['data']['keys'], cache_path)
    norm = md_ds.fit_normalization()
    for name in ('x_mean', 'x_std'):
        np.testing.assert_array_equal(norm[name][:21], baseline.normalization[name])
    for name in ('y_mean', 'y_std'):
        np.testing.assert_array_equal(norm[name], baseline.normalization[name])
    for i in (0, len(base_ds)//2, len(base_ds)-1):
        x, y = base_ds.raw_item(i); mx, my = md_ds.raw_item(i)
        np.testing.assert_array_equal(x, mx[:, :21]); np.testing.assert_array_equal(y, my)
    source = output/'frozen_runtime'
    shutil.copytree(package, source/'factr2_next', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('run_w3_md_ablation.py', 'evaluate_w3_md_ablation.py',
                 'w3_coverage_plan.py', 'w3_joint_target.py'):
        shutil.copy2(Path(__file__).resolve().parent/name, source/name)
    source_hashes = {str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file()}
    write_json(output/'frozen_runtime_sha256.json', source_hashes)
    write_json(output/'preparation.json', dict(manifest_sha256=sha256(output/'split.json'),
        experiment_sha256=sha256(output/'experiment.yaml'),
        baseline_run=str(baseline_run), baseline_model_sha256=sha256(baseline_run/'model.pt'),
        recipe_sha256=frozen_recipe_hash, cache_sha256=sha256(cache_path), shared_sources_sha256=shared,
        source_provenance=provenance, isolation=parents, windows_train=len(md_ds),
        input_size=md_ds.input_size, base_columns_and_targets_exact=True,
        feature_normalization_first21_and_targets_exact=True, deployment_ready=False))
    (output/'start_training.py').write_text(
        "import sys,runpy\nfrom pathlib import Path\n"
        "folder=Path(__file__).resolve().parent\n"
        "sys.path.insert(0,str(folder/'frozen_runtime'))\n"
        "sys.argv=[str(folder/'frozen_runtime/run_w3_md_ablation.py'),'train','--experiment',str(folder)]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n")
    print(json.dumps(dict(experiment=str(output), train_windows=len(md_ds),
        thresholds_rad_s=recipe['thresholds_rad_s'], baseline_reused=True), ensure_ascii=False, indent=2))


def verify_frozen(folder):
    audit = json.loads((folder/'preparation.json').read_text())
    for file, key in [('split.json','manifest_sha256'), ('experiment.yaml','experiment_sha256'), ('md_recipe.json','recipe_sha256'),
                      ('md_features.h5','cache_sha256')]:
        if sha256(folder/file) != audit[key]:
            raise ValueError('md_frozen_input_changed:'+file)
    for name, digest in json.loads((folder/'frozen_runtime_sha256.json').read_text()).items():
        if sha256(folder/'frozen_runtime'/name) != digest:
            raise ValueError('md_frozen_source_changed:'+name)
    return audit


def train(folder):
    folder = Path(folder).resolve(); audit = verify_frozen(folder)
    cfg = yaml.safe_load((folder/'experiment.yaml').read_text())
    baseline = load_checkpoint(Path(cfg['comparison']['baseline_run']))
    if sha256(Path(cfg['comparison']['baseline_run'])/'model.pt') != audit['baseline_model_sha256']:
        raise ValueError('md_baseline_changed')
    manifest, reports = validate_manifest(cfg['data']['manifest'], dict(side='right',
        joint_order=cfg['joint_names'], contract_version=cfg['contract_version'], sample_hz=100))
    validate_independent_manifest(manifest, reports); check_parent_isolation(reports)
    state = dict(status='preparing_training', pid=os.getpid(), deployment_ready=False,
                 automatic_deployment=False, baseline_reused=True)
    def status(**values):
        state.update(values); write_json(folder/'training_state.json', state)
    if (folder/'training_state.json').exists():
        raise ValueError('Never overwrite an attempted MD training run')
    status()
    try:
        set_seed(int(cfg['train']['seed']))
        torch.set_num_threads(int(cfg['train']['cpu_threads']))
        torch.use_deterministic_algorithms(True)
        device = torch.device(cfg['train']['device'])
        datasets = [MDDataset(manifest['splits'][role], cfg['data']['keys'], cfg['comparison']['feature_cache'])
                    for role in ('train','val')]
        norm = datasets[0].fit_normalization()
        for key in ('x_mean','x_std'):
            np.testing.assert_array_equal(norm[key][:21], baseline.normalization[key])
        for key in ('y_mean','y_std'):
            np.testing.assert_array_equal(norm[key], baseline.normalization[key])
        for ds in datasets: ds.norm = norm
        loaders = [DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=(i==0))
                   for i,ds in enumerate(datasets)]
        model = paired_initial_model(cfg['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'],
                                     weight_decay=cfg['train']['weight_decay'])
        metrics = dict(train_loss=[], val_loss=[], validation_diagnostics=[], best_epoch=None,
            loss_units='physical_nm', seed=cfg['train']['seed'],
            selection_rule='minimum validation physical Nm squared MSE; first epoch on ties',
            baseline_parameter_count=sum(p.numel() for p in baseline.model.parameters()),
            md_parameter_count=sum(p.numel() for p in model.parameters()),
            window_counts=dict(train=len(datasets[0]), val=len(datasets[1])))
        output = folder/'next_md_seed0'; output.mkdir(exist_ok=False)
        best, early_best, stale = float('inf'), float('inf'), 0
        best_state = None; started = time.monotonic()
        early = cfg['train']['early_stopping']
        for epoch in range(1, cfg['train']['epochs']+1):
            status(status='training', current_epoch=epoch, best_epoch=metrics['best_epoch'])
            tr = run_epoch(model, loaders[0], device, optimizer,
                gradient_clip=cfg['train']['gradient_clip'], output_scale=norm['y_std'])
            va, diagnostic = validation_epoch(model, loaders[1], device, norm, 'physical_nm')
            if not np.isfinite([tr,va]).all(): raise ValueError('nonfinite_loss')
            metrics['train_loss'].append(tr); metrics['val_loss'].append(va)
            metrics['validation_diagnostics'].append(diagnostic)
            if va < best:
                best = va; metrics['best_epoch'] = epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(dict(model_state_dict=best_state,input_size=126,output_size=7,history=50,
                    model=cfg['model'],schema='next_md_ablation_v1',best_epoch=epoch),output/'best.pending')
                (output/'best.pending').replace(output/'best.pt')
            write_json(output/'metrics.json', metrics)
            status(status='training', completed_epoch=epoch, best_epoch=metrics['best_epoch'],
                   last_train_mse_nm2=tr,last_val_mse_nm2=va,best_val_mse_nm2=best,
                   elapsed_seconds=time.monotonic()-started)
            print(f'NEXT+MD epoch {epoch}: train={tr:.8f} val={va:.8f} best={best:.8f}',flush=True)
            if va < early_best-early['min_delta']: early_best=va;stale=0
            else: stale+=1
            if Path(cfg['train']['stop_file']).exists():
                metrics['stop_reason']='user_requested';break
            if early['enabled'] and epoch>=early['warmup'] and stale>=early['patience']:
                metrics['early_stopped_epoch']=epoch;break
        model.load_state_dict(best_state); model.eval()
        metrics.update(training_seconds=time.monotonic()-started,
                       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        torch.save(dict(model_state_dict=best_state,input_size=126,output_size=7,history=50,
                       model=cfg['model'],schema='next_md_ablation_v1'),output/'model.pt')
        np.savez(output/'normalization.npz',**norm)
        (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        write_json(output/'metrics.json',metrics)
        write_json(output/'metadata.json',dict(schema='next_md_ablation_v1',input_size=126,output_size=7,
            side='right',joint_order=cfg['joint_names'],history=50,sample_hz=100,
            feature_order=['q','qdot','q_cmd-q','MD_threshold_major'],
            baseline_run=cfg['comparison']['baseline_run'],recipe_sha256=audit['recipe_sha256'],
            manifest_sha256=audit['manifest_sha256'],automatic_deployment=False,deployment_ready=False,
            artifact_sha256={f:sha256(output/f) for f in ['model.pt','normalization.npz','config.yaml','metrics.json']}))
        from evaluate_w3_md_ablation import evaluate
        status(status='evaluating_frozen_models',run=str(output))
        evaluate(folder)
        status(status='completed',run=str(output),best_epoch=metrics['best_epoch'],
               best_val_mse_nm2=best,report=str(folder/'COMPARISON.md'))
    except BaseException as exc:
        status(status='failed',error=str(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.manifest,args.baseline_run,args.output)
    else:train(args.experiment)


if __name__=='__main__':main()
