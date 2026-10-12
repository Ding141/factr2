#!/usr/bin/env python3
"""Train paper NEXT, gate independent free/release validation; no auto deployment."""
import argparse
import json
from pathlib import Path
import yaml
from factr2_next.w3_config import validate_w3_config
from factr2_next.training.w3_train import train
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.data_collection.quality import sha256
from evaluate_w3_baseline import evaluate_free, evaluate_release, validate_thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--acceptance', required=True, type=Path)
    p.add_argument('--release-h5', type=Path)
    p.add_argument('--annotations', type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    acceptance = yaml.safe_load(args.acceptance.read_text()); validate_thresholds(acceptance)
    acceptance_hash = sha256(args.acceptance)
    validate_w3_config(cfg, 'train')
    if (cfg['data'].get('require_independent_sessions') is not True
            or cfg['data'].get('sample_hz') != 100):
        p.error('This pipeline requires independent real 100Hz sessions')
    if bool(args.release_h5) != bool(args.annotations):
        p.error('Supply both release H5 and annotations')
    run = train(cfg); loaded = load_checkpoint(run, expected={'side':'right', 'sample_hz':100})
    if loaded.metadata.get('operating_envelope'):
        # Prepare a reviewable runtime profile with the new, narrower envelope.
        # Creating this file does not start an observer or change a running one.
        profile = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/w3/right/inference.yaml').read_text())
        profile.update(checkpoint_dir=str(run.resolve()), sample_hz=100,
                       operating_envelope=loaded.metadata['operating_envelope'])
        profile['smoothing']['sample_hz'] = 100
        (run/'inference.yaml').write_text(yaml.safe_dump(profile, sort_keys=False))
    # Freeze acceptance BEFORE training/test. Do not tune using final test.
    (run/'acceptance.yaml').write_text(yaml.safe_dump(acceptance, sort_keys=False))
    result = dict(acceptance_sha256=acceptance_hash, deployment_ready=False,
                  automatic_deployment=False)
    result['val'] = evaluate_free(loaded, cfg['data']['manifest'], 'val', acceptance)
    if result['val']['passed']:
        result['test'] = evaluate_free(loaded, cfg['data']['manifest'], 'test', acceptance)
        if result['test']['passed'] and args.release_h5:
            result['release'] = evaluate_release(loaded, args.release_h5,
                yaml.safe_load(args.annotations.read_text()), acceptance)
            result['deployment_ready'] = result['release']['passed']
    result['pending'] = [] if result['deployment_ready'] else [
        'Independent free-space and annotated contact/release validation must all pass']
    (run/'baseline_acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(run=str(run), deployment_ready=result['deployment_ready'],
                          pending=result['pending']), indent=2))
    raise SystemExit(0 if result['deployment_ready'] else 1)


if __name__ == '__main__':
    main()
