#!/usr/bin/env python3
"""Train an explicitly preliminary model and evaluate held-out test once."""
import argparse
import json
import os
from pathlib import Path
import traceback

import torch
from torch.utils.data import DataLoader
import yaml

from factr2_next.data_collection.quality import validate_manifest, write_json
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.validation_metrics import validation_epoch
from factr2_next.training.w3_train import train
from factr2_next.w3_config import validate_w3_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    validate_w3_config(cfg, 'train')
    if cfg.get('trial', {}).get('deployment_ready') is not False:
        raise ValueError('Require explicit trial profile')
    state_path = args.config.parent/'training_state.json'
    if state_path.exists():
        raise ValueError('Never overwrite an existing training task')
    state = dict(status='training', pid=os.getpid(), config=str(args.config.resolve()),
                 deployment_ready=False, automatic_deployment=False)
    write_json(state_path, state)
    try:
        run = train(cfg)
        state.update(status='evaluating_test', run=str(run.resolve()))
        write_json(state_path, state)
        manifest, _ = validate_manifest(cfg['data']['manifest'])
        loaded = load_checkpoint(run, expected={'side':'right', 'sample_hz':100})
        ds = ManifestDataset(manifest['splits']['test'], cfg['data']['keys'], 50, 512)
        ds.norm = loaded.normalization
        loss, diagnostic = validation_epoch(loaded.model,
            DataLoader(ds, batch_size=cfg['train']['batch_size']), torch.device('cpu'), ds.norm, 'physical_nm')
        write_json(run/'trial_evaluation.json', dict(test_physical_mse_nm2=loss,
            test_diagnostics=diagnostic, evaluation_scope=cfg['trial']['scope'],
            deployment_ready=False, automatic_deployment=False, pending=cfg['trial']['pending']))
        state.update(status='completed', test_physical_mse_nm2=loss)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        traceback.print_exc()
        raise
    finally:
        write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
