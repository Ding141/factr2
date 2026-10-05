#!/usr/bin/env python3
"""Check templates, or validate one runnable profile with --runtime."""
import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'factr2_next/src/factr2_next'))
from factr2_next.w3_config import validate_w3_config

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', choices=['record', 'train', 'inference', 'visualize'])
parser.add_argument('--config', type=Path)
args = parser.parse_args()
if args.runtime:
    if not args.config:
        parser.error('--runtime requires --config')
    validate_w3_config(yaml.safe_load(args.config.read_text()), args.runtime)
else:
    contract = yaml.safe_load((ROOT / 'config/w3/contract.yaml').read_text())
    assert contract['version'] == 'w3_next_v1'
    for side in ('left', 'right'):
        for purpose in ('record', 'train', 'inference', 'visualize'):
            path = ROOT / f'config/w3/{side}/{purpose}.yaml'
            validate_w3_config(yaml.safe_load(path.read_text()), purpose, require_paths=False)
            print(f'PASS {path.relative_to(ROOT)}')
