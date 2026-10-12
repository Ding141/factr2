#!/usr/bin/env python3
"""Read-only measured NEXT coverage audit; never use test values or move the arm."""
import argparse
from pathlib import Path
import json
import yaml
from factr2_next.w3_config import validate_w3_config
from factr2_next.data_collection.quality import validate_manifest, write_json
from factr2_next.training.dataset import ManifestDataset
from factr2_next.training.coverage_audit import audit_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    validate_w3_config(cfg, 'train')
    manifest, reports = validate_manifest(cfg['data']['manifest'], {
        'side':cfg['side'], 'joint_order':cfg['joint_names'],
        'contract_version':cfg['contract_version'], 'sample_hz':cfg['data'].get('sample_hz', 50)})
    if cfg['data'].get('require_independent_sessions', False):
        from factr2_next.data_collection.campaign import validate_independent_manifest
        validate_independent_manifest(manifest, reports)
    datasets = [ManifestDataset(manifest['splits'][role], cfg['data']['keys'], 50,
                               cfg['data'].get('memory_budget_mb', 512)) for role in ('train', 'val')]
    report = audit_coverage(*datasets, cfg['data']['coverage_requirements'])
    write_json(args.output, report)
    print(json.dumps({key:report[key] for key in ('passed', 'reasons', 'window_fractions')}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
