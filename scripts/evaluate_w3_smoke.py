#!/usr/bin/env python3
"""Freeze a saved best smoke checkpoint then evaluate independent val/test."""
import argparse
import json
from pathlib import Path
import torch
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.offline_eval import evaluate,acceptance
from factr2_next.data_collection.quality import validate_manifest,write_json,sha256
import yaml
p=argparse.ArgumentParser();p.add_argument('--side',choices=['left','right'],required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
run=sorted((ROOT/'runs').glob(f'next_{a.side}_synthetic_smoke_*'))[-1]
loaded=load_checkpoint(run,expected={'side':a.side,'sample_hz':50,'history':50})
manifest,_=validate_manifest(ROOT/f'data/mock/{a.side}/split.json')
out=ROOT/f'reports/generated/5/{a.side}'
summary={'checkpoint':str(run),'checkpoint_sha256':sha256(run/'model.pt'),'metrics':json.loads((run/'metrics.json').read_text())}
for split in ('val','test','train'):
    report,residual=evaluate(loaded,manifest['splits'][split],out/split,split=split)
    report['acceptance']=acceptance(report,residual,yaml.safe_load((ROOT/'config/w3/acceptance.template.yaml').read_text()),loaded.normalization['y_std'])
    assert report['acceptance']['result']=='UNCONFIGURED'
    write_json(out/split/'report.json',report)
    summary[split]={'samples':report['model']['samples'],'rmse_nm':report['model']['overall_rmse_nm'],
        'baseline_rmse_nm':report['baseline']['overall_rmse_nm'],'acceptance':'UNCONFIGURED'}
write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
