#!/usr/bin/env python3
"""Build splits from the accepted independent e2e recording sessions."""
from pathlib import Path
import json
from factr2_next.data_collection.quality import make_manifest
ROOT=Path(__file__).resolve().parents[1]
for side in ('left','right'):
    paths={}
    for split,label in [('train','continuous'),('val','val'),('test','test')]:
        report=json.loads((ROOT/f'reports/generated/4/{side}/{label}/continuous.json').read_text())
        assert report['accepted'] and report['result']=='PASS'
        paths[split]=[report['path']]
    make_manifest(ROOT/f'data/mock/{side}/split.json',paths,f'synthetic_{side}_e2e_v1')
    print(side,paths)
