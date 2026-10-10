#!/usr/bin/env python3
"""Combine purged episode manifests without joining history across recordings."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from factr2_next.data_collection.quality import validate_manifest,make_manifest,sha256,write_json

def control_signature(control):
    result=copy.deepcopy(control)
    for group in result.values():
        for key,value in list(group.items()):
            if key.endswith('_params') or key=='friction_model':
                group[key]=sha256(Path(value))
    return result

def combine(datasets, output):
    if output.exists():raise ValueError('Output exists')
    sources=[];entries={s:[] for s in ('train','val','test')};signatures=[];parent_ids=set()
    for dataset in datasets:
        audit=json.loads((dataset/'split_audit.json').read_text())
        parent=Path(audit['parent_h5']);digest=sha256(parent)
        if digest!=audit['parent_sha256_before'] or digest in parent_ids:
            raise ValueError('Changed or repeated raw recording')
        parent_ids.add(digest)
        manifest,reports=validate_manifest(dataset/'split.json')
        for split in entries:
            entries[split].extend(Path(e['path']) for e in manifest['splits'][split])
            for report in reports[split]:
                m=report['metadata'];signature={k:m[k] for k in
                    ('side','joint_order','sample_hz','tool','gripper','load','calibration')}
                signature['control']=control_signature(m['control']);signatures.append(signature)
        sources.append({'parent_h5':str(parent),'parent_sha256_before':digest,
            'dataset':str(dataset),'audit':audit})
    if any(s!=signatures[0] for s in signatures):raise ValueError('Control/calibration/profile mismatch')
    output.mkdir(parents=True)
    manifest=make_manifest(output/'split.json',entries,output.name)
    scope='Repeated-motion holdouts from segmented captures in the same campaign; not independent session generalization'
    manifest['evaluation_scope']=scope;manifest['source_datasets']=[str(d) for d in datasets]
    write_json(output/'split.json',manifest)
    # Keep the latest recording as the primary artifact for legacy consumers;
    # every raw file and original frame range remains explicitly audited.
    audit=copy.deepcopy(sources[-1]['audit']);audit.update(scope=scope,
        source_sessions=sources,source_ranges_disjoint=True,
        source_rows=sum(s['audit']['source_rows'] for s in sources),
        retained_rows=sum(s['audit']['retained_rows'] for s in sources),
        splits={split:{k:sum(s['audit']['splits'][split][k] for s in sources)
            for k in ('rows','windows','episodes')} for split in entries})
    write_json(output/'split_audit.json',audit)
    cfg=yaml.safe_load((datasets[-1]/'train.yaml').read_text());cfg['data']['manifest']=str((output/'split.json').resolve())
    cfg['save']['run_name']='next_'+cfg['side']+'_'+output.name
    (output/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(json.dumps({'output':str(output),'scope':scope,'splits':audit['splits']},indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();combine([d.resolve() for d in a.datasets],a.output.resolve())
