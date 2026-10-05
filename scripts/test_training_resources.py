#!/usr/bin/env python3
"""One-hour, 50 Hz synthetic loading budget; no windows materialized globally."""
import json
from pathlib import Path
import resource
import time
import h5py
import numpy as np
from factr2_next.data_collection.quality import sidecar_path,sha256,write_json,require
from factr2_next.training.dataset import ManifestDataset
from factr2_next.w3_config import SIGNALS
ROOT=Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'reports/generated/4/left/continuous/continuous.json').read_text())
meta=source['metadata'];n=180000
path=ROOT/'data/mock/resource_one_hour.h5'
q=np.sin(np.arange(n,dtype=np.float32)[:,None]*.02+np.arange(7,dtype=np.float32)[None,:])*.1
arrays=[q,q*.3+.1,q+.2,q*.4+1]
with h5py.File(path,'w') as h:
    h.attrs.update(schema='factr2_next_h5_v1',session_name='synthetic_resource',created_at='2026-10-05')
    ep=h.create_group('ep_0000')
    for key,a in zip(SIGNALS,arrays):
        g=ep.create_group(key);g['data']=a;g['timestamps']=10**15+np.arange(n,dtype=np.int64)*20_000_000
meta.update(session_id='synthetic_resource',episodes={'ep_0000':{'rows':n}},h5_sha256=sha256(path))
write_json(sidecar_path(path),meta)
require(path)
started=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
keys={k:k for k in SIGNALS};ds=ManifestDataset([{'path':str(path),'episodes':['ep_0000']}],keys)
normalization=ds.fit_normalization()
for i in (0,len(ds)//2,len(ds)-1):
    x,y=ds.raw_item(i)
    np.testing.assert_array_equal(x,np.concatenate((arrays[0][i:i+50],arrays[1][i:i+50],arrays[2][i:i+50]-arrays[0][i:i+50]),axis=1))
    np.testing.assert_array_equal(y,arrays[3][i+49])
report=dict(result='PASS',rows=n,windows=len(ds),history=50,load_and_norm_seconds=time.monotonic()-started,
    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,before_rss_mb=before,
    step_cache_bytes=ds.step_bytes,eager_window_bytes=len(ds)*50*21*4,memory_budget_mb=512,
    caveat='RSS includes torch runtime, fixture arrays and checker; budget covers cache, not process runtime')
write_json(ROOT/'reports/generated/5/resources.json',report);print(json.dumps(report))
