#!/usr/bin/env python3
"""Malformed runtime profiles fail before NEXT publisher creation."""
import copy
import json
from pathlib import Path
import subprocess
import yaml
from ros_test_processes import ROOT,configure_dds
configure_dds(90)
from factr2_next.data_collection.quality import write_json
out=ROOT/'reports/generated/6/config';out.mkdir(parents=True,exist_ok=True)
base=yaml.safe_load((ROOT/'config/w3/left/inference.yaml').read_text())
base['checkpoint_dir']=json.loads((ROOT/'reports/generated/5/left/summary.json').read_text())['checkpoint']
changes=[('alpha','smoothing','ema_alpha',1.2,'smoothing_alpha'),
         ('scale','normalized_contact_magnitude','scale',0,'contact_scale'),
         ('rate','smoothing','sample_hz',49,'W3 smoothing rate'),
         ('threshold','contact','low_threshold',3,'contact_scale_thresholds'),
         ('source','normalized_contact_magnitude','source','unknown','contact_source'),
         ('norm','normalized_contact_magnitude','norm','unknown','contact_norm')]
result={}
for label,section,key,value,reason in changes:
    cfg=copy.deepcopy(base);cfg[section][key]=value;file=out/(label+'.yaml');file.write_text(yaml.safe_dump(cfg))
    with open(out/(label+'.log'),'w') as log:
        p=subprocess.run(['python','-c','from factr2_next.inference.inference_node import main; main()',
            '--ros-args','-p','config_file:='+str(file)],stdout=log,stderr=subprocess.STDOUT,timeout=10)
    text=(out/(label+'.log')).read_text();assert p.returncode!=0 and reason in text,(label,text)
    assert 'Loaded NEXT checkpoint:' not in text
    result[label]={'exit_code':p.returncode,'error':reason}
write_json(out/'summary.json',{'result':'PASS','cases':result});print('PASS:',', '.join(result))
