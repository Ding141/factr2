"""Offline wide-coverage retraining and export; never moves or deploys a robot.

Extra kinematic compensation is deliberately not fitted. Historical session
roles are retained; the new test role is read only after validation selection.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from factr2_next.data_collection.quality import sha256,write_json
from prepare_w3_targeted_session import prepare as derive
from train_w3_highspeed import prepare
from finish_w3_highspeed import matched_sigma
from export_w3_frozen_sigma import export
from process_w3_continuous_campaign import mean_test_gate

ROOT=Path(__file__).resolve().parents[1]


def process(campaign,experiment,sigma_dir,base):
    campaign,experiment,sigma_dir,base=map(lambda p:Path(p).resolve(),(campaign,experiment,sigma_dir,base))
    for role in ('train','val','test'):
        stage=campaign/'plans'/role
        state=json.loads((stage/'test_state.json').read_text())
        quality=json.loads((stage/'continuous_quality.json').read_text())
        if state['status']!='completed' or not any(b['accepted'] for b in quality['blocks']):
            raise ValueError('Require completed independent sessions with complete quality-passing blocks')
        derive(Path(state['session']),campaign/'derived'/role,False,stage/'continuous_quality.json')
    # Reuse old low AND previously recorded high speed data, preserving their
    # original whole-session roles. No previous test recording becomes train.
    prepare(campaign,ROOT/'data/right/highspeed_mean_20261025_v3',
        ROOT/'models/w3/right/highspeed_observer_20261025_v1/mean_checkpoint',
        experiment,scheme='anchored_full')
    write_json(campaign/'training_status.json',dict(status='training_mean',experiment=str(experiment),
        extra_compensation='disabled_for_final_deployment',test_used_for_selection=False))
    with (experiment/'training.log').open('x') as f:
        subprocess.run([sys.executable,str(experiment/'start.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    gate=mean_test_gate(experiment)
    write_json(campaign/'mean_test_gate.json',gate)
    # Export remains an explicit research artifact even if a performance gate
    # fails; a failed gate is reported verbatim and is never rewritten to pass.
    write_json(campaign/'training_status.json',dict(status='training_matching_sigma',gate=gate,
        experiment=str(experiment),extra_compensation='disabled_for_final_deployment'))
    matched_sigma(experiment,sigma_dir)
    export(sigma_dir,base)
    from analyze_w3_wide_model import analyze
    analyze(experiment,base,campaign/"heldout_mean_comparison")
    write_json(campaign/'candidate_artifacts.json',dict(mean=str(experiment),sigma=str(sigma_dir),
        observer=str(base),compensation=None,compensation_enabled=False,deployed=False,
        performance_gate=gate,mean_metadata_sha256=sha256(base/'metadata.json')))
    write_json(campaign/'training_status.json',dict(status='complete',experiment=str(experiment),
        sigma=str(sigma_dir),observer=str(base),gate=gate,compensation_enabled=False,
        automatic_deployment=False,contact_validated=False))
    print(json.dumps(dict(observer=str(base),gate=gate,compensation_enabled=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','experiment','sigma-dir','base'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();process(a.campaign,a.experiment,a.sigma_dir,a.base)
