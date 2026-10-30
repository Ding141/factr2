"""Process a fully completed campaign; never commands or deploys to a robot."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from factr2_next.data_collection.quality import write_json,sha256
from prepare_w3_targeted_session import prepare as derive
from train_w3_highspeed import prepare as prepare_mean
from finish_w3_highspeed import matched_sigma,fit_compensation
from export_w3_frozen_sigma import export

ROOT=Path(__file__).resolve().parents[1]


def mean_test_gate(experiment):
    experiment=Path(experiment);criteria=json.loads((experiment/'deployment_criteria.json').read_text())
    report=json.loads((experiment/'evaluation/comparison.json').read_text())
    test=report['splits']['test'];old,new=test['INCUMBENT'],test['NEXT_MSE']
    passed=(new['high']['physical_mse_nm2']<=(1-criteria['minimum_new_highspeed_test_mse_improvement_fraction'])*old['high']['physical_mse_nm2'] and
            new['old']['physical_mse_nm2']<=criteria['maximum_old_test_mse_ratio']*old['old']['physical_mse_nm2'])
    joints={}
    for scope in ('old','high'):
        before=np.asarray(old[scope]['per_joint_rmse_nm']);after=np.asarray(new[scope]['per_joint_rmse_nm'])
        limit=before+np.maximum(.2*before,.02)
        joints[scope]=(after<=limit).tolist();passed &= bool(np.all(after<=limit))
    result=dict(passed=bool(passed),criteria=criteria,per_joint_pass=joints,
        highspeed_mse_ratio=new['high']['physical_mse_nm2']/old['high']['physical_mse_nm2'],
        old_mse_ratio=new['old']['physical_mse_nm2']/old['old']['physical_mse_nm2'],
        action='Candidate may proceed to matched-head and compensation checks' if passed else 'Keep incumbent; no tuning on test')
    write_json(experiment/'deployment_gate.json',result)
    return result


def process(campaign,experiment,sigma_dir,base,compensation):
    campaign,experiment,sigma_dir,base,compensation=map(lambda p:Path(p).resolve(),
        (campaign,experiment,sigma_dir,base,compensation))
    for role in ('train','val','test'):
        stage=campaign/'plans'/role;state=json.loads((stage/'test_state.json').read_text())
        quality=json.loads((stage/'continuous_quality.json').read_text())
        if state['status']!='completed' or not any(b['accepted'] for b in quality['blocks']):
            raise ValueError('Require completed recordings with whole quality-passing blocks')
    for role in ('train','val','test'):
        state=json.loads((campaign/'plans'/role/'test_state.json').read_text())
        print(derive(Path(state['session']),campaign/'derived'/role,False,
                     campaign/'plans'/role/'continuous_quality.json'),flush=True)
    prepare_mean(campaign,ROOT/'data/right/next_u_ablation_20261023_v1',
                 ROOT/'models/w3/right/frozen_sigma_observer_20261023/mean_checkpoint',experiment)
    with (experiment/'training.log').open('x') as f:
        p=subprocess.run([sys.executable,str(experiment/'start.py')],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Training failed; inspect training.log; do not resume automatically')
    gate=mean_test_gate(experiment);print(json.dumps(gate),flush=True)
    if not gate['passed']:raise RuntimeError('Independent test gate failed; preserve incumbent')
    matched_sigma(experiment,sigma_dir)
    export(sigma_dir,base)
    fit_compensation(experiment,sigma_dir,base,compensation)
    from evaluate_w3_highspeed_candidate import evaluate
    final=evaluate(experiment,sigma_dir,base,compensation,campaign/'candidate_evaluation')
    if not final['passed']:raise RuntimeError('Full pipeline regression gate failed; preserve incumbent')
    # These are candidate artifacts only; live replacement is a separate action
    # after parity, full-pipeline comparison and current-source checks.
    write_json(campaign/'candidate_artifacts.json',dict(mean=str(experiment),sigma=str(sigma_dir),
        observer=str(base),compensation=str(compensation),deployed=False,
        mean_metadata_sha256=sha256(base/'metadata.json'),
        compensation_sha256=sha256(compensation/'model.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','experiment','sigma-dir','base','compensation'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();process(a.campaign,a.experiment,a.sigma_dir,a.base,a.compensation)
