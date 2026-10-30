"""Atomically select a verified read-only joint-LSTM observation bundle."""
import argparse
import json
from pathlib import Path
import shutil

from factr2_next.data_collection.quality import sha256,write_json
from factr2_next.inference.joint_lstm_observer import load_bundle,ROOT

REPO=Path(__file__).resolve().parents[1]


def activate(bundle,record):
    bundle,record=Path(bundle).resolve(),Path(record).resolve()
    loaded,meta=load_bundle(bundle)
    if meta.get('mean_model_sha256')!=sha256(bundle/'mean_checkpoint/model.pt'):
        raise ValueError('activation_mean_hash')
    evaluation=Path(meta['evaluation'])/'comparison.json'
    if sha256(evaluation)!=meta['evaluation_sha256']:raise ValueError('activation_evaluation_hash')
    result=json.loads(evaluation.read_text())
    if set(result['splits'])!={'val','test'}:raise ValueError('activation_requires_complete_evaluation')
    if result['mean_sha256']!=meta['mean_model_sha256']:raise ValueError('activation_evaluation_mean')
    if any(reason=='stop_file' for reason in json.loads((bundle/'mean_checkpoint/metrics.json').read_text()).get('per_joint_stop_reason',[])):
        raise ValueError('activation_requires_completed_training')
    record.mkdir(parents=True,exist_ok=False)
    profile=REPO/'config/w3/right/observer_active.json'
    for original in (profile,REPO/'config/w3/right/compensation_active.json'):
        if original.exists():shutil.copy2(original,record/original.name)
    write_json(record/'rollback.json',dict(previous_observer_profile_existed=profile.exists(),
        previous_default='observe_w3_kinematic_compensation.py',new_bundle=str(bundle)))
    value=dict(schema='w3_observer_profile_v1',kind='joint_lstm',ros_domain_id=74,
        bundle=str(bundle),bundle_sha256=sha256(bundle/'bundle.json'),
        mean_model_sha256=meta['mean_model_sha256'],next_topic_root=ROOT,
        architecture='independent_joint_lstm_v1',compensation_enabled=False,
        controller_changes=False,research_only=True,contact_validated=False,
        selected_epoch_per_joint=loaded.metadata['per_joint_best_epoch'],
        trained_epochs_per_joint=json.loads((bundle/'mean_checkpoint/metrics.json').read_text())['per_joint_stopped_epoch'],
        rollback=str(record))
    write_json(profile,value);write_json(record/'activated_profile.json',value)
    print(json.dumps(value,indent=2));return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();activate(a.bundle,a.record)
