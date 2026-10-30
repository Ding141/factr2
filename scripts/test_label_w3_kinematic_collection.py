"""Offline capture identity, exact event labeling, and full-precision contracts."""
import csv
import json
from pathlib import Path

import pytest

from factr2_next.inference.kinematic_compensation import digest
from label_w3_kinematic_collection import label


@pytest.fixture
def collection(tmp_path):
    folder = tmp_path/'observation'; folder.mkdir()
    session = tmp_path/'session'; (session/'audit').mkdir(parents=True)
    model = tmp_path/'model'; model.mkdir(); (model/'model.json').write_text('{}')
    base = tmp_path/'base'; base.mkdir(); (base/'metadata.json').write_text('{}')
    (folder/'motion.yaml').write_text('frozen plan bytes')
    roles = {'train_block':'train','val_block':'val','test_block':'test'}
    def write(path, value):path.write_text(json.dumps(value))
    spec = dict(plan_sha256=digest(folder/'motion.yaml'), steps=3, block_roles=roles,
                model_dir=str(model), model_hash=digest(model/'model.json'),
                base_model_dir=str(base), base_metadata_sha256=digest(base/'metadata.json'))
    write(folder/'test_spec.json',spec)
    write(folder/'test_state.json',dict(status='completed',session=str(session)))
    write(folder/'recorder_state.json',dict(status='closed'))
    write(session/'quality.json',dict(accepted=True))
    events = [dict(name=name,status='completed',command_start_ros_ns=1_000_000_000+i*4_000_000_000,
                   command_end_ros_ns=1_000_000_000+(i+1)*4_000_000_000) for i,name in enumerate(roles)]
    write(session/'audit/motion_run.json',dict(status='completed',motion_sha256=spec['plan_sha256'],events=events))
    fields = ['source_timestamp_ns','next_state'] + [f'{key}_j{j}' for key in
              ('measured_tau_nm','base_free_pred_nm','base_external_raw_nm') for j in range(1,8)]
    with (folder/'samples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i in range(1200):
            row=dict(source_timestamp_ns=1_000_000_000+i*10_000_000,next_state='valid')
            for j in range(1,8):
                row.update({f'measured_tau_nm_j{j}':.123456789,
                            f'base_free_pred_nm_j{j}':.0123456789,
                            f'base_external_raw_nm_j{j}':.123456789-.0123456789})
            writer.writerow(row)
    return folder


def test_label_uses_exact_event_time_and_preserves_precision(collection):
    report=label(collection)
    assert report['rows']==1200 and report['residual_identity_error_nm']<1e-12
    assert report['predeclared_valid_rows']==dict(train=400,val=400,test=400)
    with (collection/'labeled_samples.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert rows[399]['exact_event']=='train_block'
    assert rows[400]['exact_event']=='val_block'
    assert rows[800]['predeclared_role']=='test'
    assert float(rows[0]['free_pred_tau_nm_j1'])==.0123456789
    assert float(rows[0]['measured_tau_nm_j1'])==.123456789


@pytest.mark.parametrize('failure',['partial_motion','changed_model','failed_quality','overlapping_events'])
def test_bad_capture_is_not_labeled(collection,failure):
    state=json.loads((collection/'test_state.json').read_text())
    session=Path(state['session'])
    if failure=='partial_motion':
        state['status']='failed';(collection/'test_state.json').write_text(json.dumps(state))
    elif failure=='changed_model':
        spec=json.loads((collection/'test_spec.json').read_text())
        (Path(spec['model_dir'])/'model.json').write_text('{"changed":true}')
    elif failure=='failed_quality':
        (session/'quality.json').write_text('{"accepted":false}')
    else:
        path=session/'audit/motion_run.json';run=json.loads(path.read_text())
        run['events'][1]['command_start_ros_ns']-=10_000_000
        path.write_text(json.dumps(run))
    with pytest.raises(ValueError):label(collection)
    assert not (collection/'labeled_samples.csv').exists()
