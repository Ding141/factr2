"""Corruption and content-addressed split regressions, without ROS hardware."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
from factr2_next.data_collection.h5_writer import H5Writer
from factr2_next.data_collection.quality import check, sha256, sidecar_path, make_manifest, validate_manifest
from factr2_next.w3_config import SIGNALS


def fixture(path, n=110, phase=0):
    meta=dict(contract_version='w3_next_v1',side='left',joint_order=[f'left_joint_{i}' for i in range(7)],
        sample_hz=50,session_id=path.stem,topics={k:dict(topic='/factr2/left/'+v[0],field=v[1]) for k,v in SIGNALS.items()},
        software_commits={'factr2':'test','w3':'test'},config_sha256='test',tool='mock_tool',gripper='excluded',
        load='none',calibration={'id':'synthetic_v1'},control={'id':'mock'},trajectory_id=str(phase),
        contact={'present':False,'label':'synthetic_free'},temperature={'source':'unavailable'},time_source='test',
        audit_paths=[],health_gate_enabled=False,source='synthetic',episodes={})
    writer=H5Writer(path,path.stem,SIGNALS,meta); writer.start_episode()
    for i in range(n):
        q=np.arange(7,dtype=np.float32)*.01 + np.sin((i+phase)*.04)*.1
        writer.append(10**15+i*20_000_000,{ 'joint_pos':q, 'joint_vel':q*.3+.1,
            'joint_cmd':q+.2, 'measured_joint_torque':q*.4+1})
    writer.close()
    return path


def refresh(path):
    p=sidecar_path(path);meta=json.loads(p.read_text());meta['h5_sha256']=sha256(path);p.write_text(json.dumps(meta))


@pytest.mark.parametrize('bad', ['missing','schema','six','eight','length','float_time','duplicate','regression',
                               'unequal','nan','inf','nan_time','low_hz','gap','hash','side','order','contract'])
def test_corruption(tmp_path,bad):
    path=fixture(tmp_path/'bad.h5')
    if bad in ('side','order','contract'):
        m=json.loads(sidecar_path(path).read_text())
        m[{'side':'side','order':'joint_order','contract':'contract_version'}[bad]]= 'wrong'
        sidecar_path(path).write_text(json.dumps(m))
    elif bad != 'hash':
        with h5py.File(path,'r+') as h:
            ep=h['ep_0000'];g=ep['joint_pos']
            if bad=='missing':del ep['joint_cmd']
            elif bad=='schema':h.attrs['schema']='wrong'
            elif bad in ('six','eight'):
                del g['data'];g.create_dataset('data',data=np.ones((110,6 if bad=='six' else 8),np.float32))
            elif bad=='length':g['data'].resize(109,axis=0)
            elif bad in ('float_time','nan_time'):
                t=g['timestamps'][:].astype(float)
                if bad=='nan_time':t[0]=float('nan')
                del g['timestamps'];g['timestamps']=t
            elif bad in ('duplicate','regression'):g['timestamps'][2]=g['timestamps'][1]-(1 if bad=='regression' else 0)
            elif bad=='unequal':g['timestamps'][2]+=1
            elif bad in ('nan','inf'):g['data'][2,0]=float(bad)
            else:
                for key in SIGNALS:
                    t=ep[key]['timestamps'][:]
                    if bad=='low_hz':t=10**15+np.arange(110,dtype=np.int64)*30_000_000
                    else:t[55:]+=100_000_000
                    ep[key]['timestamps'][:]=t
        refresh(path)
    else:
        with h5py.File(path,'r+') as h:h.attrs['changed']=1
    report=check(path)
    assert not report['accepted'] and report['errors']
    proc=subprocess.run([sys.executable,'-m','factr2_next.data_collection.quality','check',str(path)],capture_output=True,text=True)
    assert proc.returncode==1
    assert json.loads(proc.stdout)['errors']


def test_splits(tmp_path):
    paths=[fixture(tmp_path/f'{s}.h5',phase=i*10) for i,s in enumerate(('train','val','test'))]
    output=tmp_path/'manifest.json'
    make_manifest(output,dict(zip(('train','val','test'),[[x] for x in paths])),'synthetic_test')
    validate_manifest(output)
    copy=tmp_path/'copy.h5';shutil.copy2(paths[0],copy);shutil.copy2(sidecar_path(paths[0]),sidecar_path(copy))
    with pytest.raises(ValueError,match='split_overlap'):
        make_manifest(tmp_path/'bad.json',{'train':[paths[0]],'val':[copy],'test':[paths[2]]},'leak')


def test_writer_transaction(tmp_path):
    path=fixture(tmp_path/'good.h5');assert check(path)['accepted']
    with h5py.File(path,'r') as h:assert h['ep_0000/joint_pos/data'].shape==(110,7)


def test_manifest_profile_rejection(tmp_path):
    paths=[fixture(tmp_path/f'{i}.h5',phase=i) for i in range(3)]
    mpath=sidecar_path(paths[1]);m=json.loads(mpath.read_text());m['tool']='another_tool';mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='dataset_profile_mismatch'):
        make_manifest(tmp_path/'split.json',dict(zip(('train','val','test'),[[p] for p in paths])),'wrong_tool')


def test_frame_validation():
    from sensor_msgs.msg import JointState
    from factr2_next.w3_samples import frame
    names=[f'left_joint_{i}' for i in range(7)]
    specs={k:{'field':field} for k,(_,field) in SIGNALS.items()}
    msgs=[]
    for key,(_,field) in SIGNALS.items():
        m=JointState();m.name=names.copy();m.header.stamp.sec=10
        setattr(m,field,[float(i) for i in range(7)]);msgs.append(m)
    stamp,samples=frame(msgs,list(SIGNALS),specs,names,10_010_000_000)
    assert stamp==10_000_000_000 and samples['joint_vel'][6]==6
    msgs[0].name.reverse()
    with pytest.raises(ValueError,match='frame_names'):frame(msgs,list(SIGNALS),specs,names,10_010_000_000)
    msgs[0].name=names
    msgs[0].header.stamp.nanosec=1
    with pytest.raises(ValueError,match='frame_stamp'):frame(msgs,list(SIGNALS),specs,names,10_010_000_000)


def test_bad_sidecar_stays_machine_readable(tmp_path):
    path=fixture(tmp_path/'badmeta.h5')
    for content in ('[]','{"source": NaN}'):
        sidecar_path(path).write_text(content)
        report=check(path);assert not report['accepted'] and report['errors'][0].startswith('read_error:')
        proc=subprocess.run([sys.executable,'-m','factr2_next.data_collection.quality','check',str(path)],capture_output=True,text=True)
        assert proc.returncode==1 and json.loads(proc.stdout)['errors']


def test_gap_nanosecond_boundary(tmp_path):
    path=fixture(tmp_path/'boundary.h5')
    with h5py.File(path,'r+') as h:
        for key in SIGNALS:
            t=h['ep_0000'][key]['timestamps'][:];t[55:]+=20_000_001
            h['ep_0000'][key]['timestamps'][:]=t
    refresh(path)
    report=check(path)
    assert not report['accepted'] and 'ep_0000:gap' in report['errors']
