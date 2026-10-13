"""Single/dual controller loading must retain the recorded arm's physical contract."""
import copy
import json
import xml.etree.ElementTree as ET
import pytest
import yaml
from factr2_next.data_collection.campaign import arm_control_profile, control_profile, geometry_sha256


def profile(tmp_path, dual):
    folder=tmp_path/('dual' if dual else 'single');folder.mkdir()
    right=[f'right_joint_{i}' for i in range(7)]
    names=[f'left_joint_{i}' for i in range(7)]+right if dual else right
    hardware='<hardware><plugin>robot</plugin><param name="offsets_yaml">'+str(folder/'cal.yaml')+'</param></hardware>'
    description='<robot name="r"><link name="base"/><ros2_control name="system">'+hardware+''.join('<joint name="'+n+'"/>' for n in names)+'</ros2_control></robot>'
    for name in ('position','gravity'):
        params=dict(joints=names,kp_gains=[99.]*7+[1.]*7 if dual else [1.]*7,
                    robot_description=description,friction_model_yaml=str(folder/'friction.yaml'))
        (folder/(name+'.yaml')).write_text(yaml.safe_dump({'controller':{'ros__parameters':params}}))
    calibration=dict(offsets=[dict(name=n,axis_sign=1. if dual else 1,
        zero_offset=.01,slot=i%7,motor_type='motor',channel=0 if n.startswith('left') else 1) for i,n in enumerate(names)])
    (folder/'cal.yaml').write_text(yaml.safe_dump(calibration))
    (folder/'friction.yaml').write_text('coefficient: 0.1\n')
    return dict(side='right',joint_order=right,calibration=dict(path=str(folder/'cal.yaml')),
        control=dict(gains=dict(position_params=str(folder/'position.yaml')),
        feedforward=dict(gravity_params=str(folder/'gravity.yaml'),friction_model=str(folder/'friction.yaml'))))


def test_single_and_dual_same_right_arm_with_legacy_profiles_preserved(tmp_path):
    single,dual=profile(tmp_path,False),profile(tmp_path,True)
    assert control_profile(single)!=control_profile(dual)
    assert arm_control_profile(single)==arm_control_profile(dual)
    scoped=copy.deepcopy(dual);scoped['control_profile_scope']='arm_v2'
    assert control_profile(scoped)==arm_control_profile(single)


@pytest.mark.parametrize('changed', ['gain','zero','sign','inertia','friction'])
def test_actual_physical_changes_are_not_hidden(tmp_path,changed):
    single,dual=profile(tmp_path,False),profile(tmp_path,True)
    expected=arm_control_profile(single)
    if changed in ('gain','inertia'):
        p=dual['control']['gains']['position_params'];value=yaml.safe_load(open(p))
        params=value['controller']['ros__parameters']
        if changed=='gain':params['kp_gains'][-1]+=1.
        else:params['robot_description']=params['robot_description'].replace('<link name="base"/>','<link name="base"><inertial><mass value="2"/></inertial></link>')
    elif changed in ('zero','sign'):
        p=dual['calibration']['path'];value=yaml.safe_load(open(p))
        value['offsets'][-1]['zero_offset' if changed=='zero' else 'axis_sign']+=.1
    else:p=dual['control']['feedforward']['friction_model'];value={'coefficient':.2}
    with open(p,'w') as stream:yaml.safe_dump(value,stream)
    assert arm_control_profile(dual)!=expected


def test_mechanical_geometry_hash_excludes_only_control_interfaces():
    base='<robot name="r"><link name="base"/></robot>'
    control=base.replace('</robot>','<ros2_control name="x"/></robot>')
    changed=base.replace('<link name="base"/>','<link name="base"><inertial><mass value="2"/></inertial></link>')
    assert geometry_sha256(base)==geometry_sha256(control)
    assert geometry_sha256(base)!=geometry_sha256(changed)


def test_scoped_manifest_accepts_identical_right_calibration_and_rejects_a_gain_change(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'factr2_next/src/factr2_next/test'))
    from test_w3_quality import fixture
    from factr2_next.data_collection.quality import sidecar_path,sha256,make_manifest,validate_manifest
    single,dual=profile(tmp_path,False),profile(tmp_path,True)
    files=[]
    for index,physical in enumerate((single,dual,dual)):
        p=fixture(tmp_path/f'{index}.h5',hz=100,phase=index)
        meta=json.loads(sidecar_path(p).read_text())
        meta.update(physical,control_profile_scope='arm_v2',source='real',health_gate_enabled=True)
        meta['calibration']['sha256']=sha256(meta['calibration']['path'])
        for spec in meta['topics'].values():spec['topic']=spec['topic'].replace('/left/','/right/')
        sidecar_path(p).write_text(json.dumps(meta));files.append(p)
    manifest=tmp_path/'split.json'
    make_manifest(manifest,dict(zip(('train','val','test'),[[p] for p in files])),'physical_same_arm')
    p=Path(dual['control']['gains']['position_params']);value=yaml.safe_load(p.read_text())
    value['controller']['ros__parameters']['kp_gains'][-1]+=1.
    p.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError,match='dataset_profile_mismatch'):
        validate_manifest(manifest)


@pytest.mark.parametrize('failure', ['measured_motion','target_motion','stale','left_health'])
def test_stationary_left_collision_guard_rejects_changes(failure):
    import numpy as np
    from w3_coverage_motion import CoverageClient
    client=CoverageClient.__new__(CoverageClient)
    client.stationary_left_deg=np.zeros(7)
    client.state={'left_q':np.zeros(7),'left_cmd':np.zeros(7),'left_healthy':True}
    client.fresh=lambda key:True
    client.guard_stationary_left()
    if failure=='stale':client.fresh=lambda key:False
    elif failure=='left_health':client.state['left_healthy']=False
    else:client.state['left_q' if failure=='measured_motion' else 'left_cmd'][0]=np.deg2rad(3.1)
    with pytest.raises(RuntimeError,match='Left arm moved|left arm feedback|left arm motor'):
        client.guard_stationary_left()
