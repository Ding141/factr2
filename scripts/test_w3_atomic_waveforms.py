import copy

import numpy as np
import pytest

from record_w3_atomic_waveforms import KEYS,validate_frame,BASE_KEYS,validate_base_frame


def packet():
    values={k:[0.]*7 for k in KEYS}
    values.update(measured_tau_nm=[1.]*7,base_external_raw_nm=[1.]*7,
                  measured_filtered_nm=[2.]*7,corrected_free_pred_nm=[1.5]*7,
                  compensated_raw_nm=[.5]*7,external_filtered_nm=[.4]*7,correction_nm=[.1]*7)
    return dict(schema='w3_kinematic_residual_v1',joint_names=[f'right_joint_{j}' for j in range(7)],
        units='Nm',base_metadata_sha256='abc',recording_frame=values,
        raw_external_nm=[.5]*7,final_external_nm=[.4]*7,upstream_external_nm=[1.]*7,correction_nm=[.1]*7)


def test_atomic_packet_carries_all_unrounded_inputs_and_consistent_outputs():
    p=packet();p['recording_frame']['q_deg'][0]=.123456789
    a=validate_frame(p,'abc');assert a['q_deg'][0]==.123456789


@pytest.mark.parametrize('field',['missing','mixed_frame','wrong_model','nan'])
def test_bad_atomic_frame_is_rejected_before_recording(field):
    p=packet()
    if field=='missing':del p['recording_frame']['command_deg']
    elif field=='mixed_frame':p['recording_frame']['base_free_pred_nm'][0]=.1
    elif field=='wrong_model':p['base_metadata_sha256']='different'
    else:p['recording_frame']['q_deg'][0]=float('nan')
    with pytest.raises((ValueError,KeyError,AssertionError)):validate_frame(p,'abc')


def test_uncompensated_base_frame_cannot_hide_a_nonzero_correction():
    a={k:[0.]*7 for k in BASE_KEYS}
    measured=np.array([.123456789]*7,np.float32)
    free=np.array([.023456789]*7,np.float32)
    raw=(measured-free).tolist()
    a.update(measured_tau_nm=measured.tolist(),base_free_pred_nm=free.tolist(),
             base_external_raw_nm=raw,external_filtered_nm=raw)
    p=dict(schema='w3_frozen_sigma_observer_v1',joint_names=[f'right_joint_{j}' for j in range(7)],
        units='Nm',base_metadata_sha256='abc',compensation_applied=False,recording_frame=a,
        correction_nm=[0.]*7,raw_external_nm=raw,final_external_nm=raw)
    validate_base_frame(p,'abc')
    for key in ('compensation_applied','correction_nm'):
        bad=copy.deepcopy(p);bad[key]=True if key=='compensation_applied' else [.001]*7
        with pytest.raises((ValueError,AssertionError)):validate_base_frame(bad,'abc')
