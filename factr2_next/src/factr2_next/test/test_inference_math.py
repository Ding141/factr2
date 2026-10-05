import numpy as np
import pytest
from factr2_next.inference.inference_node import TorqueFilter,InferenceNode
from factr2_next.visualization.web_node import page,plot_keys


@pytest.mark.parametrize('mode',['none','ema','lowpass'])
def test_filter(mode):
    f=TorqueFilter(dict(mode=mode,ema_alpha=.2,cutoff_hz=5,sample_hz=50))
    np.testing.assert_array_equal(f.update([1.,2.]),[1.,2.])
    alpha=1 if mode=='none' else .2 if mode=='ema' else .02/(1/(2*np.pi*5)+.02)
    np.testing.assert_allclose(f.update([3.,4.]),np.array([1.,2.])+alpha*2)
    f.reset();np.testing.assert_array_equal(f.update([7.,8.]),[7.,8.])


@pytest.mark.parametrize('cfg',[{'ema_alpha':0},{'ema_alpha':1.1},{'sample_hz':0},{'cutoff_hz':0},{'ema_alpha':float('nan')},{'mode':'bad'}])
def test_filter_rejection(cfg):
    with pytest.raises(ValueError):TorqueFilter(cfg)


def test_contact_hysteresis():
    # Math methods do not need a ROS graph.
    n=InferenceNode.__new__(InferenceNode)
    n.contact_cfg={'low_threshold':1.,'high_threshold':2.};n.contact_state=False
    assert [n._update_contact_state(v) for v in [.5,2.,1.5,1.,1.5,2.5]]==[False,True,True,False,False,True]
    n.contact_magnitude_cfg={'source':'raw_external_joint_torque','norm':'l1','scale':2.}
    assert n._normalized_contact_magnitude(np.zeros(7),np.array([1.,-3.]))==2.
    n.contact_magnitude_cfg['source']='filtered_external_joint_torque'
    assert n._normalized_contact_magnitude(np.array([2.,2.]),np.zeros(7))==2.


def test_web_dynamic_seventh_and_no_feedback():
    names=[f'left_joint_{i}' for i in range(7)];html=page(names)
    assert 'left_joint_6' in html and 'ext_j7' in html and 'raw_j7' in html and 'free_j7' in html
    assert 'Feedback (Nm)' not in html and 'fb_j' not in html
    keys=plot_keys(7);assert 'ext_j7' in keys and 'raw_j7' in keys and 'free_j7' in keys
    assert 'state' in html and 'last valid' in html and 'statusEl.textContent="live"' not in html


def test_w3_cannot_disable_adapter_gate():
    from pathlib import Path
    import yaml
    from factr2_next.w3_config import validate_w3_config
    root=Path(__file__).resolve().parents[4]
    cfg=yaml.safe_load((root/'config/w3/left/inference.yaml').read_text());cfg['require_adapter_status']=False
    with pytest.raises(ValueError,match='requires adapter diagnostics'):validate_w3_config(cfg,'inference',require_paths=False)
