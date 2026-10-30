"""Offline trajectory constraints and smooth interpolation, no robot access."""
import numpy as np
import pytest

from w3_wide_highspeed import AMPLITUDE,CENTER,PROFILES,wide
from w3_continuous_highspeed import smoothness
from w3_targeted_protocol import bounds,coefficients,sample


@pytest.mark.parametrize('seed',[202611269,202612278,202611270,202612279,202611271,202612280,202612281])
def test_wide_excitation_respects_envelope_derivatives_and_continuity(seed):
    p=PROFILES[2]
    k,d=wide(CENTER,AMPLITUDE,p,seed)
    low,high,peak=bounds(k)
    assert np.all(low>=CENTER-AMPLITUDE-.1)
    assert np.all(high<=CENTER+AMPLITUDE+.1)
    limits=np.array([p[x] for x in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])[:,None]
    assert np.all(peak<=limits)
    for end in (k[0],k[-1]):
        np.testing.assert_allclose(end['position_deg'],CENTER)
        assert np.max(np.abs(end['velocity_deg_s']))==0
        assert np.max(np.abs(end['acceleration_deg_s2']))==0
    s=smoothness(k)
    assert s['interior_stopped_samples']==s['interior_knot_stops']==0
    for a,b,c in zip(k,k[1:],k[2:]):
        left,right=coefficients(a,b),coefficients(b,c)
        dt=b['time_seconds']-a['time_seconds']
        np.testing.assert_allclose(left.sum(0),right[0],atol=1e-8)
        np.testing.assert_allclose(sum(i*left[i] for i in range(1,6))/dt,
                                  right[1]/(c['time_seconds']-b['time_seconds']),atol=1e-8)
    q=sample(k,np.arange(0,k[-1]['time_seconds'],.05))[0]
    # Broad motion must improve coverage, not merely oscillate near the center.
    assert np.all(np.ptp(q,axis=0)>AMPLITUDE)
    assert d['interior_intentional_holds']==0


def test_roles_have_different_motion_phases():
    a,_=wide(CENTER,AMPLITUDE,PROFILES[1],202611269)
    b,_=wide(CENTER,AMPLITUDE,PROFILES[1],202611270)
    assert a!=b
