"""Offline checks only: no ROS publisher, controller or motor socket."""
import math

import numpy as np
import pytest

from w3_continuous_highspeed import AMPLITUDE, CENTER, PROFILES, continuous, smoothness
from w3_targeted_protocol import bounds, coefficients, sample


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('seed',[202610250,202610251,202610252])
def test_continuous_path_limits_rest_endpoints_and_no_internal_stop(profile,seed):
    knots,definition=continuous(CENTER,AMPLITUDE,profile,seed)
    low,high,peak=bounds(knots)
    # Bernstein hull is a conservative enclosure, not the actual extrema.
    assert np.all(low>=CENTER-AMPLITUDE-.05)
    assert np.all(high<=CENTER+AMPLITUDE+.05)
    q=sample(knots,np.arange(0,knots[-1]['time_seconds'],.005))[0]
    assert np.all(q>=CENTER-AMPLITUDE-1e-5)
    assert np.all(q<=CENTER+AMPLITUDE+1e-5)
    limits=np.array([profile[k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])[:,None]
    assert np.all(peak<=limits)
    for k in (knots[0],knots[-1]):
        np.testing.assert_allclose(k['position_deg'],CENTER)
        np.testing.assert_allclose(k['velocity_deg_s'],0.)
        np.testing.assert_allclose(k['acceleration_deg_s2'],0.)
    s=smoothness(knots)
    assert s['interior_stopped_samples']==s['interior_knot_stops']==0
    assert s['maximum_knot_jerk_jump_deg_s3']<.3
    assert definition['one_native_trajectory'] and definition['interior_arrival_waits']==0
    for a,b,c in zip(knots,knots[1:],knots[2:]):
        left,right=coefficients(a,b),coefficients(b,c)
        dt=b['time_seconds']-a['time_seconds']
        for d in range(3):
            value=sum(left[k]*math.factorial(k)/math.factorial(k-d)/dt**d for k in range(d,6))
            expected=right[d]*math.factorial(d)/(c['time_seconds']-b['time_seconds'])**d
            np.testing.assert_allclose(value,expected,atol=1e-8)


def test_same_seed_reproducible_but_session_phases_independent():
    first,_=continuous(CENTER,AMPLITUDE,PROFILES[1],202610250)
    again,_=continuous(CENTER,AMPLITUDE,PROFILES[1],202610250)
    other,_=continuous(CENTER,AMPLITUDE,PROFILES[1],202610251)
    assert first==again and first!=other


def test_retimming_reduces_physical_derivatives():
    p=dict(PROFILES[1],velocity_deg_s=1.,acceleration_deg_s2=.03,jerk_deg_s3=.003)
    ks,d=continuous(CENTER,AMPLITUDE,p,13)
    assert d['actual_seconds']>90
    limits=np.array([p[k] for k in ('velocity_deg_s','acceleration_deg_s2','jerk_deg_s3')])[:,None]
    assert np.all(bounds(ks)[2]<=limits)
