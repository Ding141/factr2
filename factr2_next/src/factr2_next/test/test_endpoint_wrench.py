import numpy as np
import pytest
from factr2_next.inference.endpoint_wrench import estimate_wrench, rotate_wrench_at_same_origin


def full_rank_jacobian():
    return np.c_[np.diag([.3, .4, .5, 1., 1.1, 1.2]), np.zeros(6)]


def test_known_force_and_moment_recovered_without_unit_mix():
    j = full_rank_jacobian()
    expected = np.array([3., -2., 1., .4, -.2, .1])
    for length in (.1, .4, 1.):
        value, info = estimate_wrench(j, j.T @ expected,
                                     characteristic_length_m=length, damping=0)
        np.testing.assert_allclose(value, expected, atol=1e-12)
        assert info['valid'] and info['torque_fit_error_nm'] < 1e-12


def test_singular_pose_is_unavailable_instead_of_large_spurious_force():
    j = full_rank_jacobian(); j[2] = 0
    value, info = estimate_wrench(j, np.ones(7))
    assert value is None and info['reason'] == 'near_singular'
    j[2, 2] = 1e-5
    value, info = estimate_wrench(j, np.ones(7))
    assert value is None and info['condition_number'] > 100


def test_torque_in_jacobian_nullspace_cannot_be_a_single_tool_wrench():
    j = full_rank_jacobian(); torque = np.zeros(7); torque[-1] = 1
    value, info = estimate_wrench(j, torque)
    assert value is None and info['reason'] == 'inconsistent_with_tool_contact'


def test_axes_rotate_at_tool_origin_without_moment_arm_translation():
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rotate_wrench_at_same_origin([0, 2, 0, -3, 0, 0], r),
                               [2, 0, 0, 0, 3, 0])


@pytest.mark.parametrize('change', [
    {'characteristic_length_m': 0}, {'damping': -1}, {'max_condition': 1},
    {'max_relative_residual': 0}, {'damping': float('nan')}])
def test_invalid_solver_configuration_rejected(change):
    with pytest.raises(ValueError):
        estimate_wrench(full_rank_jacobian(), np.zeros(7), **change)
