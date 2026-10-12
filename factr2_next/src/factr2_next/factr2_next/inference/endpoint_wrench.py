"""Equivalent tool-origin wrench from NEXT joint-torque residuals.

The sign convention follows NEXT's residual, not a calibrated force sensor.
Assumes all external loading acts at the selected tool origin.
"""
import numpy as np


def geometry_quality(jacobian, characteristic_length_m=.4, mode='full_wrench'):
    """Shared planner/runtime criterion; rotation of tool axes leaves SVD unchanged."""
    j = np.asarray(jacobian, dtype=float)
    if (j.shape != (6, 7) or not np.isfinite(j).all()
            or not np.isfinite(characteristic_length_m) or characteristic_length_m <= 0):
        raise ValueError('geometry_shape_or_length')
    scaled = solver_jacobian(j, characteristic_length_m, mode)
    s = np.linalg.svd(scaled.T, compute_uv=False)
    return dict(condition_number=float(s[0]/s[-1]) if s[-1] > 1e-12 else None,
                scaled_singular_values=s.tolist(), observable_rank=int(np.sum(s > 1e-8)),
                scaled_noise_gain=float(1/s[-1]) if s[-1] > 1e-12 else None)


def solver_jacobian(jacobian, length, mode):
    if mode not in ('full_wrench', 'force_only'):
        raise ValueError('Unknown endpoint estimation mode')
    scaled = jacobian.copy()
    scaled[:3] /= length
    # A point force has only three unknowns. Angular singularity alone does not
    # make those unknowns unobservable; external moments are assumed absent.
    return scaled[:3] if mode == 'force_only' else scaled


def estimate_wrench(jacobian, torque, *, characteristic_length_m=0.4,
                    damping=0.005, max_condition=100., max_relative_residual=0.35,
                    mode='full_wrench'):
    j = np.asarray(jacobian, dtype=float)
    tau = np.asarray(torque, dtype=float)
    params = np.asarray([characteristic_length_m, damping, max_condition, max_relative_residual])
    if (j.shape != (6, 7) or tau.shape != (7,) or not np.isfinite(j).all()
            or not np.isfinite(tau).all() or not np.isfinite(params).all()
            or characteristic_length_m <= 0 or damping < 0 or max_condition <= 1
            or not 0 < max_relative_residual <= 1):
        raise ValueError('wrench_shape_finite_or_solver_parameters')
    # Scale force/torque coordinates consistently before SVD and damping:
    # A @ [L*Fx, L*Fy, L*Fz, Mx, My, Mz] = tau.
    scaled = solver_jacobian(j, characteristic_length_m, mode)
    u, s, vh = np.linalg.svd(scaled.T, full_matrices=False)
    condition = float(s[0] / s[-1]) if s[-1] > 1e-12 else None
    diagnostics = {'condition_number': condition, 'scaled_singular_values': s.tolist(),
                   'mode': mode, 'observable_rank': int(np.sum(s > 1e-8)),
                   'moments_observed': mode == 'full_wrench',
                   'assumption': 'tool_point_force_zero_external_moment' if mode == 'force_only'
                                 else 'all_external_loading_at_tool_origin'}
    if condition is None or condition > max_condition:
        return None, dict(diagnostics, reason='near_singular', valid=False)
    solution = vh.T @ ((s / (s * s + damping * damping)) * (u.T @ tau))
    wrench = np.r_[solution, np.zeros(3)] if mode == 'force_only' else solution.copy()
    wrench[:3] /= characteristic_length_m
    error = float(np.linalg.norm(j.T @ wrench - tau))
    relative = error / max(float(np.linalg.norm(tau)), 0.1)
    diagnostics.update(torque_fit_error_nm=error, relative_fit_error=relative)
    if relative > max_relative_residual:
        return None, dict(diagnostics, reason='inconsistent_with_tool_contact', valid=False)
    return wrench, dict(diagnostics, reason='ok', valid=True)


def rotate_wrench_at_same_origin(wrench, rotation_base_from_tool):
    """Rotate axes only; both input/output moments are about the tool origin."""
    value = np.asarray(wrench, dtype=float)
    r = np.asarray(rotation_base_from_tool, dtype=float)
    if (value.shape != (6,) or r.shape != (3, 3) or not np.isfinite(value).all()
            or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-8)):
        raise ValueError('wrench_or_rotation_invalid')
    return np.r_[r.T @ value[:3], r.T @ value[3:]]
