"""Offline checks using actual arm geometry; no robot motion or ROS graph."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from w3_coverage_plan import ArmModel
from factr2_next.inference.endpoint_wrench import estimate_wrench, rotate_wrench_at_same_origin


def model():
    root = ET.fromstring((Path(__file__).resolve().parents[2] /
                          'dual_arm_robot/src/description/dual_arm_support/urdf/dual_arm_robot.urdf').read_text())
    for old in root.findall('ros2_control'): root.remove(old)
    hardware = ET.SubElement(root, 'ros2_control')
    for i in range(7):
        name = f'right_joint_{i}'
        bounds = root.find(f'joint[@name="{name}"]/limit')
        j = ET.SubElement(hardware, 'joint', name=name)
        for key, attr in [('urdf_lower', 'lower'), ('urdf_upper', 'upper')]:
            ET.SubElement(j, 'param', name=key).text = bounds.get(attr)
    return ArmModel(ET.tostring(root, encoding='unicode'), 'right', 'right_attachment_point')


def test_real_geometry_known_tool_wrench_and_virtual_work():
    m = model(); q = np.deg2rad([10, 20, 10, -30, 5, 8, 5])
    pose, j = m.fk(q, True)
    rotation = pose[:3, :3]
    tool = np.array([2., -1., .5, .2, -.3, .1])
    base = np.r_[rotation @ tool[:3], rotation @ tool[3:]]
    tau = j.T @ base
    estimated, info = estimate_wrench(j, tau, damping=0)
    assert info['valid']
    np.testing.assert_allclose(rotate_wrench_at_same_origin(estimated, rotation), tool, atol=1e-12)
    dq = np.array([.1, -.2, .03, .02, -.01, .03, -.04])
    np.testing.assert_allclose(tau @ dq, base @ (j @ dq), atol=1e-12)
    for i in range(7):
        perturbed = q.copy(); perturbed[i] += 1e-7
        new_pose = m.fk(perturbed)
        np.testing.assert_allclose((new_pose[:3, 3] - pose[:3, 3]) / 1e-7, j[:3, i], atol=5e-8)
        angular_matrix = ((new_pose[:3, :3] - rotation) / 1e-7) @ rotation.T
        np.testing.assert_allclose([angular_matrix[2, 1], angular_matrix[0, 2], angular_matrix[1, 0]], j[3:, i], atol=8e-8)


def test_real_geometry_zero_pose_rejected_as_singular():
    _, j = model().fk(np.zeros(7), True)
    wrench, info = estimate_wrench(j, np.ones(7))
    assert wrench is None and info['reason'] == 'near_singular'
