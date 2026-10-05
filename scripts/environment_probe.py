#!/usr/bin/env python3
import importlib
import importlib.metadata
import json
import os
import sys

assert sys.version_info[:2] == (3, 10)
assert sys.executable == os.path.abspath('.venv/bin/python')
assert not any('w3_dual_arm_ws' in p for p in sys.path)
result = {'executable': sys.executable, 'python': sys.version,
          'rmw': os.environ['RMW_IMPLEMENTATION'], 'domain': os.environ['ROS_DOMAIN_ID'],
          'prefixes': os.environ.get('AMENT_PREFIX_PATH'), 'modules': {}}
for name in ('rclpy', 'sensor_msgs', 'diagnostic_msgs', 'message_filters',
             'ament_index_python', 'numpy', 'torch', 'h5py', 'scipy', 'yaml', 'termcolor',
             'colcon_core', 'setuptools'):
    m = importlib.import_module(name)
    version = getattr(m, '__version__', None)
    if version is None:
        try:
            version = importlib.metadata.version('PyYAML' if name == 'yaml' else name)
        except importlib.metadata.PackageNotFoundError:
            version = 'ROS apt (see dpkg snapshot)'
    result['modules'][name] = {'path': m.__file__, 'version': version}
import numpy as np
import torch
from factr2_next.training.models import build_model
model = build_model({'type': 'lstm', 'state_mode': 'stateless', 'dropout': 0.1}, 21, 7, 50)
with torch.no_grad():
    y = model(torch.from_numpy(np.zeros((2, 50, 21), dtype=np.float32)))
assert y.shape == (2, 7) and torch.isfinite(y).all()
result['cpu_forward'] = list(y.shape)
print(json.dumps(result, indent=2))
