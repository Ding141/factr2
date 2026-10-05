from glob import glob
from setuptools import find_packages, setup

setup(name='factr2_w3_adapter', version='0.1.0', packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/factr2_w3_adapter']),
                  ('share/factr2_w3_adapter', ['package.xml']),
                  ('share/factr2_w3_adapter/config', glob('config/*.yaml')),
                  ('share/factr2_w3_adapter/launch', glob('launch/*.launch.py'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='dingyj', maintainer_email='dingyj@example.com',
      description='Read-only W3/NEXT standard-message adapter', license='Apache-2.0',
      extras_require={'test': ['pytest']},
      entry_points={'console_scripts': ['w3_next_adapter = factr2_w3_adapter.adapter_node:main']})
