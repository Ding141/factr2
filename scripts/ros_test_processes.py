"""Project-local process helpers for offline DDS acceptance fixtures."""
from pathlib import Path
import os
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]
W3 = Path('/home/dingyj/w3_dual_arm_ws')


def configure_dds(domain):
    os.environ.update(ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1',
                      FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT/'config/w3/dds_loopback.xml'))


def w3_process(argv, stdout):
    values = {'HOME': os.environ['HOME'], 'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8',
              'PYTHONNOUSERSITE':'1','ROS_LOG_DIR':str(W3/'log/ros'),
              'ROS_DOMAIN_ID':os.environ['ROS_DOMAIN_ID'], 'ROS_LOCALHOST_ONLY':'1',
              'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp',
              'FASTRTPS_DEFAULT_PROFILES_FILE':os.environ['FASTRTPS_DEFAULT_PROFILES_FILE']}
    # Source only Humble + W3 local overlay, then restore requested test DDS settings.
    command = ['bash','--noprofile','--norc','-c',
               'set -e; source /opt/ros/humble/setup.bash; source "$1/install/local_setup.bash"; '
               'shift; export ROS_LOCALHOST_ONLY=1; exec "$@"', 'bash', str(W3), *map(str,argv)]
    return subprocess.Popen(command, env=values, stdout=stdout, stderr=subprocess.STDOUT)


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        code = process.wait(timeout=5)
    return code
