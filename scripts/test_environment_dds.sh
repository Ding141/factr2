#!/usr/bin/env bash
set -euo pipefail
factr_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FASTRTPS_DEFAULT_PROFILES_FILE="$factr_root/config/w3/dds_loopback.xml"
mkdir -p "$factr_root/reports/generated/1"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
  ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  FASTRTPS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE" \
  ROS_LOG_DIR="$factr_root/log/ros" \
  bash --noprofile --norc -c 'source /opt/ros/humble/setup.bash; source /home/dingyj/w3_dual_arm_ws/install/local_setup.bash; export ROS_LOCALHOST_ONLY=1; exec /usr/bin/python3.10 "$1/tests/dds_probe.py" receive' bash "$factr_root" \
  > "$factr_root/reports/generated/1/dds.json" 2>&1 &
probe_pid=$!
trap 'kill "$probe_pid" 2>/dev/null || true' EXIT
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=73 "$factr_root/scripts/next.sh" python tests/dds_probe.py send
wait "$probe_pid"
cat "$factr_root/reports/generated/1/dds.json"
