#!/usr/bin/env bash
set -e
factr_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Explicit allowlist: no inherited Python, Conda, ROS or workspace overlay.
factr_dds_env=()
# An empty XML path makes Fast DDS print a parser error on every process.
if [[ -n ${FASTRTPS_DEFAULT_PROFILES_FILE:-} ]]; then
  factr_dds_env+=("FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE")
fi
exec env -i HOME="$HOME" USER="${USER:-dingyj}" LANG="${LANG:-C.UTF-8}" \
  PIP_NO_CACHE_DIR=1 PATH=/usr/bin:/bin ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}" \
  ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}" \
  RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
  "${factr_dds_env[@]}" \
  bash --noprofile --norc -c 'source "$1/scripts/next_env.sh" || exit; cd "$1"; shift; exec "$@"' bash "$factr_root" "$@"
