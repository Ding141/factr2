#!/usr/bin/env bash
# Source in a clean shell. Never combine the W3 and NEXT overlays.
_factr_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for _factr_path in ${AMENT_PREFIX_PATH//:/ } ${CMAKE_PREFIX_PATH//:/ } ${PYTHONPATH//:/ }; do
  case "$_factr_path" in
    /opt/ros/humble*|"$_factr_root"/*) ;;
    *) echo "Refusing foreign overlay: $_factr_path. Use scripts/next.sh." >&2; return 2 ;;
  esac
done
if [[ -n ${VIRTUAL_ENV:-} && $VIRTUAL_ENV != "$_factr_root/.venv" ]]; then
  echo 'Refusing a foreign venv; use scripts/next.sh.' >&2; return 2
fi
export PYTHONNOUSERSITE=1
export ROS_LOG_DIR="$_factr_root/log/ros"
export MPLCONFIGDIR="$_factr_root/log/matplotlib"
mkdir -p "$MPLCONFIGDIR"
_factr_local=${ROS_LOCALHOST_ONLY:-1}
source /opt/ros/humble/setup.bash
source "$_factr_root/.venv/bin/activate"
if [[ -f "$_factr_root/install/local_setup.bash" ]]; then
  source "$_factr_root/install/local_setup.bash"
fi
export ROS_LOCALHOST_ONLY=$_factr_local
unset _factr_path _factr_root _factr_local
