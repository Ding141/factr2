#!/usr/bin/env bash
set -euo pipefail
factr_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
unset PYTHONPATH
if [[ ! -x "$factr_root/.venv/bin/python" ]]; then
  /usr/bin/python3.10 -m venv --system-site-packages "$factr_root/.venv"
fi
"$factr_root/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
"$factr_root/.venv/bin/python" -m pip install --no-cache-dir -r "$factr_root/requirements/next-py310.lock.txt" --extra-index-url https://download.pytorch.org/whl/cpu
