#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
"$PYTHON" -u server/check_arrival_reward.py
"$PYTHON" -u run_sac_diagnostics.py --reward-profile arrival_v1 --terminal-fraction 0.0 \
  --episodes 200 --eval-interval 10 --plot-interval 100 \
  --output-dir results/lower_sac_arrival_v1 "$@"
