#!/usr/bin/env bash
# Sequential controlled runs; optional CLI arguments apply equally to both groups.
set -euo pipefail
TASK_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_PROJECT_DIR"
if [[ -x .venv/bin/python ]]; then
    TASK_PYTHON="$TASK_PROJECT_DIR/.venv/bin/python"
else
    TASK_PYTHON="${PYTHON:-python3}"
fi
echo "Group A: original uniform replay"
"$TASK_PYTHON" -u run_sac_diagnostics.py --episodes 1000 --seed 7 "$@" --terminal-fraction 0
echo "Group B: terminal replay fraction 0.25"
"$TASK_PYTHON" -u run_sac_diagnostics.py --episodes 1000 --seed 7 "$@" --terminal-fraction 0.25
echo "Both diagnostic runs completed."
