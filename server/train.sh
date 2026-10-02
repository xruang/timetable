#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -f .venv/bin/activate ]]; then
    echo "Missing .venv. Run: bash server/setup_linux.sh" >&2
    exit 1
fi
source .venv/bin/activate

config_name="${1:-lower_sac_baseline}"
if [[ $# -gt 0 ]]; then
    shift
fi

python -u run_experiment.py run "$config_name" \
    --set lower_device=cuda \
    --set upper_device=cuda \
    --set lower_checkpoint_interval=500 \
    "$@"
