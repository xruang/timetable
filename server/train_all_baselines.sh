#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

for config_name in lower_dqn_baseline lower_ddpg_baseline lower_sac_baseline; do
    echo "===== Starting ${config_name} ====="
    bash server/train.sh "$config_name"
done
