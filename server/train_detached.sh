#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

config_name="${1:-lower_sac_baseline}"
if [[ $# -gt 0 ]]; then
    shift
fi

mkdir -p server_logs
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="server_logs/${config_name}_${timestamp}.log"
pid_path="${log_path}.pid"

nohup bash server/train.sh "$config_name" "$@" >"$log_path" 2>&1 &
training_pid=$!
echo "$training_pid" >"$pid_path"

echo "Training started."
echo "PID : $training_pid"
echo "Log : $log_path"
echo "View: tail -f $log_path"
