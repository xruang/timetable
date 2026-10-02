#!/usr/bin/env bash
set -euo pipefail

python server/check_cvxpy_lower.py

python -m timetable.timetable_0601_bidirectional_online \
  --lower-backend cvxpy \
  --convex-solver CLARABEL \
  --convex-spatial-segments 40 \
  --convex-speed-limit-ratio 0.8 \
  --up-trains 1 \
  --down-trains 1 \
  --upper-episodes 1 \
  --runtime-adjustment-span 0 \
  --dwell-adjustment-span 0 \
  --upper-device cpu \
  --progress-interval 0 \
  --run-name smoke_cvxpy_mappo \
  --output-dir timetable/results_cvxpy_mappo_smoke
