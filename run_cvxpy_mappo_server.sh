#!/usr/bin/env bash
set -euo pipefail

python -m timetable.timetable_0601_bidirectional_online \
  --lower-backend cvxpy \
  --convex-solver CLARABEL \
  --convex-spatial-segments 600 \
  --convex-speed-limit-ratio 0.8 \
  --lower-arrival-tolerance 0.5 \
  --rollout-cache-quantum 0.1 \
  --runtime-adjustment-span 0.5 \
  --dwell-adjustment-span 10 \
  --up-trains 6 \
  --down-trains 6 \
  --upper-episodes 500 \
  --upper-device auto \
  --progress-interval 10 \
  --run-name cvxpy_mappo_6x6 \
  --output-dir timetable/results_cvxpy_mappo
