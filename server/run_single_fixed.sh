#!/usr/bin/env bash
# Fixed task with the original episode-based logging and timestamped output layout.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-1000}"
SEEDS="${SEEDS:-7 17 27 37 47}"
OUT="${OUT:-results/single_hc_dgnl_255t_87s_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
for seed in $SEEDS; do
  "$PYTHON" -u run_single_scenario.py \
    --direction up --section HC-DGNL --mass-t 255 --target-time-s 87 \
    --reward-profile average_speed_v1 --alpha 0.02 --device "$DEVICE" \
    --episodes "$EPISODES" --log-interval 10 --eval-interval 100 \
    --plot-interval 100 --seed "$seed" --output-dir "$OUT" \
    2>&1 | tee "$OUT/seed${seed}.log"
done
