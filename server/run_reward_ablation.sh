#!/usr/bin/env bash
# Sequential multi-seed run. No resume: existing run directories are refused.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
STEPS="${STEPS:-200000}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-7 17 27 37 47}"
PROFILES="${PROFILES:-arrival_v1 arrival_p16 continuous_p8 continuous_p16}"
OUT="${OUT:-results/reward_ablation_server_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
"$PYTHON" -u server/reward_ablation.py --preflight --output "$OUT/preflight" | tee "$OUT/preflight.log"
for seed in $SEEDS; do
  for profile in $PROFILES; do
    "$PYTHON" -u server/reward_ablation.py --profile "$profile" --seed "$seed" \
      --steps "$STEPS" --eval-steps "$EVAL_STEPS" --device "$DEVICE" --output "$OUT" \
      2>&1 | tee "$OUT/${profile}_seed${seed}.log"
  done
done
"$PYTHON" -u server/analyze_reward_ablation.py "$OUT" | tee "$OUT/analysis.log"
"$PYTHON" -u server/audit_reward_values.py "$OUT" | tee "$OUT/value_audit.log"
