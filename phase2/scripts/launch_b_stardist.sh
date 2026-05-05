#!/usr/bin/env bash
# Launch B: StarDist 2D training on phase-1 + phase-2 stack.
# Runs in /tmp/stardist_venv (Python 3.11 + tf-metal). Background, ~90 min.
#
# Output: phase2/external_models/stardist_p12_v1/ (model weights + thresholds)
# Log:    phase2/runs/_logs/train-seg-stardist-<TS>.log
set -euo pipefail
TS=$(date +%Y%m%d-%H%M%S)
LOG="phase2/runs/_logs/train-seg-stardist-${TS}.log"
mkdir -p phase2/runs/_logs phase2/external_models
echo "log: $LOG"
nohup /tmp/stardist_venv/bin/python /tmp/stardist_train.py \
  --epochs 80 \
  --steps-per-epoch 200 \
  --batch-size 4 \
  --patch-size 256 \
  --n-rays 32 \
  --include-phase1 \
  > "$LOG" 2>&1 &
echo "PID: $!"
sleep 4 && tail -20 "$LOG"
