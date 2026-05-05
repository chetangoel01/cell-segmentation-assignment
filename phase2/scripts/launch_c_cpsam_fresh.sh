#!/usr/bin/env bash
# Launch C: Cellpose-SAM (cpsam) trained from scratch — NO warm-start from
# nuclei_cosine_ep125. Phase-1 + phase-2 stack. MPS. ~75 min.
#
# Tests whether the warm-start bias is hurting the existing cpsam_phase2_p1stack
# checkpoint (which inherited from nuclei_cosine).
set -euo pipefail
TS=$(date +%Y%m%d-%H%M%S)
TRAIN_FOVS=$(printf "FOV_%03d," $(seq 101 155) | sed 's/,$//')
LOG="phase2/runs/_logs/train-seg-cpsam_fresh-${TS}.log"
mkdir -p phase2/runs/_logs
echo "log: $LOG"
nohup .venv/bin/python -m phase2 train-segmentation \
  --backend local \
  --base-model cpsam \
  --include-phase1 \
  --train-fovs "$TRAIN_FOVS" \
  --val-fovs FOV_156,FOV_157,FOV_158,FOV_159,FOV_160 \
  --epochs 60 \
  --time-budget 4h30m \
  --device mps \
  --keep-best 2 \
  --exp-name cpsam_fresh_p12 \
  --n-channels 3 \
  --bsize 256 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --weight-decay 0.1 > "$LOG" 2>&1 &
echo "PID: $!"
sleep 3 && tail -10 "$LOG"
