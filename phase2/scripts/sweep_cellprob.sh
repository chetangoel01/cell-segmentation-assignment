#!/usr/bin/env bash
# Sweep cellprob_threshold + flow_threshold on a fixed seg checkpoint.
# Outputs JSONs at /tmp/sweep-<TAG>-cp<X>-fl<Y>.json for each cell.
#
# Usage:
#   phase2/scripts/sweep_cellprob.sh TAG SEG_CKPT [--tta]
#
# Then run compare_seg.py with the resulting JSONs to find the best (cp, fl).
set -euo pipefail
TAG="${1:?TAG}"
CKPT="${2:?CKPT}"
TTA_FLAG="${3:-}"

CLF=phase2/runs/baseline-codelab-rf500-log1p-mf01
VAL_FOVS=FOV_156,FOV_157,FOV_158,FOV_159,FOV_160

echo "=========================================================="
echo "cellprob/flow sweep: TAG=$TAG  CKPT=$CKPT  TTA=$TTA_FLAG"
echo "=========================================================="

# (cp, fl) combinations to try. cp=-1 is more permissive, +0.5 conservative.
configs=(
  "-1.0 0.4"
  "-0.5 0.4"
  "0.0  0.4"
  "0.5  0.4"
  "-0.5 0.2"
  "-0.5 0.6"
)

for combo in "${configs[@]}"; do
  read -r cp fl <<<"$combo"
  out=/tmp/sweep-${TAG}-cp${cp}-fl${fl}.json
  echo
  echo "=== cp=$cp fl=$fl -> $out ==="
  .venv/bin/python phase2/scripts/validate_local.py \
    --models-dir "$CLF" \
    --seg-checkpoint "$CKPT" \
    --val-fovs "$VAL_FOVS" \
    --include-spot-density --spot-density-sigma 8.0 \
    --cellpose-diameter 0.0 \
    --cellprob-threshold "$cp" \
    --flow-threshold "$fl" \
    --nn-radius 0.0 \
    --device mps \
    $TTA_FLAG \
    --out "$out" 2>&1 | tail -8
done

echo
echo "=== sweep summary ==="
.venv/bin/python phase2/scripts/compare_seg.py \
  $(for combo in "${configs[@]}"; do
    read -r cp fl <<<"$combo"
    echo "cp${cp}fl${fl}=/tmp/sweep-${TAG}-cp${cp}-fl${fl}.json"
  done)
