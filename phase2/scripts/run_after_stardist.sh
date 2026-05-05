#!/usr/bin/env bash
# After StarDist training completes:
#   1. Run StarDist inference on val FOVs 156-160 -> per-FOV mask .npy files
#   2. Run validate_local with --masks-dir to compute ARI + frac_in_cell
#   3. Run compare_seg.py with old + cyto3 + stardist JSONs
#
# Inputs (defaults):
#   --model-dir  phase2/external_models
#   --model-name stardist_p12_v1
set -euo pipefail

MODEL_BASEDIR="${1:-phase2/external_models}"
MODEL_NAME="${2:-stardist_p12_v1}"
SPLIT_FOR_VAL="train"
VAL_FOVS="FOV_156,FOV_157,FOV_158,FOV_159,FOV_160"
MASKS_DIR="phase2/runs/stardist_val_masks"
TS=$(date +%Y%m%d-%H%M%S)
INFER_LOG="phase2/runs/_logs/infer-stardist-val-${TS}.log"
VAL_LOG="phase2/runs/_logs/validate-stardist-${TS}.log"
mkdir -p phase2/runs/_logs "$MASKS_DIR"

echo "=== STEP 1: StarDist val inference -> $MASKS_DIR ==="
/tmp/stardist_venv/bin/python /tmp/stardist_infer.py \
  --model-dir "$MODEL_BASEDIR" \
  --model-name "$MODEL_NAME" \
  --out-dir "$MASKS_DIR" \
  --fovs "$VAL_FOVS" \
  --split train 2>&1 | tee "$INFER_LOG"

echo
echo "=== STEP 2: validate_local --masks-dir $MASKS_DIR ==="
.venv/bin/python phase2/scripts/validate_local.py \
  --models-dir phase2/runs/baseline-codelab-rf500-log1p-mf01 \
  --masks-dir "$MASKS_DIR" \
  --val-fovs "$VAL_FOVS" \
  --nn-radius 0.0 \
  --device cpu \
  --out /tmp/val_stardist.json 2>&1 | tee "$VAL_LOG" | tail -30

echo
echo "=== STEP 3: compare_seg.py side-by-side ==="
.venv/bin/python phase2/scripts/compare_seg.py \
  old=/tmp/val_old_nuclei.json \
  cyto3=/tmp/val_cyto3_ep035.json \
  stardist=/tmp/val_stardist.json
