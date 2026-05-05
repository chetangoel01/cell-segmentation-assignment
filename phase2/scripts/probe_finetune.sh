#!/usr/bin/env bash
# Helper: run validate_local + infer-baseline + validate_submission for one
# (segmentation_checkpoint, classifier_dir) pair. Prints a clear pass/fail
# summary and writes outputs under phase2/runs/probe-<TAG>/.
#
# Usage: phase2/scripts/probe_finetune.sh TAG SEG_CKPT CLASSIFIER_DIR
#
# Example:
#   phase2/scripts/probe_finetune.sh slot1 \
#     phase2/runs/20260503-130529-train-seg-cpsam_phase2_p1stack/models/cpsam_phase2_p1stack_ep020 \
#     phase2/runs/baseline-codelab-p2only

set -euo pipefail

TAG="${1:?usage: probe_finetune.sh TAG SEG_CKPT CLASSIFIER_DIR}"
SEG_CKPT="${2:?missing SEG_CKPT}"
CLF_DIR="${3:?missing CLASSIFIER_DIR}"

OUT_ROOT="phase2/runs/probe-${TAG}"
mkdir -p "$OUT_ROOT"
LOCAL_VAL_OUT="$OUT_ROOT/local_val.json"
INFER_OUT="$OUT_ROOT/infer"
SUB_CSV="$INFER_OUT/submission.csv"

echo "=========================================================================="
echo "PROBE: $TAG"
echo "  seg ckpt   : $SEG_CKPT"
echo "  classifier : $CLF_DIR"
echo "  out root   : $OUT_ROOT"
echo "=========================================================================="

if [ ! -e "$SEG_CKPT" ]; then
  echo "[fatal] segmentation checkpoint not found: $SEG_CKPT" >&2
  exit 2
fi
if [ ! -d "$CLF_DIR" ]; then
  echo "[fatal] classifier dir not found: $CLF_DIR" >&2
  exit 2
fi

VAL_FOVS="FOV_156,FOV_157,FOV_158,FOV_159,FOV_160"
TEST_FOVS="FOV_E,FOV_F,FOV_G,FOV_H,FOV_I,FOV_J,FOV_K,FOV_L,FOV_M,FOV_N"

echo
echo "=== STEP 1/3: local validation on $VAL_FOVS ==="
.venv/bin/python phase2/scripts/validate_local.py \
  --models-dir "$CLF_DIR" \
  --seg-checkpoint "$SEG_CKPT" \
  --val-fovs "$VAL_FOVS" \
  --include-spot-density \
  --spot-density-sigma 8.0 \
  --cellpose-diameter 0.0 \
  --cellprob-threshold -0.5 \
  --flow-threshold 0.4 \
  --nn-radius 0.0 \
  --device "${DEVICE:-mps}" \
  --out "$LOCAL_VAL_OUT" 2>&1 | tee "$OUT_ROOT/local_val.log" | tail -40

if [ ! -f "$LOCAL_VAL_OUT" ]; then
  echo "[fatal] validate_local did not produce $LOCAL_VAL_OUT" >&2
  exit 3
fi

# Parse local mean ARI and frac_in_cell stats.
read -r MEAN_ARI MEAN_FIC_PRED MEAN_FIC_GT <<<"$(.venv/bin/python -c "
import json, statistics
d = json.load(open('$LOCAL_VAL_OUT'))
fovs = d['per_fov']
fic_pred = statistics.mean(f['frac_in_cell_pred'] for f in fovs.values())
fic_gt   = statistics.mean(f['frac_in_cell_gt'] for f in fovs.values())
print(f'{d[\"mean_ari\"]:.4f} {fic_pred:.4f} {fic_gt:.4f}')
")"

echo
echo "=== local-val summary ==="
echo "  mean ARI         : $MEAN_ARI"
echo "  frac_in_cell pred: $MEAN_FIC_PRED  (target: close to GT)"
echo "  frac_in_cell gt  : $MEAN_FIC_GT"

echo
echo "=== STEP 2/3: infer on test FOVs $TEST_FOVS ==="
.venv/bin/python -m phase2 infer-baseline --backend local \
  --models-dir "$CLF_DIR" \
  --seg-checkpoint "$SEG_CKPT" \
  --test-fovs "$TEST_FOVS" \
  --include-spot-density \
  --spot-density-sigma 8.0 \
  --cellpose-diameter 0.0 \
  --cellprob-threshold -0.5 \
  --flow-threshold 0.4 \
  --nn-radius 0.0 \
  --device "${DEVICE:-mps}" \
  --out-dir "$INFER_OUT" 2>&1 | tee "$OUT_ROOT/infer.log" | tail -40

if [ ! -f "$SUB_CSV" ]; then
  echo "[fatal] inference did not produce $SUB_CSV" >&2
  exit 4
fi

echo
echo "=== STEP 3/3: structural validate $SUB_CSV ==="
.venv/bin/python phase2/scripts/validate_submission.py "$SUB_CSV" 2>&1 | tee -a "$OUT_ROOT/infer.log"

# Test-FOV mean frac_in_cell from infer summary.
TEST_FIC=$(.venv/bin/python -c "
import json, statistics
d = json.load(open('$INFER_OUT/summary.json'))
print(f'{statistics.mean(f[\"frac_in_cell\"] for f in d[\"per_fov\"].values()):.4f}')
")

echo
echo "=========================================================================="
echo "PROBE $TAG SUMMARY"
echo "  local mean ARI       : $MEAN_ARI"
echo "  local frac_in_cell   : $MEAN_FIC_PRED  (gt $MEAN_FIC_GT)"
echo "  test  frac_in_cell   : $TEST_FIC"
echo "  submission CSV       : $SUB_CSV"
echo "=========================================================================="
echo
echo "GATES:"
echo "  [ ] mean ARI >= 0.55                  ($MEAN_ARI)"
echo "  [ ] test frac_in_cell <  0.24         ($TEST_FIC)"
echo
echo "If both pass, submit with:"
echo "  kaggle competitions submit cell-type-classification-phase-2-cs-gy-9223 \\"
echo "    -f $SUB_CSV \\"
echo "    -m \"PROBE-$TAG: ckpt=$(basename "$SEG_CKPT") clf=$(basename "$CLF_DIR") "\
"local=$MEAN_ARI fic_test=$TEST_FIC\""
