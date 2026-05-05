#!/usr/bin/env bash
# Pull a trained bundle from Modal volume and run inference using anchor masks.
#
# Usage:
#   pull_and_infer.sh <bundle_kind> <volume_subdir> <local_dir> <submit_dir>
# bundle_kind ∈ {xgb, scanvi, mlp}
#
# Example:
#   pull_and_infer.sh xgb trained/xgb-xl-d8n3000-bg40 phase2/runs/xgb-xl-d8 phase2/runs/SUBMIT_xgb_xl_d8
set -euo pipefail

KIND="$1"
VOL_SUBDIR="$2"
LOCAL_DIR="$3"
SUBMIT_DIR="$4"

VENV="/Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment/.venv/bin/python"
TEST_MASKS="phase2/runs/sota_test_masks"
TEST_FOVS="FOV_E,FOV_F,FOV_G,FOV_H,FOV_I,FOV_J,FOV_K,FOV_L,FOV_M,FOV_N"

mkdir -p "$LOCAL_DIR" "$SUBMIT_DIR"

echo "==> pulling $VOL_SUBDIR -> $LOCAL_DIR"
modal volume get cell-seg-phase2 "$VOL_SUBDIR" "$LOCAL_DIR" --force 2>&1 | tail -10

case "$KIND" in
  xgb)
    BUNDLE=$(find "$LOCAL_DIR" -name "xgb_bundle.joblib" | head -1)
    [ -z "$BUNDLE" ] && { echo "no xgb_bundle.joblib"; exit 1; }
    echo "==> infer_xgb $BUNDLE"
    "$VENV" phase2/scripts/infer_xgb.py \
      --bundle "$BUNDLE" \
      --mode test \
      --fovs "$TEST_FOVS" \
      --masks-dir "$TEST_MASKS" \
      --out-dir "$SUBMIT_DIR"
    ;;
  scanvi)
    BUNDLE=$(find "$LOCAL_DIR" -name "scanvi_bundle.joblib" | head -1)
    [ -z "$BUNDLE" ] && { echo "no scanvi_bundle.joblib"; exit 1; }
    echo "==> infer_scanvi (qry) $BUNDLE"
    "$VENV" phase2/scripts/infer_scanvi_qry.py \
      --bundle "$BUNDLE" \
      --mode test \
      --fovs "$TEST_FOVS" \
      --masks-dir "$TEST_MASKS" \
      --out-dir "$SUBMIT_DIR"
    ;;
  mlp)
    BUNDLE=$(find "$LOCAL_DIR" -name "mlp_bundle.joblib" | head -1)
    [ -z "$BUNDLE" ] && { echo "no mlp_bundle.joblib"; exit 1; }
    echo "==> infer_mlp $BUNDLE"
    "$VENV" phase2/scripts/infer_mlp_hier.py \
      --bundle "$BUNDLE" \
      --mode test \
      --fovs "$TEST_FOVS" \
      --masks-dir "$TEST_MASKS" \
      --out-dir "$SUBMIT_DIR"
    ;;
  *)
    echo "unknown kind: $KIND"; exit 1 ;;
esac

echo "==> validate $SUBMIT_DIR/submission.csv"
"$VENV" phase2/scripts/validate_submission.py "$SUBMIT_DIR/submission.csv"
