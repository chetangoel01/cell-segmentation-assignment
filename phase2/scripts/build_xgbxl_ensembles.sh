#!/usr/bin/env bash
# Build ensembles when XGB-XL bundles arrive.
# Pattern from data: 4-way kNN E2 = 0.5623, 5-way E2-PLUS = 0.5675.
# If XGB-XL scores ≥0.5483 alone, it could replace AWS-cp0.5 (0.5483) in the 5-way
# or add as 6th voter.
set -euo pipefail
VENV="/Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment/.venv/bin/python"
cd /Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment

run_vote () {
    local out="$1"; shift
    mkdir -p "$out"
    "$VENV" phase2/scripts/vote_submissions.py --out "$out/submission.csv" "$@" 2>&1 | tail -8
}

# Pick best XGB-XL by Kaggle score once submitted; fallback = first available
XGB_BEST=""
for d in SUBMIT_xgb_xl_d8 SUBMIT_xgb_xl_d6 SUBMIT_xgb_xl_d5; do
    if [ -f "phase2/runs/$d/submission.csv" ]; then
        XGB_BEST="phase2/runs/$d/submission.csv"
        break
    fi
done

if [ -z "$XGB_BEST" ]; then
    echo "[fatal] no XGB-XL submission yet"; exit 1
fi

echo "Best XGB-XL: $XGB_BEST"

# E_XGB1: 5-way kNN + XGB-XL (6 voters, kNN-stack augmented with single boosted-trees voter)
run_vote phase2/runs/SUBMIT_FINAL_5way_PLUS_XGBXL \
    phase2/runs/SUBMIT_aws001_clusfilt_bg40_dist/submission.csv \
    phase2/runs/SUBMIT_hier_aws_kp15/submission.csv \
    phase2/runs/SUBMIT_aws_k15_bg40_dist/submission.csv \
    phase2/runs/SUBMIT_sota_erode2/submission.csv \
    phase2/runs/SUBMIT_aws_cp0.5/submission.csv \
    "$XGB_BEST"

# E_XGB2: 4-way kNN + XGB-XL replacing weakest (5 voters)
run_vote phase2/runs/SUBMIT_FINAL_5way_xgb_replaces_cp0.5 \
    phase2/runs/SUBMIT_aws001_clusfilt_bg40_dist/submission.csv \
    phase2/runs/SUBMIT_hier_aws_kp15/submission.csv \
    phase2/runs/SUBMIT_aws_k15_bg40_dist/submission.csv \
    phase2/runs/SUBMIT_sota_erode2/submission.csv \
    "$XGB_BEST"

echo ""
echo "Submitting:"
"$VENV" -c "
import subprocess
for path,label in [
  ('phase2/runs/SUBMIT_FINAL_5way_PLUS_XGBXL/submission.csv', '6-way: 5-way kNN + XGB-XL (boosted trees voter)'),
  ('phase2/runs/SUBMIT_FINAL_5way_xgb_replaces_cp0.5/submission.csv', '5-way: 4 top kNN + XGB-XL replaces weakest cp0.5')
]:
    print('Submitting:', path)
"
