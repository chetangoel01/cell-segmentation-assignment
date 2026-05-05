#!/usr/bin/env bash
# After a new segmentation checkpoint is trained, build:
#   1. Validate-local on val FOVs 156-160 (mean ARI + frac_in_cell)
#   2. Two test-FOV submissions: <new_seg> + RF500 classifier; <new_seg> + ET300 classifier
#   3. Three ensembles: vote(V7, new_seg+RF500), vote(V7, new_seg+ET300), vote(V7, both)
#
# Usage:
#   phase2/scripts/run_after_seg.sh TAG SEG_CKPT
#
# Example:
#   phase2/scripts/run_after_seg.sh cyto3 \
#     phase2/runs/20260503-174912-train-seg-cyto3_p1stack/models/cyto3_p1stack_final
#
# Outputs go under phase2/runs/postseg-<TAG>/ and ensemble dirs under
# phase2/runs/ensemble-postseg-<TAG>-*/.

set -euo pipefail

TAG="${1:?usage: run_after_seg.sh TAG SEG_CKPT}"
SEG_CKPT="${2:?missing SEG_CKPT}"

CLF_RF="phase2/runs/baseline-codelab-rf500-log1p-mf01"
CLF_ET="phase2/runs/baseline-codelab-et300-l1-mf01"
V7_CSV="phase2/runs/ensemble-v7-P-Q-RF500/submission.csv"

OUT="phase2/runs/postseg-${TAG}"
mkdir -p "$OUT"
echo "=== run_after_seg TAG=$TAG seg=$SEG_CKPT ==="
echo "  out=$OUT  V7=$V7_CSV"

if [ ! -e "$SEG_CKPT" ]; then echo "[fatal] seg ckpt not found: $SEG_CKPT" >&2; exit 2; fi
if [ ! -d "$CLF_RF" ]; then echo "[fatal] $CLF_RF not found" >&2; exit 2; fi
if [ ! -d "$CLF_ET" ]; then echo "[fatal] $CLF_ET not found" >&2; exit 2; fi

# -------- 1. Probe with each classifier (val + test infer + structural validate). --------
echo
echo "=== probe ${TAG}-rf500 ==="
phase2/scripts/probe_finetune.sh "${TAG}-rf500" "$SEG_CKPT" "$CLF_RF" 2>&1 | tee "$OUT/probe_rf500.log" | tail -30 || true

echo
echo "=== probe ${TAG}-et300 ==="
phase2/scripts/probe_finetune.sh "${TAG}-et300" "$SEG_CKPT" "$CLF_ET" 2>&1 | tee "$OUT/probe_et300.log" | tail -30 || true

RF_CSV="phase2/runs/probe-${TAG}-rf500/infer/submission.csv"
ET_CSV="phase2/runs/probe-${TAG}-et300/infer/submission.csv"

echo
echo "=== submission CSVs:"
ls -la "$RF_CSV" "$ET_CSV" 2>&1 || true

# -------- 2. Vote ensembles vs V7 --------
if [ -f "$V7_CSV" ] && [ -f "$RF_CSV" ]; then
  EOUT="phase2/runs/ensemble-postseg-${TAG}-V7-RF/submission.csv"
  mkdir -p "$(dirname "$EOUT")"
  echo
  echo "=== ensemble: V7 + ${TAG}-RF500 ==="
  .venv/bin/python phase2/scripts/vote_submissions.py \
    --out "$EOUT" "$V7_CSV" "$RF_CSV" 2>&1 | tee "$OUT/ensemble_v7_rf.log" | tail -25 || true
fi

if [ -f "$V7_CSV" ] && [ -f "$ET_CSV" ]; then
  EOUT="phase2/runs/ensemble-postseg-${TAG}-V7-ET/submission.csv"
  mkdir -p "$(dirname "$EOUT")"
  echo
  echo "=== ensemble: V7 + ${TAG}-ET300 ==="
  .venv/bin/python phase2/scripts/vote_submissions.py \
    --out "$EOUT" "$V7_CSV" "$ET_CSV" 2>&1 | tee "$OUT/ensemble_v7_et.log" | tail -25 || true
fi

if [ -f "$V7_CSV" ] && [ -f "$RF_CSV" ] && [ -f "$ET_CSV" ]; then
  EOUT="phase2/runs/ensemble-postseg-${TAG}-V7-RF-ET/submission.csv"
  mkdir -p "$(dirname "$EOUT")"
  echo
  echo "=== ensemble: V7 + ${TAG}-RF500 + ${TAG}-ET300 (3-way) ==="
  .venv/bin/python phase2/scripts/vote_submissions.py \
    --out "$EOUT" "$V7_CSV" "$RF_CSV" "$ET_CSV" 2>&1 | tee "$OUT/ensemble_v7_rf_et.log" | tail -25 || true
fi

echo
echo "=========================================================================="
echo "POSTSEG $TAG done."
echo "  candidates produced:"
ls -1 phase2/runs/ensemble-postseg-${TAG}-*/submission.csv 2>/dev/null
echo
echo "  Compare frac_in_cell vs old seg (target: drop toward 0.17):"
for d in phase2/runs/probe-${TAG}-*/infer; do
  if [ -f "$d/summary.json" ]; then
    .venv/bin/python -c "
import json, statistics, sys
d = json.load(open('$d/summary.json'))
fic = statistics.mean(f['frac_in_cell'] for f in d['per_fov'].values())
print(f'  $(basename $(dirname $d)) test-mean frac_in_cell = {fic:.4f}')
"
  fi
done
echo "=========================================================================="
