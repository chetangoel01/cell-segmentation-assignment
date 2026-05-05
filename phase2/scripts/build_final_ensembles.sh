#!/usr/bin/env bash
# Build a SUITE of ensemble candidates for submission. Each ensemble combines
# proven Kaggle scorers with this session's new bundles. Anchor is SOTA
# (AWS-kNN 0.5593) unless noted.
set -euo pipefail

VENV="/Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment/.venv/bin/python"
ROOT="/Users/chetangoel/Desktop/Repositories/cell-segmentation-assignment"
cd "$ROOT"

# Inputs
SOTA="phase2/runs/SUBMIT_aws001_clusfilt_bg40_dist/submission.csv"
HIER="phase2/runs/SUBMIT_hier_aws_kp15/submission.csv"
K15="phase2/runs/SUBMIT_aws_k15_bg40_dist/submission.csv"
SOTA_ERODE="phase2/runs/SUBMIT_sota_erode2/submission.csv"
V7="phase2/runs/SUBMIT_v7_PQM_cpsam_4way/submission.csv"
CT="phase2/runs/SUBMIT_celltypist_bridge/submission.csv"
SC="phase2/runs/SUBMIT_scanvi_xl/submission.csv"
X8="phase2/runs/SUBMIT_xgb_xl_d8/submission.csv"
X6="phase2/runs/SUBMIT_xgb_xl_d6/submission.csv"
X5="phase2/runs/SUBMIT_xgb_xl_d5/submission.csv"
MLP="phase2/runs/SUBMIT_mlp_hier/submission.csv"
MLPXL="phase2/runs/SUBMIT_mlp_xl/submission.csv"

run_vote () {
    local out="$1"; shift
    mkdir -p "$out"
    echo ""
    echo "=== ENSEMBLE: $out ==="
    "$VENV" phase2/scripts/vote_submissions.py --out "$out/submission.csv" "$@" 2>&1 | tail -25 || \
        echo "  [warn] some inputs missing"
}

# E1: SOTA + best 3 new gradient/NN classifiers (heavily new-model)
run_vote phase2/runs/SUBMIT_FINAL_E1_sota_xgb_scanvi_mlp \
    "$SOTA" "$X8" "$SC" "$MLP"

# E2: SOTA + 3 kNN-style proven (homogenous boost)
run_vote phase2/runs/SUBMIT_FINAL_E2_kNN_proven \
    "$SOTA" "$HIER" "$K15" "$SOTA_ERODE"

# E3: SOTA + best new + V7 + Hier (mixed all-time best)
run_vote phase2/runs/SUBMIT_FINAL_E3_mixed_alltime \
    "$SOTA" "$X8" "$V7" "$HIER"

# E4: SOTA + scANVI + 3 XGBs (one classifier family + sota anchor)
run_vote phase2/runs/SUBMIT_FINAL_E4_xgb_sweep \
    "$SOTA" "$X8" "$X6" "$X5"

# E5: SOTA + 4 new (all this session's new bundles, anchor=SOTA)
run_vote phase2/runs/SUBMIT_FINAL_E5_all_new_with_sota \
    "$SOTA" "$X8" "$SC" "$MLP" "$MLPXL"

# E6: 5-way max diversity (kNN, XGB, NN, MLP, CT) — heaviest diversity
run_vote phase2/runs/SUBMIT_FINAL_E6_5way_max_diversity \
    "$SOTA" "$X8" "$SC" "$MLPXL" "$CT"

# E7: anchor=SOTA + just MLP + scANVI (3-way classifier-family-only)
run_vote phase2/runs/SUBMIT_FINAL_E7_sota_scanvi_mlp \
    "$SOTA" "$SC" "$MLP"

# E8: anchor=Hier + SOTA + best XGB (3-way, hierarchical anchor)
run_vote phase2/runs/SUBMIT_FINAL_E8_hier_anchor \
    "$HIER" "$SOTA" "$X8"

echo ""
echo "==================================================="
echo "ALL CANDIDATES (sort by in-cell fraction):"
echo "==================================================="
for path in \
    "$SOTA" "$HIER" "$K15" "$SOTA_ERODE" "$V7" "$CT" "$SC" "$X8" "$X6" "$X5" "$MLP" "$MLPXL" \
    phase2/runs/SUBMIT_FINAL_E*/submission.csv; do
    if [ -f "$path" ]; then
        ic=$(awk -F',' 'NR>1 && $3!="background" {n++} NR>1 {tot++} END {if (tot>0) printf "%.4f", n/tot; else print "0"}' "$path")
        echo "  $ic  $path"
    else
        echo "  ----  $path  (missing)"
    fi
done | sort -k1,1 -r
