#!/usr/bin/env bash
# Poll Modal volume for completed bundles and run inference as soon as
# each appears. Writes submission CSVs to phase2/runs/SUBMIT_<name>/.
set -o pipefail

cd "$(dirname "$0")/../.."

declare -A KIND_OF=(
  [xgb-xl-d8n3000-bg40]="xgb"
  [xgb-xl-d6n3000-bg80]="xgb"
  [xgb-xl-d5n5000-bg40]="xgb"
  [scanvi-aws-modal-xl]="scanvi"
  [mlp-hier-aws]="mlp"
)

declare -A SUBMIT_DIR_OF=(
  [xgb-xl-d8n3000-bg40]="phase2/runs/SUBMIT_xgb_xl_d8"
  [xgb-xl-d6n3000-bg80]="phase2/runs/SUBMIT_xgb_xl_d6"
  [xgb-xl-d5n5000-bg40]="phase2/runs/SUBMIT_xgb_xl_d5"
  [scanvi-aws-modal-xl]="phase2/runs/SUBMIT_scanvi_xl"
  [mlp-hier-aws]="phase2/runs/SUBMIT_mlp_hier"
)

declare -A LOCAL_DIR_OF=(
  [xgb-xl-d8n3000-bg40]="phase2/runs/xgb-xl-d8n3000-bg40"
  [xgb-xl-d6n3000-bg80]="phase2/runs/xgb-xl-d6n3000-bg80"
  [xgb-xl-d5n5000-bg40]="phase2/runs/xgb-xl-d5n5000-bg40"
  [scanvi-aws-modal-xl]="phase2/runs/scanvi-aws-modal-xl"
  [mlp-hier-aws]="phase2/runs/mlp-hier-aws"
)

declare -A BUNDLE_NAME_OF=(
  [xgb-xl-d8n3000-bg40]="xgb_bundle.joblib"
  [xgb-xl-d6n3000-bg80]="xgb_bundle.joblib"
  [xgb-xl-d5n5000-bg40]="xgb_bundle.joblib"
  [scanvi-aws-modal-xl]="scanvi_bundle.joblib"
  [mlp-hier-aws]="mlp_bundle.joblib"
)

declare -A DONE=(
  [xgb-xl-d8n3000-bg40]=0
  [xgb-xl-d6n3000-bg80]=0
  [xgb-xl-d5n5000-bg40]=0
  [scanvi-aws-modal-xl]=0
  [mlp-hier-aws]=0
)

LOG="phase2/runs/_autopull.log"
echo "==> auto_pull_infer started $(date)" | tee -a "$LOG"

iter=0
while :; do
  iter=$((iter + 1))
  all_done=1
  for name in "${!KIND_OF[@]}"; do
    if [ "${DONE[$name]}" = "1" ]; then continue; fi
    all_done=0
    bundle="${BUNDLE_NAME_OF[$name]}"
    listing=$(modal volume ls cell-seg-phase2 "trained/$name" 2>&1 || true)
    if echo "$listing" | grep -q "$bundle"; then
      echo "[$(date +%H:%M:%S)] FOUND $name -> running inference" | tee -a "$LOG"
      DONE[$name]=1
      bash phase2/scripts/pull_and_infer.sh \
        "${KIND_OF[$name]}" \
        "trained/$name" \
        "${LOCAL_DIR_OF[$name]}" \
        "${SUBMIT_DIR_OF[$name]}" 2>&1 | tee -a "$LOG" || \
        echo "[err] inference failed for $name" | tee -a "$LOG"
    fi
  done
  if [ "$all_done" = "1" ]; then
    echo "==> all bundles processed at $(date)" | tee -a "$LOG"
    break
  fi
  if [ "$iter" -gt 200 ]; then
    echo "==> giving up after 200 iterations at $(date)" | tee -a "$LOG"
    break
  fi
  sleep 30
done
