#!/usr/bin/env bash
# Submit a single submission CSV to Kaggle for the phase-2 competition.
#
# Usage: kaggle_submit.sh <path/to/submission.csv> "<message>"
set -euo pipefail

CSV="$1"
MSG="$2"
COMP="cell-type-classification-phase-2-cs-gy-9223"

if [ ! -f "$CSV" ]; then
    echo "[fatal] $CSV does not exist"
    exit 1
fi

echo "==> Submitting $CSV to $COMP"
echo "    message: $MSG"

kaggle competitions submit -c "$COMP" -f "$CSV" -m "$MSG"

echo ""
echo "==> Waiting for score..."
sleep 6
kaggle competitions submissions -c "$COMP" 2>&1 | head -5
