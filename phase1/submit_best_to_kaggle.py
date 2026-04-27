"""Find the best-swept model across all experiments and submit its submission CSV to Kaggle.

Reads best_params_<exp>.json for every experiment that has one, picks the highest
mean_ari, runs infer.py with those params, then submits via the Kaggle CLI.

Usage:
    python submit_best_to_kaggle.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

COMPETITION = "cell-type-classification-cs-gy-9223"
KAGGLE_CLI  = os.path.expanduser("~/.local/bin/kaggle")

# exp-name → spot-sigmas used at training time
SIGMAS = {
    "cyto2":            "8",
    "cyto3":            "8",
    "nuclei":           "8",
    "multiscale":       "4,8,16",
    "cyto2_aug":        "8",
    "nuclei_aug":       "8",
    "multiscale_aug":   "4,8,16",
    "cyto2_multiz":     "8",
    "cyto2_zstats":     "8",
    "nuclei_zstats":    "8",
    "cyto2_cosine":      "8",
    "nuclei_cosine":     "8",
    "cyto2_warmup":      "8",
    "multiscale_cosine": "4,8,16",
    "cyto2_long":        "8",
    "cyto2_warmup_long":  "8",
    "cyto2_warmup_lowwd": "8",
    "nuclei_warmup":      "8",
    "nuclei_warmup_long": "8",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Print what would be submitted without actually submitting")
parser.add_argument("--exp-name", default=None,
                    help="Force a specific experiment instead of picking the best")
args = parser.parse_args()

root = Path(__file__).parent

# ── Find best experiment ──────────────────────────────────────────────────────
best_exp, best_ari, best_params = None, -1.0, {}
for exp in SIGMAS:
    params_file = root / f"best_params_{exp}.json"
    if not params_file.exists():
        continue
    d = json.loads(params_file.read_text())
    ari = d["best"]["mean_ari"]
    print(f"  {exp:20s}  val ARI = {ari:.4f}")
    if ari > best_ari:
        best_ari, best_exp, best_params = ari, exp, d["best"]

if best_exp is None:
    print("No best_params_*.json found — run sweep_thresholds.py first.")
    sys.exit(1)

if args.exp_name:
    params_file = root / f"best_params_{args.exp_name}.json"
    if not params_file.exists():
        print(f"No best_params_{args.exp_name}.json — run sweep_thresholds.py first.")
        sys.exit(1)
    d = json.loads(params_file.read_text())
    best_exp, best_ari, best_params = args.exp_name, d["best"]["mean_ari"], d["best"]

print(f"\nBest model: {best_exp}  val ARI = {best_ari:.4f}")
print(f"  cellprob_threshold = {best_params['cellprob_threshold']}")
print(f"  flow_threshold     = {best_params['flow_threshold']}")

submission_csv = root / f"submission_{best_exp}.csv"

# ── Run inference if submission CSV doesn't exist ─────────────────────────────
if not submission_csv.exists():
    print(f"\nRunning inference for {best_exp}...")
    cmd = [
        sys.executable, "-u", str(root / "infer.py"),
        "--exp-name", best_exp,
        "--spot-sigmas", SIGMAS[best_exp],
        "--params-json", str(root / f"best_params_{best_exp}.json"),
    ]
    if args.dry_run:
        print("  [dry-run] would run:", " ".join(cmd))
    else:
        result = subprocess.run(cmd, cwd=str(root))
        if result.returncode != 0:
            print("Inference failed.")
            sys.exit(1)
else:
    print(f"\nSubmission CSV already exists: {submission_csv}")

# ── Submit to Kaggle ──────────────────────────────────────────────────────────
message = (
    f"{best_exp}: val ARI={best_ari:.4f}, "
    f"cellprob={best_params['cellprob_threshold']}, "
    f"flow={best_params['flow_threshold']}"
)
kaggle_cmd = [
    KAGGLE_CLI, "competitions", "submit",
    "-c", COMPETITION,
    "-f", str(submission_csv),
    "-m", message,
]
print(f"\nSubmitting to Kaggle: {' '.join(kaggle_cmd)}")
if args.dry_run:
    print("[dry-run] skipping actual submission")
else:
    subprocess.run(kaggle_cmd, check=True)
    print("\nSubmission complete.")
