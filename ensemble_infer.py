"""Ensemble multiple trained models via spot-level majority vote.

Each model independently assigns each spot to a cell (or background).
Final assignment is determined by majority vote across models.  Ties
broken in favour of the first model (highest priority = first in list).

Usage:
    python ensemble_infer.py --exp-names cyto2 cyto3 nuclei
    python ensemble_infer.py --exp-names cyto2 cyto3 nuclei multiscale \
        --spot-sigmas-map cyto2:8 cyto3:8 nuclei:8 multiscale:4,8,16

Output: submission_ensemble_<exp1>_<exp2>_....csv
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from cellpose import models as cp_models

from src.io import load_fov_images
from src.train_cellpose import compute_spot_density

parser = argparse.ArgumentParser()
parser.add_argument("--exp-names", nargs="+", required=True,
                    help="Experiment names to ensemble (order = priority for ties)")
parser.add_argument("--spot-sigmas-map", nargs="*", default=[],
                    help="Per-experiment sigma override: 'expname:4,8,16'. "
                         "Experiments not listed default to '8'.")
parser.add_argument("--cellprob-threshold", type=float, default=-1.0)
parser.add_argument("--flow-threshold", type=float, default=0.4)
args = parser.parse_args()

EXP_NAMES  = args.exp_names
DATA_ROOT  = "/scratch/cg4652/competition"
TEST_FOVS  = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]

# Build sigma map
sigma_map: dict[str, list[float]] = {e: [8.0] for e in EXP_NAMES}
for item in args.spot_sigmas_map:
    name, sigmas_str = item.split(":", 1)
    sigma_map[name] = [float(s) for s in sigmas_str.split(",")]

print(f"Ensembling {len(EXP_NAMES)} model(s): {EXP_NAMES}")
print(f"cellprob_threshold={args.cellprob_threshold}  flow_threshold={args.flow_threshold}\n")

# ── Load all models ───────────────────────────────────────────────────────────
loaded_models: list[tuple[str, cp_models.CellposeModel]] = []
for exp in EXP_NAMES:
    model_path = f"models/{exp}/cellpose_{exp}"
    if not os.path.exists(model_path):
        print(f"WARNING: model not found for '{exp}': {model_path} — skipping")
        continue
    print(f"Loading {exp} from {model_path}")
    loaded_models.append((exp, cp_models.CellposeModel(gpu=True, pretrained_model=model_path)))

if not loaded_models:
    raise SystemExit("No models could be loaded. Run train.py first.")

test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
print(f"\nTest spots: {len(test_spots):,}")

# ── Inference ────────────────────────────────────────────────────────────────
# all_assignments[fov_name] = list of 1-D arrays (one per model), each mapping
# spot index → cell label string ("background" or "<fov>_cell_<id>")
all_assignments: dict[str, list[np.ndarray]] = {fov: [] for fov in TEST_FOVS}

for exp, model in loaded_models:
    sigmas = sigma_map[exp]
    print(f"\n[{exp}] sigmas={sigmas}")
    for fov_name in TEST_FOVS:
        fov_dir = f"{DATA_ROOT}/test/{fov_name}"
        if not os.path.exists(fov_dir):
            print(f"  Missing {fov_dir}")
            continue
        dapi, polyt = load_fov_images(fov_dir)
        fov_spots = test_spots[test_spots["fov"] == fov_name]
        dapi_max  = np.max(dapi, axis=0)
        polyt_max = np.max(polyt, axis=0)
        density_channels = [compute_spot_density(fov_spots, sigma=s) for s in sigmas]
        img = np.stack([polyt_max, dapi_max, *density_channels], axis=0)

        pred_masks, _, _ = model.eval(
            img,
            diameter=0,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            channel_axis=0,
        )
        rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
        cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
        label_ids = pred_masks[rows, cols]
        labels = np.where(
            label_ids > 0,
            np.array([f"{fov_name}_cell_{v}" for v in label_ids]),
            "background",
        )
        all_assignments[fov_name].append(labels)
        print(f"  {fov_name}: {pred_masks.max()} cells, "
              f"{(label_ids > 0).sum():,}/{len(label_ids):,} spots assigned")

# ── Majority vote per spot ────────────────────────────────────────────────────
print("\nMajority voting...")
submission_parts = []

for fov_name in TEST_FOVS:
    fov_spots = test_spots[test_spots["fov"] == fov_name].copy()
    model_preds = all_assignments[fov_name]
    if not model_preds:
        continue
    n_spots = len(model_preds[0])
    final_labels = np.empty(n_spots, dtype=object)
    for i in range(n_spots):
        votes = [preds[i] for preds in model_preds]
        # majority_vote: most common; ties broken by first model (highest priority)
        winner = Counter(votes).most_common(1)[0][0]
        final_labels[i] = winner
    n_assigned = (final_labels != "background").sum()
    print(f"  {fov_name}: {n_assigned:,}/{n_spots:,} spots assigned after vote")
    submission_parts.append(pd.DataFrame({
        "spot_id": fov_spots["spot_id"].values,
        "fov": fov_name,
        "cluster_id": final_labels,
    }))

combined = pd.concat(submission_parts, ignore_index=True)
submission = (
    test_spots[["spot_id", "fov"]]
    .merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
)
submission["cluster_id"] = submission["cluster_id"].fillna("background")

out_name = "submission_ensemble_" + "_".join(EXP_NAMES) + ".csv"
submission[["spot_id", "fov", "cluster_id"]].to_csv(out_name, index=False)
print(f"\nSaved {out_name} — {len(submission):,} rows")
print(submission["cluster_id"].value_counts().head(5))
