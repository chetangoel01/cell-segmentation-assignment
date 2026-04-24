"""Evaluate multi-model ensemble on val FOVs 036-040 before committing a submission.

Loads all available trained models, runs each on val FOVs, and does spot-level
majority voting.  Reports per-model and ensemble val ARI so you can see whether
combining models beats the best individual model.

Usage:
    python ensemble_val_eval.py
    python ensemble_val_eval.py --exp-names cyto2 nuclei cyto3 multiscale
    python ensemble_val_eval.py --exp-names cyto2 multiscale \
        --spot-sigmas-map cyto2:8 multiscale:4,8,16
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from cellpose import models as cp_models
from sklearn.metrics import adjusted_rand_score

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask, compute_spot_density

parser = argparse.ArgumentParser()
parser.add_argument("--exp-names", nargs="+",
                    default=["cyto2", "nuclei", "cyto3", "multiscale"],
                    help="Models to ensemble (skips any without a trained canonical model)")
parser.add_argument("--spot-sigmas-map", nargs="*", default=[],
                    help="Per-experiment sigma override: 'expname:4,8,16'")
parser.add_argument("--cellprob-threshold", type=float, default=-1.0)
parser.add_argument("--flow-threshold", type=float, default=0.4)
args = parser.parse_args()

DATA_ROOT = "/scratch/cg4652/competition"
VAL_FOVS  = [f"FOV_{i:03d}" for i in range(36, 41)]

# Build sigma map — auto-detect multiscale from best_params or default to 8
sigma_map: dict[str, list[float]] = {}
for item in args.spot_sigmas_map:
    name, s = item.split(":", 1)
    sigma_map[name] = [float(x) for x in s.split(",")]

# ── Load models (skip missing) ────────────────────────────────────────────────
loaded: list[tuple[str, cp_models.CellposeModel, list[float], dict]] = []
for exp in args.exp_names:
    model_path = f"models/{exp}/cellpose_{exp}"
    if not os.path.exists(model_path):
        print(f"  Skipping {exp}: no trained model at {model_path}")
        continue
    sigmas = sigma_map.get(exp, [8.0])
    # Auto-detect multiscale sigmas from exp name
    if "multiscale" in exp and exp not in sigma_map:
        sigmas = [4.0, 8.0, 16.0]
    # Load best thresholds if sweep was run
    params_path = f"best_params_{exp}.json"
    thresh = {"cellprob_threshold": args.cellprob_threshold,
              "flow_threshold": args.flow_threshold}
    if os.path.exists(params_path):
        with open(params_path) as f:
            thresh = json.load(f)["best"]
        thresh = {k: thresh[k] for k in ("cellprob_threshold", "flow_threshold")}
    print(f"Loading {exp} (sigmas={sigmas}, thresholds={thresh})")
    model = cp_models.CellposeModel(gpu=True, pretrained_model=model_path)
    loaded.append((exp, model, sigmas, thresh))

if not loaded:
    raise SystemExit("No models found. Train first.")

# ── Load shared val data ──────────────────────────────────────────────────────
print("\nLoading val FOV metadata...")
meta        = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells_df    = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

# ── Run each model on all val FOVs ────────────────────────────────────────────
# per_model_preds[exp][fov] = 1-D array of integer cell IDs (0=background) per spot
per_model_preds: dict[str, dict[str, np.ndarray]] = {exp: {} for exp, *_ in loaded}
gt_ids_per_fov:  dict[str, np.ndarray] = {}
rows_per_fov:    dict[str, np.ndarray] = {}
cols_per_fov:    dict[str, np.ndarray] = {}

for fov_name in VAL_FOVS:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        continue
    dapi, polyt = load_fov_images(fov_dir)
    fov_x  = meta.loc[fov_name, "fov_x"]
    fov_y  = meta.loc[fov_name, "fov_y"]
    fov_spots = spots_train[spots_train["fov"] == fov_name]
    dapi_max  = np.max(dapi,  axis=0)
    polyt_max = np.max(polyt, axis=0)
    gt_mask   = boundaries_to_mask(cells_df, fov_name, fov_x, fov_y)
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    gt_ids_per_fov[fov_name]  = gt_mask[rows, cols]
    rows_per_fov[fov_name]    = rows
    cols_per_fov[fov_name]    = cols

    for exp, model, sigmas, thresh in loaded:
        density_ch = [compute_spot_density(fov_spots, sigma=s) for s in sigmas]
        img = np.stack([polyt_max, dapi_max, *density_ch], axis=0)
        pred_masks, _, _ = model.eval(img, diameter=0, channel_axis=0, **thresh)
        per_model_preds[exp][fov_name] = pred_masks[rows, cols]

# ── Per-model ARI ─────────────────────────────────────────────────────────────
print("\n── Per-model val ARI ──────────────────────────────────────────────────")
model_aris: dict[str, float] = {}
for exp, *_ in loaded:
    scores = []
    for fov_name in gt_ids_per_fov:
        scores.append(adjusted_rand_score(gt_ids_per_fov[fov_name],
                                          per_model_preds[exp][fov_name]))
    mean_ari = float(np.mean(scores))
    model_aris[exp] = mean_ari
    per_fov_str = "  ".join(f"{fov_name}={s:.4f}" for fov_name, s in
                            zip(gt_ids_per_fov, scores))
    print(f"  {exp:20s}: {mean_ari:.4f}   ({per_fov_str})")

# ── Ensemble combinations ─────────────────────────────────────────────────────
print("\n── Ensemble val ARI ───────────────────────────────────────────────────")

def ensemble_ari(exp_subset: list[str]) -> float:
    scores = []
    for fov_name in gt_ids_per_fov:
        # Majority vote on raw cell IDs — same caveat as before: IDs not aligned
        # across models.  Vote binary (in-cell vs background), use first model's ID.
        first_exp = exp_subset[0]
        ref_ids = per_model_preds[first_exp][fov_name]
        in_cell_votes = np.sum(
            np.stack([per_model_preds[e][fov_name] > 0 for e in exp_subset], axis=0),
            axis=0,
        )
        voted_ids = np.where(in_cell_votes >= len(exp_subset) / 2, ref_ids, 0)
        scores.append(adjusted_rand_score(gt_ids_per_fov[fov_name], voted_ids))
    return float(np.mean(scores))

exp_names = [e for e, *_ in loaded]
# All models together
if len(exp_names) > 1:
    all_ari = ensemble_ari(exp_names)
    print(f"  {'ALL (' + '+'.join(exp_names) + ')':40s}: {all_ari:.4f}")

# All pairs
if len(exp_names) >= 2:
    from itertools import combinations
    for n in range(2, len(exp_names)):
        for combo in combinations(exp_names, n):
            ari = ensemble_ari(list(combo))
            label = "+".join(combo)
            marker = " ◀ best" if ari > max(model_aris.values()) else ""
            print(f"  {label:40s}: {ari:.4f}{marker}")

print(f"\nBest individual: {max(model_aris, key=model_aris.get)} = {max(model_aris.values()):.4f}")
