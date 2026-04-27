"""Evaluate every saved checkpoint for an experiment and promote the best-val-ARI one.

Usage:
    python eval_best_checkpoint.py --exp-name cyto2
    python eval_best_checkpoint.py --exp-name multiscale --spot-sigmas 4,8,16

Reads all checkpoints named cellpose_<exp>_ep* in models/<exp>/, evaluates each on
val FOVs 036-040, then copies the best one to models/<exp>/cellpose_<exp> (the canonical
path that infer.py loads).  Updates train_state.json so subsequent training resumes
correctly if the run is not yet at 300 epochs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
from cellpose import models as cp_models
from sklearn.metrics import adjusted_rand_score

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask, compute_spot_density, compute_zstack_features

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True)
parser.add_argument("--spot-sigmas", default="8",
                    help="Comma-separated sigma(s) used during training")
parser.add_argument("--cellprob-threshold", type=float, default=-1.0,
                    help="cellprob_threshold for checkpoint eval (default -1.0; "
                         "re-run with optimal value from sweep_thresholds.py for consistency)")
parser.add_argument("--flow-threshold", type=float, default=0.4,
                    help="flow_threshold for checkpoint eval (default 0.4)")
parser.add_argument("--zstats", action="store_true",
                    help="Use 7-channel zstack input (must match training flag)")
args = parser.parse_args()

EXP_NAME    = args.exp_name
SPOT_SIGMAS = [float(s) for s in args.spot_sigmas.split(",")]
DATA_ROOT   = "/scratch/cg4652/competition"
MODEL_DIR   = f"models/{EXP_NAME}"
MODEL_NAME  = f"cellpose_{EXP_NAME}"
# Cellpose saves epoch checkpoints into a nested models/ subdir
CKPT_DIR    = os.path.join(MODEL_DIR, "models")

# ── Find all epoch checkpoints ────────────────────────────────────────────────
pattern = os.path.join(CKPT_DIR, f"{MODEL_NAME}_ep*")
ckpts = sorted(glob.glob(pattern))

canonical = os.path.join(MODEL_DIR, MODEL_NAME)
if os.path.exists(canonical) and canonical not in ckpts:
    ckpts.append(canonical)

if not ckpts:
    print(f"No checkpoints found in {MODEL_DIR}. Train first.")
    raise SystemExit(1)

print(f"Found {len(ckpts)} checkpoint(s) for experiment '{EXP_NAME}'")

# ── Load shared metadata (loaded once) ────────────────────────────────────────
print("Loading metadata and ground truth...")
meta  = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]

# Pre-load val images (expensive I/O, done once)
print("Pre-loading val FOV images...")
val_data: list[tuple] = []
for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing {fov_name}, skipping")
        continue
    dapi, polyt = load_fov_images(fov_dir)
    fov_x  = meta.loc[fov_name, "fov_x"]
    fov_y  = meta.loc[fov_name, "fov_y"]
    fov_spots = spots_train[spots_train["fov"] == fov_name]
    density_channels = [compute_spot_density(fov_spots, sigma=s) for s in SPOT_SIGMAS]
    if args.zstats:
        zf = compute_zstack_features(dapi, polyt)
        img = np.stack([zf["polyt_max"], zf["dapi_max"],
                        zf["polyt_mean"], zf["dapi_mean"],
                        zf["polyt_std"],  zf["dapi_std"],
                        *density_channels], axis=0)
    else:
        img = np.stack([np.max(polyt, axis=0), np.max(dapi, axis=0), *density_channels], axis=0)
    gt_mask = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    gt_ids = gt_mask[rows, cols]
    val_data.append((fov_name, img, rows, cols, gt_ids))

# ── Evaluate each checkpoint ──────────────────────────────────────────────────
results: list[tuple[float, str]] = []

for ckpt_path in ckpts:
    label = os.path.basename(ckpt_path)
    model = cp_models.CellposeModel(gpu=True, pretrained_model=ckpt_path)
    ari_scores = []
    for fov_name, img, rows, cols, gt_ids in val_data:
        pred_masks, _, _ = model.eval(
            img, diameter=0, cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold, channel_axis=0,
        )
        pred_ids = pred_masks[rows, cols]
        ari_scores.append(adjusted_rand_score(gt_ids, pred_ids))
    mean_ari = float(np.mean(ari_scores))
    results.append((mean_ari, ckpt_path))
    print(f"  {label}: mean ARI = {mean_ari:.4f}  ({[f'{a:.4f}' for a in ari_scores]})")

# ── Promote best checkpoint ───────────────────────────────────────────────────
results.sort(reverse=True)
best_ari, best_ckpt = results[0]
print(f"\nBest checkpoint: {os.path.basename(best_ckpt)}  ARI = {best_ari:.4f}")

if os.path.abspath(best_ckpt) != os.path.abspath(canonical):
    shutil.copy(best_ckpt, canonical)
    print(f"Promoted to {canonical}")
else:
    print("Canonical model is already the best checkpoint.")

# Save evaluation results
eval_path = os.path.join(MODEL_DIR, "checkpoint_eval.json")

with open(eval_path, "w") as f:
    json.dump([{"ckpt": c, "mean_ari": a} for a, c in results], f, indent=2)
print(f"Full results saved to {eval_path}")
