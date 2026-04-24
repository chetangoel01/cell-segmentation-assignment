"""Grid-search cellprob_threshold × flow_threshold on validation FOVs.

Usage:
    python sweep_thresholds.py --exp-name cyto2
    python sweep_thresholds.py --exp-name multiscale --spot-sigmas 4,8,16

Writes best_params_<exp>.json with optimal thresholds to use in infer.py.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from cellpose import models as cp_models
from sklearn.metrics import adjusted_rand_score

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask, compute_spot_density, compute_zstack_features

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True)
parser.add_argument("--spot-sigmas", default="8")
parser.add_argument(
    "--cellprob-thresholds", default="-3.0,-2.0,-1.5,-1.0,-0.5,0.0",
    help="Comma-separated cellprob_threshold values to sweep",
)
parser.add_argument(
    "--flow-thresholds", default="0.1,0.2,0.4,0.6,0.8",
    help="Comma-separated flow_threshold values to sweep",
)
parser.add_argument("--zstats", action="store_true",
                    help="Use 7-channel zstack input (must match training flag)")
args = parser.parse_args()

EXP_NAME         = args.exp_name
SPOT_SIGMAS      = [float(s) for s in args.spot_sigmas.split(",")]
CP_THRESHOLDS    = [float(t) for t in args.cellprob_thresholds.split(",")]
FLOW_THRESHOLDS  = [float(t) for t in args.flow_thresholds.split(",")]
DATA_ROOT        = "/scratch/cg4652/competition"
MODEL_DIR        = f"models/{EXP_NAME}"
FINAL_MODEL      = os.path.join(MODEL_DIR, f"cellpose_{EXP_NAME}")

print(f"Loading model: {FINAL_MODEL}")
model = cp_models.CellposeModel(gpu=True, pretrained_model=FINAL_MODEL)

meta        = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells       = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
val_fovs    = [f"FOV_{i:03d}" for i in range(36, 41)]

print("Pre-loading val FOV images...")
val_data = []
for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
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

print(f"\nSweeping {len(CP_THRESHOLDS)} × {len(FLOW_THRESHOLDS)} = "
      f"{len(CP_THRESHOLDS)*len(FLOW_THRESHOLDS)} combinations...\n")

grid_results = []
best_ari = -1.0
best_params: dict = {}

for cp_thresh in CP_THRESHOLDS:
    for flow_thresh in FLOW_THRESHOLDS:
        ari_scores = []
        n_cells_list = []
        for fov_name, img, rows, cols, gt_ids in val_data:
            pred_masks, _, _ = model.eval(
                img,
                diameter=0,
                cellprob_threshold=cp_thresh,
                flow_threshold=flow_thresh,
                channel_axis=0,
            )
            pred_ids = pred_masks[rows, cols]
            ari_scores.append(adjusted_rand_score(gt_ids, pred_ids))
            n_cells_list.append(int(pred_masks.max()))
        mean_ari = float(np.mean(ari_scores))
        entry = {
            "cellprob_threshold": cp_thresh,
            "flow_threshold": flow_thresh,
            "mean_ari": mean_ari,
            "per_fov": dict(zip(val_fovs[:len(ari_scores)], [round(a, 4) for a in ari_scores])),
            "mean_cells": round(float(np.mean(n_cells_list)), 1),
        }
        grid_results.append(entry)
        marker = " ◀ best" if mean_ari > best_ari else ""
        print(f"  cp={cp_thresh:+.1f}  flow={flow_thresh:.1f}  "
              f"ARI={mean_ari:.4f}  cells={entry['mean_cells']:.0f}{marker}")
        if mean_ari > best_ari:
            best_ari = mean_ari
            best_params = {"cellprob_threshold": cp_thresh, "flow_threshold": flow_thresh,
                           "mean_ari": mean_ari}

grid_results.sort(key=lambda x: x["mean_ari"], reverse=True)
out_path = f"best_params_{EXP_NAME}.json"
with open(out_path, "w") as f:
    json.dump({"best": best_params, "grid": grid_results}, f, indent=2)

print(f"\nBest: cellprob={best_params['cellprob_threshold']:+.1f}  "
      f"flow={best_params['flow_threshold']:.1f}  ARI={best_ari:.4f}")
print(f"Full grid saved to {out_path}")
