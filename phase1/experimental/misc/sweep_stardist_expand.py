"""Expand-labels sweep on a trained StarDist 2D model.

StarDist on DAPI predicts nucleus-tight masks; Kaggle ARI rewards correct
cell-membership of (mostly cytoplasmic) mRNA spots. Dilating each nucleus
outward until it meets a neighbor approximates cells without retraining.

Runs neural-net prediction once per FOV, then sweeps over expansion distances
cheaply in NumPy. Emits one submission CSV per distance plus a val-ARI table.
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="stardist_v3")
parser.add_argument("--channel", default="dapi", choices=["dapi", "polyt", "both"])
parser.add_argument("--distances", default="0,4,8,12,16,20,24,32,48",
                    help="comma-separated pixel distances to sweep")
parser.add_argument("--submissions-dir", default="/submissions")
args = parser.parse_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from stardist.models import StarDist2D              # noqa: E402
from csbdeep.utils import normalize                 # noqa: E402
from skimage.segmentation import expand_labels      # noqa: E402

EXP_NAME   = args.exp_name
DATA_ROOT  = os.environ.get("MERFISH_DATA_ROOT", "/scratch/cg4652/competition")
VAL_FOVS   = [f"FOV_{i:03d}" for i in range(36, 41)]
TEST_FOVS  = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
BASEDIR    = "models"
DISTANCES  = [int(x) for x in args.distances.split(",")]

model = StarDist2D(None, name=EXP_NAME, basedir=BASEDIR)
print(f"Loaded {EXP_NAME}  distances={DISTANCES}")

meta        = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells_df    = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv",
                          index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
test_spots  = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")


def build_input(dapi: np.ndarray, polyt: np.ndarray) -> np.ndarray:
    dapi_max  = np.max(dapi,  axis=0).astype(np.float32)
    polyt_max = np.max(polyt, axis=0).astype(np.float32)
    if args.channel == "dapi":  return dapi_max
    if args.channel == "polyt": return polyt_max
    return np.stack([dapi_max, polyt_max], axis=-1)


def predict_fov(fov_name: str, subdir: str) -> np.ndarray:
    """Run StarDist once; return (H, W) int32 mask."""
    fov_dir = f"{DATA_ROOT}/{subdir}/{fov_name}"
    dapi, polyt = load_fov_images(fov_dir)
    img = build_input(dapi, polyt)
    img = normalize(img, 1, 99.8, axis=(0, 1))
    labels, _ = model.predict_instances(img)
    return labels.astype(np.int32)


# ── Val: spot-based ARI vs rasterized GT polygon mask ────────────────────────
print("\n=== VAL sweep (FOVs 036–040) ===")
val_preds, val_gt, val_spots = {}, {}, {}
for fov_name in VAL_FOVS:
    if not os.path.exists(f"{DATA_ROOT}/train/{fov_name}"):
        continue
    val_preds[fov_name] = predict_fov(fov_name, "train")
    val_gt[fov_name]    = boundaries_to_mask(
        cells_df, fov_name,
        meta.loc[fov_name, "fov_x"], meta.loc[fov_name, "fov_y"],
        use_all_z=True,
    ).astype(np.int32)
    val_spots[fov_name] = spots_train[spots_train["fov"] == fov_name]
    print(f"  {fov_name}: pred={val_preds[fov_name].max()} cells, "
          f"gt={val_gt[fov_name].max()} cells")

val_table = []
for d in DISTANCES:
    ari_per_fov = {}
    for fov_name, pred in val_preds.items():
        expanded = pred if d == 0 else expand_labels(pred, distance=d)
        gt = val_gt[fov_name]
        fov_spots = val_spots[fov_name]
        rr = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
        cc = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
        ari_per_fov[fov_name] = adjusted_rand_score(gt[rr, cc], expanded[rr, cc])
    mean_ari = float(np.mean(list(ari_per_fov.values())))
    val_table.append((d, mean_ari, ari_per_fov))
    print(f"  d={d:3d}  mean ARI={mean_ari:.4f}   per-fov={ {k: round(v,3) for k,v in ari_per_fov.items()} }")

best_d, best_ari, _ = max(val_table, key=lambda x: x[1])
print(f"\nBest expansion: d={best_d}  val ARI={best_ari:.4f}")
baseline_ari = next((a for d,a,_ in val_table if d == 0), None)
if baseline_ari is not None:
    print(f"Baseline d=0 val ARI: {baseline_ari:.4f}")


# ── Test: emit one submission per distance ──────────────────────────────────
print("\n=== TEST submissions ===")
test_preds = {}
for fov_name in TEST_FOVS:
    if not os.path.exists(f"{DATA_ROOT}/test/{fov_name}"):
        continue
    test_preds[fov_name] = predict_fov(fov_name, "test")
    print(f"  {fov_name}: {test_preds[fov_name].max()} cells")

os.makedirs(args.submissions_dir, exist_ok=True)
for d in DISTANCES:
    parts = []
    for fov_name, pred in test_preds.items():
        expanded = pred if d == 0 else expand_labels(pred, distance=d)
        fov_spots = test_spots[test_spots["fov"] == fov_name]
        rr = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
        cc = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
        cell_ids = expanded[rr, cc]
        labels_out = np.where(
            cell_ids > 0,
            np.array([f"{fov_name}_cell_{v}" for v in cell_ids]),
            "background",
        )
        parts.append(pd.DataFrame({
            "spot_id":    fov_spots["spot_id"].values,
            "fov":        fov_name,
            "cluster_id": labels_out,
        }))
    combined   = pd.concat(parts, ignore_index=True)
    submission = (
        test_spots[["spot_id", "fov"]]
        .merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
    )
    submission["cluster_id"] = submission["cluster_id"].fillna("background")
    out_local = f"submission_{EXP_NAME}_exp{d}.csv"
    submission[["spot_id", "fov", "cluster_id"]].to_csv(out_local, index=False)
    bg_frac = (submission["cluster_id"] == "background").mean()
    shutil.copy(out_local, os.path.join(args.submissions_dir, out_local))
    print(f"  d={d:3d}  bg={bg_frac:.3f}  saved {out_local}")

print(f"\nDone. Best val d={best_d}, ARI={best_ari:.4f}")
