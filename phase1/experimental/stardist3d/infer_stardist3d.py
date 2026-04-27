"""Run fine-tuned StarDist3D on test FOVs and generate submission CSV.

The 3D model predicts labels on each z-plane; we collapse to a 2D mask by
max-projection (each cell has the same integer ID across z, so this preserves
cell footprints) and then do spot assignment the usual way.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.io import load_fov_images
from src.stardist3d import collapse_3d_labels_to_2d

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="stardist3d")
args = parser.parse_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from stardist.models import StarDist3D   # noqa: E402
from csbdeep.utils import normalize      # noqa: E402

EXP_NAME  = args.exp_name
DATA_ROOT = os.environ.get("MERFISH_DATA_ROOT", "/scratch/cg4652/competition")
TEST_FOVS = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
BASEDIR   = "models"

model_dir = os.path.join(BASEDIR, EXP_NAME)
if not os.path.exists(os.path.join(model_dir, "weights_best.h5")):
    raise SystemExit(f"No trained model at {model_dir}/weights_best.h5 — train first.")

print(f"Loading StarDist3D model: {model_dir}")
model = StarDist3D(None, name=EXP_NAME, basedir=BASEDIR)

test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
print(f"Test spots: {len(test_spots):,}")

all_parts = []
for fov_name in TEST_FOVS:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue

    dapi, _polyt = load_fov_images(fov_dir)
    img = dapi.astype(np.float32)
    img = normalize(img, 1, 99.8, axis=(0, 1, 2))

    labels_3d, _ = model.predict_instances(img)
    pred_2d = collapse_3d_labels_to_2d(labels_3d)
    print(f"{fov_name}: {pred_2d.max()} cells detected (3D volume collapsed to 2D)")

    fov_spots = test_spots[test_spots["fov"] == fov_name]
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    cell_ids = pred_2d[rows, cols]

    labels_out = np.where(
        cell_ids > 0,
        np.array([f"{fov_name}_cell_{v}" for v in cell_ids]),
        "background",
    )
    n_assigned = (labels_out != "background").sum()
    print(f"  {fov_name}: {n_assigned:,}/{len(labels_out):,} spots assigned "
          f"({100*n_assigned/len(labels_out):.1f}%)")

    all_parts.append(pd.DataFrame({
        "spot_id":    fov_spots["spot_id"].values,
        "fov":        fov_name,
        "cluster_id": labels_out,
    }))

combined   = pd.concat(all_parts, ignore_index=True)
submission = (
    test_spots[["spot_id", "fov"]]
    .merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
)
submission["cluster_id"] = submission["cluster_id"].fillna("background")

out_path = f"submission_{EXP_NAME}.csv"
submission[["spot_id", "fov", "cluster_id"]].to_csv(out_path, index=False)
print(f"\nSaved {out_path} — {len(submission):,} rows")
print(submission["cluster_id"].value_counts().head(5))
