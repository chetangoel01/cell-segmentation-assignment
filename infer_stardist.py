"""Run fine-tuned StarDist model on test FOVs and generate submission CSV."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.io import load_fov_images

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="stardist")
parser.add_argument("--channel", default="dapi", choices=["dapi", "polyt", "both"])
args = parser.parse_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from stardist.models import StarDist2D   # noqa: E402
from csbdeep.utils import normalize      # noqa: E402

EXP_NAME  = args.exp_name
DATA_ROOT = "/scratch/cg4652/competition"
TEST_FOVS = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
BASEDIR   = "models"

model_dir = os.path.join(BASEDIR, EXP_NAME)
if not os.path.exists(os.path.join(model_dir, "weights_best.h5")):
    raise SystemExit(f"No trained model at {model_dir}/weights_best.h5 — train first.")

print(f"Loading StarDist model: {model_dir}")
model = StarDist2D(None, name=EXP_NAME, basedir=BASEDIR)

test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
print(f"Test spots: {len(test_spots):,}")


def build_input(dapi: np.ndarray, polyt: np.ndarray) -> np.ndarray:
    dapi_max  = np.max(dapi,  axis=0).astype(np.float32)
    polyt_max = np.max(polyt, axis=0).astype(np.float32)
    if args.channel == "dapi":
        return dapi_max
    elif args.channel == "polyt":
        return polyt_max
    else:
        return np.stack([dapi_max, polyt_max], axis=-1)


all_parts = []
for fov_name in TEST_FOVS:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    img = build_input(dapi, polyt)
    img = normalize(img, 1, 99.8, axis=(0, 1))

    labels, _ = model.predict_instances(img)
    pred_masks = labels.astype(np.int32)
    print(f"{fov_name}: {pred_masks.max()} cells detected")

    fov_spots = test_spots[test_spots["fov"] == fov_name]
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    cell_ids = pred_masks[rows, cols]

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
