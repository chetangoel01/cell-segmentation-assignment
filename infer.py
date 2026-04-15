"""Run fine-tuned Cellpose on test FOVs and generate submission.csv."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from cellpose import models as cp_models

from src.io import load_fov_images

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="cyto2",
                    help="Experiment name matching the one used in train.py")
args = parser.parse_args()

EXP_NAME       = args.exp_name
DATA_ROOT      = "/scratch/pl2820/competition"
MODEL_SAVE_DIR = f"models/{EXP_NAME}"
MODEL_NAME     = f"cellpose_{EXP_NAME}"
TEST_FOVS      = ["FOV_041", "FOV_042", "FOV_043", "FOV_044"]

os.makedirs("logs", exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
if not os.path.exists(model_path):
    state_file = os.path.join(MODEL_SAVE_DIR, "train_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        model_path = state.get("latest_checkpoint", model_path)
        print(f"Final model not found; using latest checkpoint: {model_path}")
    else:
        raise FileNotFoundError(
            f"No model at {model_path} and no train_state.json. Run train.py first."
        )

print(f"Experiment : {EXP_NAME}")
print(f"Model      : {model_path}")
seg_model = cp_models.CellposeModel(gpu=True, pretrained_model=model_path)

# ── Load data ────────────────────────────────────────────────────────────────
test_spots   = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
sub_template = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

print(f"Test spots : {len(test_spots):,}")
print(f"Template   : {len(sub_template):,} rows")

# ── Inference ────────────────────────────────────────────────────────────────
all_cluster_ids: dict[str, int] = {}

for fov_name in TEST_FOVS:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    pred_masks, _, _ = seg_model.eval(dapi[2], diameter=30, channels=[1, 2])
    print(f"{fov_name}: {pred_masks.max()} cells detected")

    fov_spots = test_spots[test_spots["fov"] == fov_name].copy()
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    cluster_ids = pred_masks[rows, cols]

    for spot_id, cid in zip(fov_spots["spot_id"].values, cluster_ids):
        all_cluster_ids[spot_id] = int(cid)

    bg = (cluster_ids == 0).mean()
    print(f"  {len(fov_spots):,} spots, {bg:.1%} background")

print(f"\nTotal assigned: {len(all_cluster_ids):,}")

# ── Build submission ──────────────────────────────────────────────────────────
sub = sub_template.copy()
sub["cluster_id"] = sub["spot_id"].map(all_cluster_ids)

null_count = sub["cluster_id"].isna().sum()
if null_count > 0:
    print(f"WARNING: {null_count} spots not found — filling with 0 (background)")
    sub["cluster_id"] = sub["cluster_id"].fillna(0)

sub["cluster_id"] = sub["cluster_id"].astype(int)

assert len(sub) == len(sub_template), "Row count mismatch"
assert list(sub.columns) == ["spot_id", "fov", "cluster_id"]
assert sub["cluster_id"].notna().all()

out_path = f"submission_{EXP_NAME}.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved {out_path} — {len(sub):,} rows")
print(sub["cluster_id"].value_counts().head(10))
