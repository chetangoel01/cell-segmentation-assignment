"""Run fine-tuned Cellpose on test FOVs and generate submission.csv."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from cellpose import models as cp_models

from src.io import load_fov_images
from src.train_cellpose import compute_spot_density
from generate_submission import build_submission

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="cyto2",
                    help="Experiment name matching the one used in train.py")
args = parser.parse_args()

EXP_NAME       = args.exp_name
DATA_ROOT      = "/scratch/cg4652/competition"
MODEL_SAVE_DIR = f"models/{EXP_NAME}"
MODEL_NAME     = f"cellpose_{EXP_NAME}"
FINAL_MODEL    = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
TEST_FOVS      = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
CELL_DIAMETER  = 88.9

os.makedirs("logs", exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────────────
if os.path.exists(FINAL_MODEL):
    print(f"Loading fine-tuned model: {FINAL_MODEL}")
    seg_model = cp_models.CellposeModel(gpu=True, pretrained_model=FINAL_MODEL)
else:
    print(f"Fine-tuned model not found, falling back to pretrained {EXP_NAME}")
    seg_model = cp_models.CellposeModel(gpu=True, model_type=EXP_NAME)

# ── Load data ────────────────────────────────────────────────────────────────
test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
print(f"Test spots : {len(test_spots):,}")

# ── Inference ────────────────────────────────────────────────────────────────
masks: dict[str, np.ndarray] = {}

for fov_name in TEST_FOVS:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    fov_spots = test_spots[test_spots["fov"] == fov_name]
    spot_density = compute_spot_density(fov_spots)
    dapi_max = np.max(dapi, axis=0)
    polyt_max = np.max(polyt, axis=0)
    img3ch = np.stack([polyt_max, dapi_max, spot_density], axis=0)
    pred_masks, _, _ = seg_model.eval(
        img3ch, diameter=0, cellprob_threshold=-1.0, channel_axis=0,
    )
    masks[fov_name] = pred_masks
    print(f"{fov_name}: {pred_masks.max()} cells detected")

# ── Build and save submission ─────────────────────────────────────────────────
print("\nBuilding submission...")
submission = build_submission(masks, test_spots)

out_path = f"submission_{EXP_NAME}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved {out_path} — {len(submission):,} rows")
print(submission["cluster_id"].value_counts().head(10))
