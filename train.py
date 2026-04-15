"""Fine-tune Cellpose on training FOVs and evaluate on validation FOVs."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cellpose import models as cp_models, train as cp_train

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask

DATA_ROOT = "/scratch/pl2820/competition"
PIXEL_SIZE = 0.109
MODEL_SAVE_DIR = "models"
MODEL_NAME = "cellpose_finetuned"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("Loading metadata and ground truth...")
meta = pd.read_csv(f"{DATA_ROOT}/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(
    f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0
)

# Build training set (FOVs 001-035)
train_images, train_masks = [], []
train_fovs = [f"FOV_{i:03d}" for i in range(1, 36)]

for fov_name in train_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue
    try:
        dapi, polyt = load_fov_images(fov_dir)
        fov_x = meta.loc[fov_name, "fov_x"]
        fov_y = meta.loc[fov_name, "fov_y"]
        m = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
        if m.max() == 0:
            print(f"  No cells in mask: {fov_name}")
            continue
        train_images.append(np.stack([polyt[2], dapi[2]], axis=0))
        train_masks.append(m)
        print(f"  Loaded {fov_name}: {m.max()} cells")
    except Exception as exc:
        print(f"  Skipping {fov_name}: {exc}")

print(f"\nTraining on {len(train_images)} FOVs")

# Fine-tune
base_model = cp_models.CellposeModel(gpu=True, model_type="cyto2")
model_path = cp_train.train_seg(
    base_model.net,
    train_data=train_images,
    train_labels=train_masks,
    channels=[1, 2],
    save_path=MODEL_SAVE_DIR,
    n_epochs=100,
    learning_rate=0.005,
    weight_decay=1e-5,
    batch_size=8,
    model_name=MODEL_NAME,
)
print(f"\nSaved fine-tuned model: {model_path}")

# Evaluate on validation FOVs (036-040)
finetuned_model = cp_models.CellposeModel(gpu=True, pretrained_model=model_path)
val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

ari_scores = {}
for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    fov_x = meta.loc[fov_name, "fov_x"]
    fov_y = meta.loc[fov_name, "fov_y"]

    pred_masks, _, _ = finetuned_model.eval(dapi[2], diameter=30, channels=[1, 2])
    gt_mask = boundaries_to_mask(cells, fov_name, fov_x, fov_y)

    fov_spots = spots_train[spots_train["fov"] == fov_name].copy()
    px = np.clip(((fov_spots["global_x"].values - fov_x) / PIXEL_SIZE).astype(int), 0, 2047)
    py = np.clip(((fov_spots["global_y"].values - fov_y) / PIXEL_SIZE).astype(int), 0, 2047)
    pred_ids = pred_masks[py, px]
    gt_ids = gt_mask[py, px]

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(gt_ids, pred_ids)
    ari_scores[fov_name] = ari
    print(f"{fov_name}: ARI = {ari:.4f}  ({pred_masks.max()} cells detected)")

mean_ari = float(np.mean(list(ari_scores.values()))) if ari_scores else 0.0
print(f"\nMean validation ARI : {mean_ari:.4f}")
print(f"Baseline (pretrained): 0.632")
print(f"Improvement          : {mean_ari - 0.632:+.4f}")
