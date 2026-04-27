"""Zero-shot InstanSeg inference on val FOVs + test FOVs.

InstanSeg 'fluorescence_nuclei_and_cells' expects 2-channel input:
  ch0 = nuclei (DAPI), ch1 = cells/cytoplasm (polyT)
Input shape: (H, W, 2) with pixel_size in µm.

Outputs submission_instanseg.csv and prints val ARI for quick assessment.
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, '/home/cg4652/.local/lib/python3.13/site-packages')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="fluorescence_nuclei_and_cells",
                    help="InstanSeg pretrained model name")
parser.add_argument("--val-only", action="store_true",
                    help="Only run val FOVs (skip test / submission)")
args = parser.parse_args()

DATA_ROOT  = "/scratch/cg4652/competition"
PIXEL_SIZE = 0.109  # µm/px

from instanseg import InstanSeg
from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask

print(f"Loading InstanSeg model: {args.model}")
model = InstanSeg(args.model, verbosity=1)

meta        = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells_df    = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

val_fovs  = [f"FOV_{i:03d}" for i in range(36, 41)]
test_fovs = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]

def run_fov(fov_dir: str) -> np.ndarray:
    """Load FOV and return integer mask (H, W)."""
    dapi, polyt = load_fov_images(fov_dir)
    dapi_max  = np.max(dapi,  axis=0).astype(np.float32)
    polyt_max = np.max(polyt, axis=0).astype(np.float32)
    # InstanSeg expects (H, W, C): ch0=nuclei(DAPI), ch1=cells(polyT)
    img = np.stack([dapi_max, polyt_max], axis=-1)
    labeled, _ = model.eval_small_image(img, pixel_size=PIXEL_SIZE)
    # labeled is a tensor (1, C_out, H, W); squeeze to (H, W)
    import torch
    if isinstance(labeled, torch.Tensor):
        labeled = labeled.cpu().numpy()
    # shape may be (1, 1, H, W) or (1, 2, H, W) — take last channel (whole cell)
    labeled = np.squeeze(labeled)
    if labeled.ndim == 3:
        labeled = labeled[-1]  # take whole-cell mask (last channel)
    return labeled.astype(np.int32)

# ── Val evaluation ────────────────────────────────────────────────────────────
print("\n=== Validating on FOVs 036-040 ===")
ari_scores = {}
for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue
    fov_x = meta.loc[fov_name, "fov_x"]
    fov_y = meta.loc[fov_name, "fov_y"]
    fov_spots = spots_train[spots_train["fov"] == fov_name]
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)

    pred_masks = run_fov(fov_dir)
    gt_mask    = boundaries_to_mask(cells_df, fov_name, fov_x, fov_y)

    pred_ids = pred_masks[rows, cols]
    gt_ids   = gt_mask[rows, cols]
    ari = adjusted_rand_score(gt_ids, pred_ids)
    ari_scores[fov_name] = ari
    print(f"  {fov_name}: ARI={ari:.4f}  ({pred_masks.max()} cells)")

mean_ari = float(np.mean(list(ari_scores.values()))) if ari_scores else 0.0
print(f"\nMean val ARI (InstanSeg zero-shot): {mean_ari:.4f}")
print(f"Baseline (pretrained Cellpose):      0.632")
print(f"Our best Kaggle so far:              0.7588 (nuclei_cosine)")

if args.val_only:
    sys.exit(0)

# ── Test inference → submission CSV ──────────────────────────────────────────
print("\n=== Running on test FOVs ===")
test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
all_parts  = []

for fov_name in test_fovs:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue
    pred_masks = run_fov(fov_dir)
    fov_spots  = test_spots[test_spots["fov"] == fov_name]
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    cell_ids = pred_masks[rows, cols]
    labels = np.where(
        cell_ids > 0,
        np.array([f"{fov_name}_cell_{v}" for v in cell_ids]),
        "background",
    )
    n_assigned = (labels != "background").sum()
    print(f"  {fov_name}: {pred_masks.max()} cells, {n_assigned}/{len(labels)} spots assigned")
    all_parts.append(pd.DataFrame({
        "spot_id":    fov_spots["spot_id"].values,
        "fov":        fov_name,
        "cluster_id": labels,
    }))

combined   = pd.concat(all_parts, ignore_index=True)
submission = (
    test_spots[["spot_id", "fov"]]
    .merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
)
submission["cluster_id"] = submission["cluster_id"].fillna("background")
out_path = "submission_instanseg.csv"
submission[["spot_id", "fov", "cluster_id"]].to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(submission):,} rows)")
