"""Run trained U-Net on test FOVs and generate submission CSV."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.io import load_fov_images
from src.train_cellpose import compute_spot_density
from src.unet import UNet, normalize_image, predict_to_instances


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="unet")
parser.add_argument("--base-channels", type=int, default=32)
parser.add_argument("--fg-thresh", type=float, default=0.5)
parser.add_argument("--marker-thresh", type=float, default=0.7)
args = parser.parse_args()

DATA_ROOT = "/scratch/cg4652/competition"
TEST_FOVS = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
EXP = args.exp_name
ckpt_path = f"models/{EXP}/unet_best.pt"
if not os.path.exists(ckpt_path):
    raise SystemExit(f"No checkpoint at {ckpt_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  Loading {ckpt_path}")
model = UNet(in_channels=3, n_classes=3, base=args.base_channels).to(device)
sd = torch.load(ckpt_path, map_location=device)
model.load_state_dict(sd["model"])
model.eval()

test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
print(f"Test spots: {len(test_spots):,}")
print(f"Thresholds: fg={args.fg_thresh}  marker={args.marker_thresh}")

all_parts = []
for fov_name in TEST_FOVS:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue
    dapi, polyt = load_fov_images(fov_dir)
    fov_spots = test_spots[test_spots["fov"] == fov_name]
    density = compute_spot_density(fov_spots, sigma=8.0)
    img = np.stack([
        np.max(polyt, axis=0).astype(np.float32),
        np.max(dapi, axis=0).astype(np.float32),
        density.astype(np.float32),
    ], axis=0)
    img = normalize_image(img)

    with torch.no_grad():
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_mask = predict_to_instances(probs, fg_thresh=args.fg_thresh,
                                      marker_thresh=args.marker_thresh)
    n_cells = int(pred_mask.max())
    print(f"{fov_name}: {n_cells} cells detected")

    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    cell_ids = pred_mask[rows, cols]
    labels_out = np.where(
        cell_ids > 0,
        np.array([f"{fov_name}_cell_{v}" for v in cell_ids]),
        "background",
    )
    n_assigned = (labels_out != "background").sum()
    print(f"  {fov_name}: {n_assigned:,}/{len(labels_out):,} spots assigned "
          f"({100 * n_assigned / len(labels_out):.1f}%)")

    all_parts.append(pd.DataFrame({
        "spot_id": fov_spots["spot_id"].values,
        "fov": fov_name,
        "cluster_id": labels_out,
    }))

combined = pd.concat(all_parts, ignore_index=True)
submission = (
    test_spots[["spot_id", "fov"]]
    .merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
)
submission["cluster_id"] = submission["cluster_id"].fillna("background")

out_path = f"submission_{EXP}.csv"
submission[["spot_id", "fov", "cluster_id"]].to_csv(out_path, index=False)
print(f"\nSaved {out_path} — {len(submission):,} rows")
print(submission["cluster_id"].value_counts().head(5))
