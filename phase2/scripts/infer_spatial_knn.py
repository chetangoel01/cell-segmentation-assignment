"""Inference for the spatial-context kNN classifier.

Mirrors phase2/tasks/infer_baseline.py but augments per-cell gene counts
with the L1-normalized mean of the cell's k spatial neighbors (within FOV)
before calling clf.predict. The classifier bundle must contain a
'spatial_k' field (set by train_spatial_knn.py).

Usage:
    .venv/bin/python phase2/scripts/infer_spatial_knn.py \\
      --models-dir phase2/runs/spatial-knn-k5-sk5-nw05 \\
      --test-fovs FOV_E,FOV_F,...,FOV_N \\
      --seg-checkpoint phase2/external_models/cellpose_nuclei_cosine_ep125 \\
      --include-spot-density --spot-density-sigma 8.0 \\
      --cellprob-threshold -0.5 --flow-threshold 0.4 \\
      --out-dir phase2/runs/voter-spatial-knn

Reads sample_submission.csv to seed the output frame so all spots that
Kaggle expects appear (rows for un-segmented spots default to 'background').
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.src import io  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, _pick_device, LEVELS,
)

LEVELS_TUP = LEVELS  # ('class','subclass','supertype','cluster')


def _cell_centroids(masks: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of (row_centroid, col_centroid) for each label in cell_ids."""
    cents = np.zeros((len(cell_ids), 2), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        rows, cols = np.where(masks == int(cid))
        if len(rows):
            cents[i, 0] = rows.mean()
            cents[i, 1] = cols.mean()
    return cents


def _augment_with_spatial(X: np.ndarray, centroids: np.ndarray,
                           spatial_k: int, neighbor_weight: float) -> np.ndarray:
    """L1-normalize, then concat [X_norm, neighbor_weight * neighbor_mean(X_norm)]."""
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import NearestNeighbors
    X_norm = normalize(X.astype(np.float32), norm="l1")
    n = len(X_norm)
    if n < 2:
        # No neighbors possible - duplicate self
        return np.hstack([X_norm, neighbor_weight * X_norm]).astype(np.float32)
    k_eff = min(spatial_k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(centroids)
    _, idx = nn.kneighbors(centroids)
    neighbor_mean = X_norm[idx[:, 1:]].mean(axis=1)
    return np.hstack([X_norm, neighbor_weight * neighbor_mean]).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--test-fovs", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seg-checkpoint", default=None)
    p.add_argument("--include-spot-density", action="store_true")
    p.add_argument("--spot-density-sigma", type=float, default=8.0)
    p.add_argument("--cellpose-diameter", type=float, default=0.0)
    p.add_argument("--cellprob-threshold", type=float, default=-0.5)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--nn-radius", type=float, default=0.0)
    p.add_argument("--neighbor-weight", type=float, default=0.5,
                   help="Must match the value used during training.")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_fovs = [f.strip() for f in args.test_fovs.split(",") if f.strip()]
    print(f"test FOVs: {test_fovs}")

    # Load classifier bundles
    bundles = {lvl: joblib.load(Path(args.models_dir) / f"model_{lvl}.joblib") for lvl in LEVELS_TUP}
    genes = bundles["class"]["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    spatial_k = int(bundles["class"]["spatial_k"])
    print(f"  loaded clf bundles, gene vocab={len(genes)}, spatial_k={spatial_k}")

    # Cellpose
    import torch  # noqa: F401
    from cellpose import models as cp_models
    device, gpu = _pick_device(args.device)
    print(f"  device: {device}  (gpu={gpu})")
    if args.seg_checkpoint:
        ck = Path(args.seg_checkpoint)
        if not ck.exists():
            print(f"[fatal] checkpoint not found: {ck}")
            return 1
        seg_model = cp_models.CellposeModel(gpu=gpu, pretrained_model=str(ck), device=device)
    else:
        seg_model = cp_models.CellposeModel(gpu=gpu, device=device)

    # Load test spots, seed submission with all-background
    spots = pd.read_csv(io.data_root() / "test_spots.csv")
    sub = pd.DataFrame({
        "spot_id": spots["spot_id"].values,
        "fov": spots["fov"].values,
    })
    for lvl in LEVELS_TUP:
        sub[lvl] = "background"

    per_fov_stats: dict[str, dict] = {}
    t_start = time.time()

    for fov in test_fovs:
        print(f"\n=== {fov} ===")
        fov_spots = spots[spots["fov"] == fov].reset_index(drop=True)
        if len(fov_spots) == 0:
            print(f"  [skip] no spots")
            continue
        print(f"  spots: {len(fov_spots):,}")

        from phase2.src.io import load_fov_images
        dapi, polyt = load_fov_images(fov, split="test")
        dapi2d = dapi.max(axis=0).astype(np.float32)
        polyt2d = polyt.max(axis=0).astype(np.float32)
        channels = [polyt2d, dapi2d]
        if args.include_spot_density:
            from scipy.ndimage import gaussian_filter
            d = np.zeros((2048, 2048), dtype=np.float32)
            r = fov_spots["image_row"].to_numpy().astype(int).clip(0, 2047)
            c = fov_spots["image_col"].to_numpy().astype(int).clip(0, 2047)
            np.add.at(d, (r, c), 1)
            d = gaussian_filter(d, sigma=args.spot_density_sigma)
            if d.max() > 0:
                d = d / d.max() * 65535.0
            channels.append(d.astype(np.float32))
        img = np.stack(channels, axis=0).astype(np.float32)

        t0 = time.time()
        masks, _, _ = seg_model.eval(
            img, channel_axis=0, diameter=args.cellpose_diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        seg_t = time.time() - t0
        print(f"  cellpose: {int(masks.max())} cells, {seg_t:.1f}s")

        cell_ids, X, spot_label = _featurize_cells_from_mask(
            masks, fov_spots, gene_to_idx, nn_radius=args.nn_radius
        )
        if len(cell_ids) == 0:
            per_fov_stats[fov] = {"n_cells": 0, "n_spots_in_cell": 0}
            continue

        # Compute centroids and augmented features
        centroids = _cell_centroids(masks, cell_ids)
        X_aug = _augment_with_spatial(X, centroids, spatial_k, args.neighbor_weight)
        print(f"  spatial-aug: {X_aug.shape} (centroids → kNN)")

        # Predict per level
        cell_predictions = {lvl: bundles[lvl]["clf"].predict(X_aug) for lvl in LEVELS_TUP}

        # Map spots to cells (spot_label gives which cell each in-cell spot belongs to)
        label_to_idx = {int(c): i for i, c in enumerate(cell_ids)}
        # Find which submission rows correspond to this FOV's spots, in order
        fov_mask_in_sub = sub["fov"].values == fov
        fov_indices_in_sub = np.where(fov_mask_in_sub)[0]
        # spot_label aligns with fov_spots; fov_spots aligns with sub rows for this FOV
        spot_idx = np.array([label_to_idx.get(int(l), -1) for l in spot_label])
        in_cell = spot_idx >= 0
        for lvl in LEVELS_TUP:
            preds = cell_predictions[lvl]
            new_labels = np.full(len(fov_spots), "background", dtype=object)
            new_labels[in_cell] = preds[spot_idx[in_cell]]
            sub.loc[fov_indices_in_sub, lvl] = new_labels

        n_in_cell = int(in_cell.sum())
        per_fov_stats[fov] = {
            "n_cells": len(cell_ids),
            "n_spots_in_cell": n_in_cell,
            "frac_in_cell": float(n_in_cell / len(fov_spots)),
            "seg_s": seg_t,
        }
        print(f"  in-cell: {n_in_cell}/{len(fov_spots)} ({n_in_cell/len(fov_spots):.1%})")

    # Write submission
    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nWrote {len(sub):,} rows → {sub_path}")
    summary = {
        "models_dir": args.models_dir,
        "spatial_k": spatial_k,
        "neighbor_weight": args.neighbor_weight,
        "test_fovs": test_fovs,
        "wall_total_s": time.time() - t_start,
        "per_fov": per_fov_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
