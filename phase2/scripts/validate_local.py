"""Local validator: run the inference pipeline on TRAIN FOVs (where we have
ground truth polygons + cell labels) and compute spot-level ARI per the Kaggle
metric — mean ARI across (FOV, level) pairs.

Mirrors phase2/tasks/infer_baseline.py but:
  - reads from train/ split (FOVs 151-160 are our val hold-out)
  - constructs ground-truth spot labels by point-in-polygon against
    cell_boundaries_train.csv polygons + cell_labels_train.csv labels
  - computes spot-level ARI on each (FOV, level) pair and prints the mean

Usage:
    .venv/bin/python phase2/scripts/validate_local.py \\
        --models-dir   phase2/runs/<baseline-dir> \\
        --seg-checkpoint phase2/external_models/cellpose_nuclei_cosine_ep125 \\
        --include-spot-density \\
        --cellpose-diameter 0 \\
        --cellprob-threshold -1.0 \\
        --flow-threshold 0.4 \\
        --nn-radius 15 \\
        --val-fovs FOV_151,FOV_152,...,FOV_160
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make `phase2` importable when this script is run directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.src import io, coords  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, _normalize, _segment_fov, _pick_device, LEVELS,
)


def _build_gt_spot_labels(fov: str, fov_spots: pd.DataFrame,
                          cells: pd.DataFrame, labels: pd.DataFrame
                          ) -> pd.DataFrame:
    """Return fov_spots augmented with gt_class, gt_subclass, gt_supertype,
    gt_cluster columns. Spots not inside any labeled cell polygon → 'background'.

    Uses z=2 polygons only — matches `phase2/docs/codelab_13.py:rasterize_gt_mask`
    which is the reference Kaggle scoring convention.
    """
    fov_cells = labels[labels["fov"] == fov]
    spot_x = fov_spots["global_x"].to_numpy()
    spot_y = fov_spots["global_y"].to_numpy()

    out = fov_spots.copy()
    for lvl in LEVELS:
        out[f"gt_{lvl}"] = "background"

    n_assigned = 0
    for cid in fov_cells["cell_id"]:
        if cid not in cells.index:
            continue
        row = cells.loc[cid]
        poly = coords.parse_boundary_polygon(
            row.get("boundaryX_z2", ""), row.get("boundaryY_z2", "")
        )
        if poly is None:
            continue
        inside = coords.spots_in_polygon(spot_x, spot_y, poly)
        if not inside.any():
            continue
        cell_row = fov_cells[fov_cells["cell_id"] == cid].iloc[0]
        for lvl in LEVELS:
            out.loc[fov_spots.index[inside], f"gt_{lvl}"] = cell_row[f"{lvl}_label"]
        n_assigned += int(inside.sum())
    return out, n_assigned


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--val-fovs", required=True,
                   help="Comma-separated val FOVs (e.g. FOV_151,...,FOV_160).")
    p.add_argument("--seg-checkpoint", default=None)
    p.add_argument("--masks-dir", default=None,
                   help="Directory of precomputed <FOV>.npy masks (skips Cellpose). "
                        "Used to plug in StarDist or other non-Cellpose seg.")
    p.add_argument("--tta", action="store_true",
                   help="Cellpose TTA: averages flow fields over 4 rotations (augment=True). "
                        "+~30%% inference time. Typical +1-3%% ARI.")
    p.add_argument("--save-masks-dir", default=None,
                   help="Optional dir to save per-FOV mask <FOV>.npy files. Useful "
                        "for downstream mask-ensemble experiments without re-running "
                        "cellpose.")
    p.add_argument("--include-spot-density", action="store_true")
    p.add_argument("--spot-density-sigma", type=float, default=8.0)
    p.add_argument("--cellpose-diameter", type=float, default=30.0)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--nn-radius", type=float, default=0.0)
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--out", default=None,
                   help="Optional JSON file for per-FOV/per-level ARI breakdown.")
    args = p.parse_args(argv)

    val_fovs = [f.strip() for f in args.val_fovs.split(",") if f.strip()]
    print(f"val FOVs: {val_fovs}")

    # Classifier bundles + gene vocab (from train-baseline output)
    import joblib
    bundles = {lvl: joblib.load(Path(args.models_dir) / f"model_{lvl}.joblib") for lvl in LEVELS}
    genes = bundles["class"]["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    preproc = bundles["class"].get("preproc", "log1p")
    print(f"loaded classifiers, gene vocab={len(genes)}, preproc={preproc}")

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    if masks_dir is not None:
        if not masks_dir.is_dir():
            print(f"[fatal] --masks-dir {masks_dir} not a directory")
            return 1
        print(f"using precomputed masks from {masks_dir} (skipping Cellpose)")
        seg_model = None
        device, gpu = "n/a", False
    else:
        # Cellpose
        import torch  # noqa: F401
        from cellpose import models as cp_models
        device, gpu = _pick_device(args.device)
        print(f"device: {device}  (gpu={gpu})")
        if args.seg_checkpoint:
            ck = Path(args.seg_checkpoint)
            if not ck.exists():
                print(f"[fatal] checkpoint not found: {ck}")
                return 1
            print(f"  fine-tuned checkpoint: {ck}")
            seg_model = cp_models.CellposeModel(gpu=gpu, pretrained_model=str(ck), device=device)
        else:
            print("  off-the-shelf cpsam")
            seg_model = cp_models.CellposeModel(gpu=gpu, device=device)

    save_masks_dir = Path(args.save_masks_dir) if args.save_masks_dir else None
    if save_masks_dir is not None:
        save_masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"saving per-FOV masks to {save_masks_dir}")

    # Train-split inputs: spots_train.csv + cell_boundaries_train.csv + cell_labels_train.csv
    spots = pd.read_csv(io.ground_truth_dir() / "spots_train.csv")
    cells = pd.read_csv(io.ground_truth_dir() / "cell_boundaries_train.csv", index_col=0)
    labels = pd.read_csv(io.ground_truth_dir() / "cell_labels_train.csv")
    print(f"loaded gt: cells={len(cells):,} labels={len(labels):,} spots={len(spots):,}")

    from sklearn.metrics import adjusted_rand_score

    per_fov_ari: dict[str, dict[str, float]] = {}
    per_fov_meta: dict[str, dict] = {}
    overall_ari: list[float] = []

    for fov in val_fovs:
        print(f"\n=== {fov} ===")
        fov_spots = spots[spots["fov"] == fov].copy().reset_index(drop=True)
        if len(fov_spots) == 0:
            print(f"  [skip] no spots for {fov}")
            continue
        print(f"  spots in FOV: {len(fov_spots):,}")

        # GT labels via point-in-polygon
        t0 = time.time()
        fov_spots_gt, n_in_cell_gt = _build_gt_spot_labels(fov, fov_spots, cells, labels)
        gt_t = time.time() - t0
        print(f"  GT: {n_in_cell_gt}/{len(fov_spots)} spots in cells "
              f"({n_in_cell_gt/len(fov_spots):.1%})  ({gt_t:.1f}s)")

        # Segment + featurize via train split
        t0 = time.time()
        if masks_dir is not None:
            mask_path = masks_dir / f"{fov}.npy"
            if not mask_path.exists():
                print(f"  [skip] {mask_path} not found")
                continue
            masks = np.load(mask_path).astype(np.int32)
        else:
            # validate_local mirrors test-time inference but reads from train/<FOV>/
            from phase2.src.io import load_fov_images
            dapi, polyt = load_fov_images(fov, split="train")
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
            masks, _, _ = seg_model.eval(
                img, channel_axis=0, diameter=args.cellpose_diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                augment=args.tta,
            )
        seg_t = time.time() - t0

        if save_masks_dir is not None:
            np.save(save_masks_dir / f"{fov}.npy", masks.astype(np.int32))

        # Feature build with optional nn_radius post-processing
        cell_ids, X, spot_label = _featurize_cells_from_mask(
            masks, fov_spots, gene_to_idx, nn_radius=args.nn_radius
        )

        # Predict — apply preproc that matches what the classifier was trained on.
        if len(cell_ids):
            if preproc == "l1_normalize":
                from sklearn.preprocessing import normalize
                X_n = normalize(X.astype(np.float32), norm="l1")
            else:
                X_n = _normalize(X)
            cell_predictions = {lvl: bundles[lvl]["clf"].predict(X_n) for lvl in LEVELS}
        else:
            cell_predictions = {lvl: np.array([]) for lvl in LEVELS}

        # Build predicted spot labels
        label_to_idx = {int(c): i for i, c in enumerate(cell_ids)}
        spot_idx = np.array([label_to_idx.get(int(l), -1) for l in spot_label])
        in_cell_mask = spot_idx >= 0

        per_lvl: dict[str, float] = {}
        for lvl in LEVELS:
            preds = cell_predictions[lvl]
            pred_spot = np.full(len(fov_spots), "background", dtype=object)
            if len(preds):
                pred_spot[in_cell_mask] = preds[spot_idx[in_cell_mask]]
            gt_spot = fov_spots_gt[f"gt_{lvl}"].to_numpy().astype(str)
            ari = adjusted_rand_score(gt_spot, pred_spot.astype(str))
            per_lvl[lvl] = float(ari)
            overall_ari.append(ari)
        per_fov_ari[fov] = per_lvl
        per_fov_meta[fov] = {
            "n_spots": int(len(fov_spots)),
            "n_cells_segmented": int(masks.max()),
            "frac_in_cell_pred": float(in_cell_mask.mean()),
            "frac_in_cell_gt": float(n_in_cell_gt / len(fov_spots)),
            "seg_s": seg_t,
        }
        print("  per-level ARI:")
        for lvl, v in per_lvl.items():
            print(f"    {lvl:<10} {v:+.4f}")
        print(f"  pred frac_in_cell={in_cell_mask.mean():.3f}  "
              f"gt frac_in_cell={n_in_cell_gt/len(fov_spots):.3f}")

    print()
    print("=" * 72)
    print(f"MEAN ARI across {len(per_fov_ari)} FOVs × {len(LEVELS)} levels "
          f"= {np.mean(overall_ari):.4f}")
    print("=" * 72)
    print()
    print("per-FOV mean ARI (across 4 levels):")
    for fov, levels_ari in per_fov_ari.items():
        m = np.mean(list(levels_ari.values()))
        print(f"  {fov}  mean={m:+.4f}   "
              f"frac_in_cell pred={per_fov_meta[fov]['frac_in_cell_pred']:.3f} "
              f"gt={per_fov_meta[fov]['frac_in_cell_gt']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": {k: v for k, v in vars(args).items()},
            "mean_ari": float(np.mean(overall_ari)),
            "per_fov": {f: {"per_level": per_fov_ari[f], **per_fov_meta[f]} for f in per_fov_ari},
        }, indent=2))
        print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
