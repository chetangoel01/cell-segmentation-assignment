"""Score volume-augmented classifier on val FOVs (or run inference on test).

Computes per-cell volume from predicted mask (cell_area in pixels / 4 to match
GT um^3 scale roughly — phase 2 GT volume median ~ 393, mask area median can be
in the thousands. Use simple normalization vs train_volume_median saved in clf bundle).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase2.src import io, coords  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, _normalize, LEVELS,
)
from phase2.scripts.validate_local import _build_gt_spot_labels  # noqa: E402

VAL_FOVS = ["FOV_156", "FOV_157", "FOV_158", "FOV_159", "FOV_160"]
TEST_FOVS = [f"FOV_{c}" for c in "EFGHIJKLMN"]


def cell_areas(masks: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    """Return area-in-pixels for each cell in cell_ids order."""
    out = np.zeros(len(cell_ids), dtype=np.float32)
    for i, c in enumerate(cell_ids):
        out[i] = (masks == c).sum()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--mode", choices=("val", "test"), required=True)
    p.add_argument("--out-csv", default=None,
                   help="If set + mode=test: write submission CSV here.")
    p.add_argument("--vol-area-scale", type=float, default=1/8.4,
                   help="Pixel-area to volume scale. Mask areas are ~3300px median, vol target ~393.")
    args = p.parse_args()

    models = {lvl: joblib.load(Path(args.models_dir) / f"model_{lvl}.joblib") for lvl in LEVELS}
    train_vol_med = models["class"].get("train_volume_median", 393.0)
    genes = models["class"]["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"loaded vol classifier; train_vol_median={train_vol_med:.1f}")

    masks_dir = Path(args.masks_dir)
    if args.mode == "val":
        spots = pd.read_csv(ROOT / "phase2" / "data" / "train" / "ground_truth" / "spots_train.csv")
        cells = pd.read_csv(ROOT / "phase2" / "data" / "train" / "ground_truth" / "cell_boundaries_train.csv")
        cells = cells.rename(columns={cells.columns[0]: "cell_id"})
        cells.set_index("cell_id", inplace=True)
        labels = pd.read_csv(ROOT / "phase2" / "data" / "train" / "ground_truth" / "cell_labels_train.csv")
        fovs = VAL_FOVS
    else:
        spots = pd.read_csv(ROOT / "phase2" / "data" / "test_spots.csv")
        fovs = TEST_FOVS

    from sklearn.metrics import adjusted_rand_score
    rows = []
    submission_parts = []
    for fov in fovs:
        mask_path = masks_dir / f"{fov}.npy"
        if not mask_path.exists():
            print(f"  [skip] {fov} masks missing")
            continue
        masks = np.load(mask_path).astype(np.int32)
        fov_spots = spots[spots["fov"] == fov].copy()
        if args.mode == "val":
            fov_spots, _ = _build_gt_spot_labels(fov, fov_spots, cells, labels)

        cell_ids, X_counts, spot_label = _featurize_cells_from_mask(
            masks, fov_spots, gene_to_idx, nn_radius=0.0,
        )
        if cell_ids.size == 0:
            print(f"  {fov}: no cells in mask — skipping")
            if args.mode == "test":
                pred = pd.DataFrame({"spot_id": fov_spots["spot_id"].values, "fov": fov,
                                     "class": "background", "subclass": "background",
                                     "supertype": "background", "cluster": "background"})
                submission_parts.append(pred)
            continue

        # Compute predicted-cell volumes from mask area, scale to GT volume range.
        areas = cell_areas(masks, cell_ids)
        vol_pred = areas * args.vol_area_scale  # rough match
        vol_norm = (vol_pred / train_vol_med).reshape(-1, 1).astype(np.float32)

        X_norm = _normalize(X_counts, preproc="log1p")
        X_full = np.concatenate([X_norm, vol_norm], axis=1)

        # Per-spot predictions
        spot_pred = {lvl: np.array(["background"] * len(fov_spots), dtype=object) for lvl in LEVELS}
        for lvl in LEVELS:
            yh = models[lvl]["clf"].predict(X_full)
            cid_to_label = dict(zip(cell_ids, yh))
            for cid, lab in cid_to_label.items():
                spot_pred[lvl][spot_label == cid] = lab

        if args.mode == "val":
            for lvl in LEVELS:
                ari = adjusted_rand_score(fov_spots[f"gt_{lvl}"].astype(str).values, spot_pred[lvl])
                rows.append({"fov": fov, "level": lvl, "ari": ari})
            print(f"  {fov}: " + "  ".join(f"{r['level']}={r['ari']:.4f}" for r in rows[-4:]))
        else:
            df = pd.DataFrame({"spot_id": fov_spots["spot_id"].values, "fov": fov})
            for lvl in LEVELS:
                df[lvl] = spot_pred[lvl]
            submission_parts.append(df)
            n_assigned = (df["class"] != "background").sum()
            print(f"  {fov}: {n_assigned}/{len(df)} assigned ({100*n_assigned/len(df):.1f}%)")

    if args.mode == "val":
        df = pd.DataFrame(rows)
        print()
        print("=" * 60)
        for fov in fovs:
            sub = df[df.fov == fov]
            if not sub.empty:
                print(f"  {fov} mean={sub.ari.mean():.4f}")
        mean_ari = df.ari.mean() if not df.empty else 0.0
        print(f"  MEAN across {len(fovs)}x4 = {mean_ari:.4f}")
    elif args.out_csv:
        all_sub = pd.concat(submission_parts, ignore_index=True)
        # Align with sample_submission.csv ordering
        sample = pd.read_csv(ROOT / "phase2" / "data" / "sample_submission.csv")
        merged = sample[["spot_id", "fov"]].merge(
            all_sub.drop(columns=["fov"]), on="spot_id", how="left",
        )
        for lvl in LEVELS:
            merged[lvl] = merged[lvl].fillna("background")
        merged = merged[["spot_id", "fov", "class", "subclass", "supertype", "cluster"]]
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out_csv, index=False)
        print(f"\n→ {args.out_csv} ({len(merged):,} rows)")


if __name__ == "__main__":
    main()
