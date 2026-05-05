"""scANVI inference: load per-level scANVI bundles + cellpose seg, predict
4-level taxonomy per cell, project to spots, write submission CSV (test) or
spot-level ARI report (val).
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

from phase2.src import io, coords  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, _pick_device, LEVELS,
)


LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}


def _build_gt_spot_labels(fov, fov_spots, cells_df, labels_df):
    fov_cells = labels_df[labels_df["fov"] == fov]
    spot_x = fov_spots["global_x"].to_numpy()
    spot_y = fov_spots["global_y"].to_numpy()
    out = fov_spots.copy()
    for lvl in LEVELS:
        out[f"gt_{lvl}"] = "background"
    n_assigned = 0
    for cid in fov_cells["cell_id"]:
        if cid not in cells_df.index:
            continue
        row = cells_df.loc[cid]
        poly = coords.parse_boundary_polygon(row.get("boundaryX_z2", ""), row.get("boundaryY_z2", ""))
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
    p.add_argument("--bundle", required=True, help="scanvi_bundle.joblib path")
    p.add_argument("--mode", required=True, choices=("val", "test"))
    p.add_argument("--fovs", required=True)
    p.add_argument("--seg-checkpoint", default=None)
    p.add_argument("--masks-dir", default=None)
    p.add_argument("--include-spot-density", action="store_true")
    p.add_argument("--cellpose-diameter", type=float, default=0.0)
    p.add_argument("--cellprob-threshold", type=float, default=-0.5)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.bundle)
    genes = bundle["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    # Manifest dir: locally-trained bundles store it as 'manifest_dir',
    # Modal-trained bundles store the volume-relative path. Look next to the
    # bundle file first (where modal volume get drops the per_level subdir).
    bundle_parent = Path(args.bundle).parent
    candidate_local = bundle_parent / "scanvi_per_level"
    if candidate_local.is_dir():
        manifest_dir = candidate_local
    elif "manifest_dir" in bundle:
        manifest_dir = Path(bundle["manifest_dir"])
    else:
        raise FileNotFoundError(
            f"Couldn't find scanvi_per_level next to {args.bundle}; "
            f"download the manifest dir alongside the bundle."
        )
    print(f"  manifest_dir: {manifest_dir}")

    import scvi
    import anndata as ad
    print("loading scanvi models...")
    scanvi_models = {lvl: scvi.model.SCANVI.load(str(manifest_dir / lvl), adata=None)
                     for lvl in LEVELS_INTERNAL}
    print(f"  loaded {len(scanvi_models)} scanvi models")

    fov_list = [f.strip() for f in args.fovs.split(",") if f.strip()]
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    if masks_dir is None:
        import torch  # noqa
        from cellpose import models as cp_models
        device, gpu = _pick_device(args.device)
        if args.seg_checkpoint:
            seg_model = cp_models.CellposeModel(gpu=gpu, pretrained_model=args.seg_checkpoint, device=device)
        else:
            seg_model = cp_models.CellposeModel(gpu=gpu, device=device)
    else:
        seg_model = None

    # Spots
    if args.mode == "val":
        spots = pd.read_csv(io.ground_truth_dir() / "spots_train.csv")
        cells_df = pd.read_csv(io.ground_truth_dir() / "cell_boundaries_train.csv", index_col=0)
        labels_df = pd.read_csv(io.ground_truth_dir() / "cell_labels_train.csv")
    else:
        spots = pd.read_csv(io.data_root() / "test_spots.csv")
        cells_df = labels_df = None

    sample_submission_rows = []
    per_fov_ari = {}
    overall_ari = []

    for fov in fov_list:
        print(f"\n=== {fov} ===")
        fov_spots = spots[spots["fov"] == fov].copy().reset_index(drop=True)
        if len(fov_spots) == 0:
            continue

        if args.mode == "val":
            fov_spots_gt, n_in_cell_gt = _build_gt_spot_labels(fov, fov_spots, cells_df, labels_df)

        # Mask
        if masks_dir is not None:
            mask_path = masks_dir / f"{fov}.npy"
            if not mask_path.exists():
                print(f"  [skip] {mask_path} missing")
                continue
            masks = np.load(mask_path).astype(np.int32)
        else:
            split = "train" if args.mode == "val" else "test"
            from phase2.src.io import load_fov_images
            dapi, polyt = load_fov_images(fov, split=split)
            dapi2d = dapi.max(axis=0).astype(np.float32)
            polyt2d = polyt.max(axis=0).astype(np.float32)
            channels = [polyt2d, dapi2d]
            if args.include_spot_density:
                from scipy.ndimage import gaussian_filter
                d = np.zeros((2048, 2048), dtype=np.float32)
                r = fov_spots["image_row"].to_numpy().astype(int).clip(0, 2047)
                c = fov_spots["image_col"].to_numpy().astype(int).clip(0, 2047)
                np.add.at(d, (r, c), 1)
                d = gaussian_filter(d, sigma=8.0)
                if d.max() > 0:
                    d = d / d.max() * 65535.0
                channels.append(d.astype(np.float32))
            img = np.stack(channels, axis=0).astype(np.float32)
            masks, _, _ = seg_model.eval(img, channel_axis=0,
                                         diameter=args.cellpose_diameter,
                                         flow_threshold=args.flow_threshold,
                                         cellprob_threshold=args.cellprob_threshold)

        cell_ids, X, spot_label = _featurize_cells_from_mask(masks, fov_spots, gene_to_idx, nn_radius=0)
        if len(cell_ids) == 0:
            cell_predictions = {lvl: np.array([]) for lvl in LEVELS_INTERNAL}
        else:
            # scANVI requires anndata; counts must be int.
            X_int = np.maximum(np.round(X.astype(np.float32)).astype(np.int32), 0)
            qry = ad.AnnData(
                X=X_int,
                obs=pd.DataFrame({"fov": [fov] * len(cell_ids)}),
                var=pd.DataFrame(index=genes),
            )
            cell_predictions = {}
            for lvl in LEVELS_INTERNAL:
                # scANVI predict requires labels_key column with 'UNLABELED'
                qry_lvl = qry.copy()
                qry_lvl.obs[lvl] = "UNLABELED"
                preds = scanvi_models[lvl].predict(qry_lvl)
                cell_predictions[lvl] = np.asarray(preds).astype(str)

        # Project to spots
        label_to_idx = {int(c): i for i, c in enumerate(cell_ids)}
        spot_idx = np.array([label_to_idx.get(int(l), -1) for l in spot_label])
        in_cell_mask = spot_idx >= 0

        per_lvl = {}
        spot_preds = {}
        for lvl in LEVELS_INTERNAL:
            preds = cell_predictions[lvl]
            pred_spot = np.full(len(fov_spots), "background", dtype=object)
            if len(preds):
                pred_spot[in_cell_mask] = preds[spot_idx[in_cell_mask]]
            spot_preds[LEVELS_OUT[lvl]] = pred_spot
            if args.mode == "val":
                gt_spot = fov_spots_gt[f"gt_{LEVELS_OUT[lvl]}"].to_numpy().astype(str)
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(gt_spot, pred_spot.astype(str))
                per_lvl[LEVELS_OUT[lvl]] = float(ari)
                overall_ari.append(ari)

        if args.mode == "val":
            per_fov_ari[fov] = per_lvl
            print(f"  per-level: " + "  ".join(f"{k}={v:+.4f}" for k, v in per_lvl.items()))
        else:
            sub = pd.DataFrame({
                "spot_id": fov_spots["spot_id"].values if "spot_id" in fov_spots.columns else fov_spots.index.astype(str),
                "fov": fov,
                "class": spot_preds["class"].astype(str),
                "subclass": spot_preds["subclass"].astype(str),
                "supertype": spot_preds["supertype"].astype(str),
                "cluster": spot_preds["cluster"].astype(str),
            })
            sample_submission_rows.append(sub)
            print(f"  cells={int(masks.max())}  rows={len(sub)}")

    if args.mode == "val":
        mean_ari = float(np.mean(overall_ari))
        print(f"\nMEAN ARI = {mean_ari:.4f}")
        Path(out_dir / "val_metrics.json").write_text(json.dumps({
            "mean_ari": mean_ari, "per_fov": per_fov_ari}, indent=2))
    else:
        if sample_submission_rows:
            our = pd.concat(sample_submission_rows, ignore_index=True)
        else:
            our = pd.DataFrame(columns=["spot_id", "fov", "class", "subclass", "supertype", "cluster"])
        sample_path = io.data_root() / "sample_submission.csv"
        sample = pd.read_csv(sample_path, dtype={"spot_id": str})
        our["spot_id"] = our["spot_id"].astype(str)
        merged = sample.merge(our[["spot_id", "class", "subclass", "supertype", "cluster"]],
                              on="spot_id", how="left", suffixes=("", "_pred"))
        for col in ("class", "subclass", "supertype", "cluster"):
            pc = col + "_pred"
            if pc in merged.columns:
                merged[col] = merged[pc].fillna("background")
                merged.drop(columns=[pc], inplace=True)
            else:
                merged[col] = "background"
        merged = merged[sample.columns.tolist()]
        out_csv = out_dir / "submission.csv"
        merged.to_csv(out_csv, index=False)
        print(f"\n→ {out_csv} rows={len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
