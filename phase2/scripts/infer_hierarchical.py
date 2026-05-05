"""Hierarchical AWS-augmented kNN inference.

Single combined bundle (from train_hierarchical_aws.py), runs cellpose seg
or loads precomputed masks, featurizes cells, applies taxonomy-tree-masked
hierarchical voting, writes either submission CSV (--test-fovs) or per-FOV
spot-ARI report (--val-fovs).

Background gate: optionally apply a binary 'is_real_cell' classifier loaded
from --gate-bundle. If the gate predicts 'background', force all 4 levels to
'background' for that cell, bypassing the hierarchical kNN.
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
    _featurize_cells_from_mask, _segment_fov, _pick_device, LEVELS,
)
# Make GateClassifier importable so joblib can unpickle gate bundles saved
# by train_bg_gate.py (where it was the __main__ module). The class lived at
# `__main__.GateClassifier` at pickle time; we splice it in here so unpickle
# can find it.
sys.path.insert(0, str(Path(__file__).parent))
from train_bg_gate import GateClassifier  # noqa: E402, F401
import __main__
__main__.GateClassifier = GateClassifier


LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}


def _preprocess(X: np.ndarray, mode: str) -> np.ndarray:
    X = X.astype(np.float32)
    if mode == "log1p":
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return np.log1p(X / row_sum * 1e4)
    if mode == "l1_normalize":
        from sklearn.preprocessing import normalize
        return normalize(X, norm="l1")
    raise ValueError(mode)


def _weighted_vote(labels: np.ndarray, w: np.ndarray) -> str:
    score: dict[str, float] = {}
    for lab, wi in zip(labels, w):
        score[lab] = score.get(lab, 0.0) + float(wi)
    return max(score.items(), key=lambda kv: kv[1])[0]


def _build_label_score_dict(idx_i, w_i, label_arr) -> dict[str, float]:
    score: dict[str, float] = {}
    for lab, wi in zip(label_arr[idx_i], w_i):
        score[lab] = score.get(lab, 0.0) + float(wi)
    return score


def _normalize_dict(d: dict[str, float]) -> dict[str, float]:
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}


def _argmax_dict(d: dict[str, float]) -> str:
    return max(d.items(), key=lambda kv: kv[1])[0] if d else "background"


def _hierarchical_predict(X_query: np.ndarray, bundle: dict, k_predict: int,
                          return_dists: bool = False
                          ) -> dict[str, np.ndarray]:
    """For each query row, get k_predict nearest train cells from bundle's index,
    then run taxonomy-tree-masked voting at all 4 levels.

    If return_dists, also returns per-cell label-score dictionaries at each
    level (used for spatial neighbor smoothing).
    """
    nn = bundle["sklearn_knn"]
    distances, indices = nn.kneighbors(X_query, n_neighbors=k_predict)
    eps = 1e-9
    weights = 1.0 / (distances + eps)
    yt = bundle["y_train"]
    cls_arr = yt["class_label"].astype(str)
    sub_arr = yt["subclass_label"].astype(str)
    sup_arr = yt["supertype_label"].astype(str)
    clu_arr = yt["cluster_label"].astype(str)

    out = {lvl: np.empty(len(X_query), dtype=object) for lvl in LEVELS_INTERNAL}
    dists: list[dict[str, dict[str, float]]] = []  # per-cell {level: {label: score}}
    for i in range(len(X_query)):
        idx_i = indices[i]
        w_i = weights[i]
        cell_dists: dict[str, dict[str, float]] = {}

        # Class
        cls_score = _normalize_dict(_build_label_score_dict(idx_i, w_i, cls_arr))
        cls = _argmax_dict(cls_score)
        cell_dists["class_label"] = cls_score
        out["class_label"][i] = cls

        # Subclass — masked to predicted class
        nbr_cls = cls_arr[idx_i]
        m = (nbr_cls == cls)
        sub_idx = idx_i[m] if m.any() else idx_i
        sub_w = w_i[m] if m.any() else w_i
        sub_score = _normalize_dict(_build_label_score_dict(sub_idx, sub_w, sub_arr))
        sub = _argmax_dict(sub_score)
        cell_dists["subclass_label"] = sub_score
        out["subclass_label"][i] = sub

        # Supertype — masked to predicted subclass
        nbr_sub = sub_arr[idx_i]
        m = (nbr_sub == sub)
        sup_idx = idx_i[m] if m.any() else idx_i
        sup_w = w_i[m] if m.any() else w_i
        sup_score = _normalize_dict(_build_label_score_dict(sup_idx, sup_w, sup_arr))
        sup = _argmax_dict(sup_score)
        cell_dists["supertype_label"] = sup_score
        out["supertype_label"][i] = sup

        # Cluster — masked to predicted supertype
        nbr_sup = sup_arr[idx_i]
        m = (nbr_sup == sup)
        clu_idx = idx_i[m] if m.any() else idx_i
        clu_w = w_i[m] if m.any() else w_i
        clu_score = _normalize_dict(_build_label_score_dict(clu_idx, clu_w, clu_arr))
        cell_dists["cluster_label"] = clu_score
        out["cluster_label"][i] = _argmax_dict(clu_score)
        dists.append(cell_dists)

    if return_dists:
        return out, dists
    return out


def _spatial_smooth(predictions: dict[str, np.ndarray],
                    dists: list[dict[str, dict[str, float]]],
                    cell_centers_xy: np.ndarray,
                    k_neighbors: int = 8,
                    alpha: float = 0.4,
                    iterations: int = 1
                    ) -> dict[str, np.ndarray]:
    """Iteratively smooth per-cell label-score distributions across spatial
    neighbors (kNN graph over cell centroids), then re-argmax.

    For each level, for each cell:
        score_new[i] = (1 - alpha) * score_own[i] + alpha * mean(score_own[j] for j in nbrs)
    """
    if len(cell_centers_xy) <= 1:
        return predictions
    from sklearn.neighbors import NearestNeighbors
    n = len(cell_centers_xy)
    k_eff = min(k_neighbors + 1, n)
    nn_g = NearestNeighbors(n_neighbors=k_eff).fit(cell_centers_xy)
    _, nbr_idx = nn_g.kneighbors(cell_centers_xy)  # includes self at index 0

    # Initialize per-level dist matrices via the input dists
    out = dict(predictions)
    for lvl in LEVELS_INTERNAL:
        # Collect the union of label strings across all cells at this level
        all_labels = set()
        for d in dists:
            all_labels.update(d[lvl].keys())
        all_labels = sorted(all_labels)
        if not all_labels:
            continue
        lab_to_col = {l: i for i, l in enumerate(all_labels)}
        # Build (n_cells, n_labels) matrix
        scores = np.zeros((n, len(all_labels)), dtype=np.float64)
        for i, d in enumerate(dists):
            for lab, v in d[lvl].items():
                scores[i, lab_to_col[lab]] = v

        for _ in range(iterations):
            # Mean over neighbors (excluding self at idx 0 to keep own/neighbor split clean)
            nbr_means = scores[nbr_idx[:, 1:]].mean(axis=1)
            scores = (1.0 - alpha) * scores + alpha * nbr_means

        # Re-argmax
        argm = scores.argmax(axis=1)
        out[lvl] = np.array([all_labels[c] for c in argm], dtype=object)
    return out


def _build_query_features(
    X_cells: np.ndarray, bundle: dict, fov_key_for_cells: str | None
) -> np.ndarray:
    """Apply preproc and (optionally) append FOV-mean features.

    For test/val cells, the FOV-mean comes from the cells in the same FOV
    (computed on-the-fly), matching the train-time behavior of using each
    cell's FOV mean.
    """
    Xq = _preprocess(X_cells, bundle["preproc"])
    if bundle.get("fov_means_appended"):
        # Compute per-FOV mean from the FOV's segmented cells.
        if fov_key_for_cells is None:
            # No FOV identity — fall back to overall train mean (poor, but legal).
            mean_vec = bundle["X_train"][:, : Xq.shape[1]].mean(axis=0)
        else:
            mean_vec = Xq.mean(axis=0)
        block = np.tile(mean_vec, (len(Xq), 1)).astype(np.float32)
        Xq = np.concatenate([Xq, block], axis=1).astype(np.float32)
    return Xq


def _build_gt_spot_labels(fov: str, fov_spots: pd.DataFrame,
                          cells_df: pd.DataFrame, labels_df: pd.DataFrame,
                          ) -> tuple[pd.DataFrame, int]:
    """Build per-spot GT taxonomy labels by point-in-polygon test against
    z=2 polygons (matching phase2/docs/codelab_13.py rasterize_gt_mask)."""
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
    p.add_argument("--bundle", required=True, help="Path to bundle.joblib from train_hierarchical_aws.py")
    p.add_argument("--mode", required=True, choices=("val", "test"))
    p.add_argument("--fovs", required=True, help="Comma-separated FOVs to process.")
    p.add_argument("--k-predict", type=int, default=50,
                   help="Override the K used at inference (must be <= K used at train).")
    p.add_argument("--seg-checkpoint", default=None)
    p.add_argument("--masks-dir", default=None)
    p.add_argument("--save-masks-dir", default=None)
    p.add_argument("--include-spot-density", action="store_true")
    p.add_argument("--spot-density-sigma", type=float, default=8.0)
    p.add_argument("--cellpose-diameter", type=float, default=0.0)
    p.add_argument("--cellprob-threshold", type=float, default=-0.5)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--nn-radius", type=float, default=0.0)
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--gate-bundle", default=None,
                   help="Optional joblib of a binary 'is real cell vs background' classifier "
                        "trained on competition cells only. If set, the gate predicts first; "
                        "cells flagged as background bypass the hierarchical kNN.")
    p.add_argument("--smooth-k", type=int, default=0,
                   help="If >0, apply spatial neighbor smoothing on per-cell label "
                        "distributions before argmax: build kNN graph over cell "
                        "centroids (k=this), smooth label scores, re-argmax. 0 disables.")
    p.add_argument("--smooth-alpha", type=float, default=0.4,
                   help="Smoothing weight: score_new = (1-α)*own + α*neighbor_mean.")
    p.add_argument("--smooth-iters", type=int, default=1,
                   help="Number of smoothing iterations.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading bundle: {args.bundle}")
    bundle = joblib.load(args.bundle)
    genes = bundle["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"  preproc={bundle['preproc']}  k_train={bundle['k']}  "
          f"fov_means={bundle.get('fov_means_appended', False)}  "
          f"X_train={bundle['X_train'].shape}")

    gate = None
    if args.gate_bundle:
        gate = joblib.load(args.gate_bundle)
        print(f"  gate loaded: keys={list(gate.keys())}")

    fov_list = [f.strip() for f in args.fovs.split(",") if f.strip()]
    print(f"mode={args.mode}  fovs={fov_list}")

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    save_masks_dir = Path(args.save_masks_dir) if args.save_masks_dir else None
    if save_masks_dir:
        save_masks_dir.mkdir(parents=True, exist_ok=True)

    if masks_dir is None:
        import torch  # noqa
        from cellpose import models as cp_models
        device, gpu = _pick_device(args.device)
        print(f"  device={device} gpu={gpu}")
        if args.seg_checkpoint:
            seg_model = cp_models.CellposeModel(gpu=gpu, pretrained_model=args.seg_checkpoint, device=device)
        else:
            seg_model = cp_models.CellposeModel(gpu=gpu, device=device)
    else:
        seg_model = None
        print(f"  using precomputed masks from {masks_dir}")

    # Load spots
    if args.mode == "val":
        spots = pd.read_csv(io.ground_truth_dir() / "spots_train.csv")
        cells_df = pd.read_csv(io.ground_truth_dir() / "cell_boundaries_train.csv", index_col=0)
        labels_df = pd.read_csv(io.ground_truth_dir() / "cell_labels_train.csv")
    else:
        spots = pd.read_csv(io.data_root() / "test_spots.csv")
        cells_df = labels_df = None

    sample_submission_rows: list[pd.DataFrame] = []
    per_fov_ari: dict[str, dict[str, float]] = {}
    per_fov_meta: dict[str, dict] = {}
    overall_ari: list[float] = []

    for fov in fov_list:
        print(f"\n=== {fov} ===")
        fov_spots = spots[spots["fov"] == fov].copy().reset_index(drop=True)
        if len(fov_spots) == 0:
            print(f"  [skip] no spots")
            continue
        print(f"  spots: {len(fov_spots):,}")

        # Optional GT for val mode
        if args.mode == "val":
            fov_spots_gt, n_in_cell_gt = _build_gt_spot_labels(fov, fov_spots, cells_df, labels_df)
            print(f"  GT in-cell: {n_in_cell_gt}/{len(fov_spots)} ({n_in_cell_gt/len(fov_spots):.1%})")

        # Segment or load mask
        t0 = time.time()
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
                d = gaussian_filter(d, sigma=args.spot_density_sigma)
                if d.max() > 0:
                    d = d / d.max() * 65535.0
                channels.append(d.astype(np.float32))
            img = np.stack(channels, axis=0).astype(np.float32)
            masks, _, _ = seg_model.eval(
                img, channel_axis=0, diameter=args.cellpose_diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
            )
        seg_t = time.time() - t0
        if save_masks_dir:
            np.save(save_masks_dir / f"{fov}.npy", masks.astype(np.int32))

        # Featurize
        cell_ids, X, spot_label = _featurize_cells_from_mask(
            masks, fov_spots, gene_to_idx, nn_radius=args.nn_radius
        )
        if len(cell_ids) == 0:
            cell_predictions = {lvl: np.array([]) for lvl in LEVELS_INTERNAL}
        else:
            # Build query features (preproc + optional fov-mean)
            Xq = _build_query_features(X, bundle, fov_key_for_cells=fov)

            # Optional gate: predict 'is_real_cell' first.
            if gate is not None:
                gate_X = _preprocess(X, gate.get("preproc", bundle["preproc"]))
                is_real = gate["clf"].predict(gate_X)
                # ensure boolean
                is_real_bool = (np.asarray(is_real).astype(str) != "background")
            else:
                is_real_bool = np.ones(len(cell_ids), dtype=bool)

            # Hierarchical kNN on real cells; bg cells short-circuit to 'background'.
            cell_predictions = {lvl: np.full(len(cell_ids), "background", dtype=object) for lvl in LEVELS_INTERNAL}
            real_idx = np.where(is_real_bool)[0]
            if len(real_idx):
                Xq_real = Xq[real_idx]
                if args.smooth_k > 0:
                    preds, dists = _hierarchical_predict(
                        Xq_real, bundle, k_predict=args.k_predict, return_dists=True
                    )
                    # Cell centroids in image space (rows, cols) from regionprops on mask
                    from skimage.measure import regionprops
                    rps = regionprops(masks)
                    label_to_centroid = {p.label: (p.centroid[0], p.centroid[1]) for p in rps}
                    real_cell_ids = cell_ids[real_idx]
                    centers = np.array([label_to_centroid[int(c)] for c in real_cell_ids],
                                       dtype=np.float32)
                    preds = _spatial_smooth(preds, dists, centers,
                                            k_neighbors=args.smooth_k,
                                            alpha=args.smooth_alpha,
                                            iterations=args.smooth_iters)
                else:
                    preds = _hierarchical_predict(Xq_real, bundle, k_predict=args.k_predict)
                for lvl in LEVELS_INTERNAL:
                    cell_predictions[lvl][real_idx] = preds[lvl]

        # Map predictions back to spots
        label_to_idx = {int(c): i for i, c in enumerate(cell_ids)}
        spot_idx = np.array([label_to_idx.get(int(l), -1) for l in spot_label])
        in_cell_mask = spot_idx >= 0

        per_lvl: dict[str, float] = {}
        spot_preds: dict[str, np.ndarray] = {}
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
            per_fov_meta[fov] = {
                "n_spots": int(len(fov_spots)),
                "n_cells_segmented": int(masks.max()),
                "frac_in_cell_pred": float(in_cell_mask.mean()),
                "frac_in_cell_gt": float(n_in_cell_gt / len(fov_spots)),
                "seg_s": seg_t,
            }
            print(f"  per-level: " +
                  "  ".join(f"{lvl}={ari:+.4f}" for lvl, ari in per_lvl.items()))
            print(f"  pred frac_in_cell={in_cell_mask.mean():.3f} "
                  f"gt frac_in_cell={n_in_cell_gt/len(fov_spots):.3f}")
        else:
            # test mode — accumulate submission rows
            sub = pd.DataFrame({
                "spot_id": fov_spots["spot_id"].values if "spot_id" in fov_spots.columns else fov_spots.index.astype(str),
                "fov": fov,
                "class": spot_preds["class"].astype(str),
                "subclass": spot_preds["subclass"].astype(str),
                "supertype": spot_preds["supertype"].astype(str),
                "cluster": spot_preds["cluster"].astype(str),
            })
            sample_submission_rows.append(sub)
            print(f"  cells={int(masks.max())} predictions written ({len(sub)} rows)")

    if args.mode == "val":
        mean_ari = float(np.mean(overall_ari))
        print()
        print("=" * 60)
        print(f"MEAN ARI across {len(per_fov_ari)} FOVs × 4 levels = {mean_ari:.4f}")
        print("=" * 60)
        for fov, lvls in per_fov_ari.items():
            print(f"  {fov} mean={np.mean(list(lvls.values())):+.4f}")
        Path(out_dir / "val_metrics.json").write_text(json.dumps({
            "mean_ari": mean_ari,
            "per_fov": {f: {"per_level": per_fov_ari[f], **per_fov_meta[f]} for f in per_fov_ari},
            "config": {k: v for k, v in vars(args).items()},
        }, indent=2))
    else:
        # write submission CSV ordered by sample_submission template
        if sample_submission_rows:
            our = pd.concat(sample_submission_rows, ignore_index=True)
        else:
            our = pd.DataFrame(columns=["spot_id", "fov", "class", "subclass", "supertype", "cluster"])
        # Read sample_submission to enforce row order/coverage
        sample_path = io.data_root() / "sample_submission.csv"
        sample = pd.read_csv(sample_path, dtype={"spot_id": str})
        our["spot_id"] = our["spot_id"].astype(str)
        merged = sample.merge(our[["spot_id", "class", "subclass", "supertype", "cluster"]],
                              on="spot_id", how="left", suffixes=("", "_pred"))
        for col in ("class", "subclass", "supertype", "cluster"):
            pred_col = col + "_pred"
            if pred_col in merged.columns:
                merged[col] = merged[pred_col].fillna("background")
                merged.drop(columns=[pred_col], inplace=True)
            else:
                merged[col] = "background"
        # Re-order columns to match sample_submission
        merged = merged[sample.columns.tolist()]
        out_csv = out_dir / "submission.csv"
        merged.to_csv(out_csv, index=False)
        print(f"\n→ {out_csv}  rows={len(merged):,}")
        # Quick sanity
        print(f"  full-bg rows: {(merged['class']=='background').sum():,} "
              f"({(merged['class']=='background').mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
