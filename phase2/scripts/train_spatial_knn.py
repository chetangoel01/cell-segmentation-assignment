"""Spatial-context kNN: train a kNN where each cell's feature vector is
its gene counts CONCATENATED with the mean gene counts of its k nearest
spatial neighbors (within the same FOV).

This injects local-tissue context: cells of the same type often cluster
spatially, so a cell's neighbors' expression is informative about its own
type. No prior submission has used spatial info — this is the first.

Output bundle is incompatible with the standard infer_baseline path because
it expects 2x gene-count features (own ⊕ neighbor-mean). Use the matching
infer_spatial_knn.py for inference.

Train cells: phase-2 FOVs 101-150 (val on 151-160).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd

LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}


def add_spatial_context(X_norm: np.ndarray, coords: np.ndarray, fovs: np.ndarray,
                        k: int = 5, neighbor_weight: float = 0.5) -> np.ndarray:
    """For each cell, compute mean of L1-normalized gene vectors of k spatial
    nearest neighbors within the same FOV. Concatenate: [X_norm, neighbor_mean].

    Returns shape (N, 2*genes).
    """
    from sklearn.neighbors import NearestNeighbors
    n, d = X_norm.shape
    neighbor_mean = np.zeros_like(X_norm, dtype=np.float32)
    for fov in np.unique(fovs):
        idx_fov = np.where(fovs == fov)[0]
        n_fov = len(idx_fov)
        coords_fov = coords[idx_fov]
        X_fov = X_norm[idx_fov]
        if n_fov < 2:
            neighbor_mean[idx_fov] = X_fov  # alone in FOV - use self
            continue
        k_eff = min(k, n_fov - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords_fov)
        _, idx = nn.kneighbors(coords_fov)
        # idx[:, 0] is the cell itself (zero distance) - skip
        nbrs = idx[:, 1:]
        neighbor_mean[idx_fov] = X_fov[nbrs].mean(axis=1)
    return np.hstack([X_norm, neighbor_weight * neighbor_mean]).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--train-fovs", default=None,
                   help="Comma-separated train FOVs. Default: 101-150.")
    p.add_argument("--val-fovs", default=None,
                   help="Comma-separated val FOVs. Default: 151-160.")
    p.add_argument("--knn-k", type=int, default=5, help="kNN classifier k")
    p.add_argument("--spatial-k", type=int, default=5,
                   help="Number of spatial neighbors used for context features")
    p.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    p.add_argument("--neighbor-weight", type=float, default=0.5,
                   help="In concat features, scale the neighbor-mean half by this factor before concatenation. 0 = no spatial info; 1 = equal weight; <1 = down-weight spatial.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_fovs = (args.train_fovs.split(",") if args.train_fovs
                  else [f"FOV_{i:03d}" for i in range(101, 151)])
    val_fovs = (args.val_fovs.split(",") if args.val_fovs
                else [f"FOV_{i:03d}" for i in range(151, 161)])

    print(f"Loading {args.h5ad} ...")
    a = ad.read_h5ad(args.h5ad)
    X_raw = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    print(f"  shape: {a.shape}")

    train_mask = a.obs.fov.isin(train_fovs).values
    val_mask = a.obs.fov.isin(val_fovs).values
    print(f"  train cells: {train_mask.sum()}  val cells: {val_mask.sum()}")

    # L1-normalize each cell (matches codelab kNN preproc)
    from sklearn.preprocessing import normalize
    X_norm = normalize(X_raw.astype(np.float32), norm="l1")

    coords = a.obs[["center_x", "center_y"]].to_numpy(dtype=np.float32)
    fovs = a.obs["fov"].to_numpy()

    # Compute spatial-context features (uses ALL cells, including val, for
    # neighbor lookup — this is fair because at inference we use cellpose-
    # detected cells in the test FOV; we don't peek at val labels)
    print(f"Computing spatial-context features (k={args.spatial_k}) ...")
    t0 = time.time()
    X_aug = add_spatial_context(X_norm, coords, fovs, k=args.spatial_k,
                                 neighbor_weight=args.neighbor_weight)
    print(f"  augmented shape: {X_aug.shape} ({time.time() - t0:.1f}s)")

    Xtr, Xva = X_aug[train_mask], X_aug[val_mask]
    train_genes = list(a.var_names)

    # Train + eval per level
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, adjusted_rand_score

    metrics = {"k": args.knn_k, "spatial_k": args.spatial_k, "metric": args.metric,
               "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
               "n_genes": len(train_genes), "feature_dim": int(X_aug.shape[1]),
               "per_level": {}}

    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        ytr = a.obs[lvl].values[train_mask]
        yva = a.obs[lvl].values[val_mask]
        if len(set(ytr)) < 2:
            print(f"  [skip] {lvl} only has one class")
            continue
        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=args.knn_k, metric=args.metric)
        clf.fit(Xtr, ytr)
        ypr = clf.predict(Xva)
        elapsed = time.time() - t0
        acc = float(accuracy_score(yva, ypr))
        ari = float(adjusted_rand_score(yva, ypr))
        print(f"  {out_name:<10} acc={acc:.3f}  cell-ARI={ari:.3f}  ({len(set(ytr))} classes, {elapsed:.1f}s)")
        metrics["per_level"][out_name] = {
            "accuracy": acc, "ari_cells": ari,
            "n_train_cells": int(len(ytr)), "n_val_cells": int(len(yva)),
            "n_classes": int(len(set(ytr))),
        }
        # Custom bundle includes spatial_k so inference knows how to compute features
        joblib.dump({"clf": clf, "genes": train_genes,
                     "preproc": "spatial_l1", "k": args.knn_k, "metric": args.metric,
                     "spatial_k": args.spatial_k, "feature_dim": int(X_aug.shape[1])},
                    out_dir / f"model_{out_name}.joblib")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
