"""Train RF500 classifier with volume + neighborhood-density features added.

Features: [gene_counts_log1p_normalized (1147), volume_norm (1), n_spots_in_cell_norm (1)]

At inference time we substitute predicted-mask area for volume. The trained
classifier gets 1149 features instead of 1147.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-features", default="0.1")
    args = p.parse_args()

    a = ad.read_h5ad("phase2/data/train/ground_truth/counts_train.h5ad")
    X_counts = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    obs = a.obs

    train_fovs = [f"FOV_{i:03d}" for i in range(101, 151)]
    val_fovs = [f"FOV_{i:03d}" for i in range(151, 161)]

    train_mask = obs.fov.isin(train_fovs).values
    val_mask = obs.fov.isin(val_fovs).values
    print(f"train cells: {train_mask.sum()}  val cells: {val_mask.sum()}")

    # gene-count normalize: log1p(X / row_sum * 1e4)
    def normalize(X):
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e4)

    X_norm = normalize(X_counts).astype(np.float32)

    # Volume — use median of train as 1.0
    train_vol_med = obs.loc[train_mask, "volume"].median()
    print(f"train volume median: {train_vol_med:.1f}")
    vol_feature = (obs["volume"].values / train_vol_med).astype(np.float32).reshape(-1, 1)

    # Combined feature matrix: [gene_norm, vol_norm]
    X_full = np.concatenate([X_norm, vol_feature], axis=1)
    print(f"feature shape: {X_full.shape}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = list(a.var.index)

    LEVELS = ("class", "subclass", "supertype", "cluster")
    INTERNAL = {lvl: f"{lvl}_label" for lvl in LEVELS}
    metrics = {"per_level": {}, "preproc": "log1p+volume", "n_train": int(train_mask.sum()),
               "n_val": int(val_mask.sum()), "n_genes": len(genes), "n_features": int(X_full.shape[1])}

    try:
        max_feat = float(args.max_features)
    except ValueError:
        max_feat = args.max_features

    for lvl in LEVELS:
        col = INTERNAL[lvl]
        y = obs[col].values
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_features=max_feat,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_full[train_mask], y[train_mask])
        yh = clf.predict(X_full[val_mask])
        acc = (yh == y[val_mask]).mean()
        ari = adjusted_rand_score(y[val_mask], yh)
        metrics["per_level"][lvl] = {"accuracy": float(acc), "ari_cells": float(ari), "n_classes": int(np.unique(y).size)}
        print(f"  {lvl}: acc={acc:.3f} ari={ari:.3f} n_classes={np.unique(y).size}")
        joblib.dump({"clf": clf, "genes": genes, "preproc": "log1p+volume",
                     "n_extra_features": 1, "extra_feature_names": ["volume_norm"],
                     "train_volume_median": float(train_vol_med)},
                    out_dir / f"model_{lvl}.joblib")

    import json
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
