"""Train RF500 classifier on phase-2 train + AWS .003 augmentation.

For each of the 4 hierarchy levels (class, subclass, supertype, cluster), filter
AWS cells to those whose label is in phase-2's vocab for that level. This gives
us a maximum-size training set per level without introducing AWS-only labels
that could never appear in the phase-2 test set.

Outputs joblib bundles compatible with phase2/scripts/validate_local.py.
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--include-sections", default="003",
                   help="Comma-sep AWS sections to add (003 same panel; 002 panel-shifted; 001 leakage risk)")
    p.add_argument("--label-strategy", default="filter",
                   choices=("filter", "remap_to_bg"),
                   help="filter=drop AWS cells with out-of-vocab labels; remap_to_bg=label them 'background'")
    args = p.parse_args()

    print("=== Loading sources ===")
    p2 = ad.read_h5ad("phase2/data/train/ground_truth/counts_train.h5ad")
    print(f"  phase-2 h5ad: {p2.shape}")

    aws_h = ad.read_h5ad("phase2/data/external/aws/Zhuang-ABCA-4-log2.h5ad")
    aws_labels = pd.read_csv("phase2/data/external/aws/cell_metadata_with_cluster_annotation.csv",
                             dtype={"cell_label": str}).set_index("cell_label")
    print(f"  AWS h5ad: {aws_h.shape}")
    print(f"  AWS labels: {aws_labels.shape}")

    # AWS h5ad obs index = cell_label (string). Verify.
    aws_h.obs.index = aws_h.obs.index.astype(str)

    # Restrict AWS to requested sections
    sections = [f"Zhuang-ABCA-4.{s.strip()}" for s in args.include_sections.split(",") if s.strip()]
    print(f"  AWS sections requested: {sections}")
    aws_keep_ids = set(aws_labels[aws_labels.brain_section_label.isin(sections)].index)
    aws_obs_mask = aws_h.obs.index.isin(aws_keep_ids)
    print(f"  AWS cells in those sections: {aws_obs_mask.sum():,}")

    # Gene alignment: phase-2 var.index vs AWS var.gene_symbol
    aws_h.var.index = aws_h.var.gene_symbol.astype(str)
    p2_genes = list(p2.var.index)
    aws_genes = set(aws_h.var.index)
    shared_genes = [g for g in p2_genes if g in aws_genes]
    print(f"  Shared genes: {len(shared_genes)} (of phase-2's {len(p2_genes)})")

    # Build aligned gene matrices
    p2_X = p2.X.toarray() if hasattr(p2.X, "toarray") else np.asarray(p2.X)
    p2_gene_idx = [p2_genes.index(g) for g in shared_genes]
    p2_X = p2_X[:, p2_gene_idx].astype(np.float32)
    print(f"  phase-2 X: {p2_X.shape}")

    aws_X_full = aws_h.X.toarray() if hasattr(aws_h.X, "toarray") else np.asarray(aws_h.X)
    aws_X_full = aws_X_full[aws_obs_mask].astype(np.float32)
    aws_obs_subset = aws_h.obs.index[aws_obs_mask].values
    aws_gene_idx = [list(aws_h.var.index).index(g) for g in shared_genes]
    aws_X = aws_X_full[:, aws_gene_idx].astype(np.float32)
    print(f"  AWS X: {aws_X.shape}")

    # AWS labels for kept cells
    aws_keep_labels = aws_labels.loc[aws_obs_subset]
    print(f"  AWS keep labels rows: {aws_keep_labels.shape[0]:,}")

    # Train/val split for phase-2: same as codelab
    p2_train_mask = p2.obs.fov.isin([f"FOV_{i:03d}" for i in range(101, 151)]).values
    p2_val_mask = p2.obs.fov.isin([f"FOV_{i:03d}" for i in range(151, 161)]).values
    print(f"  phase-2 train cells: {p2_train_mask.sum()}, val cells: {p2_val_mask.sum()}")

    # log1p normalize gene counts
    def normalize(X):
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return np.log1p(X / rs * 1e4)

    p2_X_norm = normalize(p2_X).astype(np.float32)
    # AWS h5ad already in log2 space - need to undo? No, we treat each row's normalization
    # independently. log2 already so just renormalize.
    # Actually AWS X is log2(CPM+1). Phase-2 X is RAW counts. They're in different spaces!
    # Need to inverse-log AWS to get raw, then normalize together.
    # log2(x+1) -> x = 2^val - 1
    aws_X_raw = np.maximum(np.power(2.0, aws_X) - 1.0, 0.0).astype(np.float32)
    aws_X_norm = normalize(aws_X_raw).astype(np.float32)
    print(f"  Normalized: phase-2 {p2_X_norm.shape}, AWS {aws_X_norm.shape}")

    # Per-level training: build (X, y) for each level using filtered AWS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LEVELS = ("class", "subclass", "supertype", "cluster")
    INTERNAL = {"class": "class_label", "subclass": "subclass_label",
                "supertype": "supertype_label", "cluster": "cluster_label"}
    metrics = {"per_level": {}, "preproc": "log1p", "include_sections": sections,
               "label_strategy": args.label_strategy, "n_p2_train": int(p2_train_mask.sum()),
               "n_p2_val": int(p2_val_mask.sum()), "n_aws_pool": int(aws_X.shape[0]),
               "n_genes": len(shared_genes)}

    try:
        max_feat = float(args.max_features)
    except ValueError:
        max_feat = args.max_features

    for lvl in LEVELS:
        col = INTERNAL[lvl]
        p2_y_train = p2.obs[col].values[p2_train_mask]
        p2_y_val = p2.obs[col].values[p2_val_mask]
        p2_vocab = set(p2.obs[col].unique())

        # AWS labels at this level
        aws_y = aws_keep_labels[lvl].astype(str).values
        if args.label_strategy == "filter":
            keep_aws = np.isin(aws_y, list(p2_vocab))
        else:  # remap_to_bg
            keep_aws = np.ones(len(aws_y), dtype=bool)
            aws_y = np.where(np.isin(aws_y, list(p2_vocab)), aws_y, "background")

        aws_X_used = aws_X_norm[keep_aws]
        aws_y_used = aws_y[keep_aws]

        X_train = np.concatenate([p2_X_norm[p2_train_mask], aws_X_used], axis=0)
        y_train = np.concatenate([p2_y_train, aws_y_used])

        X_val = p2_X_norm[p2_val_mask]
        y_val = p2_y_val

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_features=max_feat,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        yh = clf.predict(X_val)
        acc = (yh == y_val).mean()
        ari = adjusted_rand_score(y_val, yh)

        n_classes_train = len(np.unique(y_train))
        n_aws_used = aws_X_used.shape[0]
        metrics["per_level"][lvl] = {
            "accuracy": float(acc), "ari_cells": float(ari),
            "n_train": int(X_train.shape[0]), "n_aws_used": int(n_aws_used),
            "n_classes_train": int(n_classes_train), "n_classes_p2_vocab": int(len(p2_vocab)),
        }
        print(f"  [{lvl}] X_train={X_train.shape} (p2={p2_train_mask.sum()} + aws={n_aws_used})  "
              f"vocab={n_classes_train}  acc={acc:.3f}  cell-ARI={ari:.3f}")

        joblib.dump({"clf": clf, "genes": shared_genes, "preproc": "log1p"},
                    out_dir / f"model_{lvl}.joblib")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
