"""Spatial-context kNN trained on phase-2 cells + AWS Allen Brain Cell Atlas
section Zhuang-ABCA-4.001 (the competition source).

Adds 32K labeled cells (no background noise) to our 5K training pool,
6x expansion. Maps AWS Ensembl gene IDs to symbols via gene.csv to align
with our 1147-gene panel (~1111 shared).

Output bundle is compatible with infer_spatial_knn.py.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_spatial_knn import add_spatial_context  # reuse the function

LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}

# AWS metadata column → our internal label name (AWS uses unprefixed names)
AWS_LABEL_MAP = {
    "class": "class_label",
    "subclass": "subclass_label",
    "supertype": "supertype_label",
    "cluster": "cluster_label",
}


def _load_aws(aws_h5ad: Path, aws_meta_csv: Path, aws_gene_csv: Path,
              sections: list[str]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return (X, obs, gene_symbols) for cells in the requested AWS sections.
    X is (N, 1122) log2-normalized expression; gene_symbols is var-aligned."""
    aws = ad.read_h5ad(aws_h5ad)
    meta = pd.read_csv(aws_meta_csv)
    gene = pd.read_csv(aws_gene_csv)

    # Map ENSMUSG → symbol via gene.csv (more robust than aws.var which may have NaN)
    ens_to_sym = dict(zip(gene.gene_identifier, gene.gene_symbol))
    gene_symbols = [ens_to_sym.get(e, e) for e in aws.var_names]

    # Filter metadata to requested sections + only labeled cells
    meta_sub = meta[meta.brain_section_label.isin(sections)].copy()
    meta_sub = meta_sub[meta_sub["class"].notna()]
    print(f"  AWS sections {sections}: {len(meta_sub):,} labeled cells")

    # Join: meta cell_label → aws.obs.index
    meta_sub["cell_label_str"] = meta_sub.cell_label.astype(str)
    h5ad_id_to_idx = {cid: i for i, cid in enumerate(aws.obs.index.astype(str))}
    meta_sub = meta_sub[meta_sub.cell_label_str.isin(h5ad_id_to_idx)]
    aws_idx = meta_sub.cell_label_str.map(h5ad_id_to_idx).to_numpy()

    X = aws.X[aws_idx]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # AWS h5ad is log2(value+1) - convert to "raw counts" via 2^x - 1
    # so we can L1-normalize consistently with our training pipeline
    X_pseudoraw = np.power(2.0, X, dtype=np.float32) - 1.0
    print(f"  AWS X shape: {X_pseudoraw.shape}, max raw: {X_pseudoraw.max():.1f}")

    return X_pseudoraw, meta_sub, gene_symbols


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--our-h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--aws-h5ad", default="phase2/data/external/aws/Zhuang-ABCA-4-log2.h5ad")
    p.add_argument("--aws-meta", default="phase2/data/external/aws/cell_metadata_with_cluster_annotation.csv")
    p.add_argument("--aws-gene", default="phase2/data/external/aws/gene.csv")
    p.add_argument("--aws-sections", default="Zhuang-ABCA-4.001",
                   help="Comma-separated AWS section IDs to include.")
    p.add_argument("--our-train-fovs", default=None,
                   help="Comma-separated train FOVs (default 101-150)")
    p.add_argument("--our-val-fovs", default=None,
                   help="Comma-separated val FOVs (default 151-160)")
    p.add_argument("--drop-background-ours", action="store_true",
                   help="Filter our training cells where class_label=='background'.")
    p.add_argument("--restrict-aws-to-our-classes", action="store_true",
                   help="Only keep AWS cells whose class label exists in our training set.")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--spatial-k", type=int, default=5)
    p.add_argument("--neighbor-weight", type=float, default=0.5)
    p.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = args.aws_sections.split(",")

    # ---- Load OUR data ----
    print(f"Loading {args.our_h5ad} ...")
    a_ours = ad.read_h5ad(args.our_h5ad)
    X_ours = a_ours.X.toarray() if hasattr(a_ours.X, "toarray") else np.asarray(a_ours.X)
    X_ours = X_ours.astype(np.float32)
    train_fovs = (args.our_train_fovs.split(",") if args.our_train_fovs
                  else [f"FOV_{i:03d}" for i in range(101, 151)])
    val_fovs = (args.our_val_fovs.split(",") if args.our_val_fovs
                else [f"FOV_{i:03d}" for i in range(151, 161)])
    train_mask = a_ours.obs.fov.isin(train_fovs).values
    val_mask = a_ours.obs.fov.isin(val_fovs).values
    if args.drop_background_ours:
        not_bg = (a_ours.obs.class_label.values != "background")
        train_mask = train_mask & not_bg
        val_mask = val_mask & not_bg
        print(f"  dropped background cells")
    print(f"  ours train: {train_mask.sum()}  val: {val_mask.sum()}")
    our_genes = list(a_ours.var_names)

    # ---- Load AWS data ----
    print(f"Loading AWS sections {sections} ...")
    X_aws, meta_aws, aws_gene_symbols = _load_aws(
        Path(args.aws_h5ad), Path(args.aws_meta), Path(args.aws_gene), sections)
    if args.restrict_aws_to_our_classes:
        our_classes = set(a_ours.obs.class_label.unique()) - {"background"}
        keep = meta_aws["class"].isin(our_classes).values
        X_aws = X_aws[keep]
        meta_aws = meta_aws.loc[keep].reset_index(drop=True)
        print(f"  restricted to our class vocab: kept {len(meta_aws):,} cells")

    # ---- Intersect gene panels ----
    aws_sym_to_idx = {s: i for i, s in enumerate(aws_gene_symbols)}
    shared_genes = [g for g in our_genes if g in aws_sym_to_idx]
    our_idx = [our_genes.index(g) for g in shared_genes]
    aws_idx = [aws_sym_to_idx[g] for g in shared_genes]
    print(f"  shared genes: {len(shared_genes)}/{len(our_genes)} (ours) vs {len(aws_gene_symbols)} (AWS)")

    # Reindex
    X_ours = X_ours[:, our_idx]
    X_aws = X_aws[:, aws_idx]

    # ---- L1-normalize each cell ----
    from sklearn.preprocessing import normalize
    X_ours_n = normalize(X_ours, norm="l1").astype(np.float32)
    X_aws_n = normalize(X_aws, norm="l1").astype(np.float32)

    # ---- Build combined train pool ----
    # Train: our train FOVs + ALL AWS section cells (AWS doesn't split for our val)
    Xtr_ours = X_ours_n[train_mask]
    Xtr = np.vstack([Xtr_ours, X_aws_n])
    print(f"  combined train: {Xtr.shape} (ours {train_mask.sum()} + AWS {len(X_aws_n)})")

    # Val: our val FOVs only (no AWS leakage)
    Xva = X_ours_n[val_mask]

    # ---- Build labels ----
    # Combine FOV info for spatial neighborhoods (AWS uses brain_section_label as FOV proxy)
    train_fovs_arr = np.concatenate([
        a_ours.obs.fov.values[train_mask],
        meta_aws.brain_section_label.values,  # AWS cells grouped by section
    ])
    val_fovs_arr = a_ours.obs.fov.values[val_mask]

    # Coordinates: our (center_x, center_y) in µm; AWS (x, y) in mm
    # Convert AWS to µm-equivalent scale by * 1000 - within-section relative dists are what matters
    train_coords = np.vstack([
        a_ours.obs[["center_x", "center_y"]].to_numpy(dtype=np.float32)[train_mask],
        meta_aws[["x", "y"]].to_numpy(dtype=np.float32) * 1000.0,
    ])
    val_coords = a_ours.obs[["center_x", "center_y"]].to_numpy(dtype=np.float32)[val_mask]

    # Per-level labels
    y_train_per_lvl: dict[str, np.ndarray] = {}
    for lvl in LEVELS_INTERNAL:
        ours_y = a_ours.obs[lvl].values[train_mask]
        aws_col = next(k for k, v in AWS_LABEL_MAP.items() if v == lvl)
        aws_y = meta_aws[aws_col].values
        y_train_per_lvl[lvl] = np.concatenate([ours_y, aws_y])

    y_val_per_lvl = {lvl: a_ours.obs[lvl].values[val_mask] for lvl in LEVELS_INTERNAL}

    # ---- Compute spatial-context features ----
    print(f"Computing spatial-context features (k={args.spatial_k}, nw={args.neighbor_weight})...")
    t0 = time.time()
    Xtr_aug = add_spatial_context(Xtr, train_coords, train_fovs_arr,
                                   k=args.spatial_k, neighbor_weight=args.neighbor_weight)
    Xva_aug = add_spatial_context(Xva, val_coords, val_fovs_arr,
                                   k=args.spatial_k, neighbor_weight=args.neighbor_weight)
    print(f"  augmented shapes: train {Xtr_aug.shape}, val {Xva_aug.shape} ({time.time() - t0:.1f}s)")

    # ---- Train + eval per level ----
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, adjusted_rand_score

    metrics = {"k": args.knn_k, "spatial_k": args.spatial_k,
               "neighbor_weight": args.neighbor_weight, "metric": args.metric,
               "n_train": int(len(Xtr_aug)), "n_val": int(len(Xva_aug)),
               "n_genes": len(shared_genes), "feature_dim": int(Xtr_aug.shape[1]),
               "aws_sections": sections, "per_level": {}}

    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        ytr = y_train_per_lvl[lvl]
        yva = y_val_per_lvl[lvl]
        if len(set(ytr)) < 2:
            print(f"  [skip] {lvl} only has one class")
            continue
        # Filter NaN labels in train (some AWS cells might have NaN at finer levels)
        valid = pd.notna(ytr)
        ytr_f = ytr[valid]
        Xtr_f = Xtr_aug[valid]
        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=args.knn_k, metric=args.metric)
        clf.fit(Xtr_f, ytr_f)
        ypr = clf.predict(Xva_aug)
        elapsed = time.time() - t0
        acc = float(accuracy_score(yva, ypr))
        ari = float(adjusted_rand_score(yva, ypr))
        print(f"  {out_name:<10} acc={acc:.3f}  cell-ARI={ari:.3f}  "
              f"({len(set(ytr_f))} classes, {len(Xtr_f):,} train, {elapsed:.1f}s)")
        metrics["per_level"][out_name] = {
            "accuracy": acc, "ari_cells": ari,
            "n_train_cells": int(len(ytr_f)), "n_val_cells": int(len(yva)),
            "n_classes": int(len(set(ytr_f))),
        }
        joblib.dump({"clf": clf, "genes": shared_genes,
                     "preproc": "spatial_l1", "k": args.knn_k, "metric": args.metric,
                     "spatial_k": args.spatial_k, "feature_dim": int(Xtr_aug.shape[1])},
                    out_dir / f"model_{out_name}.joblib")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
