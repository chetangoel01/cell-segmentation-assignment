"""Hierarchical AWS-augmented kNN trainer.

Single combined bundle. At inference, get K_LARGE neighbors once, then vote at
each taxonomy level conditional on the predicted parent label (taxonomy-tree
masking). This collapses the cluster-level vote from ~200 strings globally
to typically 3-8 clusters per supertype, which fixes the per-level decay we
saw in the flat AWS-augmented kNN (class 0.531 → cluster 0.482).

Output: phase2/runs/<exp>/bundle.joblib containing
  - 'X_train': (N, n_genes) float32 (after preproc)
  - 'y_train': dict of 4 label arrays (length N each)
  - 'genes': list of gene symbols (column order)
  - 'preproc': 'log1p' | 'l1_normalize'
  - 'fov_means': optional dict {fov_or_section -> (n_genes,) float32}
  - 'fov_assignment': optional array length N giving each train cell's FOV/section key
  - 'k': default K_LARGE
  - 'metric': 'cosine'
  - 'taxonomy': dict-of-dicts (parent label string -> set of child label strings)
        keyed by level: 'class_to_subclass', 'subclass_to_supertype', 'supertype_to_cluster'

Plus 'sklearn_knn': a fitted NearestNeighbors index over X_train for fast lookup
during inference.
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


def _load_h5ad(path: Path):
    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    return a, X


def _preprocess(X: np.ndarray, mode: str) -> np.ndarray:
    X = X.astype(np.float32)
    if mode == "log1p":
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return np.log1p(X / row_sum * 1e4)
    elif mode == "l1_normalize":
        from sklearn.preprocessing import normalize
        return normalize(X, norm="l1")
    raise ValueError(f"unknown preproc {mode!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase2-h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--phase2-train-fovs", default=None)
    p.add_argument("--phase2-val-fovs", default=None)
    p.add_argument("--include-aws", action="store_true")
    p.add_argument("--aws-h5ad", default="phase2/data/external/aws/Zhuang-ABCA-4-log2.h5ad")
    p.add_argument("--aws-metadata",
                   default="phase2/data/external/aws/cell_metadata_with_cluster_annotation.csv")
    p.add_argument("--aws-sections", default="Zhuang-ABCA-4.001")
    p.add_argument("--aws-filter-level", default="cluster",
                   choices=("class", "subclass", "supertype", "cluster", "none"))
    p.add_argument("--upsample-background", type=int, default=40)
    p.add_argument("--preproc", default="log1p", choices=("log1p", "l1_normalize"))
    p.add_argument("--k", type=int, default=50,
                   help="K_LARGE — index will be built with this many neighbors. Inference picks "
                        "subsets of these for each level vote.")
    p.add_argument("--include-fov-means", action="store_true",
                   help="Append FOV-mean expression vector to each cell's features (spatial context). "
                        "AWS cells use the section mean as their 'FOV'.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_train = (args.phase2_train_fovs.split(",") if args.phase2_train_fovs
                else [f"FOV_{i:03d}" for i in range(101, 151)])
    p2_val = (args.phase2_val_fovs.split(",") if args.phase2_val_fovs
              else [f"FOV_{i:03d}" for i in range(151, 161)])

    print(f"phase-2 h5ad: {args.phase2_h5ad}")
    a2, X2 = _load_h5ad(Path(args.phase2_h5ad))
    train_mask2 = a2.obs.fov.isin(p2_train).values
    print(f"  phase-2 train cells: {train_mask2.sum()}")

    train_genes = list(a2.var_names)
    X_train_parts: list[np.ndarray] = [X2[train_mask2]]
    y_train_parts: dict[str, list[np.ndarray]] = {
        lvl: [a2.obs[lvl].values[train_mask2]] for lvl in LEVELS_INTERNAL
    }
    fov_assignment_parts: list[np.ndarray] = [a2.obs.fov.values[train_mask2].astype(str)]

    aws_n_used = 0
    if args.include_aws:
        sections = [s.strip() for s in args.aws_sections.split(",") if s.strip()]
        print(f"AWS h5ad: {args.aws_h5ad}  sections={sections}")
        a_aws = ad.read_h5ad(args.aws_h5ad)
        meta = pd.read_csv(args.aws_metadata)
        meta = meta[meta["brain_section_label"].isin(sections)].copy()
        meta["cell_label"] = meta["cell_label"].astype(str)
        keep = a_aws.obs.index.astype(str).isin(set(meta["cell_label"]))
        a_aws = a_aws[keep].copy()
        meta_idx = meta.set_index("cell_label").loc[a_aws.obs.index.astype(str)]

        if args.aws_filter_level != "none":
            level_pairs = [("class", "class_label"),
                           ("subclass", "subclass_label"),
                           ("supertype", "supertype_label"),
                           ("cluster", "cluster_label")]
            level_idx = next(i for i, (k, _) in enumerate(level_pairs) if k == args.aws_filter_level)
            mask = np.ones(len(meta_idx), dtype=bool)
            for aws_col, comp_col in level_pairs[: level_idx + 1]:
                comp_set = set(a2.obs[comp_col].astype(str).unique()) - {"background"}
                mask &= meta_idx[aws_col].astype(str).isin(comp_set).values
            print(f"  filter='{args.aws_filter_level}' kept {int(mask.sum())} of {len(mask)}")
            a_aws = a_aws[mask].copy()
            meta_idx = meta_idx[mask]
        print(f"  AWS cells kept: {a_aws.shape[0]}")

        X_aws_log = a_aws.X.toarray() if hasattr(a_aws.X, "toarray") else np.asarray(a_aws.X)
        X_aws_lin = (np.power(2.0, X_aws_log.astype(np.float32)) - 1.0).astype(np.float32)

        aws_symbols = a_aws.var["gene_symbol"].astype(str).tolist()
        aws_sym_to_col = {g: i for i, g in enumerate(aws_symbols)}
        X_aws_aligned = np.zeros((X_aws_lin.shape[0], len(train_genes)), dtype=np.float32)
        for j, g in enumerate(train_genes):
            col = aws_sym_to_col.get(g)
            if col is not None:
                X_aws_aligned[:, j] = X_aws_lin[:, col]

        X_train_parts.append(X_aws_aligned)
        for lvl, src_col in (("class_label", "class"),
                             ("subclass_label", "subclass"),
                             ("supertype_label", "supertype"),
                             ("cluster_label", "cluster")):
            y_train_parts[lvl].append(meta_idx[src_col].astype(str).values)
        # AWS cells' "FOV" is the section name (used for fov-mean computation)
        fov_assignment_parts.append(meta_idx["brain_section_label"].astype(str).values)
        aws_n_used = X_aws_aligned.shape[0]

    if args.upsample_background > 0:
        p2_part = X_train_parts[0]
        p2_labels = y_train_parts["class_label"][0]
        bg_mask = (p2_labels == "background")
        n_bg = int(bg_mask.sum())
        if n_bg > 0:
            print(f"  upsampling background: {n_bg} × {args.upsample_background}")
            X_train_parts.append(np.tile(p2_part[bg_mask], (args.upsample_background, 1)))
            for lvl in LEVELS_INTERNAL:
                y_train_parts[lvl].append(
                    np.tile(y_train_parts[lvl][0][bg_mask], args.upsample_background)
                )
            fov_assignment_parts.append(
                np.tile(fov_assignment_parts[0][bg_mask], args.upsample_background)
            )

    X_train = np.vstack(X_train_parts).astype(np.float32)
    y_train = {lvl: np.concatenate(parts) for lvl, parts in y_train_parts.items()}
    fov_assignment = np.concatenate(fov_assignment_parts)
    print(f"final X_train: {X_train.shape}  aws_used={aws_n_used}")

    Xtr_pre = _preprocess(X_train, args.preproc)

    fov_means = None
    fov_means_appended = False
    if args.include_fov_means:
        # Compute per-FOV/section mean of the preprocessed features.
        fov_means = {}
        for fov_key in np.unique(fov_assignment):
            mask = fov_assignment == fov_key
            fov_means[str(fov_key)] = Xtr_pre[mask].mean(axis=0).astype(np.float32)
        # Build per-cell "fov-mean" feature column block.
        fov_block = np.stack([fov_means[str(k)] for k in fov_assignment], axis=0).astype(np.float32)
        Xtr_pre = np.concatenate([Xtr_pre, fov_block], axis=1).astype(np.float32)
        fov_means_appended = True
        print(f"  FOV-mean features appended: dim now {Xtr_pre.shape[1]}")

    # Build taxonomy tree (parent->children) from train labels — union of comp + AWS.
    tax = {"class_to_subclass": {}, "subclass_to_supertype": {}, "supertype_to_cluster": {}}
    for cls, sub in zip(y_train["class_label"], y_train["subclass_label"]):
        tax["class_to_subclass"].setdefault(str(cls), set()).add(str(sub))
    for sub, sup in zip(y_train["subclass_label"], y_train["supertype_label"]):
        tax["subclass_to_supertype"].setdefault(str(sub), set()).add(str(sup))
    for sup, clu in zip(y_train["supertype_label"], y_train["cluster_label"]):
        tax["supertype_to_cluster"].setdefault(str(sup), set()).add(str(clu))
    for k in tax:
        tax[k] = {p: list(c) for p, c in tax[k].items()}
    print(f"  taxonomy: class→sub keys={len(tax['class_to_subclass'])} "
          f"sub→sup keys={len(tax['subclass_to_supertype'])} "
          f"sup→clu keys={len(tax['supertype_to_cluster'])}")

    # Fit a NearestNeighbors index for fast lookup at inference.
    from sklearn.neighbors import NearestNeighbors
    t0 = time.time()
    nn_index = NearestNeighbors(n_neighbors=args.k, metric="cosine", n_jobs=-1).fit(Xtr_pre)
    print(f"  NearestNeighbors fit: {time.time()-t0:.1f}s  K={args.k}")

    # Optional: quick sanity self-eval on val FOVs (drops AWS, evaluates phase-2 val cells).
    val_mask2 = a2.obs.fov.isin(p2_val).values
    Xva = X2[val_mask2]
    Xva_pre = _preprocess(Xva, args.preproc)
    if fov_means_appended:
        # Use phase-2 val FOV-means computed from the val cells themselves
        # (matching what we do at test time: fov_means come from the FOV's segmented cells).
        val_fov_assign = a2.obs.fov.values[val_mask2].astype(str)
        val_fov_means = {}
        for fov_key in np.unique(val_fov_assign):
            mask = val_fov_assign == fov_key
            val_fov_means[fov_key] = Xva_pre[mask].mean(axis=0)
        val_fov_block = np.stack([val_fov_means[k] for k in val_fov_assign], axis=0).astype(np.float32)
        Xva_pre = np.concatenate([Xva_pre, val_fov_block], axis=1).astype(np.float32)

    # Hierarchical predict on val
    from sklearn.metrics import adjusted_rand_score
    t0 = time.time()
    distances, indices = nn_index.kneighbors(Xva_pre)  # (n_val, K)
    # Use distance-weighting: w = 1/(d+eps)
    eps = 1e-9
    weights = 1.0 / (distances + eps)
    yv = {lvl: a2.obs[lvl].values[val_mask2].astype(str) for lvl in LEVELS_INTERNAL}

    def _weighted_vote(labels: np.ndarray, w: np.ndarray) -> str:
        # labels and w are 1D arrays of equal length
        score: dict[str, float] = {}
        for lab, wi in zip(labels, w):
            score[lab] = score.get(lab, 0.0) + float(wi)
        return max(score.items(), key=lambda kv: kv[1])[0]

    pred = {lvl: [] for lvl in LEVELS_INTERNAL}
    yt = {lvl: y_train[lvl].astype(str) for lvl in LEVELS_INTERNAL}
    for i in range(len(Xva_pre)):
        idx_i = indices[i]
        w_i = weights[i]
        # Class
        nbr_cls = yt["class_label"][idx_i]
        cls = _weighted_vote(nbr_cls, w_i)
        pred["class_label"].append(cls)
        # Subclass — mask to neighbors with class == cls
        m = (nbr_cls == cls)
        sub_labels = yt["subclass_label"][idx_i][m] if m.any() else yt["subclass_label"][idx_i]
        sub_w = w_i[m] if m.any() else w_i
        sub = _weighted_vote(sub_labels, sub_w)
        pred["subclass_label"].append(sub)
        # Supertype — mask to neighbors with subclass == sub
        nbr_sub = yt["subclass_label"][idx_i]
        m = (nbr_sub == sub)
        sup_labels = yt["supertype_label"][idx_i][m] if m.any() else yt["supertype_label"][idx_i]
        sup_w = w_i[m] if m.any() else w_i
        sup = _weighted_vote(sup_labels, sup_w)
        pred["supertype_label"].append(sup)
        # Cluster — mask to neighbors with supertype == sup
        nbr_sup = yt["supertype_label"][idx_i]
        m = (nbr_sup == sup)
        clu_labels = yt["cluster_label"][idx_i][m] if m.any() else yt["cluster_label"][idx_i]
        clu_w = w_i[m] if m.any() else w_i
        pred["cluster_label"].append(_weighted_vote(clu_labels, clu_w))
    elapsed = time.time() - t0
    print(f"  hierarchical predict on val: {elapsed:.1f}s")

    metrics = {
        "k": args.k,
        "preproc": args.preproc,
        "include_aws": args.include_aws,
        "aws_n_cells": aws_n_used,
        "include_fov_means": args.include_fov_means,
        "n_train": int(X_train.shape[0]),
        "n_val": int(val_mask2.sum()),
        "n_genes": Xtr_pre.shape[1],
        "per_level": {},
    }
    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        ari = adjusted_rand_score(yv[lvl], np.asarray(pred[lvl]))
        acc = float(np.mean(yv[lvl] == np.asarray(pred[lvl])))
        metrics["per_level"][out_name] = {
            "ari_cells": float(ari),
            "accuracy": acc,
            "n_classes": int(len(set(y_train[lvl]))),
        }
        print(f"  {out_name:<10} acc={acc:.3f}  cell-ARI={ari:.3f}")

    # Save bundle. Index is large (~100MB for 92K cells × 1147 genes float32) —
    # joblib compress=3 to keep on-disk size manageable.
    bundle = {
        "X_train": Xtr_pre,
        "y_train": {lvl: y_train[lvl].astype(str) for lvl in LEVELS_INTERNAL},
        "genes": train_genes,
        "preproc": args.preproc,
        "fov_means": fov_means,
        "fov_means_appended": fov_means_appended,
        "k": args.k,
        "metric": "cosine",
        "taxonomy": tax,
        "sklearn_knn": nn_index,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib", compress=3)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}/bundle.joblib")
    print(f"→ {out_dir}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
