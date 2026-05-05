"""Hierarchy-aware XGBoost classifier on AWS-augmented data.

Adapted from `phase2/docs/codelab_13.py` "Hierarchy-aware XGBoost" section.
Predicts class → subclass | class → supertype | class, subclass → cluster |
class, subclass, supertype, by **concatenating predicted-parent probabilities
as features** (soft hierarchy, not hard masking like our kNN version).

Data: same recipe as the AWS-augmented kNN winner —
  phase-2 train cells + AWS .001 (cluster-filtered) + bg×40 upsampling +
  log1p preproc.

Output bundle: same format as train_hierarchical_aws.py so we can reuse
infer_hierarchical.py for inference.
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


def _preprocess(X: np.ndarray, mode: str = "log1p") -> np.ndarray:
    X = X.astype(np.float32)
    if mode == "log1p":
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return np.log1p(X / row_sum * 1e4)
    raise NotImplementedError(mode)


class HierXGBBundle:
    """Wrapper saving the per-level XGBoost models + label encoders + parent
    feature plumbing, exposing predict(X) → 4-level dict."""
    def __init__(self, models, encoders, feature_dim, levels_internal):
        self.models = models
        self.encoders = encoders
        self.feature_dim = feature_dim
        self.levels_internal = levels_internal

    def predict(self, X) -> dict[str, np.ndarray]:
        x = X.astype(np.float32)
        out_per_level: dict[str, np.ndarray] = {}
        prev = None
        for lvl in self.levels_internal:
            feats = x if prev is None else np.hstack([x, prev])
            pred = self.models[lvl].predict(feats)
            out_per_level[lvl] = self.encoders[lvl].inverse_transform(pred)
            prev = self.models[lvl].predict_proba(feats)
        return out_per_level


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
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_train = (args.phase2_train_fovs.split(",") if args.phase2_train_fovs
                else [f"FOV_{i:03d}" for i in range(101, 151)])
    p2_val = (args.phase2_val_fovs.split(",") if args.phase2_val_fovs
              else [f"FOV_{i:03d}" for i in range(151, 161)])

    a2 = ad.read_h5ad(args.phase2_h5ad)
    X2 = a2.X.toarray() if hasattr(a2.X, "toarray") else np.asarray(a2.X)
    train_mask2 = a2.obs.fov.isin(p2_train).values
    val_mask2 = a2.obs.fov.isin(p2_val).values

    train_genes = list(a2.var_names)
    X_train_parts = [X2[train_mask2]]
    y_parts = {lvl: [a2.obs[lvl].values[train_mask2]] for lvl in LEVELS_INTERNAL}

    if args.include_aws:
        sections = [s.strip() for s in args.aws_sections.split(",")]
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
        for lvl, src in (("class_label", "class"), ("subclass_label", "subclass"),
                         ("supertype_label", "supertype"), ("cluster_label", "cluster")):
            y_parts[lvl].append(meta_idx[src].astype(str).values)

    if args.upsample_background > 0:
        p2_part = X_train_parts[0]
        bg = (y_parts["class_label"][0] == "background")
        n_bg = int(bg.sum())
        if n_bg > 0:
            X_train_parts.append(np.tile(p2_part[bg], (args.upsample_background, 1)))
            for lvl in LEVELS_INTERNAL:
                y_parts[lvl].append(np.tile(y_parts[lvl][0][bg], args.upsample_background))

    X_train = np.vstack(X_train_parts).astype(np.float32)
    y_train = {lvl: np.concatenate(parts) for lvl, parts in y_parts.items()}
    print(f"  X_train: {X_train.shape}")

    Xtr = _preprocess(X_train, "log1p")
    Xva = _preprocess(X2[val_mask2], "log1p")
    yv = {lvl: a2.obs[lvl].values[val_mask2].astype(str) for lvl in LEVELS_INTERNAL}

    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import adjusted_rand_score

    encoders, models = {}, {}
    prev_train: np.ndarray | None = None
    prev_val: np.ndarray | None = None
    metrics: dict = {"per_level": {}}
    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        y_str = y_train[lvl].astype(str)
        le = LabelEncoder().fit(y_str)
        feats_tr = Xtr if prev_train is None else np.hstack([Xtr, prev_train])
        feats_va = Xva if prev_val is None else np.hstack([Xva, prev_val])
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_jobs=4, tree_method="hist", eval_metric="mlogloss",
        )
        t0 = time.time()
        clf.fit(feats_tr, le.transform(y_str))
        yv_str = yv[lvl]
        # Some val labels may not be in train (e.g. supertype unseen); LabelEncoder
        # will raise. Map unseen → a sentinel and fold them into ARI as-is.
        seen_classes = set(le.classes_)
        ypr = clf.predict(feats_va)
        ypr_str = le.inverse_transform(ypr)
        # Cell-level ARI on val
        ari = adjusted_rand_score(yv_str, ypr_str)
        elapsed = time.time() - t0
        print(f"  {out_name:<10} cell-ARI={ari:.3f}  ({len(le.classes_)} classes, {elapsed:.1f}s)")
        encoders[lvl] = le
        models[lvl] = clf
        prev_train = clf.predict_proba(feats_tr)
        prev_val = clf.predict_proba(feats_va)
        metrics["per_level"][out_name] = {"ari_cells": float(ari), "n_classes": int(len(le.classes_))}

    bundle = HierXGBBundle(models=models, encoders=encoders,
                           feature_dim=Xtr.shape[1], levels_internal=LEVELS_INTERNAL)
    joblib.dump({
        "type": "hier_xgb",
        "bundle": bundle,
        "genes": train_genes,
        "preproc": "log1p",
    }, out_dir / "xgb_bundle.joblib", compress=3)
    Path(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}/xgb_bundle.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
