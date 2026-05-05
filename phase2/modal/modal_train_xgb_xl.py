"""Train hierarchy-aware XGBoost on AWS-augmented data on Modal GPU.

Reads phase-2 counts_train.h5ad + AWS Zhuang-ABCA-4 h5ad/metadata from the
`cell-seg-phase2` volume, trains 4 XGBoost classifiers (class → subclass | class
→ supertype | parent → cluster | parent), saves bundle to volume.

Run:
    modal run phase2/modal/modal_train_xgb.py::run

Then download:
    mkdir -p phase2/runs/xgb-hier-aws-bg40-modal
    modal volume get cell-seg-phase2 trained/xgb-hier-aws-bg40 \\
        phase2/runs/xgb-hier-aws-bg40-modal/
"""
from __future__ import annotations

import modal

app = modal.App("phase2-train-xgb-xl")

data_vol = modal.Volume.from_name("cell-seg-phase2", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "scipy",
        "anndata", "scikit-learn", "joblib",
        "xgboost",  # GPU build
    )
)

VOLUMES = {"/root/data": data_vol}


@app.function(image=image, gpu="A10G", timeout=9000, volumes=VOLUMES)
def train_xgb(
    aws_filter_level: str = "cluster",
    upsample_background: int = 40,
    n_estimators: int = 3000,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    aws_sections: str = "Zhuang-ABCA-4.001",
    out_subdir: str = "trained/xgb-hier-aws-xl",
) -> str:
    import json
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import anndata as ad
    import joblib

    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import adjusted_rand_score

    DATA = Path("/root/data")
    P2_H5AD = DATA / "train" / "ground_truth" / "counts_train.h5ad"
    AWS_H5AD = DATA / "external" / "aws" / "Zhuang-ABCA-4-log2.h5ad"
    AWS_META = DATA / "external" / "aws" / "cell_metadata_with_cluster_annotation.csv"
    OUT = DATA / out_subdir
    OUT.mkdir(parents=True, exist_ok=True)

    LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
    LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
                  "supertype_label": "supertype", "cluster_label": "cluster"}

    p2_train = [f"FOV_{i:03d}" for i in range(101, 151)]
    p2_val = [f"FOV_{i:03d}" for i in range(151, 161)]

    print(f"loading {P2_H5AD}")
    a2 = ad.read_h5ad(P2_H5AD)
    X2 = a2.X.toarray() if hasattr(a2.X, "toarray") else np.asarray(a2.X)
    train_mask2 = a2.obs.fov.isin(p2_train).values
    val_mask2 = a2.obs.fov.isin(p2_val).values
    train_genes = list(a2.var_names)

    X_train_parts = [X2[train_mask2]]
    y_parts: dict[str, list[np.ndarray]] = {
        lvl: [a2.obs[lvl].values[train_mask2]] for lvl in LEVELS_INTERNAL
    }

    sections = [s.strip() for s in aws_sections.split(",")]
    print(f"loading AWS {sections}")
    a_aws = ad.read_h5ad(AWS_H5AD)
    meta = pd.read_csv(AWS_META)
    meta = meta[meta["brain_section_label"].isin(sections)].copy()
    meta["cell_label"] = meta["cell_label"].astype(str)
    keep = a_aws.obs.index.astype(str).isin(set(meta["cell_label"]))
    a_aws = a_aws[keep].copy()
    meta_idx = meta.set_index("cell_label").loc[a_aws.obs.index.astype(str)]

    if aws_filter_level != "none":
        level_pairs = [("class", "class_label"), ("subclass", "subclass_label"),
                       ("supertype", "supertype_label"), ("cluster", "cluster_label")]
        level_idx = next(i for i, (k, _) in enumerate(level_pairs) if k == aws_filter_level)
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

    if upsample_background > 0:
        p2_part = X_train_parts[0]
        bg = (y_parts["class_label"][0] == "background")
        n_bg = int(bg.sum())
        if n_bg > 0:
            print(f"  bg upsample: {n_bg} × {upsample_background}")
            X_train_parts.append(np.tile(p2_part[bg], (upsample_background, 1)))
            for lvl in LEVELS_INTERNAL:
                y_parts[lvl].append(np.tile(y_parts[lvl][0][bg], upsample_background))

    X_train = np.vstack(X_train_parts).astype(np.float32)
    y_train = {lvl: np.concatenate(parts) for lvl, parts in y_parts.items()}
    print(f"X_train: {X_train.shape}")

    def _log1p(X):
        X = X.astype(np.float32)
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return np.log1p(X / rs * 1e4)

    Xtr = _log1p(X_train)
    Xva = _log1p(X2[val_mask2])
    yv = {lvl: a2.obs[lvl].values[val_mask2].astype(str) for lvl in LEVELS_INTERNAL}

    encoders, models = {}, {}
    prev_train = None
    prev_val = None
    metrics: dict = {"per_level": {}}
    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        y_str = y_train[lvl].astype(str)
        le = LabelEncoder().fit(y_str)
        feats_tr = Xtr if prev_train is None else np.hstack([Xtr, prev_train])
        feats_va = Xva if prev_val is None else np.hstack([Xva, prev_val])
        # GPU XGBoost: device='cuda' for hist tree method on GPU
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate,
            device="cuda", tree_method="hist",
            eval_metric="mlogloss",
        )
        t0 = time.time()
        clf.fit(feats_tr, le.transform(y_str))
        ypr = clf.predict(feats_va)
        ypr_str = le.inverse_transform(ypr)
        ari = adjusted_rand_score(yv[lvl], ypr_str)
        elapsed = time.time() - t0
        print(f"  {out_name:<10} cell-ARI={ari:.3f} ({len(le.classes_)} classes, {elapsed:.1f}s)")
        encoders[lvl] = le
        models[lvl] = clf
        prev_train = clf.predict_proba(feats_tr)
        prev_val = clf.predict_proba(feats_va)
        metrics["per_level"][out_name] = {"ari_cells": float(ari), "n_classes": int(len(le.classes_))}

    # Save bundle in same shape as train_xgb_hier_aws.HierXGBBundle, but
    # avoid pickling the wrapper class (cross-module pickling pain). Instead
    # save just the dict of (model, encoder) per level + genes + preproc.
    bundle = {
        "type": "hier_xgb_modal",
        "models": models,
        "encoders": encoders,
        "genes": train_genes,
        "preproc": "log1p",
        "feature_dim": Xtr.shape[1],
        "levels": list(LEVELS_INTERNAL),
    }
    bundle_path = OUT / "xgb_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    data_vol.commit()
    print(f"\n→ {bundle_path}")
    return str(bundle_path)


@app.local_entrypoint()
def run():
    path = train_xgb.remote()
    print(f"trained: {path}")


@app.local_entrypoint()
def sweep():
    """Three configs in parallel — diversity for ensembling."""
    configs = [
        # (n_estimators, max_depth, lr, bg_upsample, out_subdir)
        (3000, 8, 0.05, 40, "trained/xgb-xl-d8n3000-bg40"),
        (3000, 6, 0.05, 80, "trained/xgb-xl-d6n3000-bg80"),
        (5000, 5, 0.03, 40, "trained/xgb-xl-d5n5000-bg40"),
    ]
    handles = []
    for n_est, depth, lr, bg, out in configs:
        h = train_xgb.spawn(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            upsample_background=bg, out_subdir=out,
        )
        print(f"spawned {out}: handle={h.object_id}")
        handles.append((out, h))
    for out, h in handles:
        path = h.get()
        print(f"  done {out} → {path}")
