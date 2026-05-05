"""Train scVI + scANVI on AWS-augmented data on Modal GPU.

Reads phase-2 + AWS h5ad/metadata from `cell-seg-phase2` volume, trains scVI
once, then 4 separate scANVI heads (one per taxonomy level), saves bundle.

Run:
    modal run phase2/modal/modal_train_scanvi.py::run

Then download:
    mkdir -p phase2/runs/scanvi-aws-modal
    modal volume get cell-seg-phase2 trained/scanvi-aws-modal \\
        phase2/runs/scanvi-aws-modal/
"""
from __future__ import annotations

import modal

app = modal.App("phase2-train-scanvi-xl")

data_vol = modal.Volume.from_name("cell-seg-phase2", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "scipy",
        "anndata", "scikit-learn", "joblib",
        "scvi-tools",
    )
)

VOLUMES = {"/root/data": data_vol}


@app.function(image=image, gpu="A10G", timeout=9000, volumes=VOLUMES)
def train_scanvi(
    aws_filter_level: str = "cluster",
    aws_sections: str = "Zhuang-ABCA-4.001",
    scvi_epochs: int = 300,
    scanvi_epochs: int = 120,
    n_latent: int = 64,
    n_layers: int = 3,
    out_subdir: str = "trained/scanvi-aws-modal-xl",
) -> str:
    import json
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import anndata as ad
    import joblib
    import scvi
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
    print(f"AWS cells kept: {a_aws.shape[0]}")

    train_genes = list(a2.var_names)
    X_aws_log = a_aws.X.toarray() if hasattr(a_aws.X, "toarray") else np.asarray(a_aws.X)
    X_aws_lin = (np.power(2.0, X_aws_log.astype(np.float32)) - 1.0).astype(np.float32)
    X_aws_int = np.maximum(np.round(X_aws_lin).astype(np.int32), 0)
    X2_int = np.maximum(np.round(X2.astype(np.float32)).astype(np.int32), 0)
    aws_symbols = a_aws.var["gene_symbol"].astype(str).tolist()
    aws_sym_to_col = {g: i for i, g in enumerate(aws_symbols)}
    X_aws_aligned = np.zeros((X_aws_int.shape[0], len(train_genes)), dtype=np.int32)
    for j, g in enumerate(train_genes):
        col = aws_sym_to_col.get(g)
        if col is not None:
            X_aws_aligned[:, j] = X_aws_int[:, col]

    obs_p2 = pd.DataFrame({
        "fov": a2.obs.fov.values.astype(str),
        "split": np.where(train_mask2, "train",
                          np.where(val_mask2, "val", "other")),
        "source": "phase2",
        "class_label": a2.obs.class_label.values.astype(str),
        "subclass_label": a2.obs.subclass_label.values.astype(str),
        "supertype_label": a2.obs.supertype_label.values.astype(str),
        "cluster_label": a2.obs.cluster_label.values.astype(str),
    })
    obs_aws = pd.DataFrame({
        "fov": meta_idx["brain_section_label"].astype(str).values,
        "split": "train",
        "source": "aws",
        "class_label": meta_idx["class"].astype(str).values,
        "subclass_label": meta_idx["subclass"].astype(str).values,
        "supertype_label": meta_idx["supertype"].astype(str).values,
        "cluster_label": meta_idx["cluster"].astype(str).values,
    })
    keep_p2 = obs_p2["split"].isin(["train", "val"]).values
    X_combined = np.vstack([X2_int[keep_p2], X_aws_aligned]).astype(np.int32)
    obs_combined = pd.concat(
        [obs_p2[keep_p2].reset_index(drop=True), obs_aws.reset_index(drop=True)],
        ignore_index=True,
    )
    var_combined = pd.DataFrame(index=train_genes)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, var=var_combined)
    print(f"adata: {adata.shape}")

    # ORIGINAL labels we keep separately for val eval (before masking).
    original_labels_per_level = {
        lvl: obs_combined[lvl].astype(str).values.copy() for lvl in LEVELS_INTERNAL
    }

    val_idx = (adata.obs["split"] == "val").values
    print(f"val cells: {int(val_idx.sum())}")

    # Mask val labels for semi-supervised training; do this once on adata before
    # any per-level setup_anndata.
    for lvl in LEVELS_INTERNAL:
        adata.obs[lvl] = adata.obs[lvl].astype("category")
        if "UNLABELED" not in adata.obs[lvl].cat.categories:
            adata.obs[lvl] = adata.obs[lvl].cat.add_categories(["UNLABELED"])
        adata.obs.loc[val_idx, lvl] = "UNLABELED"

    metrics: dict = {"per_level": {}}
    OUT_PER_LVL = OUT / "scanvi_per_level"
    OUT_PER_LVL.mkdir(parents=True, exist_ok=True)

    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        ad_lvl = adata.copy()
        scvi.model.SCVI.setup_anndata(ad_lvl, batch_key="fov")
        v = scvi.model.SCVI(ad_lvl, n_layers=n_layers, n_latent=n_latent)
        t0 = time.time()
        v.train(max_epochs=scvi_epochs, early_stopping=True, accelerator="gpu")
        m = scvi.model.SCANVI.from_scvi_model(
            v, adata=ad_lvl, labels_key=lvl, unlabeled_category="UNLABELED")
        m.train(max_epochs=scanvi_epochs, accelerator="gpu")
        elapsed = time.time() - t0

        # Eval on val (against original labels we saved before masking)
        ad_val = ad_lvl[val_idx].copy()
        pred_labels = np.asarray(m.predict(ad_val)).astype(str)
        gt_labels = original_labels_per_level[lvl][val_idx]
        ari = float(adjusted_rand_score(gt_labels, pred_labels))
        print(f"  {out_name:<10} cell-ARI={ari:.3f} ({elapsed:.1f}s)")
        metrics["per_level"][out_name] = {"ari_cells": ari}

        m.save(str(OUT_PER_LVL / lvl), overwrite=True, save_anndata=False)

    joblib.dump({
        "type": "scanvi_modal",
        "genes": train_genes,
        "levels": list(LEVELS_INTERNAL),
        "manifest_dir_in_volume": f"{out_subdir}/scanvi_per_level",
    }, OUT / "scanvi_bundle.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    data_vol.commit()
    print(f"\n→ {OUT}/scanvi_bundle.joblib")
    return str(OUT / "scanvi_bundle.joblib")


@app.local_entrypoint()
def run():
    path = train_scanvi.remote()
    print(f"trained: {path}")
