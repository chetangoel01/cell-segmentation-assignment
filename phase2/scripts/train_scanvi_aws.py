"""scVI + scANVI on AWS-augmented data — adapted from phase2/docs/codelab_13.py.

Trains scVI as a base VAE over phase-2 train cells + AWS .001 cells (with raw
counts in `X`), batch-keyed by FOV/section, then trains scANVI per-level with
val cells masked as 'UNLABELED'. Saves a single bundle that can be loaded by
infer_scanvi.py for inference on cellpose-segmented cells.

Per-level scANVI follows the codelab pattern (one scANVI per level), since
scANVI takes a single labels_key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc

LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase2-h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--aws-h5ad", default="phase2/data/external/aws/Zhuang-ABCA-4-log2.h5ad")
    p.add_argument("--aws-metadata",
                   default="phase2/data/external/aws/cell_metadata_with_cluster_annotation.csv")
    p.add_argument("--aws-sections", default="Zhuang-ABCA-4.001")
    p.add_argument("--aws-filter-level", default="cluster",
                   choices=("class", "subclass", "supertype", "cluster", "none"))
    p.add_argument("--phase2-train-fovs", default=None)
    p.add_argument("--phase2-val-fovs", default=None)
    p.add_argument("--scvi-epochs", type=int, default=80)
    p.add_argument("--scanvi-epochs", type=int, default=30)
    p.add_argument("--n-latent", type=int, default=20)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_train = (args.phase2_train_fovs.split(",") if args.phase2_train_fovs
                else [f"FOV_{i:03d}" for i in range(101, 151)])
    p2_val = (args.phase2_val_fovs.split(",") if args.phase2_val_fovs
              else [f"FOV_{i:03d}" for i in range(151, 161)])

    # Load phase-2
    a2 = ad.read_h5ad(args.phase2_h5ad)
    X2 = a2.X.toarray() if hasattr(a2.X, "toarray") else np.asarray(a2.X)
    train_mask2 = a2.obs.fov.isin(p2_train).values
    val_mask2 = a2.obs.fov.isin(p2_val).values
    print(f"phase-2: train={int(train_mask2.sum())} val={int(val_mask2.sum())}")

    # AWS cells
    sections = [s.strip() for s in args.aws_sections.split(",")]
    a_aws = ad.read_h5ad(args.aws_h5ad)
    meta = pd.read_csv(args.aws_metadata)
    meta = meta[meta["brain_section_label"].isin(sections)].copy()
    meta["cell_label"] = meta["cell_label"].astype(str)
    keep = a_aws.obs.index.astype(str).isin(set(meta["cell_label"]))
    a_aws = a_aws[keep].copy()
    meta_idx = meta.set_index("cell_label").loc[a_aws.obs.index.astype(str)]
    if args.aws_filter_level != "none":
        level_pairs = [("class", "class_label"), ("subclass", "subclass_label"),
                       ("supertype", "supertype_label"), ("cluster", "cluster_label")]
        level_idx = next(i for i, (k, _) in enumerate(level_pairs) if k == args.aws_filter_level)
        mask = np.ones(len(meta_idx), dtype=bool)
        for aws_col, comp_col in level_pairs[: level_idx + 1]:
            comp_set = set(a2.obs[comp_col].astype(str).unique()) - {"background"}
            mask &= meta_idx[aws_col].astype(str).isin(comp_set).values
        a_aws = a_aws[mask].copy()
        meta_idx = meta_idx[mask]
    print(f"AWS cells kept: {a_aws.shape[0]}")

    # Build a unified AnnData. AWS X is log2(normalized+1) → un-log to integer-like
    # counts (scVI requires integer-ish count data).
    X_aws_log = a_aws.X.toarray() if hasattr(a_aws.X, "toarray") else np.asarray(a_aws.X)
    X_aws_lin = (np.power(2.0, X_aws_log.astype(np.float32)) - 1.0).astype(np.float32)
    # Round and clip to non-negative integer counts
    X_aws_int = np.maximum(np.round(X_aws_lin).astype(np.int32), 0)

    # phase-2 X is non-integer too — round it
    X2_int = np.maximum(np.round(X2.astype(np.float32)).astype(np.int32), 0)

    # Reindex AWS to phase-2 gene order
    train_genes = list(a2.var_names)
    aws_symbols = a_aws.var["gene_symbol"].astype(str).tolist()
    aws_sym_to_col = {g: i for i, g in enumerate(aws_symbols)}
    X_aws_aligned = np.zeros((X_aws_int.shape[0], len(train_genes)), dtype=np.int32)
    for j, g in enumerate(train_genes):
        col = aws_sym_to_col.get(g)
        if col is not None:
            X_aws_aligned[:, j] = X_aws_int[:, col]

    # Build AnnData
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
    # Filter phase-2 to train+val only (drop 'other' if any)
    keep_p2 = obs_p2["split"].isin(["train", "val"]).values
    X_combined = np.vstack([X2_int[keep_p2], X_aws_aligned]).astype(np.int32)
    obs_combined = pd.concat([obs_p2[keep_p2].reset_index(drop=True),
                              obs_aws.reset_index(drop=True)], ignore_index=True)
    var_combined = pd.DataFrame(index=train_genes)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, var=var_combined)
    print(f"combined adata: {adata.shape}  (train={int((adata.obs.split=='train').sum())} "
          f"val={int((adata.obs.split=='val').sum())})")

    import scvi
    # Mask val labels for semi-supervised training
    val_idx = adata.obs["split"] == "val"
    for lvl in LEVELS_INTERNAL:
        # Make categorical
        adata.obs[lvl] = adata.obs[lvl].astype("category")
        if "UNLABELED" not in adata.obs[lvl].cat.categories:
            adata.obs[lvl] = adata.obs[lvl].cat.add_categories(["UNLABELED"])
        adata.obs.loc[val_idx, lvl] = "UNLABELED"

    # --- 1. Train shared scVI base ---
    scvi.model.SCVI.setup_anndata(adata, batch_key="fov")
    print("training scVI...")
    vae = scvi.model.SCVI(adata, n_layers=args.n_layers, n_latent=args.n_latent)
    vae.train(max_epochs=args.scvi_epochs, early_stopping=True, accelerator="auto")
    print("scVI done")

    # --- 2. Train scANVI per level ---
    scanvi_models: dict[str, object] = {}
    metrics: dict = {"per_level": {}}
    val_X = adata.X[val_idx.values]
    val_obs = adata.obs[val_idx.values].reset_index(drop=True)
    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        # We need a fresh scVI per level because labels_key is on it
        ad_lvl = adata.copy()
        scvi.model.SCVI.setup_anndata(ad_lvl, batch_key="fov")
        v = scvi.model.SCVI(ad_lvl, n_layers=args.n_layers, n_latent=args.n_latent)
        v.train(max_epochs=max(40, args.scvi_epochs // 2),
                early_stopping=True, accelerator="auto")
        m = scvi.model.SCANVI.from_scvi_model(
            v, adata=ad_lvl, labels_key=lvl, unlabeled_category="UNLABELED")
        m.train(max_epochs=args.scanvi_epochs, accelerator="auto")
        scanvi_models[lvl] = m

        # Quick val eval (against the masked-out true labels)
        from sklearn.metrics import adjusted_rand_score
        true_labels = adata.obs[lvl].values  # 'UNLABELED' for val
        # Build a query of just val cells
        ad_val = ad_lvl[val_idx.values].copy()
        pred_labels = m.predict(ad_val)
        gt = obs_combined.loc[val_idx.values, lvl].astype(str).values  # original labels
        # Note: obs_combined still has originals because we copied; but we modified `adata`
        # — the originals live in obs_p2. Recover them:
        original_labels = pd.concat([obs_p2[keep_p2].reset_index(drop=True),
                                     obs_aws.reset_index(drop=True)],
                                    ignore_index=True).loc[val_idx.values, lvl].astype(str).values
        ari = float(adjusted_rand_score(original_labels, pred_labels.astype(str)))
        print(f"  {out_name:<10} cell-ARI={ari:.3f} (n_val={int(val_idx.sum())})")
        metrics["per_level"][out_name] = {"ari_cells": ari}

    # Save bundle. scvi-tools models save via their own save()/load(). Save each
    # one to a subdir, plus a manifest joblib.
    manifest_dir = out_dir / "scanvi_per_level"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for lvl, m in scanvi_models.items():
        m.save(str(manifest_dir / lvl), overwrite=True, save_anndata=False)
    joblib.dump({
        "type": "scanvi",
        "genes": train_genes,
        "levels": list(LEVELS_INTERNAL),
        "manifest_dir": str(manifest_dir),
    }, out_dir / "scanvi_bundle.joblib")
    Path(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}/scanvi_bundle.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
