"""Hierarchical MLP classifier on AWS-augmented MERFISH counts (Modal GPU).

Untried approach (XGB + scANVI are also being trained, this adds ensemble
diversity). Top-down conditional MLPs:
    - L1 input  = log1p-CPM expression (1147 features)
    - L2 input  = expression || softmax(L1)
    - L3 input  = expression || softmax(L1) || softmax(L2)
    - L4 input  = expression || softmax(L1) || softmax(L2) || softmax(L3)

Each level is a 3-layer MLP with batch-norm + dropout, trained with
class-balanced cross-entropy on phase-2 + AWS Zhuang-ABCA-4.001. Val on
phase-2 FOVs 151-160 (cell-level ARI for early sanity).

Run:
    modal run --detach phase2/modal/modal_train_mlp_hier.py::run
Bundle path: trained/mlp-hier-aws/mlp_bundle.joblib
"""
from __future__ import annotations

import modal

app = modal.App("phase2-train-mlp-xl")

data_vol = modal.Volume.from_name("cell-seg-phase2", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "scipy",
        "anndata", "scikit-learn", "joblib",
        "torch", "tqdm",
    )
)

VOLUMES = {"/root/data": data_vol}


@app.function(image=image, gpu="A10G", timeout=9000, volumes=VOLUMES)
def train_mlp(
    aws_filter_level: str = "cluster",
    aws_sections: str = "Zhuang-ABCA-4.001",
    upsample_background: int = 40,
    hidden: int = 1024,
    n_layers: int = 5,
    dropout: float = 0.4,
    epochs: int = 800,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    out_subdir: str = "trained/mlp-xl-h1024-d5",
) -> str:
    import json
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import anndata as ad
    import joblib

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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

    # Normalize features to mean 0 / std 1 over training rows
    feat_mean = Xtr.mean(axis=0, keepdims=True).astype(np.float32)
    feat_std = (Xtr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    Xtr = (Xtr - feat_mean) / feat_std
    Xva = (Xva - feat_mean) / feat_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class MLPHead(nn.Module):
        def __init__(self, in_dim: int, hidden: int, n_classes: int, n_hidden: int = 3, p: float = 0.3):
            super().__init__()
            layers: list[nn.Module] = []
            prev = in_dim
            for _ in range(n_hidden):
                layers.append(nn.Linear(prev, hidden))
                layers.append(nn.BatchNorm1d(hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p))
                prev = hidden
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    encoders: dict = {}
    models: dict = {}
    metrics: dict = {"per_level": {}, "config": {
        "hidden": hidden, "n_layers": n_layers, "dropout": dropout,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "wd": weight_decay,
        "upsample_bg": upsample_background, "aws_sections": aws_sections,
    }}

    prev_train_logits: torch.Tensor | None = None
    prev_val_logits: torch.Tensor | None = None

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)

    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        y_str = y_train[lvl].astype(str)
        le = LabelEncoder().fit(y_str)
        y_idx = le.transform(y_str)
        y_idx_t = torch.from_numpy(y_idx).long().to(device)
        n_classes = len(le.classes_)

        in_dim = Xtr.shape[1] + (0 if prev_train_logits is None else prev_train_logits.shape[1])
        feats_tr = Xtr_t if prev_train_logits is None else torch.cat([Xtr_t, prev_train_logits], dim=1)
        feats_va = Xva_t if prev_val_logits is None else torch.cat([Xva_t, prev_val_logits], dim=1)

        # class-balanced weights
        counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
        weights = (counts.sum() / np.maximum(counts, 1)).clip(max=50.0)
        weights = weights / weights.mean()
        weight_t = torch.from_numpy(weights).float().to(device)

        model = MLPHead(in_dim, hidden, n_classes, n_hidden=n_layers, p=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ce = nn.CrossEntropyLoss(weight=weight_t)

        n = feats_tr.shape[0]
        t0 = time.time()
        best_val_ari = -1.0
        best_state = None
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            tot = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits = model(feats_tr[idx])
                loss = ce(logits, y_idx_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * idx.numel()
            sched.step()

            # quick val ARI every 20 epochs (cell-level)
            if (ep + 1) % 20 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    pv = model(feats_va).argmax(dim=1).cpu().numpy()
                pred_str = le.inverse_transform(pv)
                ari = float(adjusted_rand_score(yv[lvl], pred_str))
                if ari > best_val_ari:
                    best_val_ari = ari
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f"  {out_name} ep{ep + 1:3d} loss={tot / n:.3f} val-ARI={ari:+.3f} best={best_val_ari:+.3f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        # Compute "soft" prev logits for next level (use training & val features)
        model.eval()
        with torch.no_grad():
            prev_train_logits = F.softmax(model(feats_tr), dim=1)
            prev_val_logits = F.softmax(model(feats_va), dim=1)

        encoders[lvl] = le
        models[lvl] = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "in_dim": in_dim,
            "n_classes": n_classes,
            "hidden": hidden,
            "n_hidden": n_layers,
            "dropout": dropout,
        }
        elapsed = time.time() - t0
        metrics["per_level"][out_name] = {"ari_cells": best_val_ari, "n_classes": n_classes, "secs": elapsed}
        print(f"  done {out_name}: best val-ARI={best_val_ari:+.3f} ({elapsed:.0f}s)")

    bundle = {
        "type": "mlp_hier_modal",
        "models": models,
        "encoders": encoders,
        "genes": train_genes,
        "preproc": "log1p_zscore",
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "feature_dim": Xtr.shape[1],
        "levels": list(LEVELS_INTERNAL),
    }
    bundle_path = OUT / "mlp_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    data_vol.commit()
    print(f"\n→ {bundle_path}")
    return str(bundle_path)


@app.local_entrypoint()
def run():
    h = train_mlp.spawn()
    print(f"spawned mlp-hier: handle={h.object_id}")
    path = h.get()
    print(f"trained: {path}")
