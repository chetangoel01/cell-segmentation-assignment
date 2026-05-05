"""Codelab-recipe classifier baseline.

Trains kNN(k=5, metric='cosine') on L1-normalized counts directly from
counts_train.h5ad — matches the codelab and the Spatial-ID-style featurization
used by the reference pipeline.

Differences from phase2/tasks/train_baseline.py:
  - Reads counts_train.h5ad (Allen-aligned cell × gene matrix already there)
    instead of re-deriving counts from polygons + spots.
  - Uses k=5 cosine kNN, not logreg or k=25 Euclidean.
  - L1-normalize, no log transform.

Optionally combines phase-1 (FOVs 001-035) + phase-2 (FOVs 101-150) cells when
--include-phase1 is passed (after intersecting on shared genes).

Optionally stacks Allen Brain Cell Atlas (AWS Zhuang-ABCA-4) reference cells
when --include-aws is passed. Section .001 is the byte-equal source for the
competition (32,528 cells); .002/.003 are the same animal/panel and can be
opted in via --aws-sections. AWS X is `log2(normalized+1)` so we un-log
(`2^X - 1`) and then run the same `log1p(X/sum*1e4)` preprocessing as the
competition cells, putting both halves in a comparable feature space.

Output: a directory with the same shape as our existing baseline_dirs:
  model_class.joblib, model_subclass.joblib, model_supertype.joblib, model_cluster.joblib
each containing {"clf": fitted_kNN, "genes": gene_list_in_order}.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase2-h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--phase1-h5ad", default="phase1/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--include-phase1", action="store_true",
                   help="Stack phase-1 cells onto phase-2 (intersect on shared genes).")
    p.add_argument("--include-aws", action="store_true",
                   help="Stack Allen Brain Cell Atlas (AWS Zhuang-ABCA-4) reference "
                        "cells onto the train set. Section .001 = competition source.")
    p.add_argument("--aws-h5ad",
                   default="phase2/data/external/aws/Zhuang-ABCA-4-log2.h5ad")
    p.add_argument("--aws-metadata",
                   default="phase2/data/external/aws/cell_metadata_with_cluster_annotation.csv")
    p.add_argument("--aws-sections", default="Zhuang-ABCA-4.001",
                   help="Comma-separated brain_section_label values to keep. "
                        "Default = competition source only. Add .002/.003 for "
                        "same-animal cross-section augmentation.")
    p.add_argument("--aws-keep-classes-not-in-comp", action="store_true",
                   help="By default we drop AWS cells whose class is absent from "
                        "the phase-2 competition train set, since predictions on "
                        "those classes are guaranteed-wrong on test FOVs. Pass "
                        "this flag to keep them.")
    p.add_argument("--aws-filter-level", default="cluster",
                   choices=("class", "subclass", "supertype", "cluster", "none"),
                   help="Stricter filter: only keep AWS cells whose label at this "
                        "level (and all coarser levels) exists in competition "
                        "train. 'cluster' is strictest — guarantees no AWS-only "
                        "label strings leak into kNN predictions. 'none' disables "
                        "(use --aws-keep-classes-not-in-comp instead).")
    p.add_argument("--upsample-background", type=int, default=0,
                   help="Replicate competition 'background' train cells N extra "
                        "times. Counters AWS overwhelming bg in cosine kNN. "
                        "Recommend ~20 when --include-aws.")
    p.add_argument("--train-fov-prefix", default="FOV_",
                   help="(diagnostic only)")
    p.add_argument("--phase2-train-fovs", default=None,
                   help="Comma-separated phase-2 FOVs to train on. Default: 101-150.")
    p.add_argument("--phase2-val-fovs", default=None,
                   help="Comma-separated phase-2 FOVs to val on. Default: 151-160.")
    p.add_argument("--phase1-train-fovs", default=None,
                   help="Comma-separated phase-1 FOVs to train on. Default: 001-035 if --include-phase1.")
    p.add_argument("--knn-k", type=int, default=5,
                   help="k for kNN. Codelab uses 5.")
    p.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    p.add_argument("--weights", default="uniform", choices=("uniform", "distance"))
    p.add_argument("--classifier", default="knn", choices=("knn", "rf", "hgb", "et", "vote_rf_et", "lr", "vote_rf_lr", "mlp"))
    p.add_argument("--lr-c", type=float, default=1.0)
    p.add_argument("--mlp-hidden", type=int, default=128, help="MLP hidden layer width")
    p.add_argument("--mlp-alpha", type=float, default=1e-3, help="MLP L2 regularization")
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-class-weight", default=None, choices=(None, "balanced", "balanced_subsample"))
    p.add_argument("--rf-max-features", default="sqrt",
                   help="RF max_features: 'sqrt', 'log2', or a float fraction.")
    p.add_argument("--rf-min-samples-leaf", type=int, default=1)
    p.add_argument("--rf-criterion", default="gini", choices=("gini", "entropy", "log_loss"))
    p.add_argument("--pca-components", type=int, default=0,
                   help="If >0, prepend PCA(n_components) to the classifier as a Pipeline.")
    p.add_argument("--preproc", default="l1_normalize", choices=("l1_normalize", "log1p"))
    p.add_argument("--drop-background", action="store_true",
                   help="Exclude cells with class_label=='background' from train AND val.")
    p.add_argument("--hgb-max-iter", type=int, default=200)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_train = (args.phase2_train_fovs.split(",") if args.phase2_train_fovs
                else [f"FOV_{i:03d}" for i in range(101, 151)])
    p2_val = (args.phase2_val_fovs.split(",") if args.phase2_val_fovs
              else [f"FOV_{i:03d}" for i in range(151, 161)])
    p1_train = (args.phase1_train_fovs.split(",") if args.phase1_train_fovs
                else [f"FOV_{i:03d}" for i in range(1, 36)])

    print(f"phase-2 h5ad: {args.phase2_h5ad}")
    a2, X2 = _load_h5ad(Path(args.phase2_h5ad))
    print(f"  shape: {a2.shape}, FOVs: {a2.obs.fov.nunique()}")
    print(f"  classes/subclasses/supertypes/clusters: "
          f"{a2.obs.class_label.nunique()}/{a2.obs.subclass_label.nunique()}/"
          f"{a2.obs.supertype_label.nunique()}/{a2.obs.cluster_label.nunique()}")

    train_mask2 = a2.obs.fov.isin(p2_train).values
    val_mask2 = a2.obs.fov.isin(p2_val).values
    if args.drop_background:
        not_bg = (a2.obs.class_label.values != "background")
        n_dropped_train = int(train_mask2.sum() - (train_mask2 & not_bg).sum())
        n_dropped_val = int(val_mask2.sum() - (val_mask2 & not_bg).sum())
        train_mask2 = train_mask2 & not_bg
        val_mask2 = val_mask2 & not_bg
        print(f"  --drop-background: removed {n_dropped_train} train + {n_dropped_val} val cells")
    print(f"  phase-2 train cells: {train_mask2.sum()}  val cells: {val_mask2.sum()}")

    X_train_parts = [X2[train_mask2]]
    y_train_parts = {lvl: [a2.obs[lvl].values[train_mask2]] for lvl in LEVELS_INTERNAL}
    train_genes = list(a2.var_names)

    if args.include_phase1:
        p1_path = Path(args.phase1_h5ad)
        if not p1_path.exists():
            print(f"[fatal] --include-phase1 set but {p1_path} not found")
            return 1
        print(f"phase-1 h5ad: {p1_path}")
        a1, X1 = _load_h5ad(p1_path)
        print(f"  shape: {a1.shape}, FOVs: {a1.obs.fov.nunique()}")
        train_mask1 = a1.obs.fov.isin(p1_train).values
        print(f"  phase-1 train cells: {train_mask1.sum()}")

        # Intersect gene vocab — re-index phase-1 to phase-2 genes (drop missing).
        p1_genes = list(a1.var_names)
        shared = [g for g in train_genes if g in set(p1_genes)]
        print(f"  shared genes: {len(shared)}/{len(train_genes)} (phase-2) vs {len(p1_genes)} (phase-1)")
        # Reindex both to shared gene order
        p2_idx = [train_genes.index(g) for g in shared]
        p1_idx = [p1_genes.index(g) for g in shared]
        # Replace train (phase-2 reindexed) and append phase-1
        X_train_parts = [X2[train_mask2][:, p2_idx], X1[train_mask1][:, p1_idx]]
        for lvl in LEVELS_INTERNAL:
            y_train_parts[lvl].append(a1.obs[lvl].values[train_mask1])
        # Also reindex val to shared genes
        X_val = X2[val_mask2][:, p2_idx]
        train_genes = shared
    else:
        X_val = X2[val_mask2]

    aws_n_used = 0
    if args.include_aws:
        aws_h5ad_path = Path(args.aws_h5ad)
        aws_meta_path = Path(args.aws_metadata)
        if not aws_h5ad_path.exists() or not aws_meta_path.exists():
            print(f"[fatal] --include-aws set but {aws_h5ad_path} or {aws_meta_path} missing")
            return 1
        sections = [s.strip() for s in args.aws_sections.split(",") if s.strip()]
        print(f"AWS h5ad: {aws_h5ad_path}  sections={sections}")
        a_aws = ad.read_h5ad(aws_h5ad_path)
        meta = pd.read_csv(aws_meta_path)
        meta = meta[meta["brain_section_label"].isin(sections)].copy()
        # Strings — cell_label join key is the obs index of the h5ad and the
        # `cell_label` column of the metadata CSV. Verified byte-equal in
        # phase2_extra_data memory.
        meta["cell_label"] = meta["cell_label"].astype(str)
        # Filter h5ad to cells present in metadata (drops unlabeled obs)
        keep_mask = a_aws.obs.index.astype(str).isin(set(meta["cell_label"]))
        a_aws_kept = a_aws[keep_mask].copy()
        # Order metadata to match h5ad obs order
        meta_indexed = meta.set_index("cell_label").loc[a_aws_kept.obs.index.astype(str)]
        # Optionally drop classes not in competition train (predictions on those
        # classes are guaranteed-wrong on the held-out test FOVs).
        if args.aws_filter_level != "none":
            level_pairs = [("class", "class_label"),
                           ("subclass", "subclass_label"),
                           ("supertype", "supertype_label"),
                           ("cluster", "cluster_label")]
            level_idx = next(i for i, (k, _) in enumerate(level_pairs) if k == args.aws_filter_level)
            mask = np.ones(len(meta_indexed), dtype=bool)
            for aws_col, comp_col in level_pairs[: level_idx + 1]:
                comp_set = set(a2.obs[comp_col].astype(str).unique()) - {"background"}
                mask &= meta_indexed[aws_col].astype(str).isin(comp_set).values
            n_dropped = int((~mask).sum())
            print(f"  filter='{args.aws_filter_level}': dropping {n_dropped} AWS cells with labels not in comp")
            a_aws_kept = a_aws_kept[mask].copy()
            meta_indexed = meta_indexed[mask]
        elif not args.aws_keep_classes_not_in_comp:
            comp_classes = set(a2.obs["class_label"].unique()) - {"background"}
            class_mask = meta_indexed["class"].isin(comp_classes).values
            n_dropped = int((~class_mask).sum())
            print(f"  dropping {n_dropped} AWS cells with class not in comp")
            a_aws_kept = a_aws_kept[class_mask].copy()
            meta_indexed = meta_indexed[class_mask]
        print(f"  AWS cells kept: {a_aws_kept.shape[0]}")

        # AWS X is log2(normalized+1) — un-log to get linear-scale feature
        # values comparable to the (linear) competition counts before the shared
        # log1p preprocessing applies.
        X_aws_log = a_aws_kept.X.toarray() if hasattr(a_aws_kept.X, "toarray") else np.asarray(a_aws_kept.X)
        X_aws_lin = (np.power(2.0, X_aws_log.astype(np.float32)) - 1.0).astype(np.float32)

        # Reindex AWS genes (Ensembl-id index, gene_symbol column) to current
        # train_genes order. Genes not present in AWS get a zero column.
        aws_symbols = a_aws_kept.var["gene_symbol"].astype(str).tolist()
        aws_sym_to_col = {g: i for i, g in enumerate(aws_symbols)}
        present = [g for g in train_genes if g in aws_sym_to_col]
        missing = [g for g in train_genes if g not in aws_sym_to_col]
        print(f"  AWS gene coverage: {len(present)}/{len(train_genes)}  missing={len(missing)}")
        # Build AWS X in train_genes column order
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
            y_train_parts[lvl].append(meta_indexed[src_col].astype(str).values)
        aws_n_used = X_aws_aligned.shape[0]

    # Background upsampling (applied AFTER phase-1/AWS stacking so it operates
    # on the already-stacked, gene-aligned phase-2 portion). Replicates the
    # competition 'background' class cells N extra times to counteract AWS
    # cells (which contain zero 'background' cells) overwhelming the kNN vote.
    if args.upsample_background > 0:
        # phase-2 train cells are always X_train_parts[0] (after optional
        # phase-1 reindex, X_train_parts[0] is the reindexed phase-2 half).
        p2_part = X_train_parts[0]
        # Recover the class labels matching this part (also always the first
        # in y_train_parts).
        p2_labels = y_train_parts["class_label"][0]
        bg_mask = (p2_labels == "background")
        n_bg = int(bg_mask.sum())
        if n_bg > 0:
            print(f"  upsampling background: replicating {n_bg} bg cells × {args.upsample_background}")
            X_train_parts.append(np.tile(p2_part[bg_mask], (args.upsample_background, 1)))
            for lvl in LEVELS_INTERNAL:
                y_train_parts[lvl].append(
                    np.tile(y_train_parts[lvl][0][bg_mask], args.upsample_background)
                )

    # Stack all training data
    X_train = np.vstack(X_train_parts).astype(np.float32)
    y_train = {lvl: np.concatenate(parts) for lvl, parts in y_train_parts.items()}
    y_val = {lvl: a2.obs[lvl].values[val_mask2] for lvl in LEVELS_INTERNAL}
    print(f"final training: {X_train.shape}  val: {X_val.shape}  aws_used={aws_n_used}")

    # Preprocessing (must match validate_local.py inference path)
    if args.preproc == "log1p":
        def _l1p(X):
            X = X.astype(np.float32)
            row_sum = X.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            return np.log1p(X / row_sum * 1e4)
        Xtr = _l1p(X_train)
        Xva = _l1p(X_val)
    else:
        from sklearn.preprocessing import normalize
        Xtr = normalize(X_train, norm="l1")
        Xva = normalize(X_val.astype(np.float32), norm="l1")

    # Train + eval per level
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, adjusted_rand_score

    metrics = {"k": args.knn_k, "metric": args.metric,
               "include_phase1": args.include_phase1,
               "include_aws": args.include_aws,
               "aws_sections": args.aws_sections if args.include_aws else None,
               "aws_n_cells": aws_n_used,
               "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
               "n_genes": len(train_genes), "per_level": {}}
    for lvl in LEVELS_INTERNAL:
        out_name = LEVELS_OUT[lvl]
        ytr = y_train[lvl]
        yva = y_val[lvl]
        if len(set(ytr)) < 2:
            print(f"  [skip] {lvl} only has one class")
            continue
        t0 = time.time()
        try:
            mf = float(args.rf_max_features)
        except ValueError:
            mf = args.rf_max_features
        if args.classifier == "rf":
            base = RandomForestClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1,
                                          random_state=0, class_weight=args.rf_class_weight,
                                          max_features=mf, min_samples_leaf=args.rf_min_samples_leaf,
                                          criterion=args.rf_criterion)
        elif args.classifier == "et":
            base = ExtraTreesClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1,
                                        random_state=0, class_weight=args.rf_class_weight,
                                        max_features=mf)
        elif args.classifier == "hgb":
            base = HistGradientBoostingClassifier(max_iter=args.hgb_max_iter, random_state=0)
        elif args.classifier == "vote_rf_et":
            from sklearn.ensemble import VotingClassifier
            base = VotingClassifier(estimators=[
                ("rf", RandomForestClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1, random_state=0,
                                              class_weight=args.rf_class_weight, max_features=mf)),
                ("et", ExtraTreesClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1, random_state=1,
                                            class_weight=args.rf_class_weight, max_features=mf)),
            ], voting="soft", n_jobs=1)
        elif args.classifier == "lr":
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(C=args.lr_c, max_iter=2000, n_jobs=-1)
        elif args.classifier == "mlp":
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            base = Pipeline([
                ("scale", StandardScaler(with_mean=True)),
                ("mlp", MLPClassifier(hidden_layer_sizes=(args.mlp_hidden,),
                                      alpha=args.mlp_alpha,
                                      max_iter=100,
                                      random_state=0)),
            ])
        elif args.classifier == "vote_rf_lr":
            from sklearn.ensemble import VotingClassifier
            from sklearn.linear_model import LogisticRegression
            base = VotingClassifier(estimators=[
                ("rf", RandomForestClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1, random_state=0,
                                              class_weight=args.rf_class_weight)),
                ("lr", LogisticRegression(C=args.lr_c, max_iter=2000, n_jobs=-1)),
            ], voting="soft", n_jobs=1)
        else:
            base = KNeighborsClassifier(n_neighbors=args.knn_k, metric=args.metric, weights=args.weights)
        if args.pca_components > 0:
            from sklearn.decomposition import PCA
            from sklearn.pipeline import Pipeline
            clf = Pipeline([("pca", PCA(n_components=args.pca_components, random_state=0)), ("clf", base)])
        else:
            clf = base
        clf.fit(Xtr, ytr)
        ypr = clf.predict(Xva)
        elapsed = time.time() - t0
        acc = accuracy_score(yva, ypr)
        ari = adjusted_rand_score(yva, ypr)
        print(f"  {out_name:<10} acc={acc:.3f}  cell-ARI={ari:.3f}  ({len(set(ytr))} classes, {elapsed:.1f}s)")
        metrics["per_level"][out_name] = {
            "accuracy": float(acc), "ari_cells": float(ari),
            "n_train_cells": int(len(ytr)), "n_val_cells": int(len(yva)),
            "n_classes": int(len(set(ytr))),
        }
        joblib.dump({"clf": clf, "genes": train_genes,
                     "preproc": args.preproc, "k": args.knn_k, "metric": args.metric},
                    out_dir / f"model_{out_name}.joblib")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ {out_dir}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
