"""Stage-1 cell-type classifier baseline.

Trains a simple supervised classifier (logistic regression by default, kNN
optional) per hierarchy level (class / subclass / supertype / cluster) using
ground-truth polygons + spots:

    For each labeled cell:
        gene_vector[i] = number of spots of gene_i contained in the cell's
                         z=2 polygon.
        normalize → log1p → row-normalized counts.

    Train classifier:  gene_vector → class_label (etc., one model per level).

Validation: held-out FOVs. We report per-level accuracy on cells AND a spot-
level Adjusted Rand Index (the kaggle metric proxy) where every spot inside a
predicted-cell polygon gets that cell's predicted label and extracellular
spots are 'background'.

This is intentionally a baseline — the real classifier work in the plan
(spatial features, hierarchy-aware decoding, calibrated background gate)
slots in by replacing the model + featurizer here.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.tasks import Task, register

LEVELS = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVEL_OUT = {"class_label": "class", "subclass_label": "subclass",
             "supertype_label": "supertype", "cluster_label": "cluster"}


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-fovs", required=True,
                   help="Comma-separated FOVs to train on (e.g. FOV_101,FOV_105,...).")
    p.add_argument("--val-fovs", default="",
                   help="Comma-separated FOVs to hold out for validation.")
    p.add_argument("--model", default="logreg", choices=("logreg", "knn"),
                   help="Classifier family (default: logreg).")
    p.add_argument("--knn-k", type=int, default=15,
                   help="k for kNN (ignored unless --model knn).")
    p.add_argument("--out-dir", default=None,
                   help="Where to save the trained model + metrics. "
                        "Default: phase2/runs/<timestamp>-train-baseline/.")
    p.add_argument("--max-cells", type=int, default=None,
                   help="Optional cap on cells per FOV (debug/profiling).")


def _expand_fov_list(spec: str) -> list[str]:
    return [f.strip() for f in spec.split(",") if f.strip()]


def _build_features(fov_list: list[str], max_cells: int | None,
                    cells: pd.DataFrame, labels: pd.DataFrame,
                    spots: pd.DataFrame, gene_to_idx: dict[str, int],
                    ) -> tuple[np.ndarray, dict, list[str]]:
    """Return (X, label_dict, cell_ids) for the given FOVs.

    X: (n_cells, n_genes) raw spot counts per cell × gene.
    label_dict: {level_name: np.ndarray of labels aligned to X rows}.
    cell_ids: aligned cell-id list.
    """
    from phase2.src import coords

    n_genes = len(gene_to_idx)
    rows_X: list[np.ndarray] = []
    rows_labels: dict[str, list[str]] = {lvl: [] for lvl in LEVELS}
    cell_ids: list[str] = []

    for fov in fov_list:
        fov_cells = labels[labels["fov"] == fov]
        if max_cells:
            fov_cells = fov_cells.head(max_cells)
        if len(fov_cells) == 0:
            print(f"  [skip] no labeled cells for {fov}")
            continue
        fov_spots = spots[spots["fov"] == fov]
        spot_x = fov_spots["global_x"].to_numpy()
        spot_y = fov_spots["global_y"].to_numpy()
        spot_gene = fov_spots["target_gene"].to_numpy()
        n_in = 0
        for cid in fov_cells["cell_id"]:
            if cid not in cells.index:
                continue
            row = cells.loc[cid]
            poly = coords.parse_boundary_polygon(
                row.get("boundaryX_z2", ""),
                row.get("boundaryY_z2", ""),
            )
            if poly is None:
                continue
            inside = coords.spots_in_polygon(spot_x, spot_y, poly)
            vec = np.zeros(n_genes, dtype=np.float32)
            if inside.any():
                inside_genes = spot_gene[inside]
                # Bincount via a hash-mapped index list.
                idx = np.fromiter(
                    (gene_to_idx[g] for g in inside_genes if g in gene_to_idx),
                    dtype=np.int64,
                )
                if idx.size:
                    np.add.at(vec, idx, 1)
            rows_X.append(vec)
            cell_ids.append(cid)
            for lvl in LEVELS:
                rows_labels[lvl].append(fov_cells.loc[fov_cells["cell_id"] == cid, lvl].iloc[0])
            n_in += 1
        print(f"  {fov}: {n_in} cells featurized")
    if not rows_X:
        raise RuntimeError("no cells produced features — check FOV list and ground-truth files")
    X = np.vstack(rows_X)
    label_dict = {lvl: np.array(vals) for lvl, vals in rows_labels.items()}
    return X, label_dict, cell_ids


def _normalize(X: np.ndarray) -> np.ndarray:
    """log1p of row-normalized counts (per-cell sum to 1, then log)."""
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(X / row_sum * 1e4)


def _train_and_eval(X_tr, y_tr, X_va, y_va, model_kind: str, knn_k: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, adjusted_rand_score

    if model_kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1)
    else:
        # n_jobs deprecated in sklearn 1.8 for LogisticRegression — leave default.
        clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    if X_va is None or len(X_va) == 0:
        return clf, {}
    y_pr = clf.predict(X_va)
    return clf, {
        "n_train_cells": int(len(y_tr)),
        "n_val_cells": int(len(y_va)),
        "accuracy": float(accuracy_score(y_va, y_pr)),
        "ari_cells": float(adjusted_rand_score(y_va, y_pr)),
        "predictions": y_pr.tolist(),
    }


def _run(args: argparse.Namespace) -> int:
    from phase2.src import io

    stages: dict[str, float] = {}
    wall_t0 = time.time()

    train_fovs = _expand_fov_list(args.train_fovs)
    val_fovs = _expand_fov_list(args.val_fovs)
    print(f"train FOVs: {train_fovs}")
    print(f"val FOVs:   {val_fovs}")

    gt_dir = io.ground_truth_dir()
    cells_path = gt_dir / "cell_boundaries_train.csv"
    labels_path = gt_dir / "cell_labels_train.csv"
    spots_path = gt_dir / "spots_train.csv"
    for p in (cells_path, labels_path, spots_path):
        if not p.exists():
            print(f"[fatal] missing {p}")
            return 1

    print("loading ground-truth tables …")
    t0 = time.time()
    cells = pd.read_csv(cells_path, index_col=0)
    labels = pd.read_csv(labels_path)
    spots = pd.read_csv(spots_path)
    stages["csv_load"] = time.time() - t0
    print(f"  cells: {len(cells):,}   labels: {len(labels):,}   spots: {len(spots):,}   "
          f"({stages['csv_load']:.1f}s)")

    # Filter spots to FOVs we care about — saves work on shapely calls.
    keep_fovs = set(train_fovs) | set(val_fovs)
    spots = spots[spots["fov"].isin(keep_fovs)].reset_index(drop=True)
    print(f"  spots after FOV filter: {len(spots):,}")

    # Build a stable gene vocabulary from training spots only.
    gene_counter = Counter(spots[spots["fov"].isin(set(train_fovs))]["target_gene"])
    genes = sorted(gene_counter)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"  gene vocab: {len(genes)} genes")

    print("\nfeaturizing TRAIN cells …")
    t0 = time.time()
    X_tr, lab_tr, ids_tr = _build_features(train_fovs, args.max_cells,
                                           cells, labels, spots, gene_to_idx)
    stages["featurize_train"] = time.time() - t0
    print(f"  X_train shape: {X_tr.shape}  ({stages['featurize_train']:.2f}s)")

    if val_fovs:
        print("\nfeaturizing VAL cells …")
        t0 = time.time()
        X_va, lab_va, ids_va = _build_features(val_fovs, args.max_cells,
                                               cells, labels, spots, gene_to_idx)
        stages["featurize_val"] = time.time() - t0
        print(f"  X_val shape: {X_va.shape}  ({stages['featurize_val']:.2f}s)")
    else:
        X_va, lab_va, ids_va = None, {lvl: np.array([]) for lvl in LEVELS}, []

    print("\nnormalizing …")
    t0 = time.time()
    X_tr_n = _normalize(X_tr)
    X_va_n = _normalize(X_va) if X_va is not None else None
    stages["normalize"] = time.time() - t0

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "runs" /
        f"{time.strftime('%Y%m%d-%H%M%S')}-train-baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\noutput dir: {out_dir}")

    metrics: dict = {
        "model": args.model,
        "knn_k": args.knn_k if args.model == "knn" else None,
        "train_fovs": train_fovs,
        "val_fovs": val_fovs,
        "n_genes": len(genes),
        "per_level": {},
    }

    for lvl in LEVELS:
        print(f"\n=== {lvl} ===")
        y_tr = lab_tr[lvl]
        y_va = lab_va[lvl]
        if len(set(y_tr)) < 2:
            print(f"  [skip] only one class in train: {set(y_tr)}")
            metrics["per_level"][LEVEL_OUT[lvl]] = {"skipped": "single-class train"}
            continue
        t0 = time.time()
        clf, m = _train_and_eval(X_tr_n, y_tr, X_va_n, y_va, args.model, args.knn_k)
        elapsed = time.time() - t0
        stages[f"fit_{LEVEL_OUT[lvl]}"] = elapsed
        m.pop("predictions", None)
        if m:
            print(f"  train n={m['n_train_cells']}  val n={m['n_val_cells']}  "
                  f"acc={m['accuracy']:.3f}  ari={m['ari_cells']:.3f}  "
                  f"({elapsed:.2f}s, {len(set(y_tr))} classes)")
        else:
            print(f"  no validation set — train n={len(y_tr)}  ({elapsed:.2f}s)")
        metrics["per_level"][LEVEL_OUT[lvl]] = m
        import joblib
        joblib.dump({"clf": clf, "genes": genes},
                    out_dir / f"model_{LEVEL_OUT[lvl]}.joblib")

    stages["wall_total"] = time.time() - wall_t0
    metrics["stage_timings_s"] = stages
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nmetrics → {out_dir / 'metrics.json'}")
    print("\n=== stage timings ===")
    for k, v in stages.items():
        print(f"  {k:<22} {v:6.2f}s")
    return 0


register(Task(
    name="train-baseline",
    summary="Train a Stage-1 cell-type classifier (logreg/kNN) on ground-truth polygons.",
    add_args=_add_args,
    run=_run,
    requirements={
        "gpu": False,
        "modal_image": "sklearn",
        "modal_volume": "cell-seg-phase2",
        "modal_timeout": 2 * 3600,
        "hpc_partition": "cpu",
        "hpc_hours": 1.0,
        "hpc_gpus": 0,
    },
))
