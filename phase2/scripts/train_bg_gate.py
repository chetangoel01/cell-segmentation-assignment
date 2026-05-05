"""Train a binary 'is real cell vs background' gate on competition cells only.

The phase-2 train h5ad has 1,466 cells with class_label='background' (cells where
Allen couldn't confidently transfer a label) and ~3,000 confidently-labeled cells
across 10 real classes. This script trains a binary classifier (RF or logreg)
on (counts → is_real_cell) using only the phase-2 train cells (no AWS — AWS has
zero bg cells, so it'd never see the negative class).

At inference, the gate predicts each segmented cell as either 'real' or
'background'. Cells flagged as background bypass the hierarchical kNN and
get all 4 levels set to 'background'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np


class GateClassifier:
    """Wraps a binary sklearn classifier so predict() returns 'real'/'background'
    string labels above/below a probability threshold. Module-level so joblib
    can pickle it."""
    def __init__(self, inner, threshold: float):
        self.inner = inner
        self.threshold = threshold

    def predict(self, X):
        p = self.inner.predict_proba(X)[:, 1]
        return np.where(p >= self.threshold, "real", "background").astype(object)

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


def _preprocess(X: np.ndarray, mode: str) -> np.ndarray:
    X = X.astype(np.float32)
    if mode == "log1p":
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return np.log1p(X / row_sum * 1e4)
    if mode == "l1_normalize":
        from sklearn.preprocessing import normalize
        return normalize(X, norm="l1")
    raise ValueError(mode)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase2-h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    p.add_argument("--phase2-train-fovs", default=None)
    p.add_argument("--phase2-val-fovs", default=None)
    p.add_argument("--preproc", default="log1p", choices=("log1p", "l1_normalize"))
    p.add_argument("--classifier", default="rf", choices=("rf", "lr"))
    p.add_argument("--rf-n-estimators", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for predicting 'real cell'. "
                        "Lower → more cells flagged as real (less aggressive bg). "
                        "Higher → more cells demoted to bg.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_train = (args.phase2_train_fovs.split(",") if args.phase2_train_fovs
                else [f"FOV_{i:03d}" for i in range(101, 151)])
    p2_val = (args.phase2_val_fovs.split(",") if args.phase2_val_fovs
              else [f"FOV_{i:03d}" for i in range(151, 161)])

    a = ad.read_h5ad(args.phase2_h5ad)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    train_mask = a.obs.fov.isin(p2_train).values
    val_mask = a.obs.fov.isin(p2_val).values
    Xtr = _preprocess(X[train_mask], args.preproc)
    Xva = _preprocess(X[val_mask], args.preproc)
    # Binary label: 'real' (any non-background class) vs 'background'
    ytr = (a.obs.class_label.values[train_mask] != "background").astype(int)
    yva = (a.obs.class_label.values[val_mask] != "background").astype(int)
    print(f"  train: {len(ytr)} cells, real={int(ytr.sum())} bg={int((1-ytr).sum())}")
    print(f"  val:   {len(yva)} cells, real={int(yva.sum())} bg={int((1-yva).sum())}")

    if args.classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1, random_state=0,
                                     class_weight="balanced")
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr, ytr)
    proba_va = clf.predict_proba(Xva)[:, 1]  # P(real)
    pred_va = (proba_va >= args.threshold).astype(int)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(yva, pred_va)
    pr, rc, f1, _ = precision_recall_fscore_support(yva, pred_va, average="binary")
    print(f"  acc={acc:.3f}  precision(real)={pr:.3f}  recall(real)={rc:.3f}  f1={f1:.3f}")
    # Quick threshold sweep print
    print("  threshold sweep (proba >= t → real):")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        p_t = (proba_va >= t).astype(int)
        a_t = accuracy_score(yva, p_t)
        pr_t, rc_t, _, _ = precision_recall_fscore_support(yva, p_t, average="binary")
        n_real = int(p_t.sum())
        print(f"    t={t}: acc={a_t:.3f} prec={pr_t:.3f} rec={rc_t:.3f} n_real={n_real}/{len(p_t)}")

    # Save in the gate bundle format expected by infer_hierarchical.py.
    # The infer script does: gate['clf'].predict(gate_X) and treats anything
    # not equal to 'background' as real.
    bundle = {
        "clf": GateClassifier(clf, args.threshold),
        "preproc": args.preproc,
        "threshold": args.threshold,
        "config": {k: v for k, v in vars(args).items()},
    }
    joblib.dump(bundle, out_dir / "gate.joblib", compress=3)
    Path(out_dir / "metrics.json").write_text(json.dumps({
        "acc": acc, "precision_real": pr, "recall_real": rc, "f1": f1,
    }, indent=2))
    print(f"\n→ {out_dir}/gate.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
