"""sanity_gate.py — Plan B distribution-drift gate for submissions.

Run after inference, before submitting to Kaggle. Compares predicted
per-class distribution on test FOVs to the training-data prior, plus
in-cell coverage. Emits a green/yellow/red verdict.

Background: today (2026-05-02) two RF submissions tanked from local val
~0.58 → Kaggle 0.49 because RF over-predicted base-rate-dominant classes
on test FOVs that come from a different brain region. The training prior
(35% background, 17% IT-ET Glut, etc.) is FOR THE TRAINING REGION; test
FOVs may have very different mix. Drift > 0.5x training fraction is a
red flag — though we can't prove the test region matches training, the
signal is at least "this classifier inherits training base rates."

Usage:
  .venv/bin/python phase2/scripts/sanity_gate.py phase2/runs/<dir>/submission.csv

Optional comparators:
  --vs phase2/runs/sweep-P-codelab-nuclei_cosine-cp-0.5/submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

LEVELS = ("class", "subclass", "supertype", "cluster")

# Hard thresholds (calibrated against the train+val FOVs 101-160 distribution
# and the Kaggle-known kNN baseline P submission).
TRAIN_BG_FRAC_RANGE = (0.20, 0.50)        # Background in the train h5ad varies 0.21-0.50 per FOV
PER_CLASS_DRIFT_MULT = 0.5                  # |pred_frac - train_frac| / train_frac < this
IN_CELL_COVERAGE_RANGE = (0.18, 0.32)      # GT frac_in_cell on val was 0.21-0.29
CROSS_CLF_DISAGREEMENT_RATIO = 2.0         # |a - b| / max(a, b) flag if pred class differs > this


def _load_train_dist(h5ad_path: Path) -> dict[str, float]:
    a = ad.read_h5ad(h5ad_path)
    counts = a.obs.class_label.value_counts(normalize=True).to_dict()
    return counts


def _check_drift(pred_frac: dict[str, float], train_frac: dict[str, float],
                 mult: float = PER_CLASS_DRIFT_MULT) -> list[tuple[str, float, float, str]]:
    """Return list of (class, train_frac, pred_frac, severity) for each class.
    Severity: 'OK' | 'WARN' | 'FAIL'."""
    rows = []
    all_classes = set(pred_frac) | set(train_frac)
    for cls in sorted(all_classes):
        tf = train_frac.get(cls, 0.0)
        pf = pred_frac.get(cls, 0.0)
        if tf < 0.01 and pf < 0.01:
            severity = "OK"  # both negligible
        elif tf < 0.005:
            severity = "WARN" if pf > 0.02 else "OK"  # appearing where it didn't exist
        else:
            ratio = abs(pf - tf) / tf
            if ratio > mult * 2:
                severity = "FAIL"
            elif ratio > mult:
                severity = "WARN"
            else:
                severity = "OK"
        rows.append((cls, tf, pf, severity))
    return rows


def _gate_submission(sub_path: Path, train_dist: dict[str, float]) -> dict:
    sub = pd.read_csv(sub_path)
    n = len(sub)

    # In-cell coverage
    in_cell = sub["class"] != "background"
    in_cell_frac = float(in_cell.mean())

    # Pred class distribution among in-cell rows
    pred_class_counts = sub.loc[in_cell, "class"].value_counts(normalize=True).to_dict()
    # Background fraction across whole submission
    bg_frac = 1.0 - in_cell_frac

    # Verdicts
    verdicts: list[tuple[str, str, str]] = []  # (check_name, severity, detail)

    # 1. In-cell coverage range
    lo, hi = IN_CELL_COVERAGE_RANGE
    if lo <= in_cell_frac <= hi:
        verdicts.append(("in_cell_coverage", "OK", f"{in_cell_frac:.3f} in [{lo}, {hi}]"))
    else:
        sev = "FAIL" if (in_cell_frac < lo - 0.05 or in_cell_frac > hi + 0.05) else "WARN"
        verdicts.append(("in_cell_coverage", sev,
                         f"{in_cell_frac:.3f} OUTSIDE [{lo}, {hi}]"))

    # 2. Background fraction (segmentation determines this with kNN; classifier can shift it)
    bg_lo, bg_hi = TRAIN_BG_FRAC_RANGE
    expected_bg_max = 1.0 - lo  # max bg frac if min in-cell coverage is met
    if bg_frac > 0.85:
        verdicts.append(("background_frac", "FAIL",
                         f"{bg_frac:.3f} too high - classifier over-predicting background"))
    else:
        verdicts.append(("background_frac", "OK", f"{bg_frac:.3f}"))

    # 3. Per-class drift on in-cell rows
    drift_rows = _check_drift(pred_class_counts, {k: v for k, v in train_dist.items()
                                                  if k != "background"})
    # Renormalize train_dist excluding background for in-cell-only comparison
    bg_train = train_dist.get("background", 0.0)
    train_in_cell_dist = {k: v / (1.0 - bg_train) for k, v in train_dist.items()
                          if k != "background"}
    drift_rows = _check_drift(pred_class_counts, train_in_cell_dist)
    n_fail = sum(1 for *_, s in drift_rows if s == "FAIL")
    n_warn = sum(1 for *_, s in drift_rows if s == "WARN")
    if n_fail:
        verdicts.append(("per_class_drift", "FAIL",
                         f"{n_fail} classes drift > {PER_CLASS_DRIFT_MULT*2:.0%}, "
                         f"{n_warn} drift > {PER_CLASS_DRIFT_MULT:.0%}"))
    elif n_warn:
        verdicts.append(("per_class_drift", "WARN",
                         f"{n_warn} classes drift > {PER_CLASS_DRIFT_MULT:.0%}"))
    else:
        verdicts.append(("per_class_drift", "OK", "all classes within drift tolerance"))

    # Overall verdict
    severities = [v[1] for v in verdicts]
    if "FAIL" in severities:
        overall = "RED"
    elif "WARN" in severities:
        overall = "YELLOW"
    else:
        overall = "GREEN"

    return {
        "path": str(sub_path),
        "n_rows": n,
        "in_cell_frac": in_cell_frac,
        "bg_frac": bg_frac,
        "verdicts": verdicts,
        "drift_table": drift_rows,
        "pred_class_counts": pred_class_counts,
        "train_in_cell_dist": train_in_cell_dist,
        "overall": overall,
    }


def _print_report(report: dict, comparator: dict | None = None) -> None:
    print("=" * 78)
    print(f"  SANITY GATE :: {report['path']}")
    print("=" * 78)
    print(f"  rows           = {report['n_rows']:,}")
    print(f"  in_cell_frac   = {report['in_cell_frac']:.4f}")
    print(f"  bg_frac        = {report['bg_frac']:.4f}")
    print()
    print("  Per-class drift (in-cell rows only, vs training prior):")
    print(f"  {'class':<22} {'train':>8} {'pred':>8} {'verdict':>8}")
    for cls, tf, pf, sev in sorted(report["drift_table"], key=lambda r: -r[1]):
        marker = " " if sev == "OK" else "!" if sev == "WARN" else "X"
        print(f"  {marker} {cls:<20} {tf:>8.3f} {pf:>8.3f} {sev:>8}")
    print()
    print("  Verdicts:")
    for name, sev, detail in report["verdicts"]:
        marker = " " if sev == "OK" else "!" if sev == "WARN" else "X"
        print(f"  {marker} {name:<22} {sev:<5} {detail}")

    if comparator is not None:
        print()
        print(f"  Cross-classifier comparison vs {comparator['path']}:")
        print(f"  {'class':<22} {'this':>8} {'other':>8} {'ratio':>8}")
        all_classes = sorted(set(report["pred_class_counts"]) | set(comparator["pred_class_counts"]))
        for cls in all_classes:
            a = report["pred_class_counts"].get(cls, 0.0)
            b = comparator["pred_class_counts"].get(cls, 0.0)
            if max(a, b) < 0.005:
                continue
            ratio = max(a, b) / max(min(a, b), 1e-6)
            marker = "X" if ratio > CROSS_CLF_DISAGREEMENT_RATIO else " "
            print(f"  {marker} {cls:<20} {a:>8.3f} {b:>8.3f} {ratio:>8.1f}x")

    print()
    color = {"GREEN": "OK", "YELLOW": "REVIEW", "RED": "DO NOT SUBMIT"}[report["overall"]]
    print(f"  OVERALL :: {report['overall']}  ({color})")
    print("=" * 78)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission", help="Path to submission.csv")
    p.add_argument("--vs", default=None,
                   help="Comparator submission CSV (e.g. P baseline). Adds cross-clf table.")
    p.add_argument("--train-h5ad", default="phase2/data/train/ground_truth/counts_train.h5ad")
    args = p.parse_args(argv)

    train_dist = _load_train_dist(Path(args.train_h5ad))
    report = _gate_submission(Path(args.submission), train_dist)

    comparator_report = None
    if args.vs:
        comparator_report = _gate_submission(Path(args.vs), train_dist)

    _print_report(report, comparator_report)
    return 0 if report["overall"] in ("GREEN", "YELLOW") else 1


if __name__ == "__main__":
    sys.exit(main())
