"""Ensemble trained models via majority vote on SUBMISSION CSVs (not masks).

Works across architectures (Cellpose + StarDist + any future model) because
it votes on the in-cell / background decision per spot — not on the specific
cell ID string, which is arbitrary across independent runs.  When a spot is
"in-cell" per majority, the priority submission's cluster_id string is used.

This is the fix for the subtle bug in `ensemble_infer.py` where majority-voting
directly on `FOV_A_cell_<N>` strings degenerates — different models label the
same physical cell with different integer IDs, so the votes are almost always
unique and the "ensemble" collapses to the first-listed model.

Usage:
    python ensemble_submissions.py \\
        --submissions submission_stardist_v1.csv submission_cyto2_warmup_long.csv \\
        --output submission_ensemble_stardist_plus_cyto2.csv

First-listed submission has priority:
  - Its cluster_id is used when the spot is voted "in-cell".
  - Its cluster_id is used to break ties in the binary vote.

Validate first:
    python ensemble_submissions.py \\
        --submissions submission_stardist_v1.csv submission_cyto2_warmup_long.csv \\
        --val-mode  # scores against spots_train.csv GT on FOVs 036-040
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--submissions", nargs="+", required=True,
                    help="Paths to submission CSVs to ensemble (order = priority)")
parser.add_argument("--output", default="submission_ensemble.csv",
                    help="Output CSV path")
parser.add_argument("--val-mode", action="store_true",
                    help="Score each input + the ensemble on val FOVs 036-040 "
                         "using spots_train.csv GT instead of generating a submission")
parser.add_argument("--data-root",
                    default=os.environ.get("MERFISH_DATA_ROOT",
                                           "/scratch/cg4652/competition"),
                    help="Competition data root. Defaults to $MERFISH_DATA_ROOT "
                         "(Modal mount) or HPC scratch path.")
args = parser.parse_args()


def load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"spot_id", "cluster_id"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns spot_id, cluster_id (got {list(df.columns)})")
    return df[["spot_id", "cluster_id"]].copy()


print(f"Loading {len(args.submissions)} submissions (priority order):")
subs = []
for p in args.submissions:
    df = load_submission(p)
    print(f"  {os.path.basename(p):50s} "
          f"{len(df):,} rows, "
          f"{(df['cluster_id'] != 'background').sum():,} in-cell")
    subs.append(df)

# ── Align on spot_id ──────────────────────────────────────────────────────────
merged = subs[0].rename(columns={"cluster_id": "c0"})
for i, df in enumerate(subs[1:], start=1):
    merged = merged.merge(df.rename(columns={"cluster_id": f"c{i}"}),
                          on="spot_id", how="outer")

# Any missing cluster_id = background (conservative)
for i in range(len(subs)):
    merged[f"c{i}"] = merged[f"c{i}"].fillna("background")

# ── Majority vote: binary in-cell / background ────────────────────────────────
n = len(subs)
in_cell_votes = np.zeros(len(merged), dtype=int)
for i in range(n):
    in_cell_votes += (merged[f"c{i}"].values != "background").astype(int)

in_cell_majority = in_cell_votes >= (n / 2 + 1e-9)  # strict majority; tie=>bg
# When tied at exactly n/2 (even n), defer to priority model's decision:
if n % 2 == 0:
    tied = in_cell_votes == n // 2
    priority_in_cell = merged["c0"].values != "background"
    in_cell_majority = np.where(tied, priority_in_cell, in_cell_majority)

# ── Assemble final labels ─────────────────────────────────────────────────────
# When in-cell: use priority (c0) cluster_id UNLESS c0 said background, in which
# case use the first subsequent submission that said in-cell.  This keeps IDs
# consistent with the strongest single submission wherever possible.
final = np.where(in_cell_majority,
                 merged["c0"].values,
                 "background").astype(object)

# Fix the edge case: spot is voted in-cell but priority said background
needs_fallback = in_cell_majority & (merged["c0"].values == "background")
if needs_fallback.any():
    for i in range(1, n):
        sub_ids = merged[f"c{i}"].values
        take = needs_fallback & (sub_ids != "background")
        final = np.where(take, sub_ids, final)
        needs_fallback = needs_fallback & ~take

out = pd.DataFrame({
    "spot_id":    merged["spot_id"].values,
    "cluster_id": final,
})

# Recover FOV column from priority submission
priority_df = pd.read_csv(args.submissions[0])
if "fov" in priority_df.columns:
    out = out.merge(priority_df[["spot_id", "fov"]], on="spot_id", how="left")
    out = out[["spot_id", "fov", "cluster_id"]]

print(f"\nEnsemble result:")
print(f"  total spots     : {len(out):,}")
print(f"  in-cell spots   : {(out['cluster_id'] != 'background').sum():,}")
print(f"  background      : {(out['cluster_id'] == 'background').sum():,}")
print(f"  unique clusters : {out['cluster_id'].nunique():,}")

# ── Validation mode: score against spots_train on FOVs 036-040 ────────────────
if args.val_mode:
    print("\n== Val-mode scoring against spots_train.csv GT (FOVs 036-040) ==")
    from sklearn.metrics import adjusted_rand_score
    from src.train_cellpose import boundaries_to_mask
    from src.io import load_fov_images

    meta  = pd.read_csv(f"{args.data_root}/reference/fov_metadata.csv").set_index("fov")
    cells = pd.read_csv(f"{args.data_root}/train/ground_truth/cell_boundaries_train.csv",
                         index_col=0)
    spots = pd.read_csv(f"{args.data_root}/train/ground_truth/spots_train.csv")
    val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]

    # Build GT assignments via rasterized polygon lookup (same as train.py val)
    gt_rows = []
    for fov in val_fovs:
        fov_x = meta.loc[fov, "fov_x"]
        fov_y = meta.loc[fov, "fov_y"]
        gt_mask = boundaries_to_mask(cells, fov, fov_x, fov_y)
        fs = spots[spots["fov"] == fov].copy()
        r = np.clip(fs["image_row"].values.astype(int), 0, 2047)
        c = np.clip(fs["image_col"].values.astype(int), 0, 2047)
        gt_ids = gt_mask[r, c]
        fs["gt_cluster_id"] = np.where(gt_ids > 0,
                                        [f"{fov}_cell_{v}" for v in gt_ids],
                                        "background")
        gt_rows.append(fs[["spot_id", "fov", "gt_cluster_id"]])
    gt = pd.concat(gt_rows, ignore_index=True)

    def score_df(sub_df: pd.DataFrame, name: str) -> float:
        m = gt.merge(sub_df, on="spot_id", how="left", suffixes=("", "_pred"))
        m["cluster_id"] = m["cluster_id"].fillna("background")
        per_fov = []
        for fov in val_fovs:
            g = m[m["fov"] == fov]
            if len(g) == 0:
                continue
            per_fov.append(adjusted_rand_score(g["gt_cluster_id"].astype(str),
                                                g["cluster_id"].astype(str)))
        mean = float(np.mean(per_fov))
        per_fov_str = "  ".join(f"{fov}={a:.4f}"
                                 for fov, a in zip(val_fovs, per_fov))
        print(f"  {name:40s} mean={mean:.4f}   {per_fov_str}")
        return mean

    for path in args.submissions:
        df = load_submission(path)
        # Attach fov from original CSV
        orig = pd.read_csv(path)
        if "fov" in orig.columns:
            df = df.merge(orig[["spot_id", "fov"]], on="spot_id", how="left")
        score_df(df, os.path.basename(path))

    score_df(out, "ENSEMBLE")

out.to_csv(args.output, index=False)
print(f"\nSaved {args.output}")
