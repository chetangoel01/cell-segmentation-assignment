"""Majority-vote merge multiple submission CSVs into one ensemble submission.

Each input CSV has columns: spot_id, fov, cluster_id.
For each spot, pick the most-common non-background cluster_id across inputs;
fall back to "background" only if every model said background (or there is a tie
with background winning on raw count).

Usage:
    python merge_submissions.py \
        submission_stardist_v3.csv submission_cyto2_warmup_modal.csv \
        --out submission_ensemble.csv

Tie-breaking favours non-background labels so a single confident vote still wins
over several "background" votes — cells are rare in pixel space so the
background class dominates and would otherwise drown out the segmenters.
"""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("csvs", nargs="+", help="Input submission CSVs to merge")
parser.add_argument("--out", default="submission_ensemble.csv")
args = parser.parse_args()

dfs = [pd.read_csv(p) for p in args.csvs]
for p, df in zip(args.csvs, dfs):
    print(f"{p}: {len(df):,} rows, {df['cluster_id'].nunique()} unique labels, "
          f"{(df['cluster_id'] == 'background').mean():.1%} background")

# Align all frames on spot_id order from the first CSV (they should already match).
base = dfs[0][["spot_id", "fov"]].copy()
labels = pd.DataFrame({args.csvs[i]: df["cluster_id"].values for i, df in enumerate(dfs)})

def vote(row):
    counts = Counter(row)
    # Prefer the most common non-background if any model picked a cell.
    non_bg = [(k, v) for k, v in counts.items() if k != "background"]
    if non_bg:
        return max(non_bg, key=lambda kv: kv[1])[0]
    return "background"

base["cluster_id"] = labels.apply(vote, axis=1)
base.to_csv(args.out, index=False)

print(f"\nWrote {args.out}: {len(base):,} rows")
print(f"  Background: {(base['cluster_id'] == 'background').mean():.1%}")
print(f"  Unique labels: {base['cluster_id'].nunique()}")
