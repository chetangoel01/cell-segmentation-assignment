"""vote_submissions.py — Plan C triangulated voting across N submission CSVs.

Per spot per level: majority vote. Tie-breaker: the FIRST submission listed
(treat it as the "anchor" / baseline). All submissions must have identical
spot_id/fov ordering (which they do when produced by infer_baseline on the
same test_spots.csv).

Usage:
    .venv/bin/python phase2/scripts/vote_submissions.py \\
        --out phase2/runs/voted-submission/submission.csv \\
        phase2/runs/sweep-P-codelab-nuclei_cosine-cp-0.5/submission.csv \\
        phase2/runs/<knn-k15>/submission.csv \\
        phase2/runs/<rf300>/submission.csv

Anchor = first arg. When all N disagree (no majority), anchor wins.
Reports per-level agreement statistics.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ("class", "subclass", "supertype", "cluster")


def _vote_column(arrays: list[np.ndarray], anchor_idx: int = 0) -> tuple[np.ndarray, dict]:
    """Majority vote across N parallel arrays. Returns (voted, stats).

    Tie-breaker: anchor (arrays[anchor_idx]) wins on no-majority rows.
    """
    n = len(arrays[0])
    out = np.empty(n, dtype=object)
    stats = {
        "unanimous": 0,        # all N agree
        "majority": 0,         # >N/2 agree on one value
        "tie_anchor_wins": 0,  # no majority -> anchor used
    }
    threshold = len(arrays) // 2 + 1  # strict majority (e.g. 2/3)
    stacked = np.vstack(arrays)  # (N, n)
    for i in range(n):
        col = stacked[:, i]
        # Fast path: all equal
        if (col == col[0]).all():
            out[i] = col[0]
            stats["unanimous"] += 1
            continue
        c = Counter(col.tolist())
        top_val, top_n = c.most_common(1)[0]
        if top_n >= threshold:
            out[i] = top_val
            stats["majority"] += 1
        else:
            out[i] = arrays[anchor_idx][i]
            stats["tie_anchor_wins"] += 1
    return out, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submissions", nargs="+", help="2+ submission CSVs (first = anchor / tiebreaker)")
    p.add_argument("--out", required=True, help="Output voted submission CSV path")
    args = p.parse_args(argv)

    if len(args.submissions) < 2:
        print("[fatal] need at least 2 submissions to vote")
        return 1

    paths = [Path(s) for s in args.submissions]
    print(f"Loading {len(paths)} submissions:")
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        marker = " (ANCHOR)" if i == 0 else ""
        print(f"  [{i}] {path}{marker}  rows={len(df):,}")
        dfs.append(df)

    # Verify alignment
    base = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        if len(df) != len(base):
            print(f"[fatal] row count mismatch: {paths[0]}={len(base)} vs {paths[i]}={len(df)}")
            return 1
        if not (df.spot_id == base.spot_id).all() or not (df.fov == base.fov).all():
            print(f"[fatal] spot_id or fov misaligned between {paths[0]} and {paths[i]}")
            return 1

    out = base[["spot_id", "fov"]].copy()
    print(f"\nVoting per spot per level:")
    for lvl in LEVELS:
        arrays = [df[lvl].to_numpy().astype(str) for df in dfs]
        voted, stats = _vote_column(arrays, anchor_idx=0)
        out[lvl] = voted
        n = len(voted)
        print(f"  {lvl:<10} unanimous={stats['unanimous']:>7,} ({stats['unanimous']/n:.1%})  "
              f"majority={stats['majority']:>7,} ({stats['majority']/n:.1%})  "
              f"anchor-tiebreak={stats['tie_anchor_wins']:>5,} ({stats['tie_anchor_wins']/n:.1%})")

    # Pairwise agreement diagnostic
    print(f"\nPairwise agreement vs anchor [{paths[0].name}]:")
    print(f"  {'level':<10}", "".join(f" vs[{i}] {paths[i].parent.name[-20:]:<20}" for i in range(1, len(paths))))
    for lvl in LEVELS:
        agreements = []
        for i in range(1, len(paths)):
            agree = (dfs[0][lvl].to_numpy() == dfs[i][lvl].to_numpy()).mean()
            agreements.append(f"          {agree:.4f}                ")
        print(f"  {lvl:<10}", "".join(agreements))

    # In-cell coverage of voted result
    in_cell = (out["class"] != "background").mean()
    print(f"\nVoted in-cell fraction: {in_cell:.4f}")
    for i, df in enumerate(dfs):
        ic = (df["class"] != "background").mean()
        print(f"  vs input [{i}] in-cell: {ic:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out):,} rows -> {out_path}")

    # Structural validation — fail loud if the voted CSV is malformed.
    try:
        from validate_submission import validate as _validate_sub  # local import
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from validate_submission import validate as _validate_sub  # type: ignore
    ok, msg = _validate_sub(out_path)
    print(f"validate_submission: {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        print(f"[fatal] structural validation failed — do NOT submit {out_path}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
