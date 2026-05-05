"""Tuple-based plurality voting — vote on the FULL (class, subclass, supertype, cluster)
4-tuple per spot, instead of voting each level independently.

Compared to vote_submissions.py:
- Independent: votes each level, can produce taxonomically inconsistent combinations
- Tuple: votes full label combinations, taxonomy is preserved by construction

When voters disagree, the highest-vote tuple wins. Tie-breaker: anchor's tuple.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ("class", "subclass", "supertype", "cluster")


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

    base = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        if len(df) != len(base):
            print(f"[fatal] row count mismatch")
            return 1
        if not (df.spot_id == base.spot_id).all() or not (df.fov == base.fov).all():
            print(f"[fatal] spot_id or fov misaligned")
            return 1

    # Build tuple arrays per voter
    tuples_per_voter = []
    for df in dfs:
        tup = np.empty(len(df), dtype=object)
        for i, (c, s, su, cl) in enumerate(zip(df["class"].astype(str), df["subclass"].astype(str),
                                                df["supertype"].astype(str), df["cluster"].astype(str))):
            tup[i] = (c, s, su, cl)
        tuples_per_voter.append(tup)

    n = len(base)
    out_tuples = np.empty(n, dtype=object)
    threshold = len(dfs) // 2 + 1
    stats = {"unanimous": 0, "majority": 0, "tie_anchor": 0}

    for spot_i in range(n):
        col = [tv[spot_i] for tv in tuples_per_voter]
        if all(t == col[0] for t in col):
            out_tuples[spot_i] = col[0]
            stats["unanimous"] += 1
            continue
        c = Counter(col)
        top_val, top_n = c.most_common(1)[0]
        if top_n >= threshold:
            out_tuples[spot_i] = top_val
            stats["majority"] += 1
        else:
            out_tuples[spot_i] = col[0]  # anchor wins ties
            stats["tie_anchor"] += 1

    print(f"\nTuple-vote stats: unanimous={stats['unanimous']:,} ({stats['unanimous']/n:.1%})  "
          f"majority={stats['majority']:,} ({stats['majority']/n:.1%})  "
          f"tie-anchor={stats['tie_anchor']:,} ({stats['tie_anchor']/n:.1%})")

    out = base[["spot_id", "fov"]].copy()
    out["class"] = [t[0] for t in out_tuples]
    out["subclass"] = [t[1] for t in out_tuples]
    out["supertype"] = [t[2] for t in out_tuples]
    out["cluster"] = [t[3] for t in out_tuples]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows -> {out_path}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_submission import validate as _validate_sub
    ok, msg = _validate_sub(out_path)
    print(f"validate_submission: {'PASS' if ok else 'FAIL'}  {msg}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
