"""Structural validator for Phase-2 Kaggle submission CSVs.

Verifies that a submission CSV is structurally valid against
phase2/data/sample_submission.csv before it gets uploaded. Five hard checks +
one advisory:

  1. Column order is exactly [spot_id, fov, class, subclass, supertype, cluster].
  2. Row count matches sample_submission.csv.
  3. spot_id sequence matches sample_submission.csv element-wise (in order).
  4. Every row's fov is in {FOV_E … FOV_N}.
  5. No NaN / null / empty value in any of the 4 hierarchy-level columns.

Advisory (printed but doesn't fail): count of half-background rows. Kaggle's
metric is per-level ARI averaged across 4 levels, so hierarchy-inconsistent
labels (class != "background" but cluster == "background") are accepted —
verified against the May-1 PQM ensemble that scored 0.5375.

Exit code 0 on PASS, 1 on FAIL (with first-failing row index + reason),
2 on bad invocation. Designed to wrap the tail of any submission writer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["spot_id", "fov", "class", "subclass", "supertype", "cluster"]
LEVELS = ["class", "subclass", "supertype", "cluster"]
EXPECTED_FOVS = {f"FOV_{c}" for c in "EFGHIJKLMN"}
DEFAULT_SAMPLE = Path("phase2/data/sample_submission.csv")


def validate(submission_path: Path, sample_path: Path = DEFAULT_SAMPLE) -> tuple[bool, str]:
    """Returns (ok, message). On failure, message includes the first failing row."""
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)

    # 1. Column order
    if list(sub.columns) != EXPECTED_COLUMNS:
        return False, (
            f"column order mismatch — got {list(sub.columns)}, "
            f"expected {EXPECTED_COLUMNS}"
        )

    # 2. Row count
    if len(sub) != len(sample):
        return False, (
            f"row count mismatch — got {len(sub)}, expected {len(sample)} "
            f"(sample_submission.csv)"
        )

    # 3. spot_id sequence (element-wise, in order)
    sub_ids = sub["spot_id"].values
    sample_ids = sample["spot_id"].values
    if not (sub_ids == sample_ids).all():
        first_diff = int((sub_ids != sample_ids).argmax())
        return False, (
            f"spot_id mismatch at row {first_diff}: "
            f"submission={sub_ids[first_diff]!r} vs sample={sample_ids[first_diff]!r}"
        )

    # 4. FOVs
    bad_fov = sub[~sub["fov"].isin(EXPECTED_FOVS)]
    if len(bad_fov):
        idx = int(bad_fov.index[0])
        return False, (
            f"unexpected FOV at row {idx}: {bad_fov['fov'].iloc[0]!r} "
            f"(expected one of {sorted(EXPECTED_FOVS)})"
        )

    # 5. No NaN / null / empty in any level column.
    for lvl in LEVELS:
        col = sub[lvl]
        # null / NaN / non-string sentinel
        bad = col.isna() | (col.astype(str).str.strip() == "")
        if bad.any():
            idx = int(bad.idxmax())
            row = sub.loc[idx]
            return False, (
                f"empty/NaN at row {idx} (spot_id={row['spot_id']!r}) "
                f"in column {lvl!r}: value={row[lvl]!r}"
            )

    # Advisory only — half-background row count.
    is_bg = sub[LEVELS].eq("background")
    n_full_bg = int(is_bg.all(axis=1).sum())
    n_any_bg = int(is_bg.any(axis=1).sum())
    n_half_bg = n_any_bg - n_full_bg
    pct_full_bg = n_full_bg / len(sub)

    return True, (
        f"OK  rows={len(sub):,}  fovs={sub['fov'].nunique()}  "
        f"full_background={n_full_bg:,} ({pct_full_bg:.1%})  "
        f"half_background={n_half_bg:,} (advisory)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission", help="Path to submission CSV to validate.")
    p.add_argument("--sample", default=str(DEFAULT_SAMPLE),
                   help=f"Path to sample_submission.csv (default: {DEFAULT_SAMPLE}).")
    args = p.parse_args(argv)

    sub_path = Path(args.submission)
    sample_path = Path(args.sample)
    if not sub_path.exists():
        print(f"FAIL  submission not found: {sub_path}", file=sys.stderr)
        return 2
    if not sample_path.exists():
        print(f"FAIL  sample_submission not found: {sample_path}", file=sys.stderr)
        return 2

    ok, msg = validate(sub_path, sample_path)
    if ok:
        print(f"PASS  {sub_path}  {msg}")
        return 0
    print(f"FAIL  {sub_path}  {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
