"""DEPRECATED — replaced by phase2/scripts/vote_submissions.py.

Both scripts implement plurality voting across submission CSVs (per spot, per
level). vote_submissions.py is the kept implementation: it has the same
semantics, plus per-level agreement statistics and a positional-arg CLI that
matches the autoresearch invocation pattern.

This stub fails fast so any forgotten caller gets a clear redirect.
"""
from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "ensemble_submissions.py is deprecated. Use vote_submissions.py:\n"
        "    .venv/bin/python phase2/scripts/vote_submissions.py "
        "--out OUTPUT INPUT1 INPUT2 [INPUT3 …]\n"
        "(first positional arg is the anchor / tiebreaker.)\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
