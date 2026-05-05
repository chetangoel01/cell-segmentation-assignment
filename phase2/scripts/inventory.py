"""Inventory script — track what's been pulled, what's missing, what to do next.

Walks $MERFISH_DATA_ROOT and reports status across three data sources:
  1. Course-distributed data (train/, test/, reference/, ground_truth/)
  2. BIL extras (competition_extras/ — additional tiles from the source session)
  3. AWS (aws/ — Zhuang-ABCA-4 metadata + h5ad)

Validates against expected file sizes from `fetch_bil.py` constants and reports
OK / PARTIAL / MISSING per file.

Usage:
  python3 phase2/scripts/inventory.py
  python3 phase2/scripts/inventory.py --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Reuse data_root() from src/io.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io import data_root  # noqa: E402


# ---- expected sizes (verified 2026-04-28 via HEAD against BIL/AWS) ----

EXPECTED_DAX_ROUND = 142_606_336      # 17-frame round file
EXPECTED_DAX_FIDUCIAL = 41_943_040    # 5-frame fiducial
EXPECTED_DAX_PREIMAGE = 226_492_416   # 27-frame preimage

EXPECTED_BIL_SUPPORT = {
    "tiled_positions.txt":                                  40_590,
    "spots_220912_wb3_sa2_2_5z18R_merfish5.csv":            1_811_431_246,
    "cell_boundaries_220912_wb3_sa2_2_5z18R_merfish5.csv":  3_623_666_112,
    "counts_mouse4_sagittal.h5ad":                          1_014_137_600,
    "codebook_32bit_v2.csv":                                92_715,
    "dataorganization_220912_wb3_sa2_2_5z18R_merfish5.csv":  9_060,
}

EXPECTED_AWS = {
    "cell_metadata_with_cluster_annotation.csv":  50_545_200,
    "cell_metadata.csv":                          37_955_304,
    "gene.csv":                                   84_677,
    "Zhuang-ABCA-4-log2.h5ad":                    106_739_752,
    "Zhuang-ABCA-4-raw.h5ad":                     64_247_492,
}

COMPETITION_TRAIN_TILES = set(f"{i:03d}" for i in range(101, 161))


# ---- status types ----

@dataclass
class FileStatus:
    name: str
    path: Path
    expected_bytes: int | None
    actual_bytes: int | None
    status: str  # OK | PARTIAL | MISSING | UNKNOWN_SIZE
    note: str = ""


@dataclass
class GroupReport:
    title: str
    description: str = ""
    files: list[FileStatus] = field(default_factory=list)
    summary: str = ""


def status_of(path: Path, expected: int | None) -> FileStatus:
    if not path.exists():
        return FileStatus(name=path.name, path=path, expected_bytes=expected,
                          actual_bytes=None, status="MISSING")
    actual = path.stat().st_size
    if expected is None:
        return FileStatus(name=path.name, path=path, expected_bytes=None,
                          actual_bytes=actual, status="UNKNOWN_SIZE")
    if actual == expected:
        return FileStatus(name=path.name, path=path, expected_bytes=expected,
                          actual_bytes=actual, status="OK")
    return FileStatus(name=path.name, path=path, expected_bytes=expected,
                      actual_bytes=actual, status="PARTIAL",
                      note=f"got {actual:,} of {expected:,} bytes ({actual/expected:.1%})")


# ---- collectors ----

def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def collect_course_data(root: Path) -> GroupReport:
    """Course-distributed data — train FOVs, test FOVs, reference, ground_truth."""
    report = GroupReport(title="course/local",
                         description="data from the course (train + test + ground truth)")

    # Train FOV dirs (expect 101..160 = 60 dirs)
    train_dir = root / "train"
    train_fovs = sorted([p for p in train_dir.glob("FOV_*") if p.is_dir()]) if train_dir.exists() else []
    n_train = len(train_fovs)
    train_size = sum(f.stat().st_size for d in train_fovs for f in d.glob("*.dax"))
    report.files.append(FileStatus(
        name="train/FOV_*/", path=train_dir,
        expected_bytes=None, actual_bytes=train_size,
        status="OK" if n_train >= 60 else ("PARTIAL" if n_train > 0 else "MISSING"),
        note=f"{n_train}/60 FOV dirs, {fmt_bytes(train_size)}"))

    # Test FOV dirs (expect 10 lettered dirs E..N)
    test_dir = root / "test"
    test_fovs = sorted([p for p in test_dir.glob("FOV_*") if p.is_dir()]) if test_dir.exists() else []
    n_test = len(test_fovs)
    test_size = sum(f.stat().st_size for d in test_fovs for f in d.glob("*.dax"))
    report.files.append(FileStatus(
        name="test/FOV_*/", path=test_dir,
        expected_bytes=None, actual_bytes=test_size,
        status="OK" if n_test >= 10 else ("PARTIAL" if n_test > 0 else "MISSING"),
        note=f"{n_test}/10 FOV dirs, {fmt_bytes(test_size)}"))

    # Reference / ground truth
    for rel in ["reference/fov_metadata.csv", "train/ground_truth/spots_train.csv",
                "train/ground_truth/cell_boundaries_train.csv",
                "train/ground_truth/cell_labels_train.csv",
                "train/ground_truth/counts_train.h5ad"]:
        p = root / rel
        report.files.append(status_of(p, expected=None))

    summary_n_ok = sum(1 for f in report.files if f.status == "OK")
    summary_n = len(report.files)
    report.summary = f"{summary_n_ok}/{summary_n} present"
    return report


def collect_bil_extras(root: Path) -> GroupReport:
    """BIL extras pulled via fetch-source (additional tiles from competition source session)."""
    report = GroupReport(title="bil/extras",
                         description="additional tiles from competition source session (BIL)")

    extras_root = root / "external" / "competition_extras" / "data"
    dax = sorted(extras_root.glob("*.dax")) if extras_root.exists() else []
    if not dax:
        report.summary = "not started — run `fetch_bil.py fetch-source ...` on HPC"
        return report
    # Group by tile id (3- or 4-digit)
    import re
    tile_re = re.compile(r"_(\d{3,4})[._]")
    by_tile: dict[str, list[Path]] = {}
    for f in dax:
        m = tile_re.search(f.name)
        if not m: continue
        tile = m.group(1).zfill(3)
        by_tile.setdefault(tile, []).append(f)

    n_tiles = len(by_tile)
    n_files = len(dax)
    total_size = sum(f.stat().st_size for f in dax)
    suspect = [t for t in by_tile if t in COMPETITION_TRAIN_TILES]

    # Validate file sizes match expected
    bad_size = []
    for f in dax:
        sz = f.stat().st_size
        if sz not in (EXPECTED_DAX_ROUND, EXPECTED_DAX_FIDUCIAL, EXPECTED_DAX_PREIMAGE):
            bad_size.append(f.name)

    report.files.append(FileStatus(
        name="extras/data/", path=extras_root,
        expected_bytes=None, actual_bytes=total_size,
        status="OK" if (n_tiles > 0 and not bad_size and not suspect) else "PARTIAL",
        note=f"{n_tiles} tiles, {n_files} files, {fmt_bytes(total_size)}"
             + (f" — WARNING: train tiles present in extras: {sorted(suspect)}" if suspect else "")
             + (f" — {len(bad_size)} files have unexpected sizes" if bad_size else "")))
    report.summary = f"{n_tiles} extra tile(s) pulled, {fmt_bytes(total_size)}"
    return report


def collect_bil_support(root: Path) -> GroupReport:
    report = GroupReport(title="bil/support",
                         description="cell labels / boundaries / spots / positions from BIL")
    support_root = root / "external" / "competition_support"

    expected_files = list(EXPECTED_BIL_SUPPORT.items())
    found_any = False
    for fname, expected_size in expected_files:
        p = support_root / fname
        st = status_of(p, expected_size)
        report.files.append(st)
        if st.status != "MISSING":
            found_any = True

    if not found_any:
        report.summary = "not started — run `fetch_bil.py fetch-support` (HPC recommended; spots+boundaries are 5+ GB throttled)"
    else:
        ok = sum(1 for f in report.files if f.status == "OK")
        report.summary = f"{ok}/{len(report.files)} files present"
    return report


def collect_aws(root: Path) -> GroupReport:
    report = GroupReport(title="aws/Zhuang-ABCA-4",
                         description="AWS S3 mirror — fast cell labels + h5ad")
    aws_root = root / "external" / "aws"

    found_any = False
    for fname, expected_size in EXPECTED_AWS.items():
        p = aws_root / fname
        st = status_of(p, expected_size)
        report.files.append(st)
        if st.status != "MISSING":
            found_any = True

    if not found_any:
        report.summary = "not started — run `fetch_bil.py fetch-aws` (~157 MB at AWS speed = ~10 sec)"
    else:
        ok = sum(1 for f in report.files if f.status == "OK")
        report.summary = f"{ok}/{len(report.files)} files present"
    return report


def collect_matched_pool(root: Path) -> GroupReport:
    """BIL matched-pool sessions (sa1_sample1 etc.) — only present if user pulled them."""
    report = GroupReport(title="bil/matched-pool",
                         description="other wb3+merfish5 sessions (cross-animal mouse3, same-animal sa2_sample3)")
    ext = root / "external"
    if not ext.exists():
        report.summary = "no external dir"
        return report

    samples = [d for d in ext.glob("sa*_sample*") if d.is_dir()]
    populated = []
    for s in samples:
        data_dir = s / "data"
        if not data_dir.exists():
            continue
        n_dax = sum(1 for _ in data_dir.glob("*.dax"))
        if n_dax == 0:
            continue  # ignore empty dirs (leftover from dry-runs)
        sz = sum(f.stat().st_size for f in data_dir.glob("*.dax"))
        populated.append(s)
        report.files.append(FileStatus(
            name=s.name, path=data_dir, expected_bytes=None, actual_bytes=sz,
            status="OK", note=f"{n_dax} .dax files, {fmt_bytes(sz)}"))
    if not populated:
        report.summary = "none pulled"
        return report
    report.summary = f"{len(populated)} session(s) populated"
    return report


# ---- next-step heuristics ----

def next_steps(reports: list[GroupReport]) -> list[str]:
    by_title = {r.title: r for r in reports}
    steps = []
    aws = by_title["aws/Zhuang-ABCA-4"]
    if any(f.status == "MISSING" for f in aws.files if "log2" in f.name or "annotation" in f.name):
        steps.append("CLASSIFIER PATH (fast, ~157 MB at 18 MB/s): `python3 phase2/scripts/fetch_bil.py fetch-aws`")
    extras = by_title["bil/extras"]
    if not extras.files:
        steps.append("EXTRA TILES (HPC only — needs test FOV indices first): "
                     "discover via `python3 phase2/scripts/fetch_bil.py verify FOV_E ... FOV_N --split test`, "
                     "then `fetch-source --exclude-tiles <test ids> --adjacent --rounds-only --limit 60`")
    course = by_title["course/local"]
    if any(f.status == "MISSING" for f in course.files if "ground_truth" in str(f.path)):
        steps.append("LOCAL ground_truth files missing — sync from HPC `/scratch/pl2820/competition_phase2/`")
    return steps


# ---- rendering ----

def render_text(reports: list[GroupReport], steps: list[str]) -> None:
    print("=== Phase 2 data inventory ===")
    print(f"root: {data_root()}\n")

    for r in reports:
        marker_counts = {"OK": 0, "PARTIAL": 0, "MISSING": 0, "UNKNOWN_SIZE": 0}
        for f in r.files:
            marker_counts[f.status] = marker_counts.get(f.status, 0) + 1
        print(f"[{r.title}] {r.description}")
        for f in r.files:
            sym = {"OK": "OK ", "PARTIAL": "PART", "MISSING": "MISS", "UNKNOWN_SIZE": "??  "}[f.status]
            size_s = fmt_bytes(f.actual_bytes) if f.actual_bytes is not None else fmt_bytes(f.expected_bytes)
            line = f"  [{sym}] {f.name:55s} {size_s:>10s}"
            if f.note:
                line += f"  {f.note}"
            print(line)
        print(f"  -> {r.summary}\n")

    if steps:
        print("=== next steps ===")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")


def render_json(reports: list[GroupReport], steps: list[str]) -> str:
    obj = {
        "root": str(data_root()),
        "groups": [
            {
                "title": r.title,
                "description": r.description,
                "summary": r.summary,
                "files": [
                    {"name": f.name, "path": str(f.path), "status": f.status,
                     "actual_bytes": f.actual_bytes, "expected_bytes": f.expected_bytes,
                     "note": f.note}
                    for f in r.files
                ],
            }
            for r in reports
        ],
        "next_steps": steps,
    }
    return json.dumps(obj, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args()

    root = data_root()
    reports = [
        collect_course_data(root),
        collect_aws(root),
        collect_bil_extras(root),
        collect_bil_support(root),
        collect_matched_pool(root),
    ]
    steps = next_steps(reports)

    if args.json:
        print(render_json(reports, steps))
    else:
        render_text(reports, steps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
