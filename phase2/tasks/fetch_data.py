"""Fetch-data task: pull AWS + BIL bundles into the configured data root.

Wraps phase2/scripts/fetch_bil.py. When run via the modal backend the data
lands in the `cell-seg-phase2` volume at /root/data; when run locally it
lands in $MERFISH_DATA_ROOT (or repo's phase2/data/ as a fallback).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from phase2.tasks import Task, register

# Maps friendly --target names to fetch_bil.py subcommands. Targets that
# need extra positional/keyword args go through their own --target value.
TARGETS = {
    "aws":     ["fetch-aws"],
    "support": ["fetch-support"],
    "counts":  ["fetch-counts"],
}


def _add_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--target", choices=sorted(TARGETS),
                   help="Named bundle to fetch (calls the matching fetch_bil subcommand).")
    g.add_argument("--probe", action="store_true",
                   help="HEAD a few AWS/BIL URLs to test reachability, no download.")
    g.add_argument("--ls", action="store_true",
                   help="List the contents of the data root.")
    g.add_argument("--verify", metavar="FOVS",
                   help="Comma-separated FOV list to verify against the BIL canonical sizes.")
    p.add_argument("--split", default="train",
                   help="Used with --verify: train|test (default: train).")


def _run(args: argparse.Namespace) -> int:
    from phase2.src import io
    root = io.data_root()
    root.mkdir(parents=True, exist_ok=True)

    if args.probe:
        return _probe()

    if args.ls:
        print(f"=== contents of {root} ===")
        return subprocess.call(["ls", "-laRh", str(root)])

    if args.verify:
        fovs = [f.strip() for f in args.verify.split(",") if f.strip()]
        return _verify_local(fovs, args.split)

    # Otherwise: shell out to fetch_bil.py with the named subcommand.
    subcmd = TARGETS[args.target]
    here = Path(__file__).resolve().parents[1]
    script = here / "scripts" / "fetch_bil.py"
    cmd = [sys.executable, str(script), *subcmd]
    env = {**os.environ, "MERFISH_DATA_ROOT": str(root)}
    print(f">>> {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def _probe() -> int:
    """HEAD a few canonical URLs from whatever host we're on."""
    import time
    import urllib.request

    urls = [
        ("AWS gene.csv (85 KB)",
         "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
         "metadata/Zhuang-ABCA-4/20241115/gene.csv"),
        ("BIL codebook (93 KB)",
         "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
         "additional_files/codebook_32bit_v2.csv"),
    ]
    print("=== egress IP ===")
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=10) as r:
            print(r.read().decode())
    except Exception as e:
        print(f"(unknown: {e})")
    print("\n=== HEAD probes ===")
    for label, url in urls:
        t0 = time.time()
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "phase2-runner/1.0")
            with urllib.request.urlopen(req, timeout=20) as r:
                cl = r.headers.get("Content-Length")
                print(f"{label}\n  → {r.status}  Content-Length={cl}  ({(time.time()-t0)*1000:.0f}ms)")
        except Exception as e:
            print(f"{label}\n  → FAIL: {type(e).__name__}: {e}")
    return 0


def _verify_local(fovs: list[str], split: str) -> int:
    """Local size-only verify. Mirrors the modal verify but skips network."""
    import re
    from phase2.src import io

    CANONICAL = {"round": 142_606_336, "fiducial": 41_943_040, "preimage": 226_492_416}

    def classify(name: str) -> str | None:
        if "473s5-408s5" in name:
            return "preimage"
        if name.startswith("Epi-750s1-635s1-545s1_"):
            return "fiducial"
        if name.startswith("Epi-750s5-635s5-545s1_"):
            return "round"
        return None

    base = io.train_dir() if split == "train" else io.test_dir()
    overall_ok = True
    tile_re = re.compile(r"_(\d{3,4})[._]")
    for fov in fovs:
        folder = base / fov
        print(f"\n========== {fov} ==========")
        if not folder.is_dir():
            print(f"  MISSING — {folder}")
            overall_ok = False
            continue
        files = sorted(folder.glob("*.dax"))
        if not files:
            print(f"  MISSING — no .dax in {folder}")
            overall_ok = False
            continue
        m_match = m_mis = m_unknown = 0
        for f in files:
            kind = classify(f.name)
            expected = CANONICAL.get(kind) if kind else None
            sz = f.stat().st_size
            if expected is None:
                print(f"  UNKNOWN  {f.name}")
                m_unknown += 1
            elif sz != expected:
                print(f"  BAD-SIZE {f.name}  local={sz:,}  expected({kind})={expected:,}")
                m_mis += 1
            else:
                print(f"  CANON-OK {f.name}  {sz:,} B  ({kind})")
                m_match += 1
        print(f"summary: {m_match} canon-ok, {m_mis} bad, {m_unknown} unknown of {len(files)}")
        if m_mis or m_unknown or not m_match:
            overall_ok = False
    print(f"\n{'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


register(Task(
    name="fetch-data",
    summary="Pull AWS / BIL bundles into the configured data root.",
    add_args=_add_args,
    run=_run,
    requirements={
        "gpu": False,
        "modal_image": "fetch",
        "modal_volume": "cell-seg-phase2",
        "modal_timeout": 6 * 3600,
        "hpc_partition": "cpu",
        "hpc_hours": 6.0,
        "hpc_gpus": 0,
    },
))
