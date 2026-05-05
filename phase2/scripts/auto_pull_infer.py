#!/usr/bin/env python3
"""Poll Modal volume for completed bundles, run inference + write submission CSV.

Watches `trained/<name>/<bundle>` paths on `cell-seg-phase2` volume; when a
bundle appears, downloads it and runs the matching inference script. Each
bundle runs independently — order doesn't matter.

Usage:  python phase2/scripts/auto_pull_infer.py
Stops after all bundles processed or 200 polls (~100 min).
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

JOBS = [
    {
        "name": "xgb-xl-d8n3000-bg40",
        "kind": "xgb",
        "bundle": "xgb_bundle.joblib",
        "local_dir": "phase2/runs/xgb-xl-d8n3000-bg40",
        "submit_dir": "phase2/runs/SUBMIT_xgb_xl_d8",
    },
    {
        "name": "xgb-xl-d6n3000-bg80",
        "kind": "xgb",
        "bundle": "xgb_bundle.joblib",
        "local_dir": "phase2/runs/xgb-xl-d6n3000-bg80",
        "submit_dir": "phase2/runs/SUBMIT_xgb_xl_d6",
    },
    {
        "name": "xgb-xl-d5n5000-bg40",
        "kind": "xgb",
        "bundle": "xgb_bundle.joblib",
        "local_dir": "phase2/runs/xgb-xl-d5n5000-bg40",
        "submit_dir": "phase2/runs/SUBMIT_xgb_xl_d5",
    },
    {
        "name": "scanvi-aws-modal-xl",
        "kind": "scanvi",
        "bundle": "scanvi_bundle.joblib",
        "local_dir": "phase2/runs/scanvi-aws-modal-xl",
        "submit_dir": "phase2/runs/SUBMIT_scanvi_xl",
    },
    {
        "name": "mlp-hier-aws",
        "kind": "mlp",
        "bundle": "mlp_bundle.joblib",
        "local_dir": "phase2/runs/mlp-hier-aws",
        "submit_dir": "phase2/runs/SUBMIT_mlp_hier",
    },
    {
        "name": "mlp-xl-h1024-d5",
        "kind": "mlp",
        "bundle": "mlp_bundle.joblib",
        "local_dir": "phase2/runs/mlp-xl-h1024-d5",
        "submit_dir": "phase2/runs/SUBMIT_mlp_xl",
    },
]


def now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def log(msg: str, log_path: Path) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def bundle_present(name: str, bundle: str) -> bool:
    """Bundle is 'present' only when BOTH the bundle file and metrics.json
    have been committed — metrics.json is written last, after data_vol.commit()."""
    r = subprocess.run(
        ["modal", "volume", "ls", "cell-seg-phase2", f"trained/{name}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return False
    return (bundle in r.stdout) and ("metrics.json" in r.stdout)


def run_inference(job: dict) -> bool:
    cmd = [
        "bash", "phase2/scripts/pull_and_infer.sh",
        job["kind"], f"trained/{job['name']}", job["local_dir"], job["submit_dir"],
    ]
    p = subprocess.run(cmd, cwd=ROOT)
    return p.returncode == 0


def main() -> int:
    log_path = ROOT / "phase2" / "runs" / "_autopull.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    log("started", log_path)

    for it in range(200):
        progress = False
        for job in JOBS:
            if job["name"] in done:
                continue
            if bundle_present(job["name"], job["bundle"]):
                log(f"FOUND {job['name']} -> running inference", log_path)
                ok = run_inference(job)
                msg = "OK" if ok else "FAILED"
                log(f"{msg} {job['name']} -> {job['submit_dir']}/submission.csv", log_path)
                done.add(job["name"])
                progress = True
        if len(done) == len(JOBS):
            log("all jobs processed", log_path)
            return 0
        if not progress:
            time.sleep(30)
    log(f"timeout after 200 iterations; done={done}", log_path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
