"""HPC (NYU Cloud Burst) backend: render a SLURM sbatch and submit it.

Strategy: the sbatch wraps `python -m phase2 <task> ... --backend local` so
the task code itself doesn't need to know it's on HPC — it just runs as a
local job inside the Singularity container on the compute node.

If `sbatch` isn't on PATH (e.g. you're invoking from your laptop), we still
render the script to `phase2/.runs/<ts>-<task>/job.sbatch` and print the
exact `sbatch <path>` command for you to run after rsync'ing to the cluster.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from phase2.tasks import Task

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={hours}
{gres_line}#SBATCH --mem={mem}
#SBATCH --output={out_log}
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail
cd {repo_dir}

export MERFISH_DATA_ROOT="{data_root}"

# Singularity overlay carries the python env; see ~/.claude/skills/nyu-hpc/SKILL.md.
OVERLAY="{overlay}"
SIF="{sif}"

singularity exec --nv \\
    --overlay "$OVERLAY":ro \\
    "$SIF" \\
    bash -lc "source /ext3/env.sh && python -m phase2 {task_argv} --backend local"
"""


def _strip_runner_args(args: argparse.Namespace) -> list[str]:
    """Reconstruct the task argv minus runner-only flags (backend, dry-run, …)."""
    out: list[str] = [args.task]
    skip = {"task", "backend", "dry_run", "gpus", "hours", "list"}
    for k, v in vars(args).items():
        if k in skip or v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        else:
            out.extend([flag, str(v)])
    return out


def launch(task: Task, args: argparse.Namespace) -> int:
    req = task.requirements
    repo_dir = Path(__file__).resolve().parents[2]
    runs_root = repo_dir / "phase2" / ".runs"
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / f"{ts}-{task.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve resource asks (CLI override > task requirement > sane default).
    hours = args.hours if args.hours is not None else req.get("hpc_hours", 1.0)
    gpus = args.gpus if args.gpus is not None else req.get("hpc_gpus", 0)
    partition = req.get("hpc_partition", "cpu")
    # Omit --gres entirely for CPU jobs (some SLURM clusters reject gpu:0).
    gres_line = f"#SBATCH --gres=gpu:{gpus}\n" if gpus else ""
    mem = "32G" if not gpus else "64G"

    # Format hours as HH:MM:00.
    h = int(hours)
    m = int(round((hours - h) * 60))
    time_str = f"{h:02d}:{m:02d}:00"

    sbatch_text = TEMPLATE.format(
        job_name=f"p2-{task.name}",
        account=os.environ.get("HPC_ACCOUNT", "cs_gy_9223-2026sp"),
        partition=partition,
        hours=time_str,
        mem=mem,
        out_log=str(run_dir / "slurm-%j.out"),
        repo_dir="/home/cg4652/cell-segmentation-assignment",
        data_root="/scratch/pl2820/competition_phase2",
        overlay="/scratch/cg4652/overlay-15GB-500K.ext3",
        sif="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.sif",
        task_argv=" ".join(shlex.quote(t) for t in _strip_runner_args(args)),
        gres_line=gres_line,
    )

    sbatch_path = run_dir / "job.sbatch"
    sbatch_path.write_text(sbatch_text)
    print(f"sbatch rendered → {sbatch_path}")

    if args.dry_run:
        print("\n--- begin sbatch ---")
        print(sbatch_text)
        print("--- end sbatch ---")
        print(f"\n[dry-run] would submit with: sbatch {sbatch_path}")
        return 0

    if not shutil.which("sbatch"):
        print("sbatch not on PATH — assuming we're off-cluster.")
        print(f"To submit from the cluster:\n  rsync this repo to ~/cell-segmentation-assignment/")
        print(f"  then: sbatch {sbatch_path}")
        return 0

    res = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    print(res.stdout, end="")
    if res.returncode != 0:
        print(res.stderr, end="")
    return res.returncode
