"""Unified phase-2 entry point.

    python -m phase2 <task> [task-args] [--backend {local,hpc,modal}] [--dry-run]

Tasks live in phase2/tasks/. Backends live in phase2/backends/. Each task is
backend-agnostic; backends know how to launch any task.

Examples:
    python -m phase2 smoke FOV_101
    python -m phase2 fetch-data --target aws --backend modal
    python -m phase2 train-baseline --train-fovs FOV_101,FOV_105 --val-fovs FOV_110

Use `python -m phase2 --list` to see registered tasks and their help.
"""
from __future__ import annotations

import argparse
import sys

from phase2.tasks import TASK_REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m phase2",
        description="Unified phase-2 task runner (local / HPC / Modal).",
    )
    p.add_argument("--list", action="store_true",
                   help="List registered tasks and exit.")
    sub = p.add_subparsers(dest="task", metavar="TASK")
    for name, task in sorted(TASK_REGISTRY.items()):
        sp = sub.add_parser(name, help=task.summary, description=task.summary)
        task.add_args(sp)
        sp.add_argument("--backend", choices=("local", "hpc", "modal"),
                        default="local",
                        help="Where to run this task (default: local).")
        sp.add_argument("--dry-run", action="store_true",
                        help="Show what would be launched without running.")
        # Backend-shared knobs (not all backends use all of these).
        sp.add_argument("--gpus", type=int, default=None,
                        help="GPU count override (HPC/Modal only).")
        sp.add_argument("--hours", type=float, default=None,
                        help="Wall-time hours override (HPC only).")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list or not args.task:
        print("Available tasks:")
        for name, task in sorted(TASK_REGISTRY.items()):
            print(f"  {name:<18} {task.summary}")
        print("\nRun `python -m phase2 <task> --help` for task-specific args.")
        return 0 if args.list else 1

    task = TASK_REGISTRY[args.task]
    backend_name = args.backend

    # Lazy import — keeps the local path free of modal/HPC imports.
    if backend_name == "local":
        from phase2.backends import local as backend
    elif backend_name == "hpc":
        from phase2.backends import hpc as backend
    elif backend_name == "modal":
        from phase2.backends import modal as backend
    else:
        raise ValueError(f"unknown backend {backend_name!r}")

    return backend.launch(task, args)


if __name__ == "__main__":
    sys.exit(main())
