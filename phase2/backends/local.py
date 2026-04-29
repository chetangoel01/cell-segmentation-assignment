"""Local backend: just runs the task in the current process."""
from __future__ import annotations

import argparse

from phase2.tasks import Task


def launch(task: Task, args: argparse.Namespace) -> int:
    if args.dry_run:
        print(f"[dry-run] would run task {task.name!r} locally with args:")
        for k, v in sorted(vars(args).items()):
            if k in ("backend", "dry_run", "task"):
                continue
            print(f"  --{k}={v!r}")
        return 0
    rv = task.run(args)
    return rv if isinstance(rv, int) else 0
