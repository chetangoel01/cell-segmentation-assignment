"""Modal backend: build an app per launch, mount the data volume, run the task.

Each task's `requirements["modal_image"]` selects an image preset; the runner
calls a single generic `_remote_runner(task_name, args_dict)` that re-imports
the task module *inside* the container and runs it the same way the local
backend does (so the task body is unchanged).

This is a deliberate departure from `phase2/modal/modal_fetch_data.py`'s
older pattern where each verb had its own `@app.local_entrypoint`. The new
shape: one CLI, three backends, one Modal app definition reused across
tasks.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from phase2.tasks import Task

REPO = Path(__file__).resolve().parents[2]


def _image_for(preset: str):
    """Return a modal.Image configured for the given preset."""
    import modal

    base = (
        modal.Image.debian_slim(python_version="3.11")
        # Always include shared library code + tasks so the in-container
        # runner can `from phase2.tasks import TASK_REGISTRY`.
        .add_local_dir(str(REPO / "phase2" / "src"), "/root/repo/phase2/src", copy=True)
        .add_local_dir(str(REPO / "phase2" / "tasks"), "/root/repo/phase2/tasks", copy=True)
        .add_local_file(str(REPO / "phase2" / "__init__.py"), "/root/repo/phase2/__init__.py", copy=True)
        .add_local_file(str(REPO / "phase2" / "__main__.py"), "/root/repo/phase2/__main__.py", copy=True)
        .add_local_dir(str(REPO / "phase2" / "scripts"), "/root/repo/phase2/scripts", copy=True)
        .pip_install("numpy", "pandas", "shapely")
        .env({"PYTHONPATH": "/root/repo"})
    )
    if preset == "default" or preset == "fetch":
        return base
    if preset == "sklearn":
        return base.pip_install("scikit-learn", "joblib")
    raise ValueError(f"unknown modal_image preset {preset!r}")


def _strip_runner_args(args: argparse.Namespace) -> dict:
    """Pack the task args as a dict to ship to the remote runner."""
    skip = {"task", "backend", "dry_run", "gpus", "hours", "list"}
    return {k: v for k, v in vars(args).items() if k not in skip and v is not None}


def launch(task: Task, args: argparse.Namespace) -> int:
    req = task.requirements
    vol_name = req.get("modal_volume", "cell-seg-phase2")
    timeout = req.get("modal_timeout", 3600)
    gpu = args.gpus if args.gpus is not None else (req.get("modal_gpu") if req.get("gpu") else None)

    if args.dry_run:
        # Don't import modal in dry-run — keeps the cli usable on machines
        # without the modal SDK installed (e.g. before `pip install modal`).
        print(f"[dry-run] modal app for task {task.name!r}:")
        print(f"  image preset: {req.get('modal_image', 'default')}")
        print(f"  volume:       {vol_name}  →  /root/data")
        print(f"  gpu:          {gpu}")
        print(f"  timeout:      {timeout}s")
        print(f"  task args:    {_strip_runner_args(args)}")
        return 0

    import modal
    image = _image_for(req.get("modal_image", "default"))
    vol = modal.Volume.from_name(vol_name, create_if_missing=False)
    app = modal.App(f"phase2-{task.name}")

    @app.function(image=image, volumes={"/root/data": vol},
                  timeout=timeout, gpu=gpu)
    def _remote_runner(task_name: str, task_args: dict) -> int:
        import os
        import sys as _sys
        os.environ["MERFISH_DATA_ROOT"] = "/root/data"
        if "/root/repo" not in _sys.path:
            _sys.path.insert(0, "/root/repo")

        from phase2.tasks import TASK_REGISTRY
        t = TASK_REGISTRY[task_name]
        # Rehydrate Namespace from dict.
        ns = argparse.Namespace(**task_args)
        rv = t.run(ns)
        return rv if isinstance(rv, int) else 0

    with app.run():
        rv = _remote_runner.remote(task.name, _strip_runner_args(args))
    print(f"modal task {task.name!r} returned {rv}")
    return int(rv)
