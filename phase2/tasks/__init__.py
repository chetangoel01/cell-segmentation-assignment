"""Phase-2 task registry.

A Task is a backend-agnostic unit of work. Backends know how to launch any
Task; tasks know nothing about backends. This separation is the whole point
of the runner.

To add a new task:
  1. Create phase2/tasks/<name>.py exporting a top-level `TASK = Task(...)`.
  2. Import it in `_register_builtins` below.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Task:
    name: str
    summary: str
    add_args: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int | None]
    requirements: dict = field(default_factory=dict)
    """Advisory hints for backend defaults. Recognized keys:
        gpu: bool                     — needs a GPU
        modal_image: str              — image preset name (see backends/modal.py)
        modal_gpu: str | None         — e.g. "A10G", "T4", or None
        modal_volume: str             — Modal volume name to mount at /root/data
        modal_timeout: int            — seconds
        hpc_partition: str | None     — SLURM partition
        hpc_hours: float              — default wall-clock
        hpc_gpus: int                 — default GPU count
    """


TASK_REGISTRY: dict[str, Task] = {}


def register(task: Task) -> Task:
    if task.name in TASK_REGISTRY:
        raise ValueError(f"task {task.name!r} already registered")
    TASK_REGISTRY[task.name] = task
    return task


def _register_builtins() -> None:
    # Imports are inside the function so plain `from phase2.tasks import Task`
    # doesn't trigger every task module's import-time work.
    from phase2.tasks import smoke, fetch_data, train_baseline  # noqa: F401


_register_builtins()
