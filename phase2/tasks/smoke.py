"""Smoke-test task: load a single FOV, verify coords + spots-in-polygon."""
from __future__ import annotations

import argparse

from phase2.tasks import Task, register


def _add_args(p: argparse.ArgumentParser) -> None:
    # Flagged (not positional) so cross-backend argv round-tripping is
    # unambiguous when we re-invoke `python -m phase2 smoke …` inside
    # SLURM / Modal containers.
    p.add_argument("--fov", default="FOV_101",
                   help="FOV name, e.g. FOV_101 (default).")


def _run(args: argparse.Namespace) -> int:
    from phase2.scripts.fov101_smoke_test import main as _smoke_main
    return _smoke_main(args.fov)


register(Task(
    name="smoke",
    summary="Sanity-check a FOV's image, spots, and polygon coord convention.",
    add_args=_add_args,
    run=_run,
    requirements={
        "gpu": False,
        "modal_image": "default",
        "modal_volume": "cell-seg-phase2",
        "modal_timeout": 600,
        "hpc_partition": "cpu",
        "hpc_hours": 0.25,
        "hpc_gpus": 0,
    },
))
