# Phase 2 Unified Runner — Design

**Date:** 2026-04-29
**Status:** Approved (approach A)
**Owner:** Chetan

## Problem

Phase 2 needs to run the same logical work — data fetch, smoke test,
classifier training, inference — across three execution backends: laptop
(local), NYU Cloud Burst (HPC via SLURM), and Modal (containerized cloud).
Today the few existing pieces are scattered: `modal_fetch_data.py` lives
inside an `@app.local_entrypoint` shape, `fov101_smoke_test.py` is a one-shot
local script, and there is no HPC entry point at all. As classifier code
lands, this will get worse.

## Goal

One CLI verb-and-task surface that dispatches the same task to any of the
three backends, while keeping each backend's specific concerns
(SLURM directives, Modal images, local Python imports) honest and visible —
not hidden behind a leaky abstraction.

```
python -m phase2 <task> [task-args] --backend {local,hpc,modal}
```

Examples (target shape):
```
python -m phase2 smoke FOV_101 --backend local
python -m phase2 fetch-data --target support --backend modal
python -m phase2 train-baseline --train-fovs FOV_101,FOV_105 \
    --val-fovs FOV_110 --backend local
python -m phase2 train-baseline --train-fovs FOV_101..FOV_155 \
    --val-fovs FOV_160 --backend hpc --gpus 0 --hours 1
```

## Non-goals

- **`--backend auto`**: no clever queue-aware backend picking. The user
  chooses where to run.
- **Cross-backend orchestration**: no "train on HPC, evaluate locally,
  push results back to Modal" flow built into the runner. If you need it,
  chain commands by hand.
- **Refactoring phase 1**: phase 1 is frozen. Only phase 2 lives behind
  this runner.

## Architecture

Three layers:

```
phase2/
├── __main__.py        # `python -m phase2 ...` dispatch
├── tasks/             # Pure task logic — no backend awareness
│   ├── __init__.py    # TASK_REGISTRY + Task dataclass
│   ├── smoke.py
│   ├── fetch_data.py
│   └── train_baseline.py
├── backends/          # Each backend knows how to launch ANY task
│   ├── __init__.py    # BACKENDS dict
│   ├── local.py       # imports task module, calls task.run()
│   ├── hpc.py         # renders sbatch from template, submits
│   └── modal.py       # builds modal.App, calls remote function
├── src/               # (existing) io.py, coords.py — shared library code
└── ...
```

### Task contract

Each task module exports a `Task` instance:

```python
# phase2/tasks/smoke.py
from phase2.tasks import Task

def _add_args(p):
    p.add_argument("fov", default="FOV_101", nargs="?")

def _run(args):
    from phase2.scripts.fov101_smoke_test import main
    return main(args.fov)

TASK = Task(
    name="smoke",
    summary="Sanity-check a FOV's image+spots+polygons.",
    add_args=_add_args,
    run=_run,
    requirements={"gpu": False, "modal_image": "default", "hpc_partition": "cpu"},
)
```

`requirements` is **advisory** — backends read it to pick defaults
(GPU on/off, image, partition), but the user can always override on the CLI.

### Backend contract

```python
# phase2/backends/local.py
def launch(task: Task, args: argparse.Namespace) -> int:
    return task.run(args) or 0
```

```python
# phase2/backends/hpc.py
def launch(task, args) -> int:
    sbatch_text = render_template(task, args)        # SLURM directives + python -m phase2 <task> ...
    return submit_sbatch(sbatch_text)                # writes to phase2/.runs/<ts>/job.sbatch
```

```python
# phase2/backends/modal.py
def launch(task, args) -> int:
    app = build_app(task, args)                      # modal.App + image + volumes per task.requirements
    with app.run():                                   # remote call
        return _runner.remote(task.name, vars(args))
```

### Data root resolution per backend

`phase2/src/io.py::data_root()` already honors `MERFISH_DATA_ROOT`. The
backends set it explicitly:

| Backend | `MERFISH_DATA_ROOT` |
|---------|---------------------|
| local   | repo's `phase2/data/` (existing default) |
| hpc     | `/scratch/pl2820/competition_phase2` (read-only) + `/scratch/cg4652/phase2/` for outputs |
| modal   | `/root/data` (mount of `cell-seg-phase2` volume) |

Output dirs (`models/`, `submissions/`) follow the same pattern but write to
backend-specific locations (laptop: `phase2/runs/`; HPC: scratch; Modal:
volume).

### HPC sbatch templating

A single Jinja-free f-string template at
`phase2/backends/hpc/template.sbatch` covers the common shape (account,
partition, time, GPU ask, Singularity overlay, repo path, `python -m phase2
<task> --backend local` inside the container).

We render to `phase2/.runs/<timestamp>-<task>/job.sbatch` and submit with
`sbatch`. Logs land in the same dir.

### Modal app construction

Building the app per-task at launch time (vs declaring `@app.function`
modules ahead) lets us:
- pick the image from `task.requirements["modal_image"]`
- attach the right volume(s)
- set GPU on/off per task
- call ONE generic `_runner.remote(task_name, args_dict)` that re-imports
  the task inside the container

This is a small departure from the current `modal_fetch_data.py` pattern
(which has `@app.local_entrypoint` per verb) — but it's worth it: no more
duplicating `modal run …::fetch / ::ping / ::ls` for every task.

The migrated `fetch-data` task collapses the existing `ping`, `run`, `all`,
`ls`, `verify` entrypoints into subcommand-style flags
(`--probe`, `--target {aws|support|all}`, `--ls`, `--verify <fovs>`).

## Components delivered in this PR

1. `phase2/__main__.py` + `phase2/tasks/__init__.py` (Task dataclass,
   registry, dispatch)
2. `phase2/backends/{local,hpc,modal}.py`
3. `phase2/backends/hpc/template.sbatch`
4. `phase2/tasks/smoke.py` — wraps existing `fov101_smoke_test.main`
5. `phase2/tasks/fetch_data.py` — port of `modal_fetch_data.py` shape into
   the task contract; legacy `phase2/modal/modal_fetch_data.py` retained
   as a thin shim that calls the new task with `--backend modal` so any
   existing `modal run …` muscle memory keeps working
6. `phase2/tasks/train_baseline.py` — Stage-1 cell-type classifier:
   build per-cell gene-expression vectors from polygons + spots, train an
   sklearn `LogisticRegression` (and optional `KNeighborsClassifier`) per
   hierarchy level, FOV-level held-out validation, save model + per-level
   ARI to `phase2/runs/<timestamp>/`
7. Smoke-fix: `phase2/src/coords.py::parse_boundary_polygon` now handles
   `NaN`/non-string input (some cells have polygons only at a subset of
   z-planes; fixes a real crash hit during today's smoke run on FOV_101).

## Testing strategy

End-to-end checks done as part of this work, all from repo root:

```
# Local backend
python -m phase2 smoke FOV_101                        # ~4s
python -m phase2 train-baseline \
    --train-fovs FOV_101,FOV_105,FOV_110,FOV_120,FOV_130,FOV_135,FOV_140,FOV_150,FOV_155 \
    --val-fovs FOV_160                                # ~30s estimate

# HPC backend — DRY RUN ONLY (no submit)
python -m phase2 train-baseline --backend hpc --dry-run
    → prints rendered sbatch, exits 0

# Modal backend — DRY RUN ONLY
python -m phase2 fetch-data --target aws --backend modal --dry-run
    → prints app config (image, volume, GPU=False), exits 0
```

Live HPC submission and live Modal launches are deferred to the user (cost
+ network).

## Risks / caveats

- The Modal task wrapper that re-imports task code inside the container
  needs the task module to be `add_local`'d into the image. Solved by a
  helper that bundles `phase2/{src,tasks}/` into every Modal image.
- HPC sbatch template assumes the existing Singularity overlay layout from
  `~/.claude/skills/nyu-hpc/SKILL.md`. If that changes, the template
  changes — but it's a single file.
- The kNN/LR baseline trains on cells (not spots). Per the kaggle metric
  (ARI over spots), we evaluate by mapping each spot to its containing
  polygon → that polygon's predicted label, leaving extracellular spots as
  `background`. This matches phase 1's known-good convention.
