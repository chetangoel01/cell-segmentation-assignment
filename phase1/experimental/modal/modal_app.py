"""Modal app for the cell-segmentation pipeline.

Volumes (all already exist in your account)
-------------------------------------------
- `cell-seg-data`        mounted at /root/data        (competition dataset)
- `cell-seg-models`      mounted at /root/models      (trained checkpoints)
- `cell-seg-submissions` mounted at /root/workspace   (CWD — submissions, logs, best_params land here)

At runtime, `/root/workspace/models` is a symlink to `/root/models`, so every script that
writes to the relative `models/<exp>/` path reaches the models volume, and every script
that writes `submission_*.csv` / `logs/` / `best_params_*.json` to CWD lands in the
submissions volume.

Usage
-----
    # Inference (writes submission_<exp>.csv to submissions volume)
    modal run modal_app.py::infer --exp-name cyto2_warmup_modal
    modal run modal_app.py::infer_stardist --exp-name stardist_v3

    # Val-set failure-mode diagnostic (writes logs/diagnose_<exp>/*.png + summary.json)
    modal run modal_app.py::diagnose --exp-name cyto2_warmup_modal
    modal run modal_app.py::diagnose --exp-name stardist_v3 --arch stardist

    # Heterogeneous ensemble on existing submission CSVs — enter them priority-first
    modal run modal_app.py::ensemble \\
        --submissions "submission_stardist_v3.csv submission_cyto2_warmup_modal.csv" \\
        --output submission_stardist_plus_cyto2.csv
    # add --val-mode to also score each + the ensemble on val FOVs 036-040

    # Training (A100, detached so the terminal returns immediately)
    modal run --detach modal_app.py::train --exp-name cyto2 --base-model cyto2 --epochs 300
    modal run --detach modal_app.py::train_stardist --exp-name stardist_v4 --epochs 200

    # Post-training tools
    modal run modal_app.py::sweep --exp-name cyto2
    modal run modal_app.py::pick_best --exp-name cyto2

    # Drop into the container to inspect files / download artifacts
    modal run modal_app.py::shell

Notes
-----
- `--extra "<flags>"` forwards extra CLI flags to the underlying script
  (e.g. `--extra "--tta --prob-refine"`).
- `MERFISH_DATA_ROOT=/root/data` is injected.
- `best_params_<exp>.json` files baked into the image are auto-copied into the submissions
  volume on first run so `--params-json best_params_<exp>.json` resolves.
"""
from __future__ import annotations

import shlex
import modal

app = modal.App("cell-seg")

data_vol   = modal.Volume.from_name("cell-seg-data")
models_vol = modal.Volume.from_name("cell-seg-models")
subs_vol   = modal.Volume.from_name("cell-seg-submissions")

# ── Image ────────────────────────────────────────────────────────────────────
# Cellpose v4 + StarDist + PyTorch + TensorFlow (StarDist needs TF).
# Slim debian + CUDA wheels for torch; TF gets its bundled CUDA libs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    # Pin torch + NCCL explicitly.  Without the pin, pip picks torch 2.11 whose
    # bundled libtorch_cuda.so needs `ncclCommWindowDeregister` from a newer NCCL
    # than the cu12 wheel provides, causing ImportError at runtime.
    # torch==2.5.1+cu121 is the last line that works cleanly with TF 2.15's CUDA 12.2.
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2",
        "pandas>=2.0",
        "scipy",
        "scikit-image>=0.21",
        "scikit-learn>=1.3",
        "shapely>=2.0",
        "matplotlib>=3.7",
        "tifffile",
        "anndata>=0.10",
        "h5py",
        "tqdm",
        "cellpose>=4.0",
        # StarDist stack.  Drop [and-cuda] — TF 2.15 will coexist with torch's CUDA
        # and fall back to CPU for any op torch isn't using, which is fine for the
        # 2048×2048 inference we do here.
        "stardist>=0.9",
        "csbdeep",
        "tensorflow==2.15.*",
    )
    .add_local_dir(
        ".",
        "/root/repo",
        ignore=[
            "__pycache__",
            "*.pyc",
            ".git/**",
            ".venv/**",
            ".claude/**",
            "models/**",
            "logs/**",
            "submission_*.csv",
            "*.ipynb_checkpoints/**",
        ],
    )
)

VOLUMES = {
    "/root/data":      data_vol,
    "/root/models":    models_vol,
    "/root/workspace": subs_vol,
}
ENV = {"MERFISH_DATA_ROOT": "/root/data"}


def _run(script: str, argv: list[str]) -> None:
    """Shared helper: reload volumes, set up CWD, invoke a repo script.

    - CWD is /root/workspace (cell-seg-submissions volume). Scripts' relative writes
      (submission_*.csv, logs/, best_params_*.json) land here automatically.
    - /root/workspace/models → symlink to /root/models (cell-seg-models volume), so
      scripts' relative `models/<exp>/` reads/writes hit the models volume.
    - Baked `best_params_*.json` files from the image are copied to CWD on first run.
    """
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    data_vol.reload()
    models_vol.reload()
    subs_vol.reload()

    workspace = Path("/root/workspace")
    workspace.mkdir(exist_ok=True)

    models_link = workspace / "models"
    if not models_link.exists():
        models_link.symlink_to("/root/models")
    os.chdir(workspace)

    for p in Path("/root/repo").glob("best_params_*.json"):
        dst = workspace / p.name
        if not dst.exists():
            shutil.copy(p, dst)

    os.environ["MERFISH_DATA_ROOT"] = "/root/data"
    cmd = [sys.executable, "-u", f"/root/repo/{script}", *argv]
    print(f">>> {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd)
    models_vol.commit()
    subs_vol.commit()
    if result.returncode != 0:
        raise SystemExit(f"{script} exited with code {result.returncode}")


def _split(extra: str) -> list[str]:
    """Shell-split --extra so spaces inside quotes survive."""
    return shlex.split(extra) if extra else []


# ── Remote workers ───────────────────────────────────────────────────────────
# Inference / eval / sweep → A10G (cheaper, plenty for 2048×2048 Cellpose eval).
# Training → A100 (80GB Cellpose batches fit comfortably; long runs want the headroom).
# Submission-level ensembling is pure pandas → CPU-only.

@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _infer_remote(exp_name: str, extra: str):
    _run("infer.py", ["--exp-name", exp_name, *_split(extra)])


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _infer_stardist_remote(exp_name: str, extra: str):
    _run("infer_stardist.py", ["--exp-name", exp_name, *_split(extra)])


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _diagnose_remote(exp_name: str, arch: str, extra: str):
    _run("diagnose_val_errors.py",
         ["--exp-name", exp_name, "--arch", arch, *_split(extra)])


@app.function(image=image, timeout=3600, volumes=VOLUMES, env=ENV)
def _ensemble_remote(submissions: str, output: str, val_mode: bool):
    argv = ["--output", output, "--submissions", *shlex.split(submissions)]
    if val_mode:
        argv.append("--val-mode")
    _run("ensemble_submissions.py", argv)


@app.function(image=image, gpu="A100", timeout=21600, volumes=VOLUMES, env=ENV)
def _train_remote(exp_name: str, base_model: str, epochs: int, extra: str):
    _run("train.py",
         ["--exp-name", exp_name, "--base-model", base_model,
          "--epochs", str(epochs), *_split(extra)])


@app.function(image=image, gpu="A100", timeout=21600, volumes=VOLUMES, env=ENV)
def _train_stardist_remote(exp_name: str, epochs: int, extra: str):
    _run("train_stardist.py",
         ["--exp-name", exp_name, "--epochs", str(epochs), *_split(extra)])


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _sweep_remote(exp_name: str, spot_sigmas: str, extra: str):
    _run("sweep_thresholds.py",
         ["--exp-name", exp_name, "--spot-sigmas", spot_sigmas, *_split(extra)])


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _pick_best_remote(exp_name: str, spot_sigmas: str, extra: str):
    _run("eval_best_checkpoint.py",
         ["--exp-name", exp_name, "--spot-sigmas", spot_sigmas, *_split(extra)])


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _ensemble_val_remote(exp_names: str, extra: str):
    _run("ensemble_val_eval.py",
         ["--exp-names", *shlex.split(exp_names), *_split(extra)])


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES, env=ENV)
def _ensemble_infer_remote(exp_names: str, extra: str):
    _run("ensemble_infer.py",
         ["--exp-names", *shlex.split(exp_names), *_split(extra)])


@app.function(image=image, timeout=3600, volumes=VOLUMES)
def _shell_remote():
    import subprocess
    subprocess.run(["/bin/bash", "-l"])


# ── Local entrypoints (these are what `modal run` calls) ─────────────────────
@app.local_entrypoint()
def infer(exp_name: str, extra: str = ""):
    """Run Cellpose inference on test FOVs — writes submission_<exp>.csv to workspace."""
    _infer_remote.remote(exp_name, extra)


@app.local_entrypoint()
def infer_stardist(exp_name: str = "stardist", extra: str = ""):
    """Run StarDist inference on test FOVs — writes submission_<exp>.csv to workspace."""
    _infer_stardist_remote.remote(exp_name, extra)


@app.local_entrypoint()
def diagnose(exp_name: str, arch: str = "cellpose", extra: str = ""):
    """Val-set failure-mode diagnostic — writes overlays + summary.json to logs/diagnose_<exp>/."""
    _diagnose_remote.remote(exp_name, arch, extra)


@app.local_entrypoint()
def ensemble(submissions: str, output: str = "submission_ensemble.csv",
             val_mode: bool = False):
    """Spot-level majority-vote across submission CSVs.  `submissions` is a
    space-separated list of filenames inside /root/workspace/."""
    _ensemble_remote.remote(submissions, output, val_mode)


@app.local_entrypoint()
def train(exp_name: str, base_model: str = "cyto2", epochs: int = 300,
          extra: str = ""):
    """Fine-tune Cellpose.  Checkpoints land in /root/workspace/models/<exp-name>/."""
    _train_remote.remote(exp_name, base_model, epochs, extra)


@app.local_entrypoint()
def train_stardist(exp_name: str = "stardist", epochs: int = 200, extra: str = ""):
    """Fine-tune StarDist."""
    _train_stardist_remote.remote(exp_name, epochs, extra)


@app.local_entrypoint()
def sweep(exp_name: str, spot_sigmas: str = "8", extra: str = ""):
    """Grid-search cellprob×flow thresholds — writes best_params_<exp>.json."""
    _sweep_remote.remote(exp_name, spot_sigmas, extra)


@app.local_entrypoint()
def pick_best(exp_name: str, spot_sigmas: str = "8", extra: str = ""):
    """Evaluate every epoch checkpoint, promote the best to canonical name."""
    _pick_best_remote.remote(exp_name, spot_sigmas, extra)


@app.local_entrypoint()
def ensemble_val(exp_names: str, extra: str = ""):
    """Run Cellpose-only ensemble eval on val FOVs 036-040.  `exp_names` is
    space-separated."""
    _ensemble_val_remote.remote(exp_names, extra)


@app.local_entrypoint()
def ensemble_infer(exp_names: str, extra: str = ""):
    """Cellpose-only ensemble inference (note: this script has a voting bug
    documented in ensemble_submissions.py — prefer `ensemble` for heterogeneous
    architectures)."""
    _ensemble_infer_remote.remote(exp_names, extra)


@app.local_entrypoint()
def shell():
    """Drop into a bash shell inside the container with both volumes mounted —
    useful for inspecting /root/workspace/ or copying files."""
    _shell_remote.remote()
