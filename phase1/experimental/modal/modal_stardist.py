"""Run StarDist2D fine-tuning and/or inference on Modal (A100).

Volumes (shared with modal_cellpose.py):
    cell-seg-data         (read-only data)            → /data
    cell-seg-models       (persistent checkpoints)    → /models
    cell-seg-submissions  (persistent submission CSVs) → /submissions

Usage:

    # Train a fresh run (StarDist @ 28 ep is the current Kaggle leader at 0.7627 —
    # pushing past that HURT Kaggle even though val ARI kept rising, so explore
    # short runs first).
    modal run --detach modal_stardist.py \
        --exp-name stardist_v3 --epochs 28 --channel dapi

    # Train longer, then compare Kaggle at multiple snapshots
    modal run --detach modal_stardist.py \
        --exp-name stardist_long --epochs 200 --channel dapi

    # Infer-only against an existing trained run
    modal run modal_stardist.py \
        --exp-name stardist_v3 --run-train false --run-infer true

Note: StarDist uses Keras ModelCheckpoint to save `weights_best.h5` (highest
val metric seen during training) and `weights_last.h5`.  Running infer() reads
`weights_best.h5` by default.
"""
from __future__ import annotations

import modal

app = modal.App("cell-seg-stardist")

data_vol        = modal.Volume.from_name("cell-seg-data")
models_vol      = modal.Volume.from_name("cell-seg-models",      create_if_missing=True)
submissions_vol = modal.Volume.from_name("cell-seg-submissions", create_if_missing=True)

# StarDist requires TensorFlow + CUDA.  Use debian_slim and pip-install TF with
# bundled CUDA wheels rather than the `tensorflow/tensorflow:gpu` image — the
# TF image already ships a Python under /usr/local which conflicts with Modal's
# `add_python` injection ("copy_tree: cp command failed" during image build).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "procps")
    .pip_install(
        "tensorflow[and-cuda]==2.15.0",
        "stardist",
        "csbdeep",
        "numpy<2.0",
        "pandas",
        "scipy",
        "scikit-image>=0.21",
        "scikit-learn>=1.3",
        "shapely>=2.0",
        "anndata>=0.10",
        "tifffile",
        "matplotlib",
        extra_index_url="https://pypi.nvidia.com",
    )
    .add_local_dir(
        ".",
        remote_path="/root/repo",
        ignore=[
            "models/**",
            "notebooks/**",
            "tests/**",
            ".venv/**",
            ".git/**",
            ".claude/**",
            "plans/**",
            "docs/**",
            "logs/**",
            "__pycache__/**",
            "*.ipynb",
            "sample_submission.csv",
            "test_spots.csv",
            ".DS_Store",
        ],
    )
)


def _setup_workdir():
    import os
    os.chdir("/root/repo")
    os.environ["MERFISH_DATA_ROOT"] = "/data"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if not os.path.islink("models") and not os.path.exists("models"):
        os.symlink("/models", "models")


def _periodic_commit(stop_evt, interval_s: int = 300):
    import time
    while not stop_evt.wait(interval_s):
        try:
            models_vol.commit()
            print(f"[periodic commit] models volume committed ({time.strftime('%H:%M:%S')})", flush=True)
        except Exception as exc:
            print(f"[periodic commit] error: {exc}", flush=True)


# ── Training ────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,
    volumes={
        "/data":        data_vol,
        "/models":      models_vol,
        "/submissions": submissions_vol,
    },
)
def train(
    exp_name: str,
    epochs: int = 28,
    steps_per_epoch: int = 100,
    channel: str = "dapi",
    resume: bool = False,
    all_fovs: bool = False,
):
    import subprocess, sys, threading
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [
        sys.executable, "-u", "train_stardist.py",
        "--exp-name",        exp_name,
        "--epochs",          str(epochs),
        "--steps-per-epoch", str(steps_per_epoch),
        "--channel",         channel,
    ]
    if resume:    cmd.append("--resume")
    if all_fovs:  cmd.append("--all-fovs")

    print(f"[train-stardist] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)

    stop_evt = threading.Event()
    committer = threading.Thread(target=_periodic_commit, args=(stop_evt,), daemon=True)
    committer.start()
    try:
        subprocess.check_call(cmd)
    finally:
        stop_evt.set()
        models_vol.commit()
        print(f"[train-stardist] final commit for exp={exp_name}", flush=True)


# ── Inference → submission.csv ─────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,
    volumes={
        "/data":        data_vol,
        "/models":      models_vol,
        "/submissions": submissions_vol,
    },
)
def infer(exp_name: str, channel: str = "dapi"):
    import os, shutil, subprocess, sys
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [
        sys.executable, "-u", "infer_stardist.py",
        "--exp-name", exp_name,
        "--channel",  channel,
    ]
    print(f"[infer-stardist] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    sub = f"submission_{exp_name}.csv"
    if os.path.exists(sub):
        shutil.copy(sub, f"/submissions/{sub}")
        submissions_vol.commit()
        print(f"[infer-stardist] saved {sub} to cell-seg-submissions", flush=True)
    else:
        print(f"[infer-stardist] WARNING: {sub} was not created")


# ── Expand-labels sweep ────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,
    volumes={
        "/data":        data_vol,
        "/models":      models_vol,
        "/submissions": submissions_vol,
    },
)
def expand_sweep(
    exp_name: str,
    channel: str = "dapi",
    distances: str = "0,4,8,12,16,20,24,32,48",
):
    import subprocess, sys
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [
        sys.executable, "-u", "sweep_stardist_expand.py",
        "--exp-name",  exp_name,
        "--channel",   channel,
        "--distances", distances,
    ]
    print(f"[expand-sweep] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    submissions_vol.commit()
    print(f"[expand-sweep] submissions committed", flush=True)


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    exp_name: str,
    epochs: int = 28,
    steps_per_epoch: int = 100,
    channel: str = "dapi",
    resume: bool = False,
    all_fovs: bool = False,
    run_train: bool = True,
    run_infer: bool = True,
):
    if run_train:
        train.remote(
            exp_name=exp_name,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            channel=channel,
            resume=resume,
            all_fovs=all_fovs,
        )
    if run_infer:
        infer.remote(exp_name=exp_name, channel=channel)
