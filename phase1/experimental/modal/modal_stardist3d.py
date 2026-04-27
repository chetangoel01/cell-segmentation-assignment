"""Run StarDist3D fine-tuning and/or inference on Modal (L40S).

Volumes (shared with modal_stardist.py):
    cell-seg-data         (read-only data)             → /data
    cell-seg-models       (persistent checkpoints)     → /models
    cell-seg-submissions  (persistent submission CSVs) → /submissions

Usage:

    modal run --detach modal_stardist3d.py \
        --exp-name stardist3d_v1 --epochs 30 --all-fovs
"""
from __future__ import annotations

import modal

app = modal.App("cell-seg-stardist3d")

data_vol        = modal.Volume.from_name("cell-seg-data")
models_vol      = modal.Volume.from_name("cell-seg-models",      create_if_missing=True)
submissions_vol = modal.Volume.from_name("cell-seg-submissions", create_if_missing=True)

# Same TF-CUDA image as modal_stardist.py (tensorrt-libs pulled from NVIDIA's
# PyPI index). Intentionally a separate image object so changes here don't
# pull the stardist-2D app into an unintentional rebuild.
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
    epochs: int = 30,
    steps_per_epoch: int = 100,
    n_rays: int = 96,
    patch_xy: int = 256,
    batch_size: int = 1,
    all_fovs: bool = False,
    resume: bool = False,
):
    import subprocess, sys, threading
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [
        sys.executable, "-u", "train_stardist3d.py",
        "--exp-name",        exp_name,
        "--epochs",          str(epochs),
        "--steps-per-epoch", str(steps_per_epoch),
        "--n-rays",          str(n_rays),
        "--patch-xy",        str(patch_xy),
        "--batch-size",      str(batch_size),
    ]
    if all_fovs: cmd.append("--all-fovs")
    if resume:   cmd.append("--resume")

    print(f"[train-stardist3d] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)

    stop_evt = threading.Event()
    committer = threading.Thread(target=_periodic_commit, args=(stop_evt,), daemon=True)
    committer.start()
    try:
        subprocess.check_call(cmd)
    finally:
        stop_evt.set()
        models_vol.commit()
        print(f"[train-stardist3d] final commit for exp={exp_name}", flush=True)


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
def infer(exp_name: str):
    import os, shutil, subprocess, sys
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [sys.executable, "-u", "infer_stardist3d.py", "--exp-name", exp_name]
    print(f"[infer-stardist3d] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    sub = f"submission_{exp_name}.csv"
    if os.path.exists(sub):
        shutil.copy(sub, f"/submissions/{sub}")
        submissions_vol.commit()
        print(f"[infer-stardist3d] saved {sub} to cell-seg-submissions", flush=True)
    else:
        print(f"[infer-stardist3d] WARNING: {sub} was not created")


@app.local_entrypoint()
def main(
    exp_name: str,
    epochs: int = 30,
    steps_per_epoch: int = 100,
    n_rays: int = 96,
    patch_xy: int = 256,
    batch_size: int = 1,
    all_fovs: bool = False,
    resume: bool = False,
    run_train: bool = True,
    run_infer: bool = True,
):
    if run_train:
        train.remote(
            exp_name=exp_name,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            n_rays=n_rays,
            patch_xy=patch_xy,
            batch_size=batch_size,
            all_fovs=all_fovs,
            resume=resume,
        )
    if run_infer:
        infer.remote(exp_name=exp_name)
