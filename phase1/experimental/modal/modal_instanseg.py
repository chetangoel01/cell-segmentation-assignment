"""Run InstanSeg zero-shot inference on Modal (L40S).

No training — uses the pretrained `fluorescence_nuclei_and_cells` weights that
the InstanSeg PyPI package auto-downloads from HuggingFace on first call.

Volumes:
    cell-seg-data         (read-only data)            → /data
    cell-seg-submissions  (persistent submission CSVs) → /submissions

Usage:

    modal run modal_instanseg.py --exp-name instanseg_v1
"""
from __future__ import annotations

import modal

app = modal.App("cell-seg-instanseg")

data_vol        = modal.Volume.from_name("cell-seg-data")
submissions_vol = modal.Volume.from_name("cell-seg-submissions", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel", add_python="3.11"
    )
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "instanseg-torch",
        "numpy<2.0",
        "pandas",
        "scipy",
        "scikit-image>=0.21",
        "scikit-learn>=1.3",
        "shapely>=2.0",
        "tifffile",
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


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,
    volumes={
        "/data":        data_vol,
        "/submissions": submissions_vol,
    },
)
def infer(exp_name: str = "instanseg_v1"):
    import os, shutil, subprocess, sys
    _setup_workdir()
    data_vol.reload()

    cmd = [sys.executable, "-u", "infer_instanseg.py"]
    print(f"[infer-instanseg] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    # infer_instanseg.py writes to `submission_instanseg.csv` — rename to include exp tag
    src = "submission_instanseg.csv"
    dst = f"submission_{exp_name}.csv"
    if os.path.exists(src):
        shutil.copy(src, f"/submissions/{dst}")
        submissions_vol.commit()
        print(f"[infer-instanseg] saved {dst} to cell-seg-submissions", flush=True)
    else:
        print(f"[infer-instanseg] WARNING: {src} was not created")


@app.local_entrypoint()
def main(exp_name: str = "instanseg_v1"):
    infer.remote(exp_name=exp_name)
