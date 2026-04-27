"""Run Cellpose fine-tuning and/or inference on Modal (A100).

Volumes:
    cell-seg-data         (read-only data, pre-uploaded) → /data
    cell-seg-models       (persistent model checkpoints) → /models
    cell-seg-submissions  (persistent submission CSVs)   → /submissions

Usage examples:

    # Train-only (5h timeout, detached so laptop can close)
    modal run --detach modal_cellpose.py \
        --exp-name cyto2_modal --base-model cyto2 \
        --epochs 500 --lr-schedule warmup_cosine --augment true \
        --run-infer false

    # Train + infer in one go (most common)
    modal run --detach modal_cellpose.py \
        --exp-name cyto2_long_modal --base-model cyto2 \
        --epochs 500 --augment true

    # Infer only (reuse already-trained checkpoint in the models volume)
    modal run modal_cellpose.py \
        --exp-name cyto2_long_modal \
        --run-train false --run-infer true \
        --cellprob-threshold -1.5 --flow-threshold 0.4

Outputs:
    Checkpoints   → volume `cell-seg-models` under `<exp-name>/`
    Submission    → volume `cell-seg-submissions` as `submission_<exp-name>.csv`

Pull artefacts locally:
    modal volume get cell-seg-submissions submission_<exp>.csv .
    modal volume get cell-seg-models <exp>/ ./models/
"""
from __future__ import annotations

import modal

app = modal.App("cell-seg-cellpose")

data_vol        = modal.Volume.from_name("cell-seg-data")
models_vol      = modal.Volume.from_name("cell-seg-models",      create_if_missing=True)
submissions_vol = modal.Volume.from_name("cell-seg-submissions", create_if_missing=True)

# ── Image: Cellpose v4 (torch-based). ───────────────────────────────────────────
# Using the Modal pytorch base so CUDA is already wired up for the A100.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel", add_python="3.11"
    )
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "cellpose>=4.0",
        "numpy<2.0",
        "pandas",
        "scipy",
        "scikit-image>=0.21",
        "scikit-learn>=1.3",
        "shapely>=2.0",
        "anndata>=0.10",
        "tifffile",
        "matplotlib",
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
    """Arrange `/root/repo` so the existing scripts run unmodified.

    - cwd is the repo root, so `src.*` imports and `models/<exp>/` relative paths work.
    - `models/` becomes a symlink to the persistent Modal volume at `/models`.
    - `MERFISH_DATA_ROOT=/data` so the env-var lookup added to train/infer scripts picks up the mounted volume.
    """
    import os
    os.chdir("/root/repo")
    os.environ["MERFISH_DATA_ROOT"] = "/data"
    if not os.path.islink("models") and not os.path.exists("models"):
        os.symlink("/models", "models")


def _periodic_commit(stop_evt, interval_s: int = 300):
    """Commit the models volume every `interval_s` seconds so long runs are safe."""
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
    timeout=60 * 60 * 8,  # 8h — plenty for 500 epochs on 35 FOVs
    volumes={
        "/data":        data_vol,
        "/models":      models_vol,
        "/submissions": submissions_vol,
    },
)
def train(
    exp_name: str,
    base_model: str = "cyto2",
    epochs: int = 500,
    lr_schedule: str = "warmup_cosine",
    weight_decay: float = 0.1,
    spot_sigmas: str = "8",
    augment: bool = True,
    all_fovs: bool = False,
    multi_z: bool = False,
    use_union_z: bool = False,
    zstats: bool = False,
):
    import subprocess, sys, threading
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [
        sys.executable, "-u", "train.py",
        "--base-model",   base_model,
        "--exp-name",     exp_name,
        "--epochs",       str(epochs),
        "--lr-schedule",  lr_schedule,
        "--weight-decay", str(weight_decay),
        "--spot-sigmas",  spot_sigmas,
    ]
    if augment:      cmd.append("--augment")
    if all_fovs:     cmd.append("--all-fovs")
    if multi_z:      cmd.append("--multi-z")
    if use_union_z:  cmd.append("--use-union-z")
    if zstats:       cmd.append("--zstats")

    print(f"[train] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)

    stop_evt = threading.Event()
    committer = threading.Thread(target=_periodic_commit, args=(stop_evt,), daemon=True)
    committer.start()
    try:
        subprocess.check_call(cmd)
    finally:
        stop_evt.set()
        models_vol.commit()
        print(f"[train] final commit done for exp={exp_name}", flush=True)


# ── Inference → submission.csv ─────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,  # 2h — inference + threshold post-processing
    volumes={
        "/data":        data_vol,
        "/models":      models_vol,
        "/submissions": submissions_vol,
    },
)
def infer(
    exp_name: str,
    spot_sigmas: str = "8",
    cellprob_threshold: float = -1.0,
    flow_threshold: float = 0.4,
    params_json: str | None = None,
    tta: bool = False,
    nn_radius: int = 0,
    stitch_threshold: float = 0.0,
    prob_refine: bool = False,
):
    import os, shutil, subprocess, sys
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    cmd = [
        sys.executable, "-u", "infer.py",
        "--exp-name",            exp_name,
        "--spot-sigmas",         spot_sigmas,
        "--cellprob-threshold",  str(cellprob_threshold),
        "--flow-threshold",      str(flow_threshold),
    ]
    if params_json:         cmd += ["--params-json", params_json]
    if tta:                 cmd.append("--tta")
    if nn_radius > 0:       cmd += ["--nn-radius", str(nn_radius)]
    if stitch_threshold:    cmd += ["--stitch-threshold", str(stitch_threshold)]
    if prob_refine:         cmd.append("--prob-refine")

    print(f"[infer] exp={exp_name}  cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    # infer.py names output `submission_<suffix>.csv` where suffix encodes exp+flags.
    # Copy every newly-produced submission_*.csv into the submissions volume.
    copied = []
    for fname in os.listdir("."):
        if fname.startswith("submission_") and fname.endswith(".csv") and exp_name in fname:
            shutil.copy(fname, f"/submissions/{fname}")
            copied.append(fname)
    submissions_vol.commit()
    print(f"[infer] copied to cell-seg-submissions volume: {copied}", flush=True)


# ── Validation helpers — run only if you need them ─────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,
    volumes={"/data": data_vol, "/models": models_vol},
)
def eval_best_checkpoint(exp_name: str, spot_sigmas: str = "8"):
    """Evaluate all epoch checkpoints and promote the best one to canonical path."""
    import subprocess, sys
    _setup_workdir()
    models_vol.reload()

    subprocess.check_call([
        sys.executable, "-u", "eval_best_checkpoint.py",
        "--exp-name",    exp_name,
        "--spot-sigmas", spot_sigmas,
    ])
    models_vol.commit()


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={"/data": data_vol, "/models": models_vol},
)
def sweep_thresholds(exp_name: str, spot_sigmas: str = "8"):
    """Grid-search cellprob × flow thresholds; writes best_params_<exp>.json into /models."""
    import os, shutil, subprocess, sys
    _setup_workdir()
    models_vol.reload()

    subprocess.check_call([
        sys.executable, "-u", "sweep_thresholds.py",
        "--exp-name",    exp_name,
        "--spot-sigmas", spot_sigmas,
    ])
    # Move best_params JSON into the models volume so later infer() calls can load it
    src = f"best_params_{exp_name}.json"
    if os.path.exists(src):
        shutil.copy(src, f"/models/{src}")
        models_vol.commit()
        print(f"[sweep] saved {src} to cell-seg-models volume")


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
def ensemble_infer(exp_names: str, spot_sigmas_map: str = ""):
    """Spot-level majority-vote ensemble across multiple trained models.

    Args are comma-separated strings (Modal CLI doesn't support list[str] params):
        exp_names:       "cyto2_long,nuclei,multiscale"
        spot_sigmas_map: "cyto2_long:8,multiscale:4,8,16"  (semicolon between entries)
    """
    import os, shutil, subprocess, sys
    _setup_workdir()
    data_vol.reload()
    models_vol.reload()

    names = [e.strip() for e in exp_names.split(",") if e.strip()]
    cmd = [sys.executable, "-u", "ensemble_infer.py", "--exp-names", *names]
    if spot_sigmas_map:
        # Entries separated by ';' so each entry can still contain commas (sigma lists).
        entries = [e.strip() for e in spot_sigmas_map.split(";") if e.strip()]
        cmd += ["--spot-sigmas-map", *entries]

    print(f"[ensemble] cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    for fname in os.listdir("."):
        if fname.startswith("submission_ensemble") and fname.endswith(".csv"):
            shutil.copy(fname, f"/submissions/{fname}")
    submissions_vol.commit()


# ── Local entrypoint (modal run ...) ───────────────────────────────────────────

@app.local_entrypoint()
def main(
    exp_name: str,
    base_model: str = "cyto2",
    epochs: int = 500,
    lr_schedule: str = "warmup_cosine",
    weight_decay: float = 0.1,
    spot_sigmas: str = "8",
    augment: bool = True,
    all_fovs: bool = False,
    multi_z: bool = False,
    use_union_z: bool = False,
    zstats: bool = False,
    run_train: bool = True,
    run_infer: bool = True,
    cellprob_threshold: float = -1.0,
    flow_threshold: float = 0.4,
    params_json: str = "",
):
    if run_train:
        train.remote(
            exp_name=exp_name,
            base_model=base_model,
            epochs=epochs,
            lr_schedule=lr_schedule,
            weight_decay=weight_decay,
            spot_sigmas=spot_sigmas,
            augment=augment,
            all_fovs=all_fovs,
            multi_z=multi_z,
            use_union_z=use_union_z,
            zstats=zstats,
        )
    if run_infer:
        infer.remote(
            exp_name=exp_name,
            spot_sigmas=spot_sigmas,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            params_json=params_json or None,
        )
