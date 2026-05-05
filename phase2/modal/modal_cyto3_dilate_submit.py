"""Modal GPU runner for the cyto3_p1stack_ep035 + dilate1 RF500 submission.

Run from repo root:

    modal run phase2/modal/modal_cyto3_dilate_submit.py

Prereq files in the `cell-seg-phase2` volume:
    /models/cyto3_p1stack_ep035
    /models/baseline-codelab-rf500-log1p-mf01/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal


REPO = Path(__file__).resolve().parents[2]
APP_NAME = "phase2-cyto3-dilate-submit"
VOL_NAME = "cell-seg-phase2"
TEST_FOVS = "FOV_E,FOV_F,FOV_G,FOV_H,FOV_I,FOV_J,FOV_K,FOV_L,FOV_M,FOV_N"

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(VOL_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install(
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "scikit-image",
        "cellpose>=4.0",
        "torch",
    )
    .add_local_dir(str(REPO / "phase2" / "src"), "/root/repo/phase2/src", copy=True)
    .add_local_dir(str(REPO / "phase2" / "tasks"), "/root/repo/phase2/tasks", copy=True)
    .add_local_dir(str(REPO / "phase2" / "scripts"), "/root/repo/phase2/scripts", copy=True)
    .add_local_file(str(REPO / "phase2" / "__init__.py"), "/root/repo/phase2/__init__.py", copy=True)
    .add_local_file(str(REPO / "phase2" / "__main__.py"), "/root/repo/phase2/__main__.py", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/repo",
            "MERFISH_DATA_ROOT": "/root/data",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp",
        }
    )
)


def _run(cmd: list[str]) -> None:
    print(">>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, volumes={"/root/data": data_vol}, gpu="L4", timeout=7200)
def run_remote() -> dict:
    data_vol.reload()

    ckpt = Path("/root/data/models/cyto3_p1stack_ep035")
    clf = Path("/root/data/models/baseline-codelab-rf500-log1p-mf01")
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not clf.is_dir():
        raise FileNotFoundError(clf)

    root = Path("/root/data/codex-cyto3-z2-cp-1-dilate1-rf500")
    raw_masks = root / "raw_masks"
    dilated_masks = root / "dilate1_masks"
    submit_dir = root / "submission"
    root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    _run(
        [
            py,
            "-u",
            "/root/repo/phase2/scripts/infer_cellpose_masks.py",
            "--fovs",
            TEST_FOVS,
            "--split",
            "test",
            "--out-dir",
            str(raw_masks),
            "--seg-checkpoint",
            str(ckpt),
            "--include-spot-density",
            "--spot-density-sigma",
            "8.0",
            "--cellpose-diameter",
            "0.0",
            "--cellprob-threshold",
            "-1.0",
            "--flow-threshold",
            "0.4",
            "--preprocess",
            "none",
            "--device",
            "cuda",
        ]
    )
    _run(
        [
            py,
            "-u",
            "/root/repo/phase2/scripts/postprocess_masks.py",
            "--masks-dir",
            str(raw_masks),
            "--out-dir",
            str(dilated_masks),
            "--fovs",
            TEST_FOVS,
            "--split",
            "test",
            "--morph",
            "dilate",
            "--radius",
            "1",
        ]
    )
    _run(
        [
            py,
            "-m",
            "phase2",
            "infer-baseline",
            "--backend",
            "local",
            "--models-dir",
            str(clf),
            "--masks-dir",
            str(dilated_masks),
            "--test-fovs",
            TEST_FOVS,
            "--out-dir",
            str(submit_dir),
        ]
    )

    data_vol.commit()
    summary_path = submit_dir / "summary.json"
    return {
        "root": str(root),
        "submission": str(submit_dir / "submission.csv"),
        "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
    }


@app.local_entrypoint()
def main():
    result = run_remote.remote()
    print(json.dumps(result, indent=2))
