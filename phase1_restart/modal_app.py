"""Modal entrypoint for MEDIAR zero-shot inference and fine-tune.

Why Modal: MEDIAR's pickled weights were saved against an older PyTorch ecosystem
(SMP 0.3.3, monai 1.3.0, numpy 1.24, Python 3.10). The Mac venv is on Python 3.14
with current SMP, which has multiple API drifts that make local inference brittle.
Pinning everything inside a Modal container is the path of least resistance.

Volumes:
  cell-seg-data         (existing, Mac-uploaded MERFISH FOVs at /root/data)
  cell-seg-phase1-models (created by this app, holds main_model.pt and ckpts)

Functions:
  upload_mediar_weights   — one-time: copy main_model.pt from local repo into volume
  zero_shot               — run pretrained MEDIAR on a list of FOVs, return masks
  fine_tune               — fine-tune from main_model.pt, save best ckpt to volume
  infer_with_checkpoint   — run a fine-tuned ckpt on test FOVs

Run from Mac:
  modal run phase1_restart/modal_app.py::upload_mediar_weights
  modal run phase1_restart/modal_app.py::zero_shot --split val
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

# Image with MEDIAR's pinned deps. Python 3.10 because SMP 0.3.3 / numpy 1.24 don't have
# wheels for newer Pythons, and we don't want to chase another version-rot rabbit hole.
mediar_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.24.3",
        "scipy==1.12.0",
        "scikit-image",
        "monai==1.3.0",
        "segmentation_models_pytorch==0.3.3",
        "tifffile==2023.4.12",
        "fastremap==1.14.1",
        "numba==0.57.1",
        "timm",  # MEDIAR uses an older timm path; recent versions of timm
                 # ship the path so this works as long as we don't pin too high
        "huggingface_hub",
        "pandas",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Lee-Gihun/MEDIAR.git /opt/MEDIAR",
    )
    .add_local_file(
        "phase1_restart/external/mediar_inference.py",
        remote_path="/root/predictpy/mediar_inference.py",
    )
)

app = modal.App("phase1-restart-mediar", image=mediar_image)
data_volume = modal.Volume.from_name("cell-seg-data", create_if_missing=False)
models_volume = modal.Volume.from_name(
    "cell-seg-phase1-models", create_if_missing=True
)


@app.function(volumes={"/root/models": models_volume}, timeout=120)
def verify_weights() -> dict:
    """Verify main_model.pt is in the volume after `modal volume put` ran from Mac."""
    p = Path("/root/models/mediar/main_model.pt")
    if not p.exists():
        return {"ok": False, "reason": f"{p} missing — run `modal volume put cell-seg-phase1-models main_model.pt /mediar/main_model.pt` first"}
    return {"ok": True, "path": str(p), "size_mb": p.stat().st_size / 1e6}


def _build_mediar_model(device: str = "cpu"):
    """Load main_model.pt with the correct __main__ aliases (the pickle expects
    SegformerGH and friends in __main__)."""
    import sys
    import torch

    # timm >= 1.0 moved timm.models.layers -> timm.layers. The pickle was made
    # against an older timm; alias the old paths to the new ones.
    try:
        import timm.layers.drop  # noqa: F401
        sys.modules.setdefault("timm.models.layers", sys.modules["timm.layers"])
        sys.modules.setdefault("timm.models.layers.drop", sys.modules["timm.layers.drop"])
    except ImportError:
        pass

    # The pickle references SegformerGH from the user's predict.py at training time.
    # HF Space's predict.py defines SegformerGH at module scope; we vendor a copy.
    sys.path.insert(0, "/root/predictpy")
    import mediar_inference  # type: ignore
    sys.modules["__main__"] = mediar_inference

    model = torch.load(
        "/root/models/mediar/main_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.eval()
    model.to(device)
    return model, mediar_inference


def _to_mediar_input(image):
    """numpy (C=2, H, W) [polyT, DAPI] -> torch (1, 3, H, W) for MEDIAR."""
    import numpy as np
    import torch
    polyT = image[0]
    dapi = image[1]
    third = ((polyT + dapi) / 2.0).astype(np.float32)
    hwc = np.stack([polyT, dapi, third], axis=-1).astype(np.float32)
    mn, mx = hwc.min(), hwc.max()
    if mx > mn:
        hwc = (hwc - mn) / (mx - mn)
    chw = np.moveaxis(hwc, -1, 0)
    return torch.from_numpy(chw).unsqueeze(0).float()


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/root/models": models_volume},
)
def predict_one(image_bytes: bytes, image_shape: tuple[int, ...]) -> bytes:
    """Run MEDIAR pretrained on one image (C, H, W) numpy float32. Returns mask bytes."""
    import numpy as np
    import torch

    image = np.frombuffer(image_bytes, dtype=np.float32).reshape(image_shape).copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, mi = _build_mediar_model(device=device)
    x = _to_mediar_input(image).to(device)

    with torch.no_grad():
        outputs = mi.sliding_window_inference(
            x, roi_size=512, sw_batch_size=4, predictor=model,
            padding_mode="reflect", mode="gaussian", overlap=0.5, device="cpu",
        )
        outputs = outputs.cpu().squeeze()
        mask = mi.post_process(outputs.numpy(), device)

    return np.asarray(mask).astype(np.int32).tobytes()


@app.local_entrypoint()
def verify():
    """Verify weights present on the volume (after `modal volume put`)."""
    print(json.dumps(verify_weights.remote()))


@app.local_entrypoint()
def predict_smoke(fov: str = "FOV_001"):
    """Smoke: run zero-shot MEDIAR on a single FOV via Modal, save mask locally."""
    import sys
    import numpy as np
    sys.path.insert(0, ".")
    from phase1_restart.pilot.data import load_fov_channels
    img = load_fov_channels(fov, channels=["polyT", "DAPI"])
    print(f"loaded {fov} shape={img.shape}; sending to Modal ...")
    mask_bytes = predict_one.remote(img.tobytes(), img.shape)
    mask = np.frombuffer(mask_bytes, dtype=np.int32).reshape(img.shape[1], img.shape[2])
    out_path = Path(f"phase1_restart/outputs/zero_shot/mediar/smoke/{fov}_mask.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask)
    print(f"wrote {out_path}: max_label={mask.max()}, n_unique={len(np.unique(mask))}")


@app.local_entrypoint()
def predict_split(split: str = "val", out_subdir: str = ""):
    """Run zero-shot MEDIAR on every FOV in a split, save masks locally.

    splits: val | test_proxy | test
    """
    import sys
    import time
    import numpy as np
    sys.path.insert(0, ".")
    from phase1_restart.pilot.data import list_fovs, load_fov_channels

    fovs = list_fovs(split)
    sub = out_subdir or split
    out_root = Path(f"phase1_restart/outputs/zero_shot/mediar/{sub}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"running MEDIAR zero-shot on {len(fovs)} FOVs in split={split}: {fovs}")

    for fov in fovs:
        t0 = time.time()
        out_path = out_root / f"{fov}_mask.npy"
        if out_path.exists():
            print(f"  {fov}: cached at {out_path}, skipping")
            continue
        img = load_fov_channels(fov, channels=["polyT", "DAPI"])
        mask_bytes = predict_one.remote(img.tobytes(), img.shape)
        mask = np.frombuffer(mask_bytes, dtype=np.int32).reshape(img.shape[1], img.shape[2])
        np.save(out_path, mask)
        print(f"  {fov}: max_label={mask.max()}, n_cells={len(np.unique(mask)) - 1}, took {time.time() - t0:.1f}s")
    print(f"done. masks at {out_root}")
