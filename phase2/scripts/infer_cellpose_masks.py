"""Run Cellpose/CPSAM inference and save per-FOV mask .npy files.

This is intentionally mask-only. It lets us test segmentation backbones,
thresholds, channel choices, and image preprocessing without touching the
classifier path or local-validation metric.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.src import io
from phase2.tasks.infer_baseline import _pick_device, _spot_density


def _add_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fovs", required=True, help="Comma-separated FOVs.")
    p.add_argument("--split", default="train", choices=("train", "test"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--seg-checkpoint", default=None,
                   help="Fine-tuned Cellpose/CPSAM checkpoint. If omitted, uses off-the-shelf cpsam.")
    p.add_argument("--include-spot-density", action="store_true")
    p.add_argument("--spot-density-sigma", type=float, default=8.0)
    p.add_argument("--cellpose-diameter", type=float, default=0.0)
    p.add_argument("--cellprob-threshold", type=float, default=-0.5)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--preprocess", default="none", choices=("none", "pclip", "clahe"),
                   help="Inference-time preprocessing for polyT/DAPI channels only.")
    p.add_argument("--clahe-clip-limit", type=float, default=0.01)
    return p.parse_args()


def _scale01(x: np.ndarray, lo: float = 1.0, hi: float = 99.8) -> np.ndarray:
    a, b = np.percentile(x, (lo, hi))
    if b <= a:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x.astype(np.float32) - a) / (b - a), 0.0, 1.0)
    return y.astype(np.float32)


def _preprocess_channel(x: np.ndarray, mode: str, clahe_clip_limit: float) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32)
    y = _scale01(x)
    if mode == "clahe":
        from skimage import exposure
        y = exposure.equalize_adapthist(y, clip_limit=clahe_clip_limit).astype(np.float32)
    return (y * 65535.0).astype(np.float32)


def _load_spots(split: str) -> pd.DataFrame:
    if split == "test":
        return pd.read_csv(io.data_root() / "test_spots.csv")
    return pd.read_csv(io.ground_truth_dir() / "spots_train.csv")


def main() -> int:
    args = _add_args()
    fovs = [f.strip() for f in args.fovs.split(",") if f.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from cellpose import models as cp_models

    device, gpu = _pick_device(args.device)
    print(f"device={device} gpu={gpu}", flush=True)
    if args.seg_checkpoint:
        ckpt = Path(args.seg_checkpoint)
        if not ckpt.exists():
            print(f"[fatal] missing checkpoint: {ckpt}")
            return 2
        print(f"checkpoint={ckpt}", flush=True)
        model = cp_models.CellposeModel(gpu=gpu, pretrained_model=str(ckpt), device=device)
    else:
        print("checkpoint=(off-the-shelf cpsam)", flush=True)
        model = cp_models.CellposeModel(gpu=gpu, device=device)

    spots = _load_spots(args.split)
    print(f"split={args.split} fovs={fovs}", flush=True)
    print(f"preprocess={args.preprocess} include_spot_density={args.include_spot_density}", flush=True)

    summary: dict[str, dict] = {}
    for fov in fovs:
        print(f"\n=== {fov} ===", flush=True)
        t0 = time.time()
        dapi, polyt = io.load_fov_images(fov, split=args.split)
        dapi2d = dapi.max(axis=0).astype(np.float32)
        polyt2d = polyt.max(axis=0).astype(np.float32)
        fov_spots = spots[spots["fov"] == fov].copy()

        channels = [
            _preprocess_channel(polyt2d, args.preprocess, args.clahe_clip_limit),
            _preprocess_channel(dapi2d, args.preprocess, args.clahe_clip_limit),
        ]
        if args.include_spot_density:
            channels.append(_spot_density(fov_spots, sigma=args.spot_density_sigma))
        img = np.stack(channels, axis=0).astype(np.float32)
        load_s = time.time() - t0

        t0 = time.time()
        masks, _flows, _styles = model.eval(
            img,
            channel_axis=0,
            diameter=args.cellpose_diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        seg_s = time.time() - t0
        masks = masks.astype(np.int32)
        np.save(out_dir / f"{fov}.npy", masks)
        rows = fov_spots["image_row"].to_numpy().astype(int).clip(0, masks.shape[0] - 1)
        cols = fov_spots["image_col"].to_numpy().astype(int).clip(0, masks.shape[1] - 1)
        frac_in_cell = float((masks[rows, cols] > 0).mean()) if len(fov_spots) else 0.0
        summary[fov] = {
            "n_cells": int(masks.max()),
            "n_spots": int(len(fov_spots)),
            "frac_in_cell": frac_in_cell,
            "load_s": load_s,
            "seg_s": seg_s,
        }
        print(f"cells={int(masks.max())} frac_in_cell={frac_in_cell:.4f} load={load_s:.1f}s seg={seg_s:.1f}s", flush=True)

    import json
    (out_dir / "mask_summary.json").write_text(json.dumps({
        "args": vars(args),
        "per_fov": summary,
    }, indent=2))
    print(f"\nwrote masks -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
