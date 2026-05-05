"""Run cellpose-SAM (cpsam) inference on phase-2 train/test FOVs and save masks.

Bypasses infer_baseline.py's model-load step that's been hanging. This is
a minimal cellpose-only call.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from phase2.src import io  # noqa: E402


def _spot_density(fov_spots, sigma: float, image_size: int = 2048) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    rows = np.clip(fov_spots["image_row"].astype(int).values, 0, image_size - 1)
    cols = np.clip(fov_spots["image_col"].astype(int).values, 0, image_size - 1)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    for r, c in zip(rows, cols):
        canvas[r, c] += 1.0
    return gaussian_filter(canvas, sigma=sigma, mode="constant").astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fovs", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--phase", default="phase2")
    p.add_argument("--data-root", default=None)
    p.add_argument("--cellprob-threshold", type=float, default=-1.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--diameter", type=float, default=0.0)
    p.add_argument("--include-spot-density", action="store_true")
    p.add_argument("--spot-density-sigma", type=float, default=8.0)
    p.add_argument("--device", default="mps")
    p.add_argument("--checkpoint", default=None,
                   help="Path to fine-tuned cpsam checkpoint (default: zero-shot)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    import torch
    from cellpose import models as cp_models

    if args.device == "mps":
        gpu = torch.backends.mps.is_available()
        device = torch.device("mps") if gpu else torch.device("cpu")
    elif args.device == "cuda":
        gpu = torch.cuda.is_available()
        device = torch.device("cuda") if gpu else torch.device("cpu")
    else:
        gpu = False
        device = torch.device("cpu")

    t0 = time.time()
    if args.checkpoint:
        print(f"[cpsam-infer] loading fine-tuned ckpt: {args.checkpoint}", flush=True)
        model = cp_models.CellposeModel(gpu=gpu, pretrained_model=str(args.checkpoint), device=device)
    else:
        print(f"[cpsam-infer] loading cpsam zero-shot, device={device} gpu={gpu}", flush=True)
        model = cp_models.CellposeModel(gpu=gpu, device=device)
    print(f"[cpsam-infer] model loaded in {time.time()-t0:.1f}s", flush=True)

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = ROOT / args.phase / "data"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fovs = [f.strip() for f in args.fovs.split(",") if f.strip()]

    # Load test/train spots once if needed for density channel
    spots_csv = None
    if args.include_spot_density:
        if args.split == "test":
            spots_csv = data_root / "test_spots.csv"
        else:
            spots_csv = data_root / "train" / "ground_truth" / "spots_train.csv"
        import pandas as pd
        all_spots = pd.read_csv(spots_csv)

    for fov in fovs:
        t0 = time.time()
        try:
            dapi, polyt = io.load_fov_images(fov, split=args.split)
        except FileNotFoundError as e:
            print(f"[cpsam-infer] {fov}: {e}", flush=True)
            continue
        polyt2d = polyt.max(axis=0).astype(np.float32)
        dapi2d = dapi.max(axis=0).astype(np.float32)
        channels = [polyt2d, dapi2d]
        if args.include_spot_density:
            fov_spots = all_spots[all_spots["fov"] == fov]
            channels.append(_spot_density(fov_spots, sigma=args.spot_density_sigma))
        img = np.stack(channels, axis=0).astype(np.float32)
        masks, _flows, _styles = model.eval(
            img, channel_axis=0, diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        masks = masks.astype(np.int32)
        out_path = out_dir / f"{fov}.npy"
        np.save(out_path, masks)
        print(f"[cpsam-infer] {fov}: {int(masks.max())} cells in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
