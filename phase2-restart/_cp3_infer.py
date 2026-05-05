"""Cellpose 3.x inference for phase-2 FOVs. Run with /tmp/cp3_venv/bin/python.

Saves <FOV>.npy masks (int32) into --out-dir. Uses 2-channel input
[polyT_max, DAPI_max] which is what cyto3/cyto2 expect (cyto channel + nucleus channel).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "phase2"))
sys.path.insert(0, str(ROOT / "phase1"))

from src import io as p2_io  # noqa: E402  uses phase2/src/io.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fovs", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--phase", choices=("phase1", "phase2"), default="phase2")
    p.add_argument("--data-root", default=None,
                   help="Override data root. Defaults to phase2/data or phase1/data.")
    p.add_argument("--model", default="cyto3", choices=("cyto3", "cyto2", "nuclei", "tissuenet_cp3"))
    p.add_argument("--diameter", type=float, default=0.0,
                   help="0 = auto-estimate")
    p.add_argument("--cellprob-threshold", type=float, default=-1.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    args = p.parse_args()

    from cellpose import models as cp_models
    import torch
    if args.device == "auto":
        gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    else:
        gpu = args.device != "cpu"

    print(f"[cp3-infer] loading {args.model}, gpu={gpu}")
    if args.model == "nuclei":
        model = cp_models.Cellpose(model_type=args.model, gpu=gpu)
        channels = [0, 0]  # single channel
    elif args.model == "tissuenet_cp3":
        model = cp_models.CellposeModel(model_type=args.model, gpu=gpu)
        channels = [1, 2]  # if it's a 2-channel model
    else:
        model = cp_models.Cellpose(model_type=args.model, gpu=gpu)
        channels = [1, 2]  # cyto on ch 1, nuclei on ch 2 (1-indexed)

    fovs = [s.strip() for s in args.fovs.split(",") if s.strip()]
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = ROOT / args.phase / "data"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fov in fovs:
        fov_dir = data_root / args.split / fov
        if not fov_dir.exists():
            print(f"[cp3-infer] missing {fov_dir}, skip")
            continue
        print(f"[cp3-infer] {fov} ({fov_dir})")
        dapi, polyt = p2_io.load_fov_images(str(fov_dir))
        # For cyto3: 2-channel image with cyto first, then nucleus.
        # We use polyT_max (signal of cell extent) as cyto, DAPI_max as nucleus.
        polyt_max = np.max(polyt, axis=0).astype(np.float32)
        dapi_max = np.max(dapi, axis=0).astype(np.float32)
        if args.model == "nuclei":
            # nuclei: single channel, just DAPI
            img = dapi_max
        else:
            # 2-channel: cyto=polyT, nucleus=DAPI; cellpose expects (C, H, W) or (H, W, C)
            img = np.stack([polyt_max, dapi_max], axis=0)  # (2, H, W)

        # Cellpose 3.x eval API
        masks, flows, styles, diams = model.eval(
            img,
            diameter=args.diameter if args.diameter > 0 else None,
            channels=channels,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
        )
        masks = masks.astype(np.int32)
        out_path = out_dir / f"{fov}.npy"
        np.save(out_path, masks)
        n_cells = int(masks.max())
        print(f"[cp3-infer]   {fov}: {n_cells} cells -> {out_path}")


if __name__ == "__main__":
    main()
