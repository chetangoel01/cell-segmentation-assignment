"""Mask-level ensembling for segmentation models.

Strategy: for each pixel, decide whether it's "cell" via majority vote across
input masks (binary cell / no-cell). Then take connected components of the
voted "cell" pixels, assigning a fresh integer label per component.

This is decorrelated from any one seg's labeling (we don't need to align
cell IDs across models — we just take the union of pixels that ≥K models
think are inside *some* cell, then re-segment by connected components).

Two modes:
  - intersect (K=N): only keep pixels ALL models agree on (tight masks)
  - majority (K=N//2+1): keep pixels at least half models agree on (default)
  - union (K=1): keep pixels ANY model thinks is a cell (permissive)

Usage:
  python phase2/scripts/mask_ensemble.py \\
      --inputs cyto3=phase2/runs/cyto3_val_masks stardist=phase2/runs/stardist_val_masks \\
      --out-dir phase2/runs/ensemble_val_masks \\
      --mode majority
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label


def load_mask(path: Path) -> np.ndarray:
    return np.load(path).astype(np.int32)


def ensemble_pixel(masks: list[np.ndarray], min_agree: int) -> np.ndarray:
    """Return integer mask. Pixels with >=min_agree masks marking it as cell
    become foreground; connected components then become labels."""
    h, w = masks[0].shape
    binary = np.stack([m > 0 for m in masks], axis=0)  # (N, H, W)
    agree = binary.sum(axis=0)  # (H, W) — N agreeing pixels
    fg = (agree >= min_agree).astype(np.uint8)
    # Connected components.
    labeled, n = cc_label(fg)
    return labeled.astype(np.int32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Pairs name=dir/. Each dir contains <FOV>.npy masks.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fovs", default="FOV_156,FOV_157,FOV_158,FOV_159,FOV_160")
    p.add_argument("--mode", default="majority", choices=("union", "majority", "intersect"))
    args = p.parse_args()

    pairs = []
    for inp in args.inputs:
        if "=" not in inp:
            raise SystemExit(f"bad --inputs item: {inp}; expected name=dir")
        name, d = inp.split("=", 1)
        pairs.append((name, Path(d)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fovs = [s.strip() for s in args.fovs.split(",") if s.strip()]
    n = len(pairs)
    if args.mode == "union":
        min_agree = 1
    elif args.mode == "intersect":
        min_agree = n
    else:
        min_agree = (n // 2) + 1

    print(f"ensembling {n} sources, mode={args.mode} (min_agree={min_agree})")
    for name, d in pairs:
        print(f"  [{name}] {d}")

    print()
    for fov in fovs:
        masks = []
        for name, d in pairs:
            mp = d / f"{fov}.npy"
            if not mp.exists():
                print(f"  [skip] {fov}: missing {mp}")
                masks = None
                break
            masks.append(load_mask(mp))
        if masks is None:
            continue
        ens = ensemble_pixel(masks, min_agree)
        n_cells = int(ens.max())
        # Compare to inputs
        per_input = ", ".join(f"{n}={int(m.max())}" for (n, _), m in zip(pairs, masks))
        print(f"  {fov}: ensemble={n_cells} cells   ({per_input})")
        np.save(out_dir / f"{fov}.npy", ens)


if __name__ == "__main__":
    main()
