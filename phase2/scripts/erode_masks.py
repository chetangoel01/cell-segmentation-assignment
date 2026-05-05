"""Erode all instance masks in a directory by N pixels (preserving instance IDs).

Used to shrink over-predicted cell footprints to better match test gt fic.
Codelab uses expand_labels(25) to grow nuclei into cytoplasm; we go the
other direction — shrinking already-too-large masks.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from skimage.morphology import erosion, disk


def erode_instance_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode each instance label independently by `radius` px, preserving IDs."""
    if radius <= 0:
        return mask
    out = np.zeros_like(mask)
    selem = disk(radius)
    for cid in np.unique(mask):
        if cid == 0:
            continue
        cell_mask = (mask == cid).astype(np.uint8)
        eroded = erosion(cell_mask, selem)
        out[eroded > 0] = cid
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--radius", type=int, default=3)
    args = p.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(in_dir.glob("*.npy")):
        m = np.load(fp).astype(np.int32)
        e = erode_instance_mask(m, args.radius)
        np.save(out_dir / fp.name, e)
        print(f"{fp.name}: cells {m.max()} → {e.max()}, "
              f"in-mask px {(m>0).sum()} → {(e>0).sum()} "
              f"(ratio {(e>0).sum()/max(1,(m>0).sum()):.3f})")


if __name__ == "__main__":
    main()
