"""Postprocess per-FOV instance masks without rerunning segmentation.

This is for inference-time segmentation experiments: shrink/grow masks, remove
tiny objects, or keep only masks that have enough transcript support.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.src import io


def _compact(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask)
    ids = ids[ids > 0]
    out = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
        out[mask == old_id] = new_id
    return out


def _morph(mask: np.ndarray, mode: str, radius: int) -> np.ndarray:
    if mode == "none" or radius <= 0 or mask.max() == 0:
        return mask.astype(np.int32, copy=False)
    if mode == "dilate":
        from skimage.segmentation import expand_labels
        return _compact(expand_labels(mask, distance=radius).astype(np.int32))

    from scipy import ndimage as ndi
    structure = ndi.generate_binary_structure(2, 1)
    out = np.zeros_like(mask, dtype=np.int32)
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        cell = mask == cell_id
        if mode == "erode":
            moved = ndi.binary_erosion(cell, structure=structure, iterations=radius)
        elif mode == "open":
            moved = ndi.binary_opening(cell, structure=structure, iterations=radius)
        elif mode == "close":
            moved = ndi.binary_closing(cell, structure=structure, iterations=radius)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out[moved] = int(cell_id)
    return _compact(out)


def _filter_by_area(mask: np.ndarray, min_area: int, max_area: int | None) -> np.ndarray:
    if min_area <= 0 and max_area is None:
        return mask
    out = mask.copy()
    ids, counts = np.unique(mask[mask > 0], return_counts=True)
    for cid, area in zip(ids, counts):
        if area < min_area or (max_area is not None and area > max_area):
            out[out == cid] = 0
    return _compact(out)


def _filter_by_spots(mask: np.ndarray, spots: pd.DataFrame, min_spots: int) -> np.ndarray:
    if min_spots <= 0 or mask.max() == 0:
        return mask
    rows = spots["image_row"].to_numpy().astype(int).clip(0, mask.shape[0] - 1)
    cols = spots["image_col"].to_numpy().astype(int).clip(0, mask.shape[1] - 1)
    labels = mask[rows, cols]
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    keep = set(ids[counts >= min_spots].tolist())
    out = mask.copy()
    for cid in np.unique(mask):
        if cid > 0 and cid not in keep:
            out[out == cid] = 0
    return _compact(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fovs", required=True)
    p.add_argument("--split", default="train", choices=("train", "test"))
    p.add_argument("--morph", default="none", choices=("none", "erode", "dilate", "open", "close"))
    p.add_argument("--radius", type=int, default=0)
    p.add_argument("--min-area", type=int, default=0)
    p.add_argument("--max-area", type=int, default=None)
    p.add_argument("--min-spots", type=int, default=0)
    args = p.parse_args()

    in_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fovs = [f.strip() for f in args.fovs.split(",") if f.strip()]
    spots = pd.read_csv(io.data_root() / "test_spots.csv") if args.split == "test" else pd.read_csv(io.ground_truth_dir() / "spots_train.csv")

    summary: dict[str, dict] = {}
    for fov in fovs:
        src = in_dir / f"{fov}.npy"
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        mask0 = np.load(src).astype(np.int32)
        fov_spots = spots[spots["fov"] == fov]
        mask = _morph(mask0, args.morph, args.radius)
        mask = _filter_by_area(mask, args.min_area, args.max_area)
        mask = _filter_by_spots(mask, fov_spots, args.min_spots)
        np.save(out_dir / f"{fov}.npy", mask)

        rows = fov_spots["image_row"].to_numpy().astype(int).clip(0, mask.shape[0] - 1)
        cols = fov_spots["image_col"].to_numpy().astype(int).clip(0, mask.shape[1] - 1)
        frac0 = float((mask0[rows, cols] > 0).mean()) if len(fov_spots) else 0.0
        frac = float((mask[rows, cols] > 0).mean()) if len(fov_spots) else 0.0
        summary[fov] = {
            "cells_in": int(mask0.max()),
            "cells_out": int(mask.max()),
            "frac_in_cell_in": frac0,
            "frac_in_cell_out": frac,
        }
        print(f"{fov}: cells {int(mask0.max())}->{int(mask.max())} frac {frac0:.4f}->{frac:.4f}")

    (out_dir / "postprocess_summary.json").write_text(json.dumps({
        "args": vars(args),
        "per_fov": summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
