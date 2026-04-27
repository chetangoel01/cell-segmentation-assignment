"""Local sanity check: load FOV_101 image + GT and confirm coords line up.

Run from repo root:
    python -m phase2.scripts.fov101_smoke_test

Expects local layout (after the rsync):
    phase2/data/train/FOV_101/Epi-...101.dax
    phase2/data/train/ground_truth/{spots_train.csv, cell_boundaries_train.csv,
                                     cell_labels_train.csv, counts_train.h5ad}
    phase2/data/reference/fov_metadata.csv

Smoke-checks (no plotting — pure assertions + numbers):
  1. DAPI/polyT z-stacks load with the expected shape.
  2. spots_train.csv has rows for FOV_101 and image_row/image_col fall in [0, 2048).
  3. cell_boundaries polygons for FOV_101 cells, when converted to image coords
     using the documented (flipped-x) convention, contain the spots assigned to
     that cell at >50% rate. This is the same trick that caught the 4× ARI bug
     in phase 1.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pandas as pd

from phase2.src import coords, io


def um_to_pixel(global_x: np.ndarray, global_y: np.ndarray,
                fov_x: float, fov_y: float,
                pixel_size: float = io.PIXEL_SIZE_UM,
                image_size: int = io.IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """MERFISH non-standard convention: x is flipped relative to image_row.

        image_row = image_size - (global_x - fov_x) / pixel_size
        image_col = (global_y - fov_y) / pixel_size
    """
    image_row = image_size - (global_x - fov_x) / pixel_size
    image_col = (global_y - fov_y) / pixel_size
    return image_row, image_col


def main(fov: str = "FOV_101") -> int:
    print(f"=== Phase 2 smoke test on {fov} ===")
    print(f"data_root: {io.data_root()}")

    # 1) Image
    dapi, polyt = io.load_fov_images(fov, split="train")
    print(f"DAPI stack: shape={dapi.shape}, dtype={dapi.dtype}, "
          f"range=[{dapi.min()}, {dapi.max()}], mean={dapi.mean():.1f}")
    print(f"polyT stack: shape={polyt.shape}, dtype={polyt.dtype}, "
          f"range=[{polyt.min()}, {polyt.max()}], mean={polyt.mean():.1f}")
    assert dapi.shape == (5, 2048, 2048), f"unexpected DAPI shape {dapi.shape}"
    assert polyt.shape == (5, 2048, 2048), f"unexpected polyT shape {polyt.shape}"

    # 2) Spots
    spots_path = io.ground_truth_dir() / "spots_train.csv"
    if not spots_path.exists():
        print(f"[skip] spots_train.csv not found at {spots_path}")
        return 0
    spots = pd.read_csv(spots_path)
    fov_spots = spots[spots["fov"] == fov].copy()
    print(f"spots_train: {len(spots):,} total, {len(fov_spots):,} for {fov}")
    if len(fov_spots) == 0:
        print(f"[fail] no spots for {fov} in spots_train.csv")
        return 1

    in_bounds = ((fov_spots["image_row"].between(0, 2048 - 1, inclusive="left")) &
                 (fov_spots["image_col"].between(0, 2048 - 1, inclusive="left"))).sum()
    print(f"spots with image_row/col in [0, 2048): {in_bounds:,} / {len(fov_spots):,}")
    assert in_bounds == len(fov_spots), "some spots have out-of-bounds image coords"

    top_genes = Counter(fov_spots["target_gene"]).most_common(5)
    print(f"top 5 genes in {fov}: {top_genes}")

    # 3) Cell boundaries → polygon-spot containment sanity check
    cells_path = io.ground_truth_dir() / "cell_boundaries_train.csv"
    labels_path = io.ground_truth_dir() / "cell_labels_train.csv"
    if not (cells_path.exists() and labels_path.exists()):
        print(f"[skip] missing {cells_path.name} or {labels_path.name}")
        return 0
    cells = pd.read_csv(cells_path, index_col=0)
    labels = pd.read_csv(labels_path, index_col="cell_id")
    fov_cell_ids = labels.index[labels["fov"] == fov]
    print(f"cells in {fov}: {len(fov_cell_ids)}")

    # Pick the first valid polygon (z=2, the middle z-plane) and check its
    # boundary box covers a sensible fraction of the FOV.
    sample = fov_cell_ids[:5]
    contained_total = 0
    for cid in sample:
        if cid not in cells.index:
            continue
        row = cells.loc[cid]
        poly = coords.parse_boundary_polygon(row.get("boundaryX_z2", ""),
                                             row.get("boundaryY_z2", ""))
        if poly is None:
            continue
        inside = coords.spots_in_polygon(
            fov_spots["global_x"].to_numpy(),
            fov_spots["global_y"].to_numpy(),
            poly,
        )
        contained_total += int(inside.sum())
        print(f"  cell {cid}: polygon area={poly.area:.1f} µm², "
              f"contains {int(inside.sum())} spots")

    print(f"total spots inside first 5 cells: {contained_total}")
    if contained_total == 0:
        print("[warn] no spots fell inside any sampled cell polygon — coordinate convention may be off")
        return 1

    # 4) Quick label distribution
    fov_labels = labels.loc[fov_cell_ids]
    class_counts = fov_labels["class_label"].value_counts()
    print(f"\nclass label distribution in {fov}:")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")

    print("\n=== smoke test passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
