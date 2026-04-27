"""3D ground-truth rasterization for StarDist3D.

The 2D helper in src.train_cellpose.boundaries_to_mask flattens the 5 per-z
polygons into a single (H, W) mask. StarDist3D needs a (Z=5, H, W) volume
where the *same* cell has the *same* integer ID on every z-plane it appears
on. This is the whole point of going 3D: the network gets to learn that
pixels at different z belonging to one cell are a single 3D object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.draw import polygon as draw_polygon

from src.coords import parse_boundary_polygon

N_Z = 5
DEFAULT_IMAGE_SIZE = 2048
DEFAULT_PIXEL_SIZE = 0.109


def boundaries_to_mask_3d(
    cell_boundaries_df: pd.DataFrame,
    fov_x: float,
    fov_y: float,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    filter_z: int = 2,
) -> np.ndarray:
    """Rasterize per-z polygons to a (Z, H, W) int32 volume with stable cell IDs.

    Args:
        cell_boundaries_df: loaded from cell_boundaries_train.csv (index_col=0).
        fov_x, fov_y: FOV origin in µm (from fov_metadata.csv).
        filter_z: z-plane used for the spatial pre-filter (centroid in FOV?).
            Defaults to the middle plane (z=2) which has the most reliable labels.

    Returns:
        (N_Z, image_size, image_size) int32 array. 0 = background,
        1..N = cell IDs. Every plane shares the same ID-to-cell mapping.
    """
    fov_x_max = fov_x + image_size * pixel_size
    fov_y_max = fov_y + image_size * pixel_size
    volume = np.zeros((N_Z, image_size, image_size), dtype=np.int32)
    cell_int = 1

    def _px_coords(xs_um, ys_um):
        row_px = np.clip(
            np.array([image_size - 1 - (x - fov_x) / pixel_size for x in xs_um]),
            0, image_size - 1,
        )
        col_px = np.clip(
            np.array([(y - fov_y) / pixel_size for y in ys_um]),
            0, image_size - 1,
        )
        return row_px, col_px

    for _cell_id, row in cell_boundaries_df.iterrows():
        # Cheap centroid pre-filter using the reference z-plane.
        xs_ref = row.get(f"boundaryX_z{filter_z}", "")
        ys_ref = row.get(f"boundaryY_z{filter_z}", "")
        xs_ref = str(xs_ref) if pd.notna(xs_ref) else ""
        ys_ref = str(ys_ref) if pd.notna(ys_ref) else ""
        if not xs_ref or not ys_ref:
            continue
        try:
            _xs = [float(v) for v in xs_ref.split(",") if v.strip()]
            _ys = [float(v) for v in ys_ref.split(",") if v.strip()]
        except ValueError:
            continue
        if not _xs or not _ys:
            continue
        cx = sum(_xs) / len(_xs)
        cy = sum(_ys) / len(_ys)
        if not (fov_x <= cx < fov_x_max and fov_y <= cy < fov_y_max):
            continue

        # Draw this cell's polygon on every z-plane where it's defined,
        # reusing the same integer ID.
        drew_any = False
        for z in range(N_Z):
            xs_str = row.get(f"boundaryX_z{z}", "")
            ys_str = row.get(f"boundaryY_z{z}", "")
            xs_str = str(xs_str) if pd.notna(xs_str) else ""
            ys_str = str(ys_str) if pd.notna(ys_str) else ""
            if not xs_str or not ys_str:
                continue
            poly = parse_boundary_polygon(xs_str, ys_str)
            if poly is None:
                continue
            xs_um, ys_um = poly.exterior.xy
            rr, cc = draw_polygon(*_px_coords(xs_um, ys_um), shape=(image_size, image_size))
            volume[z, rr, cc] = cell_int
            drew_any = True

        if drew_any:
            cell_int += 1

    return volume


def collapse_3d_labels_to_2d(labels_3d: np.ndarray) -> np.ndarray:
    """Project a (Z, H, W) int label mask to (H, W) via per-pixel max.

    Since every cell has the same integer ID across all z-planes it appears on,
    max-over-z recovers the cell's 2D footprint. Overlapping distinct cells at
    different z (uncommon at the ~1.5 µm z-spacing used here) are assigned to
    whichever cell has the higher ID — arbitrary but deterministic.
    """
    return np.max(labels_3d, axis=0).astype(np.int32)
