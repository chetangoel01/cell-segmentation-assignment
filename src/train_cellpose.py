from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure
from skimage.draw import polygon as draw_polygon
from shapely.geometry import Polygon as ShapelyPolygon

from src.coords import parse_boundary_polygon


def boundaries_to_mask(
    cell_boundaries_df: pd.DataFrame,
    fov_name: str,
    fov_x: float,
    fov_y: float,
    pixel_size: float = 0.109,
    image_size: int = 2048,
    z_plane: int = 2,
) -> np.ndarray:
    """Convert GT cell boundary polygons for one FOV/z-plane to an integer mask.

    Args:
        cell_boundaries_df: cell_boundaries_train.csv loaded with index_col=0.
            Index entries look like 'FOV_001_cell_1'. Columns 'boundaryX_z{z}' and
            'boundaryY_z{z}' hold comma-separated µm coordinates.
        fov_name: e.g. "FOV_001"
        fov_x, fov_y: FOV origin in µm (from fov_metadata.csv)
        pixel_size: µm per pixel (default 0.109)
        image_size: pixel dimension of the square image (default 2048)
        z_plane: which z-plane to use (0-4, default 2)
    Returns:
        (image_size, image_size) int32 array. 0 = background, 1..N = cell integer IDs.
    """
    fov_cells = cell_boundaries_df[cell_boundaries_df.index.str.startswith(fov_name)]
    mask = np.zeros((image_size, image_size), dtype=np.int32)
    cell_int = 1

    for _cell_id, row in fov_cells.iterrows():
        xs_str = row.get(f"boundaryX_z{z_plane}", "")
        ys_str = row.get(f"boundaryY_z{z_plane}", "")
        xs_str = str(xs_str) if pd.notna(xs_str) else ""
        ys_str = str(ys_str) if pd.notna(ys_str) else ""
        poly = parse_boundary_polygon(xs_str, ys_str)
        if poly is None:
            continue
        xs_um, ys_um = poly.exterior.xy
        col_px = np.clip(
            np.array([(x - fov_x) / pixel_size for x in xs_um]), 0, image_size - 1
        )
        row_px = np.clip(
            np.array([(y - fov_y) / pixel_size for y in ys_um]), 0, image_size - 1
        )
        rr, cc = draw_polygon(row_px, col_px, shape=(image_size, image_size))
        mask[rr, cc] = cell_int
        cell_int += 1

    return mask


def masks_to_polygons(
    masks: np.ndarray,
    fov_x: float,
    fov_y: float,
    pixel_size: float = 0.109,
) -> dict:
    """Convert a Cellpose integer mask array to a dict of cell_id -> Shapely Polygon in µm.

    Args:
        masks: (H, W) int array from Cellpose, 0=background, 1..N=cells
        fov_x, fov_y: FOV origin in µm
        pixel_size: µm per pixel
    Returns:
        dict mapping "cellpose_{i}" -> Shapely Polygon in µm coordinates
    """
    polygons = {}
    for cell_int in range(1, int(masks.max()) + 1):
        cell_mask = (masks == cell_int).astype(np.uint8)
        contours = measure.find_contours(cell_mask, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)  # find_contours returns (row, col) = (y, x)
        xs_um = fov_x + contour[:, 1] * pixel_size
        ys_um = fov_y + contour[:, 0] * pixel_size
        poly = ShapelyPolygon(zip(xs_um, ys_um))
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons[f"cellpose_{cell_int}"] = poly
    return polygons
