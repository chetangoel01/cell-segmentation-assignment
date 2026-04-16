from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.draw import polygon as draw_polygon
from shapely.geometry import Polygon as ShapelyPolygon

from src.coords import parse_boundary_polygon


def compute_spot_density(
    spots_df: pd.DataFrame,
    image_size: int = 2048,
    sigma: float = 8.0,
) -> np.ndarray:
    """Build a smoothed mRNA spot density map as a uint16 image.

    High density = many mRNA molecules nearby = likely inside a cell.
    sigma=8px ≈ 0.87µm, chosen to roughly match nucleus radius.
    """
    density = np.zeros((image_size, image_size), dtype=np.float32)
    rows = spots_df["image_row"].values.astype(int).clip(0, image_size - 1)
    cols = spots_df["image_col"].values.astype(int).clip(0, image_size - 1)
    np.add.at(density, (rows, cols), 1)
    density = gaussian_filter(density, sigma=sigma)
    if density.max() > 0:
        density = (density / density.max() * 65535).astype(np.uint16)
    return density


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

    Cells are identified spatially: any cell whose boundary centroid falls within
    [fov_x, fov_x + image_size*pixel_size) x [fov_y, fov_y + image_size*pixel_size)
    is included. The ``fov_name`` argument is retained for API compatibility.

    Args:
        cell_boundaries_df: cell_boundaries_train.csv loaded with index_col=0.
            Columns 'boundaryX_z{z}' and 'boundaryY_z{z}' hold comma-separated µm coords.
        fov_name: FOV name (unused; kept for backward compatibility with call sites).
        fov_x, fov_y: FOV origin in µm (from fov_metadata.csv)
        pixel_size: µm per pixel (default 0.109)
        image_size: pixel dimension of the square image (default 2048)
        z_plane: which z-plane to use (0-4, default 2)
    Returns:
        (image_size, image_size) int32 array. 0 = background, 1..N = cell integer IDs.
    """
    fov_x_max = fov_x + image_size * pixel_size
    fov_y_max = fov_y + image_size * pixel_size
    mask = np.zeros((image_size, image_size), dtype=np.int32)
    cell_int = 1

    for _cell_id, row in cell_boundaries_df.iterrows():
        xs_str = row.get(f"boundaryX_z{z_plane}", "")
        ys_str = row.get(f"boundaryY_z{z_plane}", "")
        xs_str = str(xs_str) if pd.notna(xs_str) else ""
        ys_str = str(ys_str) if pd.notna(ys_str) else ""
        if not xs_str or not ys_str:
            continue

        # Spatial pre-filter: cheap centroid check before building Shapely polygon.
        # Cell IDs in the CSV are not FOV-prefixed, so we use spatial filtering.
        try:
            _xs = [float(v) for v in xs_str.split(",") if v.strip()]
            _ys = [float(v) for v in ys_str.split(",") if v.strip()]
        except ValueError:
            continue
        if not _xs or not _ys:
            continue
        cx = sum(_xs) / len(_xs)
        cy = sum(_ys) / len(_ys)
        if not (fov_x <= cx < fov_x_max and fov_y <= cy < fov_y_max):
            continue

        poly = parse_boundary_polygon(xs_str, ys_str)
        if poly is None:
            continue
        xs_um, ys_um = poly.exterior.xy
        # MERFISH convention: x-axis is flipped in image space
        # image_row = (image_size - 1) - (x - fov_x) / pixel_size
        # image_col = (y - fov_y) / pixel_size
        row_px = np.clip(
            np.array([image_size - 1 - (x - fov_x) / pixel_size for x in xs_um]), 0, image_size - 1
        )
        col_px = np.clip(
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
