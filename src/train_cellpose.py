from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.draw import polygon as draw_polygon
from shapely.geometry import Polygon as ShapelyPolygon

from src.coords import parse_boundary_polygon


def augment_training_data(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    intensity_jitter: float = 0.25,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Expand training data by all 8 flip/rotation symmetries + intensity jitter.

    Each (image, mask) pair produces 8 augmented copies (4 rotations × 2 flips),
    multiplying the effective dataset size by 8.  Intensity jitter is applied
    independently per channel and per copy so the model sees varied staining.

    Args:
        images: list of (C, H, W) float/uint16 arrays
        masks:  list of (H, W) int32 arrays
        intensity_jitter: max multiplicative perturbation per channel (e.g. 0.25
                          means scale ∈ [0.75, 1.25])
        rng: optional numpy random Generator for reproducibility
    Returns:
        (aug_images, aug_masks) — each 8× longer than the inputs
    """
    if rng is None:
        rng = np.random.default_rng()

    aug_images: list[np.ndarray] = []
    aug_masks:  list[np.ndarray] = []

    for img, mask in zip(images, masks):
        for k in range(4):
            rot_img  = np.rot90(img,  k=k, axes=(1, 2))
            rot_mask = np.rot90(mask, k=k)
            for flip in (False, True):
                aug_img  = np.flip(rot_img,  axis=2).copy() if flip else rot_img.copy()
                aug_mask = np.flip(rot_mask, axis=1).copy() if flip else rot_mask.copy()
                if intensity_jitter > 0:
                    # Per-channel multiplicative jitter
                    scales = rng.uniform(
                        1 - intensity_jitter, 1 + intensity_jitter,
                        size=(aug_img.shape[0], 1, 1),
                    ).astype(aug_img.dtype)
                    aug_img = np.clip(aug_img * scales, 0, None).astype(aug_img.dtype)
                aug_images.append(aug_img)
                aug_masks.append(aug_mask)

    return aug_images, aug_masks


def compute_zstack_features(
    dapi: np.ndarray,
    polyt: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-channel z-stack statistics as additional input channels.

    Args:
        dapi:  (Z, H, W) uint16 — DAPI z-stack
        polyt: (Z, H, W) uint16 — polyT z-stack
    Returns:
        dict with keys: dapi_max, dapi_mean, dapi_std, polyt_max, polyt_mean, polyt_std
        All values are (H, W) float32 arrays normalised to [0, 65535] range.
    """
    def norm(arr: np.ndarray) -> np.ndarray:
        a = arr.astype(np.float32)
        mx = a.max()
        return (a / mx * 65535).astype(np.float32) if mx > 0 else a

    return {
        "dapi_max":   norm(np.max(dapi,  axis=0)),
        "dapi_mean":  norm(np.mean(dapi, axis=0)),
        "dapi_std":   norm(np.std(dapi,  axis=0)),
        "polyt_max":  norm(np.max(polyt,  axis=0)),
        "polyt_mean": norm(np.mean(polyt, axis=0)),
        "polyt_std":  norm(np.std(polyt,  axis=0)),
    }


def compute_spot_density(
    spots_df: pd.DataFrame,
    image_size: int = 2048,
    sigma: float = 8.0,
    *,
    normalize: bool = True,
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
    if normalize and density.max() > 0:
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
    use_all_z: bool = False,
) -> np.ndarray:
    """Convert GT cell boundary polygons for one FOV to an integer mask.

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
        z_plane: which z-plane to use (0-4, default 2). Ignored when use_all_z=True.
        use_all_z: when True, draw the union of all 5 z-plane boundaries for each cell.
            This aligns the GT footprint with a max-projection image (which captures
            signal from all z-planes). Spatial filtering always uses z=2 as reference.
    Returns:
        (image_size, image_size) int32 array. 0 = background, 1..N = cell integer IDs.
    """
    fov_x_max = fov_x + image_size * pixel_size
    fov_y_max = fov_y + image_size * pixel_size
    mask = np.zeros((image_size, image_size), dtype=np.int32)
    cell_int = 1

    # Use z=2 as the spatial reference for filtering (most reliable middle plane).
    # When use_all_z=False, filter_z equals z_plane (original behaviour).
    filter_z = 2 if use_all_z else z_plane
    draw_z_planes = list(range(5)) if use_all_z else [z_plane]

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
        # Spatial pre-filter: cheap centroid check on the reference z-plane.
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

        # Draw polygon(s) for this cell — one z-plane or all five.
        cell_drawn = False
        for z in draw_z_planes:
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
            mask[rr, cc] = cell_int
            cell_drawn = True

        if cell_drawn:
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
