from __future__ import annotations

from typing import Optional

import numpy as np
from shapely.geometry import Polygon


def pixel_to_um(px: float, py: float, fov_x: float, fov_y: float, pixel_size: float = 0.109):
    """Convert pixel (px, py) within a FOV to global µm coordinates."""
    return fov_x + px * pixel_size, fov_y + py * pixel_size


def parse_boundary_polygon(xs_str: str, ys_str: str) -> Optional[Polygon]:
    """Parse comma-separated boundary coordinate strings into a Shapely Polygon."""
    if not xs_str or not ys_str:
        return None
    xs = [float(v) for v in xs_str.split(",") if v != ""]
    ys = [float(v) for v in ys_str.split(",") if v != ""]
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    poly = Polygon(zip(xs, ys))
    if poly.is_empty or poly.area == 0:
        return None

    if not poly.is_valid:
        try:
            # Shapely 2.x
            from shapely.validation import make_valid

            poly = make_valid(poly)
        except Exception:
            # Last resort: attempt fix by buffering
            poly = poly.buffer(0)

    if poly.is_empty or poly.area == 0:
        return None
    return poly


def spots_in_polygon(spot_x: np.ndarray, spot_y: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Return boolean array: True if spot (x, y) is inside polygon."""
    # Prefer Shapely 2.x vectorized predicate if available.
    try:
        from shapely import contains_xy

        return contains_xy(polygon, spot_x, spot_y)
    except Exception:
        try:
            from shapely.vectorized import contains

            return contains(polygon, spot_x, spot_y)
        except Exception:
            # Fallback: slow, but correct.
            from shapely.geometry import Point

            return np.array([polygon.contains(Point(x, y)) for x, y in zip(spot_x, spot_y)])

