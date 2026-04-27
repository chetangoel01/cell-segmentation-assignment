from __future__ import annotations

from typing import Optional

import numpy as np
from shapely.geometry import Polygon


def parse_boundary_polygon(xs_str: str, ys_str: str) -> Optional[Polygon]:
    """Parse comma-separated boundaryX/boundaryY strings into a Shapely Polygon."""
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
            from shapely.validation import make_valid
            poly = make_valid(poly)
        except Exception:
            poly = poly.buffer(0)
    if poly.is_empty or poly.area == 0:
        return None
    return poly


def spots_in_polygon(spot_x: np.ndarray, spot_y: np.ndarray, polygon: Polygon) -> np.ndarray:
    try:
        from shapely import contains_xy
        return contains_xy(polygon, spot_x, spot_y)
    except Exception:
        try:
            from shapely.vectorized import contains
            return contains(polygon, spot_x, spot_y)
        except Exception:
            from shapely.geometry import Point
            return np.array([polygon.contains(Point(x, y)) for x, y in zip(spot_x, spot_y)])
