from __future__ import annotations

from typing import Dict

import pandas as pd
from shapely.geometry import Polygon

from src.coords import spots_in_polygon


def assign_spots_to_cells(spots: pd.DataFrame, cell_polygons: Dict[str, Polygon]) -> Dict[str, str]:
    """Assign each spot to a cell by point-in-polygon test.

    Args:
        spots: DataFrame with columns spot_id, global_x, global_y (µm)
        cell_polygons: dict mapping cell_id -> Shapely Polygon in µm coordinates
    Returns:
        dict mapping spot_id -> cell_id (or "background")
    """
    if spots.empty:
        return {}

    assignments = {sid: "background" for sid in spots["spot_id"].tolist()}
    spot_x = spots["global_x"].to_numpy()
    spot_y = spots["global_y"].to_numpy()
    spot_ids = spots["spot_id"].to_numpy()

    for cell_id, polygon in cell_polygons.items():
        if polygon is None:
            continue
        inside = spots_in_polygon(spot_x, spot_y, polygon)
        for sid in spot_ids[inside]:
            assignments[str(sid)] = cell_id  # last cell wins on overlap (rare)

    return assignments

