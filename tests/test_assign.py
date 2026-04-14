import pandas as pd
from shapely.geometry import Polygon
from src.assign import assign_spots_to_cells


def test_assign_spots_basic():
    polygons = {
        "cell_A": Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        "cell_B": Polygon([(10, 10), (14, 10), (14, 14), (10, 14)]),
    }
    spots = pd.DataFrame(
        {
            "spot_id": ["s0", "s1", "s2"],
            "global_x": [2.0, 12.0, 50.0],
            "global_y": [2.0, 12.0, 50.0],
        }
    )
    result = assign_spots_to_cells(spots, polygons)
    assert result["s0"] == "cell_A"
    assert result["s1"] == "cell_B"
    assert result["s2"] == "background"


def test_assign_all_background_when_no_cells():
    spots = pd.DataFrame(
        {
            "spot_id": ["s0"],
            "global_x": [999.0],
            "global_y": [999.0],
        }
    )
    result = assign_spots_to_cells(spots, {})
    assert result["s0"] == "background"

