import numpy as np
import pandas as pd

from phase1_restart.pilot.eval import (
    assign_spots_to_mask,
    compute_per_fov_ari,
    mean_ari,
)


def test_assign_spots_to_mask_simple_with_spot_id():
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 1
    mask[7:9, 7:9] = 2
    spots = pd.DataFrame(
        {
            "spot_id": ["s0", "s1", "s2"],
            "fov": ["FOV_001"] * 3,
            "image_row": [3, 8, 0],
            "image_col": [3, 8, 0],
        }
    )
    out = assign_spots_to_mask(spots, mask, fov="FOV_001")
    assert list(out["cluster_id"]) == ["FOV_001_1", "FOV_001_2", "background"]
    assert list(out["spot_id"]) == ["s0", "s1", "s2"]


def test_assign_spots_synthesizes_spot_id_when_missing():
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 1
    spots = pd.DataFrame(
        {
            "fov": ["FOV_001"] * 2,
            "image_row": [3, 0],
            "image_col": [3, 0],
        }
    )
    out = assign_spots_to_mask(spots, mask, fov="FOV_001")
    assert list(out["spot_id"]) == ["row_0", "row_1"]


def test_perfect_ari_is_1():
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 1
    spots = pd.DataFrame(
        {
            "spot_id": ["s0", "s1"],
            "fov": ["FOV_001"] * 2,
            "image_row": [3, 0],
            "image_col": [3, 0],
        }
    )
    pred = assign_spots_to_mask(spots, mask, fov="FOV_001")
    truth = pred.copy()
    ari = compute_per_fov_ari(truth, pred)
    assert ari["FOV_001"] == 1.0
    assert mean_ari(ari) == 1.0


def test_mean_ari_empty():
    assert mean_ari({}) == 0.0
