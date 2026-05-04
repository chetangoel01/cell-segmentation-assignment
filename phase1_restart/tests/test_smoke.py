import numpy as np

from phase1_restart.scripts.smoke import in_cell_ratio


def test_in_cell_ratio_strong_signal():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1:3, 1:3] = 1
    dapi = np.array(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 4.0, 4.0, 0.1],
            [0.1, 4.0, 4.0, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    r = in_cell_ratio(mask, dapi)
    assert 30.0 < r < 50.0  # in-cell mean=4.0, out-cell mean~0.1


def test_in_cell_ratio_no_cells():
    mask = np.zeros((4, 4), dtype=np.int32)
    dapi = np.ones((4, 4), dtype=np.float32)
    assert in_cell_ratio(mask, dapi) == 0.0


def test_in_cell_ratio_no_background():
    mask = np.ones((4, 4), dtype=np.int32)
    dapi = np.ones((4, 4), dtype=np.float32)
    assert in_cell_ratio(mask, dapi) == 0.0
