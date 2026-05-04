import numpy as np
import pandas as pd

from phase1_restart.pilot.data import (
    FOV_SPLITS,
    fov_origin,
    list_fovs,
    load_fov_channels,
    load_train_spots,
    spot_density_map,
)


def test_splits_disjoint_and_complete():
    train, val, tp, test = (
        FOV_SPLITS["train"],
        FOV_SPLITS["val"],
        FOV_SPLITS["test_proxy"],
        FOV_SPLITS["test"],
    )
    all_train_set = set(train) | set(val) | set(tp)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(tp)) == 0
    assert len(set(val) & set(tp)) == 0
    assert all_train_set == {f"FOV_{i:03d}" for i in range(1, 41)}
    assert set(test) == {"FOV_A", "FOV_B", "FOV_C", "FOV_D"}


def test_list_fovs_test_set():
    assert list_fovs("test") == ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]


def test_fov_origin_returns_tuple():
    fx, fy, ps = fov_origin("FOV_001")
    assert isinstance(fx, float) and isinstance(fy, float)
    assert ps == 0.109


def test_spot_density_returns_2048_float32():
    sd = spot_density_map("FOV_001", sigma=8.0)
    assert sd.shape == (2048, 2048)
    assert sd.dtype == np.float32
    assert sd.max() > 0


def test_load_fov_channels_dapi_polyt_returns_chw_float():
    img = load_fov_channels("FOV_001", channels=["DAPI", "polyT"])
    assert img.shape == (2, 2048, 2048)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_load_fov_channels_with_spot_density():
    img = load_fov_channels("FOV_001", channels=["polyT", "DAPI", "spot_density"])
    assert img.shape == (3, 2048, 2048)
    assert img.dtype == np.float32


def test_load_train_spots_synthesizes_spot_id():
    df = load_train_spots("FOV_001")
    assert "spot_id" in df.columns
    assert len(set(df["spot_id"])) == len(df)
    assert df["spot_id"].iloc[0] == "train_FOV_001_0"
