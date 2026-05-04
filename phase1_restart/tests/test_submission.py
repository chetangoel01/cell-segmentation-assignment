import numpy as np
import pandas as pd
import pytest

from phase1_restart.pilot.submission import (
    build_submission,
    validate_submission,
    write_submission,
)


def test_build_submission_uses_pre_computed_coords():
    test_spots = pd.DataFrame(
        {
            "spot_id": ["spot_0", "spot_1", "spot_2"],
            "fov": ["FOV_A", "FOV_A", "FOV_B"],
            "image_row": [3, 100, 5],
            "image_col": [3, 100, 5],
        }
    )
    masks = {
        "FOV_A": np.zeros((10, 10), dtype=np.int32),
        "FOV_B": np.zeros((10, 10), dtype=np.int32),
    }
    masks["FOV_A"][2:5, 2:5] = 1
    masks["FOV_B"][4:7, 4:7] = 1
    out = build_submission(test_spots, masks)
    assert list(out.columns) == ["spot_id", "fov", "cluster_id"]
    assert out.loc[out["spot_id"] == "spot_0", "cluster_id"].iloc[0] == "FOV_A_1"
    assert out.loc[out["spot_id"] == "spot_1", "cluster_id"].iloc[0] == "background"
    assert out.loc[out["spot_id"] == "spot_2", "cluster_id"].iloc[0] == "FOV_B_1"


def test_validate_submission_ok(tmp_path):
    sample = pd.DataFrame(
        {
            "spot_id": ["spot_0", "spot_1", "spot_2", "spot_3"],
            "fov": ["FOV_A"] * 4,
            "cluster_id": ["background"] * 4,
        }
    )
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)
    good = sample.copy()
    good["cluster_id"] = ["FOV_A_1", "background", "FOV_A_2", "background"]
    validate_submission(good, sample_csv)


def test_validate_submission_row_count_mismatch(tmp_path):
    sample = pd.DataFrame(
        {
            "spot_id": ["spot_0", "spot_1", "spot_2", "spot_3"],
            "fov": ["FOV_A"] * 4,
            "cluster_id": ["background"] * 4,
        }
    )
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)
    bad = sample.iloc[:3].copy()
    with pytest.raises(ValueError, match="row count"):
        validate_submission(bad, sample_csv)


def test_validate_submission_non_string_cluster_id(tmp_path):
    sample = pd.DataFrame(
        {
            "spot_id": ["spot_0", "spot_1"],
            "fov": ["FOV_A", "FOV_A"],
            "cluster_id": ["background", "background"],
        }
    )
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)
    bad = sample.copy()
    bad["cluster_id"] = [1, 2]
    with pytest.raises(ValueError, match="non-empty string"):
        validate_submission(bad, sample_csv)


def test_write_submission(tmp_path):
    df = pd.DataFrame(
        {"spot_id": ["spot_0"], "fov": ["FOV_A"], "cluster_id": ["background"]}
    )
    out = write_submission(df, tmp_path / "sub.csv")
    assert out.exists()
    rt = pd.read_csv(out)
    assert list(rt.columns) == ["spot_id", "fov", "cluster_id"]
