"""Mask → Phase-1 Kaggle CSV. Uses pre-computed image_row/image_col from test_spots.csv."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from phase1_restart.pilot.eval import assign_spots_to_mask


def build_submission(test_spots: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """test_spots: must have spot_id, fov, image_row, image_col.
    masks: {fov_name: (H, W) int32}. Missing FOVs → all spots in that FOV → "background".
    """
    parts: list[pd.DataFrame] = []
    for fov, df in test_spots.groupby("fov"):
        if fov in masks:
            parts.append(assign_spots_to_mask(test_spots, masks[fov], fov=fov))
        else:
            sub = df.assign(cluster_id="background")
            parts.append(sub[["spot_id", "fov", "cluster_id"]])
    out = pd.concat(parts, ignore_index=True)
    # Preserve sample_submission ordering by spot_id (which encodes original row order
    # via "spot_<n>"). Sort by the integer suffix to keep sample-submission alignment.
    out["_sort"] = out["spot_id"].str.extract(r"spot_(\d+)").astype(int)
    out = out.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return out[["spot_id", "fov", "cluster_id"]]


def validate_submission(submission: pd.DataFrame, sample_path: Path) -> None:
    """Hard-fail if structure doesn't match sample_submission."""
    sample = pd.read_csv(sample_path)
    if len(submission) != len(sample):
        raise ValueError(
            f"row count mismatch: submission={len(submission)} sample={len(sample)}"
        )
    if not (submission["spot_id"].values == sample["spot_id"].values).all():
        raise ValueError("spot_id sequence does not match sample submission")
    if not submission["cluster_id"].apply(lambda x: isinstance(x, str) and len(x) > 0).all():
        raise ValueError("cluster_id must be non-empty string for every row")
    if not set(submission["fov"]).issubset(set(sample["fov"])):
        raise ValueError(
            f"unexpected fov values: {set(submission['fov']) - set(sample['fov'])}"
        )


def write_submission(submission: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    return out_path
