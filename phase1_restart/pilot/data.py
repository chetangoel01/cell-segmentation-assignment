"""FOV loaders, channel assembly, splits, GT cache.

Reuses phase1/src/io and phase1/src/train_cellpose as read-only imports.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.src import io as p1_io  # type: ignore  # noqa: E402

PHASE1_DATA = REPO_ROOT / "phase1" / "data"
TRAIN_DIR = PHASE1_DATA / "train"
TEST_DIR = PHASE1_DATA / "test"
REFERENCE_DIR = PHASE1_DATA / "reference"
SPOTS_TRAIN = TRAIN_DIR / "ground_truth" / "spots_train.csv"
SPOTS_TEST = PHASE1_DATA / "test_spots.csv"
FOV_METADATA = REFERENCE_DIR / "fov_metadata.csv"
SAMPLE_SUBMISSION = PHASE1_DATA / "sample_submission.csv"
BOUNDARIES_TRAIN = TRAIN_DIR / "ground_truth" / "cell_boundaries_train.csv"

PIXEL_SIZE = 0.109  # µm/px
IMAGE_SIZE = 2048

FOV_SPLITS: dict[str, list[str]] = {
    "train": [f"FOV_{i:03d}" for i in range(1, 31)],
    "val": [f"FOV_{i:03d}" for i in range(36, 41)],
    "test_proxy": [f"FOV_{i:03d}" for i in range(31, 36)],
    "test": ["FOV_A", "FOV_B", "FOV_C", "FOV_D"],
}


def list_fovs(split: str) -> list[str]:
    return FOV_SPLITS[split]


def fov_origin(fov: str) -> tuple[float, float, float]:
    """Returns (fov_x, fov_y, pixel_size) from fov_metadata.csv."""
    md = pd.read_csv(FOV_METADATA)
    row = md[md["fov"] == fov]
    if row.empty:
        raise KeyError(f"FOV {fov} not in {FOV_METADATA}")
    return float(row["fov_x"].iloc[0]), float(row["fov_y"].iloc[0]), float(row["pixel_size"].iloc[0])


def _max_project(zstack: np.ndarray) -> np.ndarray:
    return zstack.max(axis=0).astype(np.float32)


def _normalize_unit(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1.0, 99.5])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _is_train_fov(fov: str) -> bool:
    return fov.startswith("FOV_") and fov[4:].isdigit()


def spot_density_map(fov: str, sigma: float = 8.0) -> np.ndarray:
    spots_csv = SPOTS_TRAIN if _is_train_fov(fov) else SPOTS_TEST
    df = pd.read_csv(spots_csv)
    df_fov = df[df["fov"] == fov]
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    rows = df_fov["image_row"].astype(int).clip(0, IMAGE_SIZE - 1).values
    cols = df_fov["image_col"].astype(int).clip(0, IMAGE_SIZE - 1).values
    np.add.at(canvas, (rows, cols), 1.0)
    blurred = gaussian_filter(canvas, sigma=sigma).astype(np.float32)
    return _normalize_unit(blurred)


def load_fov_channels(fov: str, channels: list[str]) -> np.ndarray:
    """Returns (C, H, W) float32 in [0, 1], channel order = ``channels``."""
    fov_dir = (TRAIN_DIR if _is_train_fov(fov) else TEST_DIR) / fov
    dapi_z, polyt_z = p1_io.load_fov_images(str(fov_dir), fov=fov)
    dapi = _normalize_unit(_max_project(dapi_z))
    polyt = _normalize_unit(_max_project(polyt_z))
    band: dict[str, np.ndarray] = {"DAPI": dapi, "polyT": polyt}
    if "spot_density" in channels:
        band["spot_density"] = spot_density_map(fov, sigma=8.0)
    return np.stack([band[c] for c in channels], axis=0).astype(np.float32)


def load_train_spots(fov: str) -> pd.DataFrame:
    """Train spots have no spot_id column. We synthesize one from row position so that
    truth and pred eval frames can be merged consistently. Train spot ids are not
    submitted to Kaggle — only used for local val ARI."""
    df = pd.read_csv(SPOTS_TRAIN)
    df = df[df["fov"] == fov].reset_index(drop=True)
    df.insert(0, "spot_id", [f"train_{fov}_{i}" for i in range(len(df))])
    return df


def load_test_spots() -> pd.DataFrame:
    """Test spots have spot_id baked in."""
    return pd.read_csv(SPOTS_TEST)
