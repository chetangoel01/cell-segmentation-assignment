from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Frame indices within the 27-frame Epi file (from dataorganization.csv)
DAPI_FRAMES = [6, 11, 16, 21, 26]   # z0..z4, 405 nm
POLYT_FRAMES = [5, 10, 15, 20, 25]  # z0..z4, 488 nm

PIXEL_SIZE_UM = 0.109
IMAGE_SIZE = 2048


def data_root() -> Path:
    """Return MERFISH dataset root.

    Resolution order:
      1. $MERFISH_DATA_ROOT
      2. <repo>/phase2/data/  (local dev layout)
    """
    env = os.environ.get("MERFISH_DATA_ROOT")
    if env:
        return Path(env)
    # __file__ is .../phase2/src/io.py → repo/phase2/data
    return Path(__file__).resolve().parents[1] / "data"


def train_dir() -> Path:
    return data_root() / "train"


def test_dir() -> Path:
    return data_root() / "test"


def reference_dir() -> Path:
    return data_root() / "reference"


def ground_truth_dir() -> Path:
    return train_dir() / "ground_truth"


def load_dax(path: str | Path, n_pixels: int = IMAGE_SIZE) -> np.ndarray:
    """Load a raw .dax file as (n_frames, n_pixels, n_pixels) uint16."""
    raw = np.fromfile(str(path), dtype=np.uint16)
    n_frames = raw.size // (n_pixels * n_pixels)
    if n_frames * (n_pixels * n_pixels) != raw.size:
        raise ValueError(
            f"DAX size {raw.size} not divisible by {n_pixels}x{n_pixels}; "
            f"got remainder {raw.size % (n_pixels * n_pixels)}"
        )
    return raw.reshape(n_frames, n_pixels, n_pixels)


def get_dapi_stack(raw: np.ndarray) -> np.ndarray:
    return raw[DAPI_FRAMES]


def get_polyt_stack(raw: np.ndarray) -> np.ndarray:
    return raw[POLYT_FRAMES]


def _parse_fov_id(fov: str) -> str:
    """'FOV_101' -> '101', 'FOV_E' -> 'E'. Numeric IDs are zero-padded to 3."""
    fov = fov.strip()
    if fov.upper().startswith("FOV_"):
        fov = fov.split("_", 1)[1]
    return fov.zfill(3) if fov.isdigit() else fov


def fov_dir(fov: str, split: str = "train") -> Path:
    """Path to a single FOV folder. split ∈ {train, test}."""
    base = train_dir() if split == "train" else test_dir()
    return base / fov


def find_epi_file(fov: str, split: str = "train") -> Path:
    """Locate the multichannel Epi .dax (27 frames: DAPI + polyT + fiducial + 2 gene)."""
    folder = fov_dir(fov, split=split)
    fov_id = _parse_fov_id(fov)
    pattern = f"Epi-750s5-635s5-545s1-473s5-408s5_{fov_id}.dax"
    matches = sorted(folder.glob(pattern))
    if not matches:
        existing = sorted(p.name for p in folder.glob("*.dax"))[:5]
        hint = f" Found .dax files: {existing}" if existing else ""
        raise FileNotFoundError(f"No Epi file matching {pattern} in {folder}.{hint}")
    return matches[0]


def load_fov_images(fov: str, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load (dapi_stack, polyt_stack) for one FOV — both (5, 2048, 2048) uint16."""
    epi = find_epi_file(fov, split=split)
    raw = load_dax(epi)
    return get_dapi_stack(raw), get_polyt_stack(raw)
