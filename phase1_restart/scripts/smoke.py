"""Coord-sanity gate. Runs pretrained Cellpose baseline on FOV_001, asserts:

  (1) in-cell DAPI mean / out-of-cell DAPI mean >= 2.0  — primary coord-convention test
  (2) per-FOV ARI vs polygon GT in [0.40, 0.85]         — secondary baseline sanity

Hard-halt on any failure: do not proceed with adapter work if coords are broken.

The pretrained Cellpose baseline used here is a coord-sanity ANCHOR ONLY, never warm-started
into any candidate or shipped submission (see design §6 decision 1).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from phase1_restart.pilot.data import (
    load_fov_channels,
    fov_origin,
    load_train_spots,
    BOUNDARIES_TRAIN,
)
from phase1_restart.pilot.eval import (
    assign_spots_to_mask,
    compute_per_fov_ari,
    mean_ari,
)


def in_cell_ratio(mask: np.ndarray, dapi: np.ndarray) -> float:
    in_cell = dapi[mask > 0]
    out_cell = dapi[mask == 0]
    if len(in_cell) == 0 or len(out_cell) == 0:
        return 0.0
    out_mean = max(float(out_cell.mean()), 1e-6)
    return float(in_cell.mean()) / out_mean


def _run_cellpose_baseline(dapi: np.ndarray, polyt: np.ndarray) -> np.ndarray:
    """Pretrained Cellpose v4 inference. Used for coord sanity only."""
    from cellpose import models

    model = models.CellposeModel(gpu=False)  # v4 ignores model_type
    img = np.stack([polyt, dapi], axis=0).astype(np.float32)
    out = model.eval(
        img,
        diameter=88.9,
        channel_axis=0,
        normalize=True,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    masks = out[0] if isinstance(out, tuple) else out
    return np.asarray(masks).astype(np.int32)


def _build_truth_for_fov(fov: str) -> pd.DataFrame:
    """Polygon GT → mask → assign_spots. Returns truth DF for compute_per_fov_ari."""
    # phase1_restart.pilot.data already pushes both REPO_ROOT and phase1/ onto sys.path
    # at import time, which is what `phase1.src.train_cellpose` needs (it does
    # `from src.coords import ...` internally).
    from phase1.src import train_cellpose as p1_tc  # type: ignore

    fx, fy, ps = fov_origin(fov)
    boundaries = pd.read_csv(BOUNDARIES_TRAIN, index_col=0)
    gt_mask = p1_tc.boundaries_to_mask(
        boundaries,
        fov_name=fov,
        fov_x=fx,
        fov_y=fy,
        pixel_size=ps,
        image_size=2048,
        z_plane=2,
    ).astype(np.int32)
    spots = load_train_spots(fov)
    truth = assign_spots_to_mask(spots, gt_mask, fov=fov)
    return truth


def main() -> int:
    fov = "FOV_001"
    print(f"loading {fov} ...", flush=True)
    img = load_fov_channels(fov, channels=["DAPI", "polyT"])
    dapi, polyt = img[0], img[1]

    print("running cellpose baseline (cyto2 / v4) ...", flush=True)
    mask = _run_cellpose_baseline(dapi, polyt)
    print(f"  cells detected: {mask.max()}")

    ratio = in_cell_ratio(mask, dapi)
    print(f"in-cell DAPI / out-of-cell DAPI = {ratio:.2f} (require >= 2.0)")
    if ratio < 2.0:
        print("FAIL: coord convention is broken — DAPI is not enriched inside masks.")
        return 1

    print("building polygon GT for FOV_001 ...", flush=True)
    truth = _build_truth_for_fov(fov)
    spots = load_train_spots(fov)
    pred = assign_spots_to_mask(spots, mask, fov=fov)

    per_fov = compute_per_fov_ari(truth, pred)
    ari = mean_ari(per_fov)
    print(f"FOV_001 ARI vs polygon GT = {ari:.4f} (require 0.40 <= ARI <= 0.85)")
    if not (0.40 <= ari <= 0.85):
        print("FAIL: cellpose baseline ARI outside expected range — likely coord bug.")
        return 1

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
