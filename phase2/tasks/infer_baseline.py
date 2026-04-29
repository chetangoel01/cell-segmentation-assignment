"""Inference task: Cellpose segmentation + classifier prediction → submission CSV.

Input:
  --models-dir   directory containing model_{class,subclass,supertype,cluster}.joblib
                 (output of `train-baseline`)
  --test-fovs    comma-separated test FOVs (e.g. FOV_E,FOV_F)

For each test FOV:
  1. Load DAPI + polyT z-stacks, max-project to 2D.
  2. Run Cellpose-SAM segmentation (MPS on Apple Silicon, CUDA on HPC,
     CPU fallback otherwise) → integer mask (2048, 2048).
  3. For each unique mask label, count spot occurrences per gene → gene vector.
  4. Predict 4 hierarchy levels for each cell.
  5. For each spot in the FOV: look up its mask label; if 0 → "background",
     else use that cell's predicted labels.

Output: submission CSV with columns spot_id, fov, class, subclass, supertype,
cluster — concatenated across the requested FOVs. Spots from FOVs not in
--test-fovs are emitted as all-"background" rows so the file aligns with
sample_submission.csv row-for-row when run on the full test set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.tasks import Task, register

LEVELS = ("class", "subclass", "supertype", "cluster")


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models-dir", required=True,
                   help="Directory of joblib models from `train-baseline`.")
    p.add_argument("--test-fovs", required=True,
                   help="Comma-separated test FOVs to segment + classify "
                        "(e.g. FOV_E,FOV_F). Other FOVs emit all-background.")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default phase2/runs/<ts>-infer-baseline/).")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"),
                   help="Cellpose device. 'auto' picks MPS/CUDA/CPU.")
    p.add_argument("--cellpose-diameter", type=float, default=30.0,
                   help="Approx cell diameter in pixels (cellpose hint).")


def _pick_device(spec: str):
    import torch
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps"), True
        if torch.cuda.is_available():
            return torch.device("cuda"), True
        return torch.device("cpu"), False
    if spec == "mps":
        return torch.device("mps"), True
    if spec == "cuda":
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _segment_fov(fov: str, model, diameter: float):
    from phase2.src import io
    dapi, polyt = io.load_fov_images(fov, split="test")
    # Max-project z-stacks → (2048, 2048).
    dapi2d = dapi.max(axis=0)
    polyt2d = polyt.max(axis=0)
    # Cellpose-SAM (cpsam): pass single multichannel image with explicit
    # channel_axis. Stack as (channels, H, W).
    img = np.stack([polyt2d, dapi2d], axis=0).astype(np.float32)
    masks, _flows, _styles = model.eval(
        img, channel_axis=0, diameter=diameter,
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks  # (2048, 2048) int32, label 0 = background


def _featurize_cells_from_mask(masks: np.ndarray, fov_spots: pd.DataFrame,
                               gene_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (cell_ids, X) where X[i] is the gene-count vector for cell i."""
    # Look up mask label for every spot.
    rows = fov_spots["image_row"].to_numpy().clip(0, masks.shape[0] - 1)
    cols = fov_spots["image_col"].to_numpy().clip(0, masks.shape[1] - 1)
    spot_label = masks[rows.astype(np.int64), cols.astype(np.int64)]
    fov_spots = fov_spots.assign(_mask_label=spot_label)

    n_genes = len(gene_to_idx)
    cell_ids = np.array(sorted(int(c) for c in np.unique(spot_label) if c != 0))
    if cell_ids.size == 0:
        return cell_ids, np.zeros((0, n_genes), dtype=np.float32), spot_label

    label_to_row = {c: i for i, c in enumerate(cell_ids)}
    X = np.zeros((len(cell_ids), n_genes), dtype=np.float32)
    inside = fov_spots[fov_spots["_mask_label"] != 0]
    for label, gene in zip(inside["_mask_label"].to_numpy(), inside["target_gene"].to_numpy()):
        gi = gene_to_idx.get(gene)
        if gi is None:
            continue
        X[label_to_row[int(label)], gi] += 1
    return cell_ids, X, spot_label


def _normalize(X: np.ndarray) -> np.ndarray:
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(X / row_sum * 1e4)


def _run(args: argparse.Namespace) -> int:
    import joblib
    from phase2.src import io

    stages: dict[str, float] = {}
    wall_t0 = time.time()

    test_fovs = [f.strip() for f in args.test_fovs.split(",") if f.strip()]
    print(f"test FOVs: {test_fovs}")

    models_dir = Path(args.models_dir)
    print(f"loading models from {models_dir} …")
    t0 = time.time()
    bundles = {lvl: joblib.load(models_dir / f"model_{lvl}.joblib") for lvl in LEVELS}
    # All four bundles share the same `genes` list (built at train time).
    genes = bundles["class"]["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    stages["load_models"] = time.time() - t0
    print(f"  loaded {len(bundles)} models, gene vocab={len(genes)}  ({stages['load_models']:.2f}s)")

    print("loading test_spots.csv …")
    t0 = time.time()
    spots = pd.read_csv(io.data_root() / "test_spots.csv")
    stages["load_spots"] = time.time() - t0
    print(f"  spots: {len(spots):,}  ({stages['load_spots']:.2f}s)")

    print("\nloading Cellpose-SAM (cpsam) …")
    t0 = time.time()
    import torch
    from cellpose import models as cp_models
    device, gpu = _pick_device(args.device)
    print(f"  device: {device}  (gpu={gpu})")
    model = cp_models.CellposeModel(gpu=gpu, device=device)
    stages["load_cellpose"] = time.time() - t0
    print(f"  ({stages['load_cellpose']:.2f}s)")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "runs" /
        f"{time.strftime('%Y%m%d-%H%M%S')}-infer-baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\noutput dir: {out_dir}")

    # Build the submission frame as background, then fill in segmented FOVs.
    sub = spots[["spot_id", "fov"]].copy()
    for lvl in LEVELS:
        sub[lvl] = "background"

    per_fov_stats: dict[str, dict] = {}
    for fov in test_fovs:
        print(f"\n=== {fov} ===")
        fov_spots = spots[spots["fov"] == fov].copy()
        print(f"  spots in FOV: {len(fov_spots):,}")

        t0 = time.time()
        masks = _segment_fov(fov, model, args.cellpose_diameter)
        seg_t = time.time() - t0
        n_cells = int(masks.max())
        print(f"  cellpose: {n_cells} cells, {seg_t:.2f}s")

        t0 = time.time()
        cell_ids, X, spot_label = _featurize_cells_from_mask(masks, fov_spots, gene_to_idx)
        feat_t = time.time() - t0
        print(f"  featurize: {len(cell_ids)} cells × {X.shape[1]} genes, {feat_t:.2f}s")

        t0 = time.time()
        if len(cell_ids):
            X_n = _normalize(X)
            cell_predictions = {lvl: bundles[lvl]["clf"].predict(X_n) for lvl in LEVELS}
        else:
            cell_predictions = {lvl: np.array([]) for lvl in LEVELS}
        pred_t = time.time() - t0
        print(f"  predict 4 levels: {pred_t:.2f}s")

        # Map spot → cell index → prediction.
        t0 = time.time()
        label_to_idx = {int(c): i for i, c in enumerate(cell_ids)}
        spot_idx = np.array([label_to_idx.get(int(l), -1) for l in spot_label])
        in_cell = spot_idx >= 0
        fov_mask = sub["fov"].to_numpy() == fov
        # Align to original spot row order in `spots` for this FOV.
        row_indexer = np.where(fov_mask)[0]
        for lvl in LEVELS:
            preds = cell_predictions[lvl]
            if len(preds):
                # default "background" already in place; overwrite only in-cell spots
                vals = np.full(len(fov_spots), "background", dtype=object)
                vals[in_cell] = preds[spot_idx[in_cell]]
                sub.loc[row_indexer, lvl] = vals
        write_t = time.time() - t0
        print(f"  spots→cells assign + write rows: {write_t:.2f}s")

        per_fov_stats[fov] = {
            "n_spots": int(len(fov_spots)),
            "n_cells_segmented": n_cells,
            "n_cells_with_spots": int(len(cell_ids)),
            "n_spots_in_cell": int(in_cell.sum()),
            "frac_in_cell": float(in_cell.mean()) if len(in_cell) else 0.0,
            "seg_s": seg_t, "feat_s": feat_t, "pred_s": pred_t, "write_s": write_t,
        }
        stages[f"fov_{fov}"] = seg_t + feat_t + pred_t + write_t

    print("\nwriting submission CSV …")
    t0 = time.time()
    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    stages["write_submission"] = time.time() - t0
    print(f"  → {sub_path}  ({stages['write_submission']:.2f}s, {len(sub):,} rows)")

    stages["wall_total"] = time.time() - wall_t0
    summary = {
        "test_fovs": test_fovs,
        "device": str(device),
        "models_dir": str(models_dir),
        "per_fov": per_fov_stats,
        "stage_timings_s": stages,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== stage timings ===")
    for k, v in stages.items():
        print(f"  {k:<22} {v:7.2f}s")
    return 0


register(Task(
    name="infer-baseline",
    summary="Segment test FOVs (Cellpose-SAM) + classify cells + write submission CSV.",
    add_args=_add_args,
    run=_run,
    requirements={
        "gpu": True,
        "modal_gpu": "T4",
        "modal_image": "cellpose",
        "modal_volume": "cell-seg-phase2",
        "modal_timeout": 2 * 3600,
        "hpc_partition": "gpu",
        "hpc_hours": 1.0,
        "hpc_gpus": 1,
    },
))
