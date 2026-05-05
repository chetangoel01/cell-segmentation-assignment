"""Phase-1 segmentation pilot harness — single file by design.

Reuses phase1/src/{io,coords,evaluate}.py unchanged.

Pipeline:
  1) build val GT for FOVs 036-040 by polygon containment (z=2 polygons)
  2) run StarDist inference (subprocess to /tmp/stardist_venv) -> .npy masks
  3) score assignment ladder:
       (a) raw mask lookup
       (b) off-mask nearest-cell rescue with global radius R
       (c) tiny-cluster cleanup (clusters < min_spots merged or backgrounded)
  4) freeze winner; compute proxy on 031-035; ship test predictions A-D

Run as `python phase2-restart/pilot.py <subcommand> [args]`. Subcommands:
  build-gt     — build val GT (cached at runs/cache/val_gt_z2.parquet)
  infer        — invoke stardist inference for given FOVs
  score        — score a mask dir + assignment variant against val GT
  test-submit  — apply frozen rule to test FOVs A-D, write submission.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1"))

from src.coords import parse_boundary_polygon  # noqa: E402

VAL_FOVS = ["FOV_036", "FOV_037", "FOV_038", "FOV_039", "FOV_040"]
PROXY_FOVS = ["FOV_031", "FOV_032", "FOV_033", "FOV_034", "FOV_035"]
TEST_FOVS = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
PIXEL_SIZE = 0.109
IMAGE_SIZE = 2048
PHASE1_DATA = ROOT / "phase1" / "data"
RUN_DIR = ROOT / "phase2-restart" / "runs"
CACHE_DIR = RUN_DIR / "cache"
STARDIST_VENV = "/tmp/stardist_venv/bin/python"
STARDIST_MODEL_DIR = ROOT / "phase2" / "external_models"
STARDIST_MODEL_NAME = "stardist_p12_v1"


def _load_fov_metadata() -> pd.DataFrame:
    return pd.read_csv(PHASE1_DATA / "reference" / "fov_metadata.csv")


def _global_to_pixel(gx: np.ndarray, gy: np.ndarray, fov_x: float, fov_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Per CLAUDE.md: image_row = 2048 − (global_x − fov_x)/0.109; col = (global_y − fov_y)/0.109."""
    rows = IMAGE_SIZE - (gx - fov_x) / PIXEL_SIZE
    cols = (gy - fov_y) / PIXEL_SIZE
    return rows.astype(int), cols.astype(int)


def build_val_gt(fov_list: list[str], z_plane: int = 2, cache: bool = True) -> pd.DataFrame:
    """Build per-spot GT cluster_id by polygon containment at z=2.

    Returns DataFrame with columns: spot_id, fov, gt_cluster_id.
    spot_id is constructed deterministically from row index in spots_train (filtered to z=z_plane).
    """
    cache_path = CACHE_DIR / f"gt_{'_'.join(fov_list)}_z{z_plane}.pkl"
    if cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    print(f"[gt] loading boundaries for z={z_plane}...")
    boundaries = pd.read_csv(PHASE1_DATA / "train" / "ground_truth" / "cell_boundaries_train.csv")
    cell_id_col = boundaries.columns[0]  # unnamed hash col
    boundaries = boundaries.rename(columns={cell_id_col: "cell_hash"})

    bx_col, by_col = f"boundaryX_z{z_plane}", f"boundaryY_z{z_plane}"
    polys = []
    for _, row in boundaries.iterrows():
        bx = row[bx_col]; by = row[by_col]
        if not isinstance(bx, str) or not isinstance(by, str):
            continue
        poly = parse_boundary_polygon(bx, by)
        if poly is not None:
            polys.append((row["cell_hash"], poly))
    print(f"[gt] parsed {len(polys)} valid polygons (out of {len(boundaries)} rows)")

    # Spatial index polygon bboxes
    poly_bboxes = np.array([p[1].bounds for p in polys])  # (n, 4): minx, miny, maxx, maxy

    print(f"[gt] loading spots for {fov_list} z={z_plane}...")
    spots = pd.read_csv(PHASE1_DATA / "train" / "ground_truth" / "spots_train.csv")
    spots = spots[(spots["fov"].isin(fov_list)) & (spots["global_z"] == float(z_plane))].copy()
    spots = spots.reset_index(drop=True)
    spots["spot_id"] = [f"gt_z{z_plane}_{i}" for i in range(len(spots))]

    print(f"[gt] {len(spots):,} spots; assigning by point-in-polygon...")
    cluster_ids = np.full(len(spots), "background", dtype=object)
    sx = spots["global_x"].values
    sy = spots["global_y"].values
    # Per-FOV bbox prune then exact contains
    for fov in fov_list:
        fov_mask = spots["fov"].values == fov
        fov_idx = np.where(fov_mask)[0]
        fx = sx[fov_idx]
        fy = sy[fov_idx]
        # bbox-only candidate filter using vectorized numpy
        # for each spot: find polygons with bbox containing the point
        for k, (chash, poly) in enumerate(polys):
            mnx, mny, mxx, mxy = poly_bboxes[k]
            in_bbox = (fx >= mnx) & (fx <= mxx) & (fy >= mny) & (fy <= mxy)
            if not in_bbox.any():
                continue
            cand_idx = fov_idx[in_bbox]
            cand_x = fx[in_bbox]
            cand_y = fy[in_bbox]
            try:
                from shapely import contains_xy
                inside = contains_xy(poly, cand_x, cand_y)
            except ImportError:
                from shapely.geometry import Point
                inside = np.array([poly.contains(Point(x, y)) for x, y in zip(cand_x, cand_y)])
            if inside.any():
                # First-write-wins (skip already-assigned to handle nested polygons deterministically)
                pick = cand_idx[inside]
                unset = cluster_ids[pick] == "background"
                cluster_ids[pick[unset]] = chash

    spots["gt_cluster_id"] = cluster_ids
    out = spots[["spot_id", "fov", "image_row", "image_col", "global_x", "global_y", "gt_cluster_id"]].copy()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(cache_path)
    print(f"[gt] frac_in_cell = {(cluster_ids != 'background').mean():.3f}")
    print(f"[gt] cells touched: {len(set(cluster_ids)) - 1}")
    print(f"[gt] cached -> {cache_path}")
    return out


def smoke_test_coords(fov: str = "FOV_001") -> dict:
    """Sanity check: build z=2 GT for one FOV, verify frac_in_cell is sensible (~0.4-0.85)."""
    df = build_val_gt([fov], z_plane=2, cache=False)
    n_total = len(df)
    n_in_cell = (df["gt_cluster_id"] != "background").sum()
    frac = n_in_cell / max(n_total, 1)
    n_cells = df.loc[df["gt_cluster_id"] != "background", "gt_cluster_id"].nunique()
    result = {"fov": fov, "n_spots": int(n_total), "n_in_cell": int(n_in_cell), "frac_in_cell": float(frac), "n_cells_touched": int(n_cells)}
    # Empirical phase-1 reality: ~25-27% in-cell across FOVs (low cell density).
    ok = 0.10 <= frac <= 0.95 and n_cells > 20
    print(f"[smoke] {result}")
    print(f"[smoke] {'PASS' if ok else 'FAIL'} - frac_in_cell {frac:.3f} (expect 0.10-0.95) and n_cells_touched={n_cells} (expect >20)")
    return result


# ---------------- StarDist subprocess shim ----------------

STARDIST_INFER_SCRIPT = ROOT / "phase2-restart" / "_stardist_infer_subproc.py"


def run_stardist_inference(fov_list: list[str], split: str, out_dir: Path) -> None:
    """Invoke stardist via /tmp/stardist_venv as a subprocess."""
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        STARDIST_VENV, str(STARDIST_INFER_SCRIPT),
        "--fovs", ",".join(fov_list),
        "--split", split,
        "--model-dir", str(STARDIST_MODEL_DIR),
        "--model-name", STARDIST_MODEL_NAME,
        "--out-dir", str(out_dir),
        "--data-root", str(PHASE1_DATA),
    ]
    print("[stardist] " + " ".join(args))
    subprocess.run(args, check=True)


# ---------------- Assignment ladder ----------------

@dataclass
class AssignConfig:
    rescue_radius: float = 0.0  # in pixels; 0 disables rung b
    cleanup_min_spots: int = 0  # 0 disables rung c


def _spots_pixel_coords(spots: pd.DataFrame, fov: str, fov_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return per-spot (rows, cols). Prefers existing image_row/image_col; falls back to global_x/y."""
    if "image_row" in spots.columns and spots["image_row"].notna().all():
        return (
            np.clip(spots["image_row"].astype(int).values, 0, IMAGE_SIZE - 1),
            np.clip(spots["image_col"].astype(int).values, 0, IMAGE_SIZE - 1),
        )
    fr = fov_meta.loc[fov_meta["fov"] == fov].iloc[0]
    rows, cols = _global_to_pixel(spots["global_x"].values, spots["global_y"].values, fr["fov_x"], fr["fov_y"])
    return np.clip(rows, 0, IMAGE_SIZE - 1), np.clip(cols, 0, IMAGE_SIZE - 1)


def assign_spots(mask: np.ndarray, spots_fov: pd.DataFrame, fov: str, fov_meta: pd.DataFrame, cfg: AssignConfig) -> np.ndarray:
    """Apply rungs (a) raw -> (b) off-mask rescue -> (c) tiny-cluster cleanup."""
    rows, cols = _spots_pixel_coords(spots_fov, fov, fov_meta)
    cell_ids = mask[rows, cols].copy()  # int

    # rung (b): off-mask rescue with global radius R
    if cfg.rescue_radius > 0:
        background_idx = np.where(cell_ids == 0)[0]
        if len(background_idx) > 0:
            from scipy.ndimage import distance_transform_edt
            cell_dt, (nr, nc) = distance_transform_edt(mask == 0, return_indices=True)
            within = cell_dt[rows[background_idx], cols[background_idx]] <= cfg.rescue_radius
            kept = background_idx[within]
            if len(kept) > 0:
                rescued = mask[nr[rows[kept], cols[kept]], nc[rows[kept], cols[kept]]]
                cell_ids[kept] = rescued

    # rung (c): tiny-cluster cleanup — clusters with < min_spots reassigned to background or merged to nearest
    if cfg.cleanup_min_spots > 0:
        unique_cells, counts = np.unique(cell_ids[cell_ids > 0], return_counts=True)
        small = unique_cells[counts < cfg.cleanup_min_spots]
        if len(small) > 0:
            # Simple rule: drop tiny clusters to background. (Merge variant deferred.)
            small_mask = np.isin(cell_ids, small)
            cell_ids[small_mask] = 0

    return cell_ids


def score_pipeline(masks_dir: Path, gt_df: pd.DataFrame, fov_meta: pd.DataFrame, cfg: AssignConfig, fovs: list[str]) -> dict:
    """Apply assignment to each FOV's mask; compute per-FOV + mean ARI vs gt_df."""
    from sklearn.metrics import adjusted_rand_score
    rows = []
    for fov in fovs:
        mask_path = masks_dir / f"{fov}.npy"
        if not mask_path.exists():
            print(f"[score] missing {mask_path}, skipping")
            continue
        mask = np.load(mask_path).astype(np.int32)
        gt_fov = gt_df[gt_df["fov"] == fov].reset_index(drop=True)
        if gt_fov.empty:
            continue
        cell_ids = assign_spots(mask, gt_fov, fov, fov_meta, cfg)
        pred_labels = np.where(cell_ids > 0, np.array([f"{fov}_cell_{v}" for v in cell_ids]), "background")
        gt_labels = gt_fov["gt_cluster_id"].astype(str).values
        ari = adjusted_rand_score(gt_labels, pred_labels)
        n_cells_pred = int(mask.max())
        n_cells_used = int(len(set(pred_labels)) - (1 if "background" in pred_labels else 0))
        frac = float((cell_ids > 0).mean())
        rows.append({"fov": fov, "ari": ari, "n_cells_pred": n_cells_pred, "n_cells_used": n_cells_used, "frac_in_cell": frac, "n_spots": len(gt_fov)})
        print(f"[score] {fov}: ARI={ari:.4f}  cells={n_cells_pred} used={n_cells_used}  fic={frac:.3f}")
    df = pd.DataFrame(rows)
    mean_ari = df["ari"].mean() if not df.empty else 0.0
    print(f"[score] mean ARI = {mean_ari:.4f}  (R={cfg.rescue_radius}, min_spots={cfg.cleanup_min_spots})")
    return {"mean_ari": float(mean_ari), "per_fov": rows, "config": {"rescue_radius": cfg.rescue_radius, "cleanup_min_spots": cfg.cleanup_min_spots}}


def write_test_submission(masks_dir: Path, fov_meta: pd.DataFrame, cfg: AssignConfig, out_path: Path) -> None:
    test_spots = pd.read_csv(PHASE1_DATA / "test_spots.csv")
    parts = []
    for fov in TEST_FOVS:
        mask_path = masks_dir / f"{fov}.npy"
        mask = np.load(mask_path).astype(np.int32)
        spots_fov = test_spots[test_spots["fov"] == fov].reset_index(drop=True)
        cell_ids = assign_spots(mask, spots_fov, fov, fov_meta, cfg)
        pred_labels = np.where(cell_ids > 0, np.array([f"{fov}_cell_{v}" for v in cell_ids]), "background")
        parts.append(pd.DataFrame({"spot_id": spots_fov["spot_id"].values, "fov": fov, "cluster_id": pred_labels}))
        print(f"[submit] {fov}: {(cell_ids > 0).mean():.3f} in-cell, {len(set(pred_labels)) - 1} unique cells")

    combined = pd.concat(parts, ignore_index=True)
    out = test_spots[["spot_id", "fov"]].merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
    out["cluster_id"] = out["cluster_id"].fillna("background")
    out.to_csv(out_path, index=False)
    print(f"[submit] wrote {out_path} ({len(out):,} rows)")


# ---------------- CLI ----------------

def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke").set_defaults(func="smoke")
    sub.add_parser("build-gt").set_defaults(func="build-gt")

    p_infer = sub.add_parser("infer")
    p_infer.set_defaults(func="infer")
    p_infer.add_argument("--split", choices=("val", "test", "proxy"), required=True)
    p_infer.add_argument("--out-dir", required=True)

    p_score = sub.add_parser("score")
    p_score.set_defaults(func="score")
    p_score.add_argument("--masks-dir", required=True)
    p_score.add_argument("--split", choices=("val", "proxy"), default="val")
    p_score.add_argument("--rescue-radius", type=float, default=0.0)
    p_score.add_argument("--cleanup-min-spots", type=int, default=0)
    p_score.add_argument("--out-json")

    p_submit = sub.add_parser("test-submit")
    p_submit.set_defaults(func="test-submit")
    p_submit.add_argument("--masks-dir", required=True)
    p_submit.add_argument("--rescue-radius", type=float, default=0.0)
    p_submit.add_argument("--cleanup-min-spots", type=int, default=0)
    p_submit.add_argument("--out", required=True)

    args = p.parse_args()
    fov_meta = _load_fov_metadata()

    if args.func == "smoke":
        smoke_test_coords()
    elif args.func == "build-gt":
        build_val_gt(VAL_FOVS, z_plane=2)
        build_val_gt(PROXY_FOVS, z_plane=2)
    elif args.func == "infer":
        fovs = {"val": VAL_FOVS, "test": TEST_FOVS, "proxy": PROXY_FOVS}[args.split]
        run_stardist_inference(fovs, args.split, Path(args.out_dir))
    elif args.func == "score":
        fovs = VAL_FOVS if args.split == "val" else PROXY_FOVS
        gt_df = build_val_gt(fovs, z_plane=2)
        cfg = AssignConfig(rescue_radius=args.rescue_radius, cleanup_min_spots=args.cleanup_min_spots)
        result = score_pipeline(Path(args.masks_dir), gt_df, fov_meta, cfg, fovs)
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(result, indent=2))
            print(f"[score] saved {args.out_json}")
    elif args.func == "test-submit":
        cfg = AssignConfig(rescue_radius=args.rescue_radius, cleanup_min_spots=args.cleanup_min_spots)
        write_test_submission(Path(args.masks_dir), fov_meta, cfg, Path(args.out))


if __name__ == "__main__":
    _cli()
