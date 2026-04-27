"""Qualitative + quantitative error analysis of a trained model on val FOVs 036-040.

Answers "where is the model wrong?" that aggregate ARI can't:
  - Missed cells (GT cell with 0 predicted pixels overlapping)
  - Spurious cells (predicted cell with 0 GT overlap)
  - Split cells (1 GT cell → multiple predicted cells)
  - Merged cells (multiple GT cells → 1 predicted cell)
  - Boundary leakage (spots in correct-cell pixels at GT boundary lost to bg)
  - Per-spot assignment outcomes: correct_cell, wrong_cell, false_positive,
    false_negative (i.e. a confusion-matrix view on spots)

Outputs:
  logs/diagnose_<exp>/<FOV>.png       — visual overlay
  logs/diagnose_<exp>/summary.json    — aggregate + per-FOV counts

Usage:
    python diagnose_val_errors.py --exp-name cyto2_warmup_long
    python diagnose_val_errors.py --exp-name stardist --arch stardist \
        --stardist-weights models/stardist/weights_best.h5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask, compute_spot_density, compute_zstack_features

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True)
parser.add_argument("--arch", default="cellpose", choices=["cellpose", "stardist"],
                    help="Model architecture to load (default: cellpose)")
parser.add_argument("--spot-sigmas", default="8",
                    help="Comma-separated sigma(s) used during training")
parser.add_argument("--params-json", default=None,
                    help="Path to best_params_<exp>.json for optimal thresholds")
parser.add_argument("--cellprob-threshold", type=float, default=-1.0)
parser.add_argument("--flow-threshold", type=float, default=0.4)
parser.add_argument("--zstats", action="store_true")
parser.add_argument("--stardist-weights", default=None,
                    help="Path to StarDist weights_best.h5 (for --arch stardist)")
parser.add_argument("--out-dir", default=None,
                    help="Output directory (default: logs/diagnose_<exp>)")
args = parser.parse_args()

EXP_NAME    = args.exp_name
SPOT_SIGMAS = [float(s) for s in args.spot_sigmas.split(",")]
# Honor MERFISH_DATA_ROOT so the same script works on HPC (/scratch/cg4652/...)
# and Modal (mounted volume, e.g. /root/data).  Same convention as notebooks.
DATA_ROOT   = os.environ.get("MERFISH_DATA_ROOT", "/scratch/cg4652/competition")
VAL_FOVS    = [f"FOV_{i:03d}" for i in range(36, 41)]
OUT_DIR     = args.out_dir or f"logs/diagnose_{EXP_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional threshold override from sweep
CP_THRESH, FL_THRESH = args.cellprob_threshold, args.flow_threshold
if args.params_json and os.path.exists(args.params_json):
    with open(args.params_json) as f:
        p = json.load(f)["best"]
    CP_THRESH, FL_THRESH = p["cellprob_threshold"], p["flow_threshold"]
    print(f"Loaded thresholds: cellprob={CP_THRESH}, flow={FL_THRESH}")


# ── Model loaders (pluggable per architecture) ────────────────────────────────
def _make_cellpose():
    from cellpose import models as cp_models
    model_path = f"models/{EXP_NAME}/cellpose_{EXP_NAME}"
    model = cp_models.CellposeModel(gpu=True, pretrained_model=model_path)

    def predict(img):
        masks, _, _ = model.eval(img, diameter=0,
                                  cellprob_threshold=CP_THRESH,
                                  flow_threshold=FL_THRESH,
                                  channel_axis=0)
        return masks
    return predict


def _make_stardist():
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    weights = args.stardist_weights or f"models/{EXP_NAME}/weights_best.h5"
    # csbdeep's load_weights() prepends self.logdir (= basedir/name) to the
    # filename we pass, so pass the basename only to avoid a doubled path.
    model = StarDist2D(None, name=os.path.basename(os.path.dirname(weights)),
                       basedir=os.path.dirname(os.path.dirname(weights)))
    model.load_weights(os.path.basename(weights))

    def predict(img):
        # StarDist uses DAPI max only, per-image 1-99.8% percentile normalized
        dapi_max = img[1]  # channel order: [polyt, dapi, ...]
        x = normalize(dapi_max, 1, 99.8)
        masks, _ = model.predict_instances(x)
        return masks
    return predict


predict_fn = _make_stardist() if args.arch == "stardist" else _make_cellpose()


# ── Load shared GT ────────────────────────────────────────────────────────────
print("Loading metadata + GT...")
meta  = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")


# ── Per-FOV diagnostic ────────────────────────────────────────────────────────
def analyze_fov(fov_name: str) -> dict:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    dapi, polyt = load_fov_images(fov_dir)
    fov_x  = meta.loc[fov_name, "fov_x"]
    fov_y  = meta.loc[fov_name, "fov_y"]
    fov_spots = spots_train[spots_train["fov"] == fov_name]
    density_ch = [compute_spot_density(fov_spots, sigma=s) for s in SPOT_SIGMAS]

    if args.zstats:
        zf = compute_zstack_features(dapi, polyt)
        img = np.stack([zf["polyt_max"], zf["dapi_max"],
                        zf["polyt_mean"], zf["dapi_mean"],
                        zf["polyt_std"],  zf["dapi_std"],
                        *density_ch], axis=0)
    else:
        dapi_max  = np.max(dapi,  axis=0)
        polyt_max = np.max(polyt, axis=0)
        img = np.stack([polyt_max, dapi_max, *density_ch], axis=0)

    gt_mask   = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
    pred_mask = predict_fn(img)

    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    gt_ids   = gt_mask[rows, cols]
    pred_ids = pred_mask[rows, cols]

    # ── Per-spot outcome classification ────────────────────────────────────────
    # Build best-matching permutation: for each predicted cell, find the GT cell
    # it overlaps most with (spot-wise), then compare spot-by-spot.
    spot_outcomes = np.empty(len(gt_ids), dtype=object)
    gt_in   = gt_ids   > 0
    pred_in = pred_ids > 0
    spot_outcomes[ gt_in &  pred_in] = "in_cell"       # refined below
    spot_outcomes[~gt_in & ~pred_in] = "correct_bg"
    spot_outcomes[ gt_in & ~pred_in] = "false_negative" # missed (should be in-cell)
    spot_outcomes[~gt_in &  pred_in] = "false_positive" # spurious assignment

    # Among in_cell/in_cell, separate "correct_cell" from "wrong_cell" via
    # spot-based cell matching: for each predicted cell, its best-matching GT cell
    # is the one sharing the most spots.
    both_in = gt_in & pred_in
    if both_in.any():
        gt_sub, pred_sub = gt_ids[both_in], pred_ids[both_in]
        # Build mapping: pred_cell → most-shared gt_cell
        from collections import Counter
        best_match: dict[int, int] = {}
        for pc in np.unique(pred_sub):
            cand = Counter(gt_sub[pred_sub == pc])
            best_match[int(pc)] = int(cand.most_common(1)[0][0])
        correct = np.array([best_match[int(pc)] == int(gc)
                            for pc, gc in zip(pred_sub, gt_sub)])
        idx = np.where(both_in)[0]
        spot_outcomes[idx[ correct]] = "correct_cell"
        spot_outcomes[idx[~correct]] = "wrong_cell"

    # ── Cell-level structural errors ───────────────────────────────────────────
    gt_cells   = [c for c in np.unique(gt_mask)   if c]
    pred_cells = [c for c in np.unique(pred_mask) if c]

    # Overlap matrix: pixels(gt_cell ∩ pred_cell)
    missed_cells   = 0  # GT cells with no predicted overlap
    spurious_cells = 0  # pred cells with no GT overlap
    split_cells    = 0  # GT cells overlapping ≥2 pred cells substantially
    merged_cells   = 0  # pred cells overlapping ≥2 GT cells substantially

    for gc in gt_cells:
        gc_pixels = gt_mask == gc
        if gc_pixels.sum() == 0:
            continue
        overlapping = np.unique(pred_mask[gc_pixels])
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) == 0:
            missed_cells += 1
        else:
            # Count "substantial" overlaps: ≥10% of GT cell area
            overlap_fracs = [(pred_mask[gc_pixels] == pc).sum() / gc_pixels.sum()
                             for pc in overlapping]
            if sum(f >= 0.1 for f in overlap_fracs) >= 2:
                split_cells += 1

    for pc in pred_cells:
        pc_pixels = pred_mask == pc
        if pc_pixels.sum() == 0:
            continue
        overlapping = np.unique(gt_mask[pc_pixels])
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) == 0:
            spurious_cells += 1
        else:
            overlap_fracs = [(gt_mask[pc_pixels] == gc).sum() / pc_pixels.sum()
                             for gc in overlapping]
            if sum(f >= 0.1 for f in overlap_fracs) >= 2:
                merged_cells += 1

    ari = float(adjusted_rand_score(gt_ids, pred_ids))

    # ── Save overlay PNG ───────────────────────────────────────────────────────
    # DAPI grayscale background + GT boundaries (green) + pred boundaries (red)
    # + spot markers colored by outcome.
    from skimage.segmentation import find_boundaries
    gt_bounds   = find_boundaries(gt_mask,   mode="outer")
    pred_bounds = find_boundaries(pred_mask, mode="outer")

    fig, ax = plt.subplots(figsize=(12, 12))
    dapi_disp = np.max(dapi, axis=0).astype(np.float32)
    dapi_disp = (dapi_disp - dapi_disp.min()) / max(dapi_disp.max() - dapi_disp.min(), 1)
    ax.imshow(dapi_disp, cmap="gray", vmin=0, vmax=0.4)

    overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    overlay[gt_bounds]   = [0.2, 1.0, 0.2, 0.8]   # green = GT
    overlay[pred_bounds] = [1.0, 0.3, 0.3, 0.8]   # red   = pred
    ax.imshow(overlay)

    color_map = {
        "correct_cell":   "#00ff00",
        "wrong_cell":     "#ff00ff",
        "correct_bg":     "#666666",
        "false_positive": "#ffaa00",
        "false_negative": "#00aaff",
    }
    # Subsample spots if > 3k for visual clarity
    show_idx = np.arange(len(spot_outcomes))
    if len(show_idx) > 3000:
        show_idx = np.random.default_rng(0).choice(show_idx, 3000, replace=False)
    for outcome, color in color_map.items():
        sel = show_idx[spot_outcomes[show_idx] == outcome]
        if len(sel):
            ax.scatter(cols[sel], rows[sel], s=3, c=color, alpha=0.7)

    legend = [
        Patch(color="#00ff00", label=f"correct_cell ({(spot_outcomes=='correct_cell').sum()})"),
        Patch(color="#ff00ff", label=f"wrong_cell ({(spot_outcomes=='wrong_cell').sum()})"),
        Patch(color="#00aaff", label=f"false_neg (pred=bg, true=cell) "
                                      f"({(spot_outcomes=='false_negative').sum()})"),
        Patch(color="#ffaa00", label=f"false_pos (pred=cell, true=bg) "
                                      f"({(spot_outcomes=='false_positive').sum()})"),
        Patch(color="#666666", label=f"correct_bg ({(spot_outcomes=='correct_bg').sum()})"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_title(f"{fov_name}  |  {EXP_NAME}  |  ARI={ari:.4f}  |  "
                 f"GT={len(gt_cells)} cells, pred={len(pred_cells)} cells  |  "
                 f"missed={missed_cells}  spurious={spurious_cells}  "
                 f"split={split_cells}  merged={merged_cells}", fontsize=10)
    ax.axis("off")
    out_png = os.path.join(OUT_DIR, f"{fov_name}.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fov_name}: ARI={ari:.4f}  GT={len(gt_cells)}  pred={len(pred_cells)}  "
          f"missed={missed_cells}  spurious={spurious_cells}  "
          f"split={split_cells}  merged={merged_cells}  →  {out_png}")

    return {
        "fov": fov_name,
        "ari": ari,
        "gt_cells": len(gt_cells),
        "pred_cells": len(pred_cells),
        "missed_cells": missed_cells,
        "spurious_cells": spurious_cells,
        "split_cells": split_cells,
        "merged_cells": merged_cells,
        "n_spots": int(len(gt_ids)),
        "correct_cell": int((spot_outcomes == "correct_cell").sum()),
        "wrong_cell":   int((spot_outcomes == "wrong_cell").sum()),
        "false_positive": int((spot_outcomes == "false_positive").sum()),
        "false_negative": int((spot_outcomes == "false_negative").sum()),
        "correct_bg":   int((spot_outcomes == "correct_bg").sum()),
    }


print(f"\n== Diagnosing '{EXP_NAME}' on val FOVs {VAL_FOVS} ==")
per_fov = [analyze_fov(f) for f in VAL_FOVS]

summary = {
    "exp": EXP_NAME,
    "arch": args.arch,
    "per_fov": per_fov,
    "mean_ari": float(np.mean([r["ari"] for r in per_fov])),
    "total_missed":   sum(r["missed_cells"]   for r in per_fov),
    "total_spurious": sum(r["spurious_cells"] for r in per_fov),
    "total_split":    sum(r["split_cells"]    for r in per_fov),
    "total_merged":   sum(r["merged_cells"]   for r in per_fov),
    "total_wrong_cell":    sum(r["wrong_cell"]    for r in per_fov),
    "total_false_positive": sum(r["false_positive"] for r in per_fov),
    "total_false_negative": sum(r["false_negative"] for r in per_fov),
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n== Summary ==")
print(f"  mean ARI          : {summary['mean_ari']:.4f}")
print(f"  missed GT cells   : {summary['total_missed']}")
print(f"  spurious cells    : {summary['total_spurious']}")
print(f"  split cells       : {summary['total_split']}")
print(f"  merged cells      : {summary['total_merged']}")
print(f"  wrong-cell spots  : {summary['total_wrong_cell']}")
print(f"  false_positive    : {summary['total_false_positive']} (pred=cell, true=bg)")
print(f"  false_negative    : {summary['total_false_negative']} (pred=bg, true=cell)")
print(f"\nOverlays + JSON → {OUT_DIR}/")
