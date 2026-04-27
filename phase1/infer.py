"""Run fine-tuned Cellpose on test FOVs and generate submission.csv."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from cellpose import models as cp_models

from src.io import load_fov_images
from src.train_cellpose import compute_spot_density

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="cyto2")
parser.add_argument("--spot-sigmas", default="8")
parser.add_argument("--cellprob-threshold", type=float, default=-1.0)
parser.add_argument("--flow-threshold", type=float, default=0.4)
parser.add_argument("--params-json", default=None,
                    help="Path to best_params_<exp>.json to auto-load optimal thresholds")
parser.add_argument("--tta", action="store_true",
                    help="Test-time augmentation: run 4 orientations, majority-vote spots")
parser.add_argument("--nn-radius", type=int, default=0,
                    help="Assign background spots to nearest cell within this pixel radius")
parser.add_argument("--prob-refine", action="store_true",
                    help="Use the raw cell-probability map from Cellpose flows to recover "
                         "background spots that fall in high-confidence cellular regions but "
                         "just outside the mask (flow convergence artifacts at boundaries). "
                         "More principled than --nn-radius because it uses the model's own "
                         "confidence signal rather than blind nearest-cell distance.")
parser.add_argument("--prob-threshold", type=float, default=0.0,
                    help="Cell probability logit threshold for --prob-refine. Spots in "
                         "background where cell_prob > this value are reassigned to the "
                         "nearest cell.  0.0 means the model thinks the region is at least "
                         "50%% likely to be cellular. (default: 0.0)")
parser.add_argument("--prob-radius", type=int, default=15,
                    help="Max pixel distance from a cell boundary for --prob-refine "
                         "reassignment.  Conservative limit prevents assigning truly "
                         "extracellular spots. (default: 15)")
parser.add_argument("--multi-z-infer", action="store_true",
                    help="Run inference on each of the 5 z-planes separately and average "
                         "the cell-probability maps.  The averaged probability drives "
                         "boundary refinement (requires --prob-refine).  Useful when the "
                         "model was trained with --multi-z.")
parser.add_argument("--stitch-z", action="store_true",
                    help="Expand the max-projection mask using z-plane agreement: run "
                         "inference on each of the 5 z-planes and fill background pixels "
                         "that are within --stitch-radius of an existing cell and assigned "
                         "as cellular in any z-plane.  Recovers cells that are visible in "
                         "focused z-planes but washed out in the max projection.")
parser.add_argument("--stitch-radius", type=int, default=20,
                    help="Max pixel distance from an existing cell for --stitch-z fill "
                         "(default: 20)")
parser.add_argument("--stitch-threshold", type=float, default=0.0,
                    help="Use Cellpose's built-in z-plane stitching: pass all 5 z-planes "
                         "as a (Z,C,H,W) stack and let Cellpose stitch cells across slices "
                         "by IOU >= this threshold.  The middle z-plane (z=2, which matches "
                         "the GT labeling plane) is used for final spot assignment.  "
                         "0.0 disables (default).  Try 0.1–0.5.")
args = parser.parse_args()

EXP_NAME    = args.exp_name
SPOT_SIGMAS = [float(s) for s in args.spot_sigmas.split(",")]
DATA_ROOT   = "/scratch/cg4652/competition"
TEST_FOVS   = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]

CELLPROB_THRESH = args.cellprob_threshold
FLOW_THRESH     = args.flow_threshold
if args.params_json and os.path.exists(args.params_json):
    with open(args.params_json) as f:
        _p = json.load(f)["best"]
    CELLPROB_THRESH = _p["cellprob_threshold"]
    FLOW_THRESH     = _p["flow_threshold"]
    print(f"Loaded thresholds from {args.params_json}: "
          f"cellprob={CELLPROB_THRESH}, flow={FLOW_THRESH}")

os.makedirs("logs", exist_ok=True)

MODEL_DIR   = f"models/{EXP_NAME}"
FINAL_MODEL = os.path.join(MODEL_DIR, f"cellpose_{EXP_NAME}")

if os.path.exists(FINAL_MODEL):
    print(f"Loading fine-tuned model: {FINAL_MODEL}")
    seg_model = cp_models.CellposeModel(gpu=True, pretrained_model=FINAL_MODEL)
else:
    print(f"Fine-tuned model not found, falling back to pretrained {EXP_NAME}")
    seg_model = cp_models.CellposeModel(gpu=True, model_type=EXP_NAME)

test_spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
print(f"Test spots: {len(test_spots):,}  TTA={'on' if args.tta else 'off'}  "
      f"nn_radius={args.nn_radius}  prob_refine={'on' if args.prob_refine else 'off'}  "
      f"multi_z_infer={'on' if args.multi_z_infer else 'off'}  "
      f"stitch_z={'on' if args.stitch_z else 'off'}")


# ── TTA orientations: (transform_image_fn, untransform_mask_fn) ──────────────
# Each pair applies a spatial transform to the image before inference and its
# inverse to the output mask so spots can be looked up in original coordinates.
TTA_OPS = [
    (lambda x: x,                              lambda m: m),
    (lambda x: np.flip(x, axis=2).copy(),      lambda m: np.flip(m, axis=1).copy()),
    (lambda x: np.flip(x, axis=1).copy(),      lambda m: np.flip(m, axis=0).copy()),
    (lambda x: np.rot90(x, k=2, axes=(1,2)).copy(), lambda m: np.rot90(m, k=2).copy()),
]


def _extract_cellprob(flows: list) -> np.ndarray | None:
    """Extract the (H, W) cell-probability logit map from Cellpose flows.

    Cellpose eval() returns flows as a list of arrays.  The cell-probability map
    is the only 2D array in that list (all other items are 3D gradient arrays).
    We locate it by shape rather than hard-coding an index so this works across
    Cellpose 2.x / 3.x / v4 API variants.
    """
    for f in flows:
        if isinstance(f, np.ndarray) and f.ndim == 2:
            return f
    return None


def run_inference(
    img: np.ndarray,
    dapi_stack: np.ndarray | None = None,
    polyt_stack: np.ndarray | None = None,
    density_ch: list | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run model; return (masks, cell_prob_map).

    When --multi-z-infer is set, the model is also run on each of the 5 individual
    z-plane images and the resulting cell-probability maps are averaged.  The averaged
    map is more robust to focal-depth variation and drives --prob-refine boundary
    recovery.  Masks still come from the max-projection run (best single input).

    dapi_stack / polyt_stack / density_ch are required only for --multi-z-infer.
    """
    eval_kwargs = dict(
        diameter=0,
        cellprob_threshold=CELLPROB_THRESH,
        flow_threshold=FLOW_THRESH,
        channel_axis=0,
    )

    if args.tta:
        # TTA: collect one mask per orientation, return stacked array for later voting
        all_masks = []
        for aug_fn, unaug_fn in TTA_OPS:
            aug_masks, _, _ = seg_model.eval(aug_fn(img), **eval_kwargs)
            all_masks.append(unaug_fn(aug_masks))
        return np.stack(all_masks, axis=0), None  # TTA doesn't support prob-refine

    masks, flows, _ = seg_model.eval(img, **eval_kwargs)
    cell_prob = _extract_cellprob(flows)

    # ── Multi-z probability averaging ─────────────────────────────────────────
    # Run the model on each individual z-plane image and average the probability
    # maps.  This gives a more stable confidence signal than max-proj alone:
    # cells well-focused in some z-planes contribute strong probability even if
    # they appear blurry in others.  Masks are still from the max-proj run.
    if args.multi_z_infer and dapi_stack is not None and cell_prob is not None:
        z_probs = [cell_prob]
        n_z = dapi_stack.shape[0]
        for z_idx in range(n_z):
            img_z = np.stack([polyt_stack[z_idx].astype(np.float32),
                              dapi_stack[z_idx].astype(np.float32),
                              *density_ch], axis=0)
            _, flows_z, _ = seg_model.eval(img_z, **eval_kwargs)
            cp_z = _extract_cellprob(flows_z)
            if cp_z is not None:
                z_probs.append(cp_z)
        cell_prob = np.mean(z_probs, axis=0)
        print(f"    multi-z-infer: averaged {len(z_probs)} probability maps")

    # ── Cellpose built-in z-plane stitching ──────────────────────────────────
    # Pass all z-planes as a (Z, C, H, W) stack; Cellpose segments each slice
    # with the 2D model and stitches cells across slices using IOU overlap.
    # The middle z-plane (z=2) is used for final 2D spot assignment since that
    # is the plane the ground truth boundaries were labeled from.
    if args.stitch_threshold > 0 and dapi_stack is not None and not args.tta:
        n_z = dapi_stack.shape[0]
        z_stack = np.stack([
            np.stack([polyt_stack[z].astype(np.float32),
                      dapi_stack[z].astype(np.float32),
                      *density_ch], axis=0)
            for z in range(n_z)
        ], axis=0)  # (Z, C, H, W)
        masks_3d, flows_3d, _ = seg_model.eval(
            z_stack,
            diameter=0,
            cellprob_threshold=CELLPROB_THRESH,
            flow_threshold=FLOW_THRESH,
            channel_axis=1,
            z_axis=0,
            stitch_threshold=args.stitch_threshold,
        )
        # masks_3d is (Z, H, W) — use middle z-plane for assignment
        mid_z = n_z // 2
        masks = masks_3d[mid_z] if masks_3d.ndim == 3 else masks_3d
        cell_prob = _extract_cellprob(flows_3d[mid_z]) if isinstance(flows_3d, list) else cell_prob
        print(f"    stitch-threshold={args.stitch_threshold}: "
              f"{int(masks.max())} cells at z={mid_z} (from {n_z}-plane stack)")

    # ── Z-plane mask stitching ────────────────────────────────────────────────
    # Run the 2D model on each individual z-plane and expand the max-proj mask
    # to cover pixels that are background in max-proj but clearly cellular in one
    # or more focused z-planes.  Only fills pixels within stitch_radius of an
    # existing cell to avoid creating isolated ghost cells in empty regions.
    if args.stitch_z and dapi_stack is not None and not args.tta:
        expanded = masks.copy()
        n_z = dapi_stack.shape[0]
        total_filled = 0
        for z_idx in range(n_z):
            img_z = np.stack([polyt_stack[z_idx].astype(np.float32),
                              dapi_stack[z_idx].astype(np.float32),
                              *density_ch], axis=0)
            masks_z, _, _ = seg_model.eval(img_z, **eval_kwargs)
            new_pixels = (expanded == 0) & (masks_z > 0)
            if new_pixels.any() and expanded.max() > 0:
                dist, (nr, nc) = distance_transform_edt(expanded == 0,
                                                        return_indices=True)
                within = dist <= args.stitch_radius
                to_fill = new_pixels & within
                if to_fill.any():
                    expanded[to_fill] = expanded[nr[to_fill], nc[to_fill]]
                    total_filled += int(to_fill.sum())
        if total_filled:
            print(f"    stitch-z: filled {total_filled:,} pixels across {n_z} z-planes "
                  f"(radius≤{args.stitch_radius}px)")
        masks = expanded

    return masks, cell_prob


def assign_spots(
    masks_or_stack: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    fov_name: str,
    cell_prob: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each spot to a cluster label string.

    TTA strategy: cell IDs are arbitrary per-mask so we cannot vote on them
    directly across orientations.  Instead we vote on the binary in-cell /
    background decision, then use orientation-0's cell ID for spots where a
    majority of orientations agree the spot is inside *some* cell.

    prob-refine: for spots in background where the model's cell-probability map
    exceeds prob_threshold (the model genuinely thinks there is a cell here but
    flow convergence left the pixel unassigned), reassign to the nearest cell
    within prob_radius pixels.  This directly targets boundary mis-assignments
    without blind distance-based recovery.
    """
    is_tta = masks_or_stack.ndim == 3  # (N_aug, H, W)

    if is_tta:
        ref_mask = masks_or_stack[0]
        ref_ids  = ref_mask[rows, cols]
        in_cell_votes = np.sum(
            np.stack([m[rows, cols] > 0 for m in masks_or_stack], axis=0),
            axis=0,
        )
        n_aug = masks_or_stack.shape[0]
        cell_ids = np.where(in_cell_votes >= (n_aug / 2), ref_ids, 0)
    else:
        cell_ids = masks_or_stack[rows, cols]
        ref_mask = masks_or_stack

    # ── Probability-guided boundary recovery ──────────────────────────────────
    # The Cellpose cell-probability logit (flows' 2D array) captures WHERE the
    # model thinks cells are before the flow-convergence step.  Spots in background
    # where cell_prob > prob_threshold are in regions the model tagged as cellular
    # but that the flow didn't assign to any mask — exactly the boundary leakage
    # we want to fix.  We then restrict to spots within prob_radius of an actual
    # cell to avoid reassigning truly extracellular spots far from any boundary.
    if args.prob_refine and cell_prob is not None:
        bg = cell_ids == 0
        if bg.any() and ref_mask.max() > 0:
            bg_rows, bg_cols = rows[bg], cols[bg]
            high_conf = cell_prob[bg_rows, bg_cols] > args.prob_threshold
            if high_conf.any():
                cell_pixels = ref_mask > 0
                dist, (nr, nc) = distance_transform_edt(~cell_pixels, return_indices=True)
                near_cell = dist[bg_rows, bg_cols] <= args.prob_radius
                should_assign = high_conf & near_cell
                if should_assign.any():
                    nn_ids = ref_mask[nr[bg_rows, bg_cols], nc[bg_rows, bg_cols]]
                    cell_ids = cell_ids.copy()
                    cell_ids[np.where(bg)[0][should_assign]] = nn_ids[should_assign]
                    print(f"    prob-refine: recovered {int(should_assign.sum()):,} spots "
                          f"(prob>{args.prob_threshold:.1f}, radius≤{args.prob_radius}px)")

    # ── Legacy distance-only fallback (nn_radius) ─────────────────────────────
    if args.nn_radius > 0:
        bg = cell_ids == 0
        if bg.any() and ref_mask.max() > 0:
            cell_pixels = ref_mask > 0
            dist, (nr, nc) = distance_transform_edt(~cell_pixels, return_indices=True)
            bg_rows, bg_cols = rows[bg], cols[bg]
            within = dist[bg_rows, bg_cols] <= args.nn_radius
            nn_ids = ref_mask[nr[bg_rows, bg_cols], nc[bg_rows, bg_cols]]
            cell_ids = cell_ids.copy()
            cell_ids[np.where(bg)[0][within]] = nn_ids[within]
            print(f"    NN fallback: recovered {int(within.sum()):,} spots")

    labels = np.where(
        cell_ids > 0,
        np.array([f"{fov_name}_cell_{v}" for v in cell_ids]),
        "background",
    )
    return labels


# ── Inference ────────────────────────────────────────────────────────────────
all_parts = []

for fov_name in TEST_FOVS:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_dir}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    fov_spots  = test_spots[test_spots["fov"] == fov_name]
    dapi_max   = np.max(dapi,  axis=0)
    polyt_max  = np.max(polyt, axis=0)
    density_ch = [compute_spot_density(fov_spots, sigma=s) for s in SPOT_SIGMAS]
    img        = np.stack([polyt_max, dapi_max, *density_ch], axis=0)

    need_stacks = args.multi_z_infer or args.stitch_z or args.stitch_threshold > 0
    masks_out, cell_prob = run_inference(
        img,
        dapi_stack=dapi if need_stacks else None,
        polyt_stack=polyt if need_stacks else None,
        density_ch=density_ch if need_stacks else None,
    )
    n_cells    = int(masks_out[0].max() if args.tta else masks_out.max())
    print(f"{fov_name}: {n_cells} cells detected")

    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    labels = assign_spots(masks_out, rows, cols, fov_name, cell_prob=cell_prob)

    n_assigned = (labels != "background").sum()
    print(f"  {fov_name}: {n_assigned:,}/{len(labels):,} spots assigned "
          f"({100*n_assigned/len(labels):.1f}%)")

    all_parts.append(pd.DataFrame({
        "spot_id":    fov_spots["spot_id"].values,
        "fov":        fov_name,
        "cluster_id": labels,
    }))

# ── Build and save submission ─────────────────────────────────────────────────
combined   = pd.concat(all_parts, ignore_index=True)
submission = (
    test_spots[["spot_id", "fov"]]
    .merge(combined[["spot_id", "cluster_id"]], on="spot_id", how="left")
)
submission["cluster_id"] = submission["cluster_id"].fillna("background")

suffix = EXP_NAME
if args.tta:
    suffix += "_tta"
if args.nn_radius > 0:
    suffix += f"_nn{args.nn_radius}"
if args.prob_refine:
    suffix += f"_probrefine{args.prob_threshold:.1f}r{args.prob_radius}"
if args.multi_z_infer:
    suffix += "_mzi"
if args.stitch_z:
    suffix += f"_sz{args.stitch_radius}"
if args.stitch_threshold > 0:
    suffix += f"_st{args.stitch_threshold}"
out_path = f"submission_{suffix}.csv"
submission[["spot_id", "fov", "cluster_id"]].to_csv(out_path, index=False)
print(f"\nSaved {out_path} — {len(submission):,} rows")
print(submission["cluster_id"].value_counts().head(5))
