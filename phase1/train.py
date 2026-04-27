"""Fine-tune Cellpose on training FOVs with checkpoint save/resume for HPC preemption."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cellpose import models as cp_models, train as cp_train

from src.io import load_fov_images
from src.train_cellpose import (augment_training_data, boundaries_to_mask,
                                 compute_spot_density, compute_zstack_features)

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="cyto2",
                    choices=["cyto2", "cyto3", "nuclei"],
                    help="Cellpose pretrained model to fine-tune from")
parser.add_argument("--exp-name", default=None,
                    help="Experiment name (defaults to base-model name)")
parser.add_argument("--spot-sigmas", default="8",
                    help="Comma-separated Gaussian sigma(s) for spot density channel(s) "
                         "(e.g. '8' or '4,8,16')")
parser.add_argument("--augment", action="store_true",
                    help="Expand training set 8x with flip/rotation symmetries + intensity jitter")
parser.add_argument("--zstats", action="store_true",
                    help="Add z-stack mean+std channels to input (dapi_mean, dapi_std, "
                         "polyt_mean, polyt_std) alongside max projection")
parser.add_argument("--use-union-z", action="store_true",
                    help="Use the union of all 5 z-plane boundaries as the GT mask for "
                         "max-projection training samples.  Fixes the systematic mismatch "
                         "where z=2-only GT underestimates cell extent visible in max-proj.")
parser.add_argument("--multi-z", action="store_true",
                    help="Generate 6 training samples per FOV instead of 1: one max-projection "
                         "sample (with union-z GT) plus 5 individual z-plane samples (each with "
                         "its own z-specific GT mask).  Gives 6x more training data from the "
                         "same 35 FOVs, improving generalisation and reducing val→test gap. "
                         "Implies --use-union-z for the max-projection sample.")
parser.add_argument("--epochs", type=int, default=300,
                    help="Total training epochs (default 300)")
parser.add_argument("--lr-schedule", default="flat",
                    choices=["flat", "cosine", "warmup_cosine"],
                    help="Learning rate schedule: flat=constant, cosine=cosine decay, "
                         "warmup_cosine=linear warmup then cosine decay")
parser.add_argument("--lr-min", type=float, default=1e-7,
                    help="Minimum LR for cosine schedule end (default 1e-7)")
parser.add_argument("--lr-peak", type=float, default=5e-5,
                    help="Peak LR for warmup_cosine (default 5e-5)")
parser.add_argument("--lr-warmup-epochs", type=int, default=20,
                    help="Number of warmup epochs for warmup_cosine (default 20)")
parser.add_argument("--weight-decay", type=float, default=0.1,
                    help="Weight decay for optimizer (default 0.1)")
parser.add_argument("--all-fovs", action="store_true",
                    help="Train on all 40 FOVs (001-040) and skip the final validation loop. "
                         "Use when val ARI has become unreliable and we want maximum training "
                         "signal for the final Kaggle submission.")
args = parser.parse_args()

EXP_NAME = args.exp_name or args.base_model
SPOT_SIGMAS = [float(s) for s in args.spot_sigmas.split(",")]

DATA_ROOT      = "/scratch/cg4652/competition"
PIXEL_SIZE     = 0.109
MODEL_SAVE_DIR = f"models/{EXP_NAME}"
MODEL_NAME     = f"cellpose_{EXP_NAME}"
TOTAL_EPOCHS   = args.epochs
CHUNK_EPOCHS   = 5    # save a checkpoint every N epochs

def get_lr(epoch: int) -> float:
    """Return the learning rate for the given epoch based on the schedule."""
    base_lr = 1e-5
    if args.lr_schedule == "flat":
        return base_lr
    elif args.lr_schedule == "cosine":
        progress = epoch / TOTAL_EPOCHS
        return args.lr_min + 0.5 * (base_lr - args.lr_min) * (1 + math.cos(math.pi * progress))
    else:  # warmup_cosine
        if epoch < args.lr_warmup_epochs:
            return base_lr + (args.lr_peak - base_lr) * epoch / max(args.lr_warmup_epochs, 1)
        progress = (epoch - args.lr_warmup_epochs) / max(TOTAL_EPOCHS - args.lr_warmup_epochs, 1)
        return args.lr_min + 0.5 * (args.lr_peak - args.lr_min) * (1 + math.cos(math.pi * progress))

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Signal handler (SIGUSR1 sent 120s before time limit via --signal) ────────
_stop_after_chunk = False

def _sigusr1_handler(signum, frame):
    global _stop_after_chunk
    print("SIGUSR1 received: will stop after current checkpoint", flush=True)
    _stop_after_chunk = True

signal.signal(signal.SIGUSR1, _sigusr1_handler)

# ── Checkpoint detection ────────────────────────────────────────────────────
state_file = os.path.join(MODEL_SAVE_DIR, "train_state.json")
completed_epochs = 0
latest_ckpt = None

if os.path.exists(state_file):
    try:
        with open(state_file) as f:
            state = json.load(f)
        completed_epochs = state.get("completed_epochs", 0)
        latest_ckpt = state.get("latest_checkpoint")
        if latest_ckpt and not os.path.exists(latest_ckpt):
            print(f"WARNING: checkpoint not found ({latest_ckpt}), restarting from scratch")
            completed_epochs = 0
            latest_ckpt = None
        else:
            print(f"Resuming from epoch {completed_epochs}/{TOTAL_EPOCHS}: {latest_ckpt}")
    except (json.JSONDecodeError, KeyError):
        print(f"WARNING: corrupt state file ({state_file}), restarting from scratch")
        completed_epochs = 0
        latest_ckpt = None
else:
    print("No checkpoint found, starting fresh")

print("Loading metadata and ground truth...")
meta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(
    f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0
)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

# Build training set (FOVs 001-035 by default, 001-040 with --all-fovs)
train_images, train_masks = [], []
train_range = (1, 41) if args.all_fovs else (1, 36)
train_fovs = [f"FOV_{i:03d}" for i in range(*train_range)]

use_union_z = args.use_union_z or args.multi_z  # multi_z implies union-z for max-proj

for fov_name in train_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue
    try:
        dapi, polyt = load_fov_images(fov_dir)
        fov_x = meta.loc[fov_name, "fov_x"]
        fov_y = meta.loc[fov_name, "fov_y"]
        fov_spots = spots_train[spots_train["fov"] == fov_name]
        density_channels = [compute_spot_density(fov_spots, sigma=s) for s in SPOT_SIGMAS]

        # ── Max-projection sample (always included) ───────────────────────────
        m_maxproj = boundaries_to_mask(cells, fov_name, fov_x, fov_y, use_all_z=use_union_z)
        if m_maxproj.max() == 0:
            print(f"  No cells in mask: {fov_name}")
            continue
        if args.zstats:
            zf = compute_zstack_features(dapi, polyt)
            img_channels = [zf["polyt_max"], zf["dapi_max"],
                            zf["polyt_mean"], zf["dapi_mean"],
                            zf["polyt_std"],  zf["dapi_std"],
                            *density_channels]
        else:
            dapi_max  = np.max(dapi,  axis=0)
            polyt_max = np.max(polyt, axis=0)
            img_channels = [polyt_max, dapi_max, *density_channels]
        train_images.append(np.stack(img_channels, axis=0))
        train_masks.append(m_maxproj)

        # ── Per-z-plane samples (only with --multi-z) ─────────────────────────
        # Each z-plane is a separate training image paired with its own z-specific
        # GT mask.  This gives 5 additional samples per FOV (6x total), and the
        # model sees cells at varied focal depths — critical for generalisation.
        # Note: spot_density is 2D (no z coordinate), so the same density map is
        # reused across z-planes.  zstats mode is max-proj only (needs full stack).
        n_z_added = 0
        if args.multi_z:
            n_z = dapi.shape[0]  # 5 z-planes
            for z_idx in range(n_z):
                m_z = boundaries_to_mask(cells, fov_name, fov_x, fov_y, z_plane=z_idx)
                if m_z.max() == 0:
                    continue
                img_channels_z = [polyt[z_idx].astype(np.float32),
                                   dapi[z_idx].astype(np.float32),
                                   *density_channels]
                train_images.append(np.stack(img_channels_z, axis=0))
                train_masks.append(m_z)
                n_z_added += 1

        suffix = f" + {n_z_added} z-plane samples" if n_z_added else ""
        print(f"  Loaded {fov_name}: {m_maxproj.max()} cells{suffix}")
    except Exception as exc:
        print(f"  Skipping {fov_name}: {exc}")

print(f"\nTraining set: {len(train_images)} samples (before augmentation)")
if args.augment:
    train_images, train_masks = augment_training_data(train_images, train_masks)
    print(f"After 8x flip/rotation augmentation: {len(train_images)} samples")

# ── Chunked training with checkpoints ───────────────────────────────────────
if completed_epochs >= TOTAL_EPOCHS:
    print("Training already complete, skipping to validation.")
    final_model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
else:
    if latest_ckpt:
        print(f"Loading checkpoint: {latest_ckpt}")
        net = cp_models.CellposeModel(gpu=True, pretrained_model=latest_ckpt).net
    else:
        net = cp_models.CellposeModel(gpu=True, model_type=args.base_model).net

    epoch = completed_epochs
    last_ckpt_path = latest_ckpt

    while epoch < TOTAL_EPOCHS:
        chunk = min(CHUNK_EPOCHS, TOTAL_EPOCHS - epoch)
        target_epoch = epoch + chunk
        ckpt_name = f"{MODEL_NAME}_ep{target_epoch:03d}"
        print(f"\nTraining epochs {epoch + 1}-{target_epoch} / {TOTAL_EPOCHS}  (checkpoint: {ckpt_name})")

        chunk_lr = get_lr(epoch)
        print(f"  LR for this chunk: {chunk_lr:.2e}")
        last_ckpt_path, train_losses, *_ = cp_train.train_seg(
            net,
            train_data=train_images,
            train_labels=train_masks,
            channel_axis=0,
            save_path=MODEL_SAVE_DIR,
            n_epochs=chunk,
            learning_rate=chunk_lr,
            weight_decay=args.weight_decay,
            batch_size=8,
            model_name=ckpt_name,
        )
        last_ckpt_path = str(last_ckpt_path)
        avg_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")

        epoch = target_epoch
        with open(state_file, "w") as f:
            json.dump({"completed_epochs": epoch, "latest_checkpoint": str(last_ckpt_path)}, f)
        print(f"  Checkpoint saved at epoch {epoch}: {last_ckpt_path}  (avg loss: {avg_loss:.4f})", flush=True)

        if _stop_after_chunk:
            print(f"Stopping at epoch {epoch} (SIGUSR1). Requeue will resume.", flush=True)
            break

        if epoch < TOTAL_EPOCHS:
            net = cp_models.CellposeModel(gpu=True, pretrained_model=last_ckpt_path).net

    # Copy final checkpoint to canonical name for infer.py to find
    final_model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    if last_ckpt_path and os.path.abspath(last_ckpt_path) != os.path.abspath(final_model_path):
        shutil.copy(last_ckpt_path, final_model_path)
    print(f"\nTraining complete. Final model: {final_model_path}")

# ── Validation on held-out FOVs (036-040) ───────────────────────────────────
if args.all_fovs:
    print("\n--all-fovs set: skipping held-out validation. Done.")
    import sys as _sys
    _sys.exit(0)

print("\nValidating on held-out FOVs (036-040)...")
finetuned_model = cp_models.CellposeModel(gpu=True, pretrained_model=final_model_path)

ari_scores = {}
val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]

for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    fov_x = meta.loc[fov_name, "fov_x"]
    fov_y = meta.loc[fov_name, "fov_y"]
    fov_spots = spots_train[spots_train["fov"] == fov_name]
    density_channels = [compute_spot_density(fov_spots, sigma=s) for s in SPOT_SIGMAS]
    if args.zstats:
        zf = compute_zstack_features(dapi, polyt)
        img_channels = [zf["polyt_max"], zf["dapi_max"],
                        zf["polyt_mean"], zf["dapi_mean"],
                        zf["polyt_std"],  zf["dapi_std"],
                        *density_channels]
    else:
        dapi_max  = np.max(dapi,  axis=0)
        polyt_max = np.max(polyt, axis=0)
        img_channels = [polyt_max, dapi_max, *density_channels]

    pred_masks, _, _ = finetuned_model.eval(
        np.stack(img_channels, axis=0),
        diameter=0, cellprob_threshold=-1.0, channel_axis=0,
    )
    gt_mask = boundaries_to_mask(cells, fov_name, fov_x, fov_y)

    fov_spots = spots_train[spots_train["fov"] == fov_name].copy()
    # Corrected MERFISH coordinate convention via pre-computed columns:
    #   image_row = 2048 - (global_x - fov_x) / pixel_size  (x-axis flipped)
    #   image_col = (global_y - fov_y) / pixel_size
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    pred_ids = pred_masks[rows, cols]
    gt_ids   = gt_mask[rows, cols]

    ari = adjusted_rand_score(gt_ids, pred_ids)
    ari_scores[fov_name] = ari
    print(f"  {fov_name}: ARI = {ari:.4f}  ({pred_masks.max()} cells)")

mean_ari = float(np.mean(list(ari_scores.values()))) if ari_scores else 0.0
print(f"\nMean validation ARI  : {mean_ari:.4f}")
print(f"Baseline (pretrained): 0.632")
print(f"Improvement          : {mean_ari - 0.632:+.4f}")
