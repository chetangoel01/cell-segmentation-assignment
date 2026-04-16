"""Fine-tune Cellpose on training FOVs with checkpoint save/resume for HPC preemption."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cellpose import models as cp_models, train as cp_train

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="cyto2",
                    choices=["cyto2", "cyto3", "nuclei"],
                    help="Cellpose pretrained model to fine-tune from")
parser.add_argument("--exp-name", default=None,
                    help="Experiment name (defaults to base-model name)")
args = parser.parse_args()

EXP_NAME = args.exp_name or args.base_model

DATA_ROOT      = "/scratch/pl2820/competition"
PIXEL_SIZE     = 0.109
MODEL_SAVE_DIR = f"models/{EXP_NAME}"
MODEL_NAME     = f"cellpose_{EXP_NAME}"
TOTAL_EPOCHS   = 100
CHUNK_EPOCHS   = 20   # save a checkpoint every N epochs

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
else:
    print("No checkpoint found, starting fresh")

print("Loading metadata and ground truth...")
meta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(
    f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0
)

# Build training set (FOVs 001-035)
train_images, train_masks = [], []
train_fovs = [f"FOV_{i:03d}" for i in range(1, 36)]

for fov_name in train_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue
    try:
        dapi, polyt = load_fov_images(fov_dir)
        fov_x = meta.loc[fov_name, "fov_x"]
        fov_y = meta.loc[fov_name, "fov_y"]
        m = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
        if m.max() == 0:
            print(f"  No cells in mask: {fov_name}")
            continue
        train_images.append(np.stack([polyt[2], dapi[2]], axis=0))
        train_masks.append(m)
        print(f"  Loaded {fov_name}: {m.max()} cells")
    except Exception as exc:
        print(f"  Skipping {fov_name}: {exc}")

print(f"\nTraining set: {len(train_images)} FOVs")

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

        last_ckpt_path = cp_train.train_seg(
            net,
            train_data=train_images,
            train_labels=train_masks,
            save_path=MODEL_SAVE_DIR,
            n_epochs=chunk,
            learning_rate=0.005,
            weight_decay=1e-5,
            batch_size=8,
            model_name=ckpt_name,
        )

        epoch = target_epoch
        with open(state_file, "w") as f:
            json.dump({"completed_epochs": epoch, "latest_checkpoint": str(last_ckpt_path)}, f)
        print(f"  Checkpoint saved at epoch {epoch}: {last_ckpt_path}")

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
print("\nValidating on held-out FOVs (036-040)...")
finetuned_model = cp_models.CellposeModel(gpu=True, pretrained_model=final_model_path)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

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

    pred_masks, _, _ = finetuned_model.eval(np.stack([polyt[2], dapi[2]], axis=0), diameter=30)
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
