"""Fine-tune StarDist 2D_versatile_fluo on MERFISH training FOVs.

Input: DAPI max-projection (single channel) — StarDist is designed for
nuclear/fluorescence segmentation and 2D_versatile_fluo expects single-channel.
We include polyT max as a second option via --channel argument.

StarDist checkpoints automatically to basedir/name/weights_best.h5 and
weights_last.h5, so preemption recovery checks for those files.
"""
from __future__ import annotations

import argparse
import os
import signal

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask, compute_spot_density

parser = argparse.ArgumentParser()
parser.add_argument("--channel", default="dapi",
                    choices=["dapi", "polyt", "both"],
                    help="Input channel(s): dapi=DAPI max, polyt=polyT max, "
                         "both=concatenate as 2-ch (requires new head)")
parser.add_argument("--exp-name", default="stardist",
                    help="Experiment name; model saved to models/<exp-name>/")
parser.add_argument("--epochs", type=int, default=200,
                    help="Total training epochs (default 200)")
parser.add_argument("--steps-per-epoch", type=int, default=100)
parser.add_argument("--resume", action="store_true",
                    help="If weights_best.h5 exists at models/<exp>/, resume training "
                         "from those weights for `epochs` more epochs instead of skipping "
                         "training and going straight to validation.")
parser.add_argument("--all-fovs", action="store_true",
                    help="Train on all 40 FOVs (001-040). Skip the held-out val loading and "
                         "the final validation loop. Use when val ARI has become unreliable and "
                         "we want maximum training signal for the final Kaggle submission.")
args = parser.parse_args()

# StarDist / TF imports after argparse so --help is fast
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from stardist.models import StarDist2D          # noqa: E402
from stardist import fill_label_holes           # noqa: E402
from csbdeep.utils import normalize             # noqa: E402

EXP_NAME   = args.exp_name
DATA_ROOT  = "/scratch/cg4652/competition"
BASEDIR    = "models"
MODEL_DIR  = os.path.join(BASEDIR, EXP_NAME)
TOTAL_EPOCHS = args.epochs

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Signal handler ────────────────────────────────────────────────────────────
_interrupted = False

def _sigusr1_handler(signum, frame):
    global _interrupted
    _interrupted = True
    print("SIGUSR1: will stop after current epoch. StarDist saves best/last weights.", flush=True)

signal.signal(signal.SIGUSR1, _sigusr1_handler)

# ── Check for existing trained model ─────────────────────────────────────────
weights_best = os.path.join(MODEL_DIR, "weights_best.h5")
weights_last = os.path.join(MODEL_DIR, "weights_last.h5")
already_trained = os.path.exists(weights_best)
if already_trained:
    print(f"Found trained model weights at {weights_best} — loading for val only")
else:
    print(f"No trained model found at {MODEL_DIR}, training from pretrained 2D_versatile_fluo")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading metadata and ground truth...")
meta   = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells  = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

def build_input(dapi: np.ndarray, polyt: np.ndarray) -> np.ndarray:
    """Build (H, W) or (H, W, C) image according to --channel flag."""
    dapi_max  = np.max(dapi,  axis=0).astype(np.float32)
    polyt_max = np.max(polyt, axis=0).astype(np.float32)
    if args.channel == "dapi":
        return dapi_max
    elif args.channel == "polyt":
        return polyt_max
    else:  # both — (H, W, 2)
        return np.stack([dapi_max, polyt_max], axis=-1)

train_images, train_masks = [], []
train_range = (1, 41) if args.all_fovs else (1, 36)
train_fovs = [f"FOV_{i:03d}" for i in range(*train_range)]

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
        img = build_input(dapi, polyt)
        # StarDist expects per-image normalization
        img = normalize(img, 1, 99.8, axis=(0, 1))
        m   = fill_label_holes(m)
        train_images.append(img)
        train_masks.append(m)
        print(f"  Loaded {fov_name}: {m.max()} cells")
    except Exception as exc:
        print(f"  Skipping {fov_name}: {exc}")

print(f"\nTraining set: {len(train_images)} FOVs")

# ── Load held-out val FOVs (036-040) early for validation_data + threshold opt ──
val_images, val_masks = [], []
if not args.all_fovs:
    print("Loading held-out val FOVs (036-040)...")
    val_fovs_list = [f"FOV_{i:03d}" for i in range(36, 41)]
    for fov_name in val_fovs_list:
        fov_dir = f"{DATA_ROOT}/train/{fov_name}"
        if not os.path.exists(fov_dir):
            continue
        dapi, polyt = load_fov_images(fov_dir)
        fov_x = meta.loc[fov_name, "fov_x"]
        fov_y = meta.loc[fov_name, "fov_y"]
        img = build_input(dapi, polyt)
        img = normalize(img, 1, 99.8, axis=(0, 1))
        m = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
        m = fill_label_holes(m)
        val_images.append(img)
        val_masks.append(m)
    print(f"Val set: {len(val_images)} FOVs")
else:
    print("--all-fovs: using 001-040 all for training, no held-out val")
    # StarDist requires validation_data — reuse last training FOV as a placeholder
    val_images = [train_images[-1]]
    val_masks  = [train_masks[-1]]

# ── Build / load model ────────────────────────────────────────────────────────
skip_training = already_trained and not args.resume
if already_trained:
    model = StarDist2D(None, name=EXP_NAME, basedir=BASEDIR)
    if args.resume:
        print(f"Resuming training for {TOTAL_EPOCHS} more epochs from {weights_best}")
else:
    # Load pretrained config and weights
    pretrained = StarDist2D.from_pretrained("2D_versatile_fluo")
    conf = pretrained.config
    conf.train_tensorboard = False  # TensorBoard not installed in this env

    if args.channel == "both":
        # 2-channel input: rebuild config with n_channel_in=2
        from stardist.models import Config2D
        conf = Config2D(
            n_rays=conf.n_rays,
            grid=conf.grid,
            n_channel_in=2,
            use_gpu=True,
            train_tensorboard=False,
        )
        model = StarDist2D(conf, name=EXP_NAME, basedir=BASEDIR)
        print("Using 2-channel input — cannot transfer pretrained weights (channel mismatch)")
    else:
        model = StarDist2D(conf, name=EXP_NAME, basedir=BASEDIR)
        # Transfer pretrained weights
        model.keras_model.set_weights(pretrained.keras_model.get_weights())
        print("Transferred 2D_versatile_fluo pretrained weights")

    del pretrained  # free memory

# ── Train ──────────────────────────────────────────────────────────────────────
if not skip_training:
    print(f"\nTraining StarDist for {TOTAL_EPOCHS} epochs, "
          f"{args.steps_per_epoch} steps/epoch  (channel={args.channel})")
    model.train(
        train_images,
        train_masks,
        validation_data=(val_images, val_masks),
        augmenter=None,  # StarDist has built-in augmentation
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=args.steps_per_epoch,
        workers=4,
    )
    if not args.all_fovs:
        model.optimize_thresholds(val_images, val_masks)
    print(f"\nTraining complete. Weights: {weights_best}")

# ── Validation on held-out FOVs 036-040 ──────────────────────────────────────
if args.all_fovs:
    print("\n--all-fovs set: skipping held-out validation. Done.")
    import sys as _sys
    _sys.exit(0)

print("\nValidating on held-out FOVs (036-040)...")

ari_scores = {}
val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]

for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        print(f"  Missing: {fov_name}")
        continue

    dapi, polyt = load_fov_images(fov_dir)
    fov_x  = meta.loc[fov_name, "fov_x"]
    fov_y  = meta.loc[fov_name, "fov_y"]
    img    = build_input(dapi, polyt)
    img    = normalize(img, 1, 99.8, axis=(0, 1))

    # StarDist returns (labels, details) — labels is integer mask
    labels, _ = model.predict_instances(img)
    pred_masks = labels.astype(np.int32)

    gt_mask   = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
    fov_spots = spots_train[spots_train["fov"] == fov_name].copy()
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
