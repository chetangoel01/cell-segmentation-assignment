"""Fine-tune StarDist3D on MERFISH DAPI z-stacks.

Input:  (Z=5, Y=2048, X=2048) DAPI volume per FOV (no max-projection).
Target: (Z=5, Y=2048, X=2048) int mask where each cell has one stable ID
        across all z-planes it occupies (see src/stardist3d.py).

Anisotropy: z-spacing = 1.5 µm vs xy pixel = 0.109 µm → ratio ~14:1, which is
extreme but StarDist3D handles it by distorting ray directions in the ray-bank.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.io import load_fov_images
from src.stardist3d import boundaries_to_mask_3d, collapse_3d_labels_to_2d

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="stardist3d")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--steps-per-epoch", type=int, default=100)
parser.add_argument("--n-rays", type=int, default=96,
                    help="Number of rays for 3D polyhedron. Default 96 (StarDist paper).")
parser.add_argument("--patch-xy", type=int, default=256,
                    help="xy patch size for training (kept small for memory); "
                         "z always spans all 5 planes.")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--all-fovs", action="store_true")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# TF imports after argparse so --help stays fast
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from stardist import Rays_GoldenSpiral, fill_label_holes  # noqa: E402
from stardist.models import Config3D, StarDist3D          # noqa: E402
from csbdeep.utils import normalize                       # noqa: E402

EXP_NAME     = args.exp_name
DATA_ROOT    = os.environ.get("MERFISH_DATA_ROOT", "/scratch/cg4652/competition")
BASEDIR      = "models"
MODEL_DIR    = os.path.join(BASEDIR, EXP_NAME)
TOTAL_EPOCHS = args.epochs

# Voxel size in µm. Passed to StarDist3D as anisotropy so the ray bank bends
# in z to match the thin-slab geometry rather than fighting it.
ANISOTROPY = (1.5, 0.109, 0.109)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Signal handler ────────────────────────────────────────────────────────────
_interrupted = False

def _sigusr1(*_):
    global _interrupted
    _interrupted = True
    print("SIGUSR1: will stop after current epoch.", flush=True)

signal.signal(signal.SIGUSR1, _sigusr1)

# ── Resume or fresh ───────────────────────────────────────────────────────────
weights_best = os.path.join(MODEL_DIR, "weights_best.h5")
already_trained = os.path.exists(weights_best)
if already_trained and not args.resume:
    print(f"Found {weights_best} — skipping training, going straight to val.")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading metadata and ground truth...")
meta        = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells_df    = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv",
                          index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

def load_fov_3d(fov_name: str, data_subdir: str = "train") -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (image: (Z, Y, X) normalized float32, mask_3d: (Z, Y, X) int32 or None).

    mask_3d is None if fov is in the test set (no GT).
    """
    fov_dir = f"{DATA_ROOT}/{data_subdir}/{fov_name}"
    dapi, _polyt = load_fov_images(fov_dir)
    # DAPI already (5, H, W) float from load_fov_images
    img = dapi.astype(np.float32)
    # Per-volume normalization (axes 0,1,2) so intensity is comparable across z.
    img = normalize(img, 1, 99.8, axis=(0, 1, 2))

    mask_3d = None
    if data_subdir == "train":
        fov_x = meta.loc[fov_name, "fov_x"]
        fov_y = meta.loc[fov_name, "fov_y"]
        mask_3d = boundaries_to_mask_3d(cells_df, fov_x, fov_y)
        # Fill any interior holes on a per-plane basis (StarDist expects solid labels)
        for z in range(mask_3d.shape[0]):
            mask_3d[z] = fill_label_holes(mask_3d[z])
    return img, mask_3d


train_range = (1, 41) if args.all_fovs else (1, 36)
train_fovs = [f"FOV_{i:03d}" for i in range(*train_range)]

train_images, train_masks = [], []
for fov_name in train_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        continue
    try:
        img, m3d = load_fov_3d(fov_name)
        if m3d is None or m3d.max() == 0:
            print(f"  No cells in {fov_name} — skipping")
            continue
        train_images.append(img)
        train_masks.append(m3d)
        print(f"  Loaded {fov_name}: {int(m3d.max())} cells, vol shape={img.shape}")
    except Exception as exc:
        print(f"  Skipping {fov_name}: {exc}")

print(f"\nTraining set: {len(train_images)} FOVs")
if len(train_images) == 0:
    sys.exit("No training data loaded — aborting.")

val_images, val_masks = [], []
if not args.all_fovs:
    print("Loading held-out val FOVs (036-040)...")
    for fov_name in [f"FOV_{i:03d}" for i in range(36, 41)]:
        fov_dir = f"{DATA_ROOT}/train/{fov_name}"
        if not os.path.exists(fov_dir):
            continue
        img, m3d = load_fov_3d(fov_name)
        val_images.append(img)
        val_masks.append(m3d)
    print(f"Val set: {len(val_images)} FOVs")
else:
    # StarDist requires validation_data — reuse last train FOV as placeholder
    val_images = [train_images[-1]]
    val_masks  = [train_masks[-1]]

# ── Config & model ────────────────────────────────────────────────────────────
if already_trained:
    model = StarDist3D(None, name=EXP_NAME, basedir=BASEDIR)
    skip_training = not args.resume
else:
    skip_training = False
    rays = Rays_GoldenSpiral(args.n_rays, anisotropy=ANISOTROPY)
    # unet_pool=(1,2,2): don't pool z — with 1.5 µm slices, pooling would
    # collapse a whole nucleus thickness per step. Side benefit: removes the
    # z % 4 == 0 constraint on train_patch_size so z=5 is allowed.
    conf = Config3D(
        rays              = rays,
        grid              = (1, 2, 2),            # output stride: keep z full resolution
        unet_pool         = (1, 2, 2),            # encoder pool: don't pool z at all
        unet_n_depth      = 2,
        anisotropy        = ANISOTROPY,
        use_gpu           = False,                # disables gputools-based preproc
                                                  # (TF still uses GPU for training);
                                                  # gputools isn't in our Modal image.
        n_channel_in      = 1,
        train_patch_size  = (5, args.patch_xy, args.patch_xy),  # full-z × xy crop
        train_batch_size  = args.batch_size,
        train_tensorboard = False,
    )
    model = StarDist3D(conf, name=EXP_NAME, basedir=BASEDIR)

# ── Train ─────────────────────────────────────────────────────────────────────
if not skip_training:
    print(f"\nTraining StarDist3D  epochs={TOTAL_EPOCHS}  "
          f"steps/epoch={args.steps_per_epoch}  patch=(5,{args.patch_xy},{args.patch_xy})  "
          f"n_rays={args.n_rays}  anisotropy={ANISOTROPY}")
    model.train(
        train_images,
        train_masks,
        validation_data=(val_images, val_masks),
        augmenter=None,             # StarDist's built-in
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=args.steps_per_epoch,
        workers=2,
    )
    if not args.all_fovs:
        model.optimize_thresholds(val_images, val_masks)
    print(f"\nTraining complete. Best weights: {weights_best}")

# ── Validation (held-out 036–040) ────────────────────────────────────────────
if args.all_fovs:
    print("\n--all-fovs set: skipping held-out validation.")
    sys.exit(0)

print("\nValidating on FOVs 036–040...")
ari_scores = {}
for fov_name in [f"FOV_{i:03d}" for i in range(36, 41)]:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        continue
    img, gt_3d = load_fov_3d(fov_name)
    labels_3d, _ = model.predict_instances(img)
    pred_2d = collapse_3d_labels_to_2d(labels_3d)
    gt_2d   = collapse_3d_labels_to_2d(gt_3d)

    fov_spots = spots_train[spots_train["fov"] == fov_name]
    rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
    cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
    ari = adjusted_rand_score(gt_2d[rows, cols], pred_2d[rows, cols])
    ari_scores[fov_name] = ari
    print(f"  {fov_name}: ARI={ari:.4f}  ({pred_2d.max()} cells)")

mean_ari = float(np.mean(list(ari_scores.values()))) if ari_scores else 0.0
print(f"\nMean val ARI (StarDist3D): {mean_ari:.4f}")
print(f"Best 2D StarDist val ARI:   ~0.83  (Kaggle 0.7627)")
