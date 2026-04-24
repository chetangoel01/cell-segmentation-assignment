"""Train vanilla U-Net for cell instance segmentation via semantic+watershed.

Same data layout and 3-channel input as the Cellpose pipeline:
    [polyT_max, DAPI_max, spot_density]  (max-projected over 5 z-planes)

Targets are 3-class: 0=background, 1=cell interior (eroded), 2=cell boundary.
At inference, watershed turns the semantic prediction into integer instance masks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, Dataset

from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask, compute_spot_density
from src.unet import (UNet, dice_loss, make_semantic_target,
                      normalize_image, predict_to_instances)


# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="unet")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--crop-size", type=int, default=512)
parser.add_argument("--crops-per-fov", type=int, default=8,
                    help="Random crops sampled per FOV per epoch")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-min", type=float, default=1e-6)
parser.add_argument("--base-channels", type=int, default=32)
parser.add_argument("--erosion-iters", type=int, default=2,
                    help="Pixels to erode each cell before defining interior class")
parser.add_argument("--ce-weight", type=float, default=0.5,
                    help="Weight on cross-entropy in CE+Dice loss (Dice gets 1-this)")
args = parser.parse_args()

DATA_ROOT = "/scratch/cg4652/competition"
EXP_NAME = args.exp_name
MODEL_DIR = f"models/{EXP_NAME}"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Class weights for cross-entropy: boundary is rare, upweight it
CLASS_WEIGHTS = torch.tensor([0.5, 1.0, 3.0], device=device)


# ── Signal handler ──────────────────────────────────────────────────────────
_stop = False
def _sigusr1(*_):
    global _stop
    print("SIGUSR1: will save and stop after this epoch", flush=True)
    _stop = True
signal.signal(signal.SIGUSR1, _sigusr1)


# ── Data loading ────────────────────────────────────────────────────────────
print("Loading metadata and ground truth...")
meta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")


def load_fov(fov_name: str, use_union_z: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image (3, 2048, 2048) float32 normalized, mask (2048, 2048) int32)."""
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    dapi, polyt = load_fov_images(fov_dir)
    fov_x = meta.loc[fov_name, "fov_x"]
    fov_y = meta.loc[fov_name, "fov_y"]
    fov_spots = spots_train[spots_train["fov"] == fov_name]
    density = compute_spot_density(fov_spots, sigma=8.0)
    img = np.stack([
        np.max(polyt, axis=0).astype(np.float32),
        np.max(dapi, axis=0).astype(np.float32),
        density.astype(np.float32),
    ], axis=0)
    img = normalize_image(img)
    mask = boundaries_to_mask(cells, fov_name, fov_x, fov_y, use_all_z=use_union_z)
    return img, mask


print("Loading training FOVs (001-035)...")
train_fovs = [f"FOV_{i:03d}" for i in range(1, 36)]
train_data = []
for fn in train_fovs:
    img, mask = load_fov(fn)
    if mask.max() == 0:
        print(f"  Skipping {fn}: no cells")
        continue
    target = make_semantic_target(mask, erosion_iters=args.erosion_iters)
    train_data.append((img, target, mask))
    print(f"  {fn}: {mask.max()} cells")

print(f"Train FOVs: {len(train_data)}")

print("Loading val FOVs (036-040)...")
val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]
val_data = []
for fn in val_fovs:
    img, mask = load_fov(fn, use_union_z=False)  # eval against z=2 GT (matches infer)
    val_data.append((fn, img, mask))


# ── Dataset (random crops) ──────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, samples, crop_size: int, n_per_fov: int):
        self.samples = samples
        self.crop_size = crop_size
        self.n_per_fov = n_per_fov

    def __len__(self):
        return len(self.samples) * self.n_per_fov

    def __getitem__(self, idx):
        img, target, _mask = self.samples[idx % len(self.samples)]
        H, W = target.shape
        c = self.crop_size
        # Bias crops toward regions with cells (≥50% chance)
        if np.random.rand() < 0.7 and target.max() > 0:
            ys, xs = np.where(target > 0)
            i = np.random.randint(len(ys))
            cy, cx = ys[i], xs[i]
            r0 = np.clip(cy - c // 2, 0, H - c)
            c0 = np.clip(cx - c // 2, 0, W - c)
        else:
            r0 = np.random.randint(0, H - c + 1)
            c0 = np.random.randint(0, W - c + 1)
        img_c = img[:, r0:r0 + c, c0:c0 + c].copy()
        tgt_c = target[r0:r0 + c, c0:c0 + c].copy()
        # Flip/rotate aug
        k = np.random.randint(4)
        img_c = np.rot90(img_c, k=k, axes=(1, 2)).copy()
        tgt_c = np.rot90(tgt_c, k=k).copy()
        if np.random.rand() < 0.5:
            img_c = np.flip(img_c, axis=2).copy()
            tgt_c = np.flip(tgt_c, axis=1).copy()
        return torch.from_numpy(img_c), torch.from_numpy(tgt_c).long()


train_ds = CropDataset(train_data, args.crop_size, args.crops_per_fov)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)


# ── Model ───────────────────────────────────────────────────────────────────
state_file = os.path.join(MODEL_DIR, "train_state.json")
ckpt_path = os.path.join(MODEL_DIR, "unet_latest.pt")
best_path = os.path.join(MODEL_DIR, "unet_best.pt")

model = UNet(in_channels=3, n_classes=3, base=args.base_channels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

start_epoch = 0
best_ari = -1.0
if os.path.exists(state_file) and os.path.exists(ckpt_path):
    try:
        state = json.load(open(state_file))
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model"])
        optimizer.load_state_dict(sd["optim"])
        start_epoch = state.get("epoch", 0)
        best_ari = state.get("best_ari", -1.0)
        print(f"Resuming from epoch {start_epoch}, best_ari={best_ari:.4f}")
    except Exception as e:
        print(f"WARNING: failed to resume ({e}), starting fresh")
        start_epoch = 0
        best_ari = -1.0


def lr_at(epoch: int) -> float:
    progress = epoch / max(args.epochs, 1)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))


def evaluate() -> float:
    model.eval()
    aris = []
    with torch.no_grad():
        for fn, img, gt_mask in val_data:
            x = torch.from_numpy(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_mask = predict_to_instances(probs)
            fov_spots = spots_train[spots_train["fov"] == fn]
            rows = np.clip(fov_spots["image_row"].values.astype(int), 0, 2047)
            cols = np.clip(fov_spots["image_col"].values.astype(int), 0, 2047)
            ari = adjusted_rand_score(gt_mask[rows, cols], pred_mask[rows, cols])
            aris.append(ari)
            print(f"    {fn}: ARI={ari:.4f}  ({pred_mask.max()} cells)")
    return float(np.mean(aris))


# ── Training loop ───────────────────────────────────────────────────────────
print(f"\nTraining {start_epoch + 1}-{args.epochs}, batch={args.batch_size}, "
      f"crop={args.crop_size}, samples/epoch={len(train_ds)}")

for epoch in range(start_epoch, args.epochs):
    model.train()
    cur_lr = lr_at(epoch)
    for g in optimizer.param_groups:
        g["lr"] = cur_lr

    losses = []
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y, weight=CLASS_WEIGHTS)
        dl = dice_loss(logits, y)
        loss = args.ce_weight * ce + (1 - args.ce_weight) * dl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = float(np.mean(losses))
    print(f"\nEpoch {epoch + 1}/{args.epochs}  lr={cur_lr:.2e}  loss={avg_loss:.4f}", flush=True)

    # Eval every 5 epochs and at the end
    do_eval = (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs or _stop
    if do_eval:
        ari = evaluate()
        print(f"  Mean val ARI: {ari:.4f}  (best so far: {max(ari, best_ari):.4f})", flush=True)
        if ari > best_ari:
            best_ari = ari
            torch.save({"model": model.state_dict()}, best_path)
            print(f"  -> Saved new best to {best_path}")

    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, ckpt_path)
    json.dump({"epoch": epoch + 1, "best_ari": best_ari}, open(state_file, "w"))

    if _stop:
        print("Stopping after SIGUSR1.")
        break

print(f"\nDone. Best val ARI: {best_ari:.4f}")
print(f"Best checkpoint: {best_path}")
