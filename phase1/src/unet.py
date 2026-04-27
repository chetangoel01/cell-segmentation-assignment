"""Vanilla 4-down/4-up U-Net for semantic segmentation (bg/interior/boundary)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import watershed


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 3, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_channels, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bot = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bot(self.pool(d4))
        u4 = self.u4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out(u1)


def make_semantic_target(mask: np.ndarray, erosion_iters: int = 2) -> np.ndarray:
    """Convert integer instance mask to 3-class semantic target.

    Classes: 0 = background, 1 = cell interior (eroded), 2 = cell boundary (ring).
    The boundary class lets the network learn to separate touching cells, which
    a plain foreground/background mask can't express.
    """
    target = np.zeros(mask.shape, dtype=np.uint8)
    if mask.max() == 0:
        return target
    cells = mask > 0
    interior = np.zeros_like(cells)
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        cm = mask == cell_id
        em = ndi.binary_erosion(cm, iterations=erosion_iters)
        interior |= em
    boundary = cells & ~interior
    target[interior] = 1
    target[boundary] = 2
    return target


def normalize_image(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Per-channel percentile normalization to [0, 1] float32."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        ch = img[c].astype(np.float32)
        lo, hi = np.percentile(ch, [p_low, p_high])
        if hi > lo:
            out[c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[c] = 0
    return out


def predict_to_instances(probs: np.ndarray, fg_thresh: float = 0.5,
                          marker_thresh: float = 0.7) -> np.ndarray:
    """Convert (3, H, W) softmax probabilities to integer instance mask via watershed.

    interior probability gives the seeds; boundary probability is the elevation
    surface that watershed walks. Foreground = interior + boundary above threshold.
    """
    interior = probs[1]
    boundary = probs[2]
    foreground = (interior + boundary) > fg_thresh
    markers, _ = ndi.label(interior > marker_thresh)
    if markers.max() == 0:
        return np.zeros(probs.shape[1:], dtype=np.int32)
    labels = watershed(boundary, markers, mask=foreground)
    return labels.astype(np.int32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Multi-class Dice loss. logits: (B, C, H, W); target: (B, H, W) long."""
    n_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target.long(), num_classes=n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * target_1h).sum(dims)
    union = probs.sum(dims) + target_1h.sum(dims)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1 - dice.mean()
