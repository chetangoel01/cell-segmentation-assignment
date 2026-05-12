"""Mini-grid of failed approaches: ΔKaggle vs the relevant submitted reference.

Numbers from report Sections 3.6 and 4.8. Each bar is the *change* from a
specific baseline, so the comparison reference is annotated.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# (label, delta, reference)
phase1 = [
    ("4-way TTA",              -0.0166, "vs StarDist 28ep"),
    ("StarDist 180 ep",        -0.0206, "vs StarDist 28ep"),
    ("NN-radius dilation",     -0.2600, "vs StarDist 28ep"),
]
phase2 = [
    ("Autoresearch RF500",     -0.0465, "vs P (nuclei_cosine+kNN)"),
    ("cpsam-zeroshot + RF500", -0.1968, "vs P (nuclei_cosine+kNN)"),
    ("3-way w/ cpsam ep20",    -0.0367, "vs PQR-V7"),
    ("HGB at cluster lvl",     -0.30,    "vs kNN local val (proxy)"),
]

fig, (a1, a2) = plt.subplots(
    1, 2, figsize=(11, 3.6),
    gridspec_kw={"width_ratios": [len(phase1), len(phase2)]}
)

for ax, items, color, title in [
    (a1, phase1, "#2C5F8D", "Phase 1: Segmentation"),
    (a2, phase2, "#A03E3E", "Phase 2: Classification"),
]:
    labels = [x[0] for x in items]
    deltas = np.array([x[1] for x in items])
    refs = [x[2] for x in items]
    y = np.arange(len(items))

    ax.barh(y, deltas, color=color, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Kaggle ARI", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    for i, (d, r) in enumerate(zip(deltas, refs)):
        # delta value: outside the bar tip, to the left
        ax.annotate(f"{d:+.3f}", xy=(d, i), xytext=(-6, 0),
                    textcoords="offset points",
                    ha="right", va="center",
                    fontsize=8.5, color="black", fontweight="bold")
        # reference: at the zero side of the bar, sitting on white background
        ax.annotate(r, xy=(0, i), xytext=(6, 0),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=7, color="#555", style="italic")

a1.set_xlim(-0.34, 0.02)
a2.set_xlim(-0.36, 0.05)

fig.suptitle(
    "Approaches that regressed on Kaggle. Reference for each Δ noted in italics.",
    fontsize=9.5, y=1.02
)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "failed_approaches.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")
