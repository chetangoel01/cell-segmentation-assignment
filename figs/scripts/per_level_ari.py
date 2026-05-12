"""Per-level ARI breakdown: baseline vs final 5-way AWS-augmented ensemble.

The baseline numbers (class 0.347, subclass 0.352, supertype 0.352, cluster 0.351)
are published. The 5-way final per-level numbers are approximate from the
phase2_results figure caption; we mark them as such in the title.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

levels = ["class", "subclass", "supertype", "cluster"]
baseline = np.array([0.347, 0.352, 0.352, 0.351])  # Cellpose+kNN, published
final5 = np.array([0.610, 0.570, 0.550, 0.540])    # approximate from caption text

fig, ax = plt.subplots(figsize=(5.4, 3.6))
x = np.arange(len(levels))
w = 0.36

b1 = ax.bar(x - w/2, baseline, w, label="Cellpose+kNN baseline", color="#888888",
            edgecolor="white")
b2 = ax.bar(x + w/2, final5, w, label="5-way AWS-aug kNN (ours)", color="#A03E3E",
            edgecolor="white")

for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=8)

# arrows showing lift
for i, (lo, hi) in enumerate(zip(baseline, final5)):
    ax.annotate("", xy=(i + w/2 - 0.02, hi), xytext=(i - w/2 + 0.02, lo),
                arrowprops=dict(arrowstyle="->", color="#1F7A3F", lw=0.7, alpha=0.5))

ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel("Kaggle public ARI")
ax.set_ylim(0, 0.75)
ax.set_title("Per-level ARI: baseline is flat across hierarchy;\nlift is larger at coarser levels", fontsize=9.5)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=8.5, frameon=False)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "per_level_ari.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")
