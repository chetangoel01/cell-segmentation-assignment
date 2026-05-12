"""Kaggle submission timeline for Phase 1 (segmentation) and Phase 2 (classification).

X = submission index within each phase (we don't have exact submission timestamps
for every entry; ordering preserves the chronology recorded in the report and
SUBMISSIONS log). Y = public Kaggle ARI. Running-best line overlaid.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# (label, kaggle_ari)  --  order is chronological within phase
phase1 = [
    ("Cellpose pretrained\n(baseline)", 0.6320),
    ("Cellpose cyto2 (3ch)", 0.7464),
    ("StarDist v2 (~180 ep)", 0.7421),
    ("StarDist + TTA", 0.7461),
    ("Cellpose nuclei_cosine", 0.7588),
    ("StarDist 28 ep (best)", 0.7627),
]

phase2 = [
    ("Cellpose+kNN\n(baseline)", 0.3510),
    ("cpsam-zeroshot + RF500", 0.3378),
    ("M: logreg-log1p", 0.4859),
    ("RF500-log1p-mf01", 0.4881),
    ("3-way (V7+SD+cpsam)", 0.4979),
    ("Q: cpsam + kNN", 0.5031),
    ("P: nuclei_cosine + kNN", 0.5346),
    ("PQM ensemble", 0.5375),
    ("PQMRF-V6", 0.5408),
    ("PQR-V7", 0.5419),
    ("V7 + PQM + cpsam_floor", 0.5421),
    ("AWS-aug kNN single", 0.5593),
    ("5-way AWS-aug (final)", 0.5675),
]

PHASE1_COLOR = "#2C5F8D"
PHASE2_COLOR = "#A03E3E"
BEST_COLOR = "#1F7A3F"
BASELINE_GREY = "#888888"

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [len(phase1), len(phase2)]}
)

for ax, data, title, color, baseline_label in [
    (ax1, phase1, "Phase 1: Segmentation", PHASE1_COLOR, "Cellpose pretrained"),
    (ax2, phase2, "Phase 2: Classification", PHASE2_COLOR, "Cellpose+kNN"),
]:
    x = np.arange(len(data))
    y = np.array([s for _, s in data])
    labels = [lab for lab, _ in data]

    # running best
    best = np.maximum.accumulate(y)

    ax.plot(x, best, "-", color=BEST_COLOR, lw=1.4, alpha=0.55, zorder=1, label="running best")
    ax.scatter(x, y, s=42, color=color, edgecolor="white", lw=0.8, zorder=3)
    # baseline marker
    ax.axhline(y[0], ls=":", color=BASELINE_GREY, lw=0.9, zorder=0)
    ax.scatter([0], [y[0]], s=80, color=BASELINE_GREY, edgecolor="white", lw=1.0, zorder=2, marker="s")

    # annotate best
    best_idx = int(np.argmax(y))
    ax.annotate(
        f"{y[best_idx]:.4f}",
        xy=(best_idx, y[best_idx]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        color=BEST_COLOR,
        fontweight="bold",
    )
    ax.annotate(
        f"baseline\n{y[0]:.3f}",
        xy=(0, y[0]),
        xytext=(8, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        color=BASELINE_GREY,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Kaggle public ARI", fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

ax1.set_ylim(0.60, 0.78)
ax2.set_ylim(0.30, 0.60)

fig.suptitle(
    "Submission progression: each dot is a Kaggle submission, in chronological order. "
    "Running-best line in green.",
    fontsize=9,
    y=1.02,
)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "training_history.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")
