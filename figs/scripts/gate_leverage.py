"""Visualize the 4.9× leverage of the background gate vs the multi-class classifier.

Structural argument: with prior p_bg = 0.83, improving the binary gate by 10 points
of accuracy moves the metric on ~83% of all spots; improving the classifier by 10
points only moves the metric on the 17% foreground subset. Slope ratio = 0.83/0.17.

We plot expected accuracy on test as we sweep one knob while holding the other
fixed; the slopes recover the 4.9× ratio.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

P_BG = 0.83  # prior probability of background on test
P_FG = 1 - P_BG

def accuracy(p_gate, p_cls):
    """Expected spot accuracy given gate accuracy p_gate and classifier accuracy p_cls.
    Simplified: gate correctly classifies foreground/background w.p. p_gate; given
    correctly gated as foreground, classifier picks the right label w.p. p_cls."""
    return p_gate * (P_BG + P_FG * p_cls)

GATE_FIXED = 0.92
CLS_FIXED = 0.80

x_gate = np.linspace(0.70, 1.0, 200)
y_vs_gate = accuracy(x_gate, CLS_FIXED)

x_cls = np.linspace(0.40, 1.0, 200)
y_vs_cls = accuracy(GATE_FIXED, x_cls)

# slopes at the operating point
slope_gate = (accuracy(0.92 + 0.01, CLS_FIXED) - accuracy(0.92, CLS_FIXED)) / 0.01
slope_cls = (accuracy(GATE_FIXED, 0.80 + 0.01) - accuracy(GATE_FIXED, 0.80)) / 0.01
ratio = slope_gate / slope_cls

fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.plot(x_gate, y_vs_gate, color="#A03E3E", lw=2.2,
        label=f"Sweep gate accuracy  (classifier fixed at {CLS_FIXED:.2f})")
ax.plot(x_cls, y_vs_cls, color="#2C5F8D", lw=2.2,
        label=f"Sweep classifier accuracy  (gate fixed at {GATE_FIXED:.2f})")

# lifts at the operating point (used in the title)
lift_gate = accuracy(GATE_FIXED + 0.10, CLS_FIXED) - accuracy(GATE_FIXED, CLS_FIXED)
lift_cls = accuracy(GATE_FIXED, CLS_FIXED + 0.10) - accuracy(GATE_FIXED, CLS_FIXED)

# slope annotations directly on each curve
ax.annotate(f"slope = {lift_gate*10:.2f}\n(+{lift_gate:.3f} per 10pt)",
            xy=(0.83, accuracy(0.83, CLS_FIXED)),
            xytext=(-100, 0), textcoords="offset points",
            color="#A03E3E", fontsize=9, fontweight="bold",
            ha="left", va="center")

ax.annotate(f"slope = {lift_cls*10:.2f}\n(+{lift_cls:.3f} per 10pt)",
            xy=(0.55, accuracy(GATE_FIXED, 0.55)),
            xytext=(0, -28), textcoords="offset points",
            color="#2C5F8D", fontsize=9, fontweight="bold",
            ha="center", va="top")

ax.set_xlabel("Accuracy of one knob (other knob held fixed)")
ax.set_ylabel("Expected per-spot accuracy on test")
ax.set_title(
    f"Gate has ~{lift_gate/lift_cls:.0f}× the leverage of the classifier:\n"
    f"a 10-point gate gain is worth +{lift_gate:.3f}; a 10-point classifier gain is +{lift_cls:.3f}.",
    fontsize=10
)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.grid(ls="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0.40, 1.02)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "gate_leverage.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")
print(f"slope ratio = {ratio:.3f}  (analytic = {P_BG/P_FG:.3f})")
