"""experiment.py — the single editable file.

The autoresearch agent edits THIS file every iteration to try a new
hyperparameter or pipeline change. Everything is fair game:

  - Segmentation knobs (cellpose hyperparameters, channel choice, post-proc)
  - Classifier (algo, k, metric, normalization, training data)
  - Spatial features (neighbor cell counts, GCN, etc.)
  - Inference pipeline (mask post-processing, spot assignment rules)

The agent does NOT edit:
  - phase2/scripts/validate_local.py   (the metric)
  - phase2/autoresearch/run_experiment.py  (the harness)
  - phase2/data/                        (the data)

Run via:
  .venv/bin/python phase2/autoresearch/run_experiment.py [--full-val]
"""
from __future__ import annotations

# =====================================================================
# Segmentation
# =====================================================================
SEG_CHECKPOINT = "phase2/external_models/cellpose_nuclei_cosine_ep125"
INCLUDE_SPOT_DENSITY = True       # 3rd channel = gaussian-blurred spot density (σ=8)
SPOT_DENSITY_SIGMA = 8.0
CELLPOSE_DIAMETER = 0.0           # 0 = auto-estimate per FOV
CELLPROB_THRESHOLD = -0.5
FLOW_THRESHOLD = 0.4
NN_RADIUS = 0.0                   # >0 reassigns nearby background spots to nearest cell

# =====================================================================
# Classifier
# =====================================================================
# 'codelab_p2' = pretrained kNN(k=5, cosine, L1) on phase-2 counts.h5ad.
#                Located at phase2/runs/baseline-codelab-p2only/.
# 'logreg_p2'  = our previous logreg baseline at phase2/runs/.../train-baseline/.
# Or set CLASSIFIER_DIR to a custom path.
CLASSIFIER_DIR = "phase2/runs/baseline-codelab-rf500-log1p-mf01"

# =====================================================================
# Post-hoc ensemble (optional)
# =====================================================================
# If non-empty, the harness will produce a plurality-vote ensemble of THIS run
# combined with these existing submission CSVs. Useful for stacking on top of
# already-validated runs without re-running them.
ENSEMBLE_WITH: list[str] = []

# =====================================================================
# Notes (free-form — agent's hypothesis for THIS iteration)
# =====================================================================
HYPOTHESIS = """
Iter29: ET300 on L1-norm with max_features=0.1. Combining the winning mf=0.1
trick with ExtraTrees (which had best class internal-val on L1 originally) and
keeping L1 preproc. Internal val crushes everything: class 0.552 (+0.023 over
RF-log1p-mf01), subclass +0.013, supertype +0.019, cluster +0.006. Mean +0.015.
First config to improve every level simultaneously.
"""
