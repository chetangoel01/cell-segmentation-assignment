"""
Custom Kaggle evaluation metric for the MERFISH segmentation competition.

Scoring logic (Adjusted Rand Index on spot-to-cell assignment):
  1. Students are given spot positions for test FOVs (no gene labels)
  2. Students segment cells from DAPI/polyT images
  3. Students assign each spot to one of their cells or "background"
  4. Students submit: spot_id, cluster_id
  5. We compare student's assignment (X) to GT assignment (Y) using ARI
  6. Score is averaged across FOVs

Score range: -1.0 (adversarial) → 0.0 (random) → 1.0 (perfect clustering)

This tests segmentation quality:
  - Bad segmentation → spots assigned to wrong cells → low ARI
  - Oversegmentation (each spot = own cell) → ARI ≈ 0
  - Undersegmentation (all spots = 1 cell) → ARI ≈ 0
  - Perfect pipeline → student clustering matches GT → ARI = 1.0

ARI is cluster-ID independent: it only checks whether pairs of spots are
grouped consistently between the two clusterings, regardless of cluster names.

Solution format (solution.csv):
  spot_id (index), fov, gt_cluster_id, Usage

Submission format (submission.csv):
  spot_id (index), fov, cluster_id
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def merfish_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Compute mean Adjusted Rand Index across FOVs.

    Parameters
    ----------
    solution : pd.DataFrame
        GT spot assignments. Index = spot_id, columns = [fov, gt_cluster_id]
    submission : pd.DataFrame
        Student spot assignments. Index = spot_id, columns = [fov, cluster_id]

    Returns
    -------
    float
        Mean ARI across all FOVs in the solution.
    """
    # Align submission to solution by spot_id
    # Missing spots in submission get 'background'
    submission = submission.reindex(solution.index)
    if submission['cluster_id'].isna().any():
        submission['cluster_id'] = submission['cluster_id'].fillna('background')

    ari_scores = []
    for fov in solution['fov'].unique():
        fov_mask = solution['fov'] == fov
        gt_labels = solution.loc[fov_mask, 'gt_cluster_id'].astype(str).values
        pred_labels = submission.loc[fov_mask, 'cluster_id'].astype(str).values

        ari = adjusted_rand_score(gt_labels, pred_labels)
        ari_scores.append(ari)

    return float(np.mean(ari_scores)) if ari_scores else 0.0


# ---------------------------------------------------------------------------
# Kaggle entrypoint
# Kaggle calls: score = score(solution, submission, row_id_column_name)
# ---------------------------------------------------------------------------
def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """Score student submissions using Adjusted Rand Index on spot-to-cell assignment."""
    # Drop Kaggle's Usage column if present
    if 'Usage' in solution.columns:
        solution = solution.drop(columns=['Usage'])
    if 'Usage' in submission.columns:
        submission = submission.drop(columns=['Usage'])

    solution = solution.set_index(row_id_column_name)
    submission = submission.set_index(row_id_column_name)

    return merfish_score(solution, submission)
