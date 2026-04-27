"""Kaggle scoring metric: mean ARI across FOVs on spot-to-cell assignment."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def merfish_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Compute mean Adjusted Rand Index across FOVs.

    Args:
        solution: GT assignments, index=spot_id, columns=[fov, gt_cluster_id]
        submission: predicted assignments, index=spot_id, columns=[fov, cluster_id]
    Returns:
        Mean ARI across all FOVs in the solution.
    """
    submission = submission.reindex(solution.index)
    if submission['cluster_id'].isna().any():
        submission['cluster_id'] = submission['cluster_id'].fillna('background')

    ari_scores = []
    for fov in solution['fov'].unique():
        fov_mask = solution['fov'] == fov
        gt_labels   = solution.loc[fov_mask, 'gt_cluster_id'].astype(str).values
        pred_labels = submission.loc[fov_mask, 'cluster_id'].astype(str).values
        ari_scores.append(adjusted_rand_score(gt_labels, pred_labels))

    return float(np.mean(ari_scores)) if ari_scores else 0.0


def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """Kaggle entrypoint — Kaggle calls score(solution, submission, row_id_column_name)."""
    for df in (solution, submission):
        if 'Usage' in df.columns:
            df.drop(columns=['Usage'], inplace=True)
    solution   = solution.set_index(row_id_column_name)
    submission = submission.set_index(row_id_column_name)
    return merfish_score(solution, submission)
