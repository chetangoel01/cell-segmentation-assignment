from __future__ import annotations

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def compute_ari(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Compute mean ARI across all FOVs present in solution.

    Both inputs are expected to have columns: spot_id, fov, cluster_id.
    """
    if solution.empty:
        return 0.0

    merged = solution.merge(submission, on=["spot_id", "fov"], suffixes=("_gt", "_pred"))
    if merged.empty:
        return 0.0

    fov_scores = []
    for _fov, group in merged.groupby("fov"):
        fov_scores.append(adjusted_rand_score(group["cluster_id_gt"], group["cluster_id_pred"]))

    return float(sum(fov_scores) / len(fov_scores)) if fov_scores else 0.0

