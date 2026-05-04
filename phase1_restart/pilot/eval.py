"""Per-FOV ARI evaluation. Spot assignment via mask lookup using pre-computed image_row/image_col."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def assign_spots_to_mask(spots: pd.DataFrame, mask: np.ndarray, fov: str) -> pd.DataFrame:
    """Look up mask label per spot. Output columns: spot_id, fov, cluster_id (str).

    cluster_id == "background" if mask label is 0; else f"{fov}_{int(label)}" (FOV-namespaced).

    If the input ``spots`` lacks ``spot_id``, one is synthesized as ``f"row_{i}"`` from
    the position within the FOV-filtered DataFrame. Only relevant for local eval on
    train spots (test_spots.csv always has spot_id).
    """
    df = spots[spots["fov"] == fov].copy().reset_index(drop=True)
    if "spot_id" not in df.columns:
        df.insert(0, "spot_id", [f"row_{i}" for i in range(len(df))])
    rows = df["image_row"].astype(int).clip(0, mask.shape[0] - 1).values
    cols = df["image_col"].astype(int).clip(0, mask.shape[1] - 1).values
    labels = mask[rows, cols]
    cluster_ids = np.where(
        labels == 0,
        np.array(["background"] * len(labels), dtype=object),
        np.array([f"{fov}_{int(l)}" for l in labels], dtype=object),
    )
    df = df.assign(cluster_id=cluster_ids)
    return df[["spot_id", "fov", "cluster_id"]]


def compute_per_fov_ari(truth: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    """Both DFs have spot_id, fov, cluster_id. Returns {fov: ARI}."""
    out: dict[str, float] = {}
    merged = truth.merge(pred, on=["spot_id", "fov"], suffixes=("_t", "_p"))
    for fov in merged["fov"].unique():
        sub = merged[merged["fov"] == fov]
        out[fov] = float(adjusted_rand_score(sub["cluster_id_t"], sub["cluster_id_p"]))
    return out


def mean_ari(per_fov: dict[str, float]) -> float:
    if not per_fov:
        return 0.0
    return float(np.mean(list(per_fov.values())))
