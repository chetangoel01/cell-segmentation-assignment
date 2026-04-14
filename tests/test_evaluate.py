import pandas as pd
from src.evaluate import compute_ari


def test_perfect_ari():
    gt = pd.DataFrame(
        {"spot_id": ["s0", "s1", "s2"], "fov": ["A", "A", "A"], "cluster_id": ["c1", "c1", "c2"]}
    )
    pred = pd.DataFrame(
        {"spot_id": ["s0", "s1", "s2"], "fov": ["A", "A", "A"], "cluster_id": ["c1", "c1", "c2"]}
    )
    assert abs(compute_ari(gt, pred) - 1.0) < 1e-6


def test_all_background_ari():
    gt = pd.DataFrame({"spot_id": ["s0", "s1"], "fov": ["A", "A"], "cluster_id": ["c1", "c2"]})
    pred = pd.DataFrame(
        {"spot_id": ["s0", "s1"], "fov": ["A", "A"], "cluster_id": ["background", "background"]}
    )
    ari = compute_ari(gt, pred)
    assert ari == 0.0

