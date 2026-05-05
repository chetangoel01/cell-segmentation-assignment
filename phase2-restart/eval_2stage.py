"""Evaluate a 2-stage classifier:
  Stage 1: original RF500 (with bg) — predicts class label including 'background'
  Stage 2: nobg RF500 — for cells predicted as non-bg, override with nobg's prediction

Score per-spot ARI at spot level on val FOVs (matches Kaggle metric).
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase2.src import io, coords  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, _normalize, LEVELS,
)
from phase2.scripts.validate_local import _build_gt_spot_labels  # noqa: E402

VAL_FOVS = ["FOV_156", "FOV_157", "FOV_158", "FOV_159", "FOV_160"]
MASKS_DIR = ROOT / "phase2" / "runs" / "cyto3_z2_cp-1.0_val_masks"  # actually cpsam
ORIG_DIR = ROOT / "phase2" / "runs" / "baseline-codelab-rf500-log1p-mf01"
NOBG_DIR = ROOT / "phase2" / "runs" / "cls-rf500-log1p-mf01-nobg"


def load_bundles(d: Path) -> tuple[dict, list, str]:
    """Returns ({level: clf}, genes, preproc)"""
    out = {}
    bundle = joblib.load(d / "model_class.joblib")
    genes = list(bundle["genes"])
    preproc = bundle.get("preproc", "log1p")
    for lvl in LEVELS:
        out[lvl] = joblib.load(d / f"model_{lvl}.joblib")["clf"]
    return out, genes, preproc


def main():
    print("loading classifiers…")
    orig_clf, orig_genes, orig_preproc = load_bundles(ORIG_DIR)
    nobg_clf, nobg_genes, nobg_preproc = load_bundles(NOBG_DIR)
    assert orig_genes == nobg_genes, "gene vocab mismatch"
    print(f"  preproc orig={orig_preproc} nobg={nobg_preproc}")
    gene_to_idx = {g: i for i, g in enumerate(orig_genes)}

    print("loading spots + GT…")
    spots = pd.read_csv(ROOT / "phase2" / "data" / "train" / "ground_truth" / "spots_train.csv")
    cells = pd.read_csv(ROOT / "phase2" / "data" / "train" / "ground_truth" / "cell_boundaries_train.csv")
    cells = cells.rename(columns={cells.columns[0]: "cell_id"})
    cells.set_index("cell_id", inplace=True)
    labels = pd.read_csv(ROOT / "phase2" / "data" / "train" / "ground_truth" / "cell_labels_train.csv")

    from sklearn.metrics import adjusted_rand_score
    fov_aris = []
    for fov in VAL_FOVS:
        fov_spots = spots[spots["fov"] == fov].copy()
        fov_spots, n_assigned = _build_gt_spot_labels(fov, fov_spots, cells, labels)

        masks = np.load(MASKS_DIR / f"{fov}.npy").astype(np.int32)
        cell_ids, X, spot_label = _featurize_cells_from_mask(masks, fov_spots, gene_to_idx)
        X_orig = _normalize(X, preproc=orig_preproc)
        X_nobg = _normalize(X, preproc=nobg_preproc)

        # Stage 1: predict with original (handles bg)
        # Stage 2: for cells predicted as non-bg, override with nobg classifier
        for lvl in LEVELS:
            yh_orig = orig_clf[lvl].predict(X_orig)
            yh_nobg = nobg_clf[lvl].predict(X_nobg)
            # 2-stage: keep yh_orig if it says 'background', else use yh_nobg
            yh = np.where(yh_orig == "background", yh_orig, yh_nobg)

            cid_to_label = dict(zip(cell_ids, yh))
            spot_pred = np.array(["background"] * len(fov_spots), dtype=object)
            for cid, lab in cid_to_label.items():
                spot_pred[spot_label == cid] = lab
            ari = adjusted_rand_score(fov_spots[f"gt_{lvl}"].astype(str).values, spot_pred)
            print(f"  {fov} [{lvl}]: ARI={ari:.4f}", end="")
            if lvl == "cluster":
                print()
            else:
                print(" |", end=" ")
            fov_aris.append({"fov": fov, "level": lvl, "ari": ari})
    df = pd.DataFrame(fov_aris)
    print()
    print("========================================")
    for fov in VAL_FOVS:
        sub = df[df.fov == fov]
        print(f"  {fov} mean={sub.ari.mean():.4f}")
    print(f"  MEAN across {len(VAL_FOVS)}x4 = {df.ari.mean():.4f}")


if __name__ == "__main__":
    main()
