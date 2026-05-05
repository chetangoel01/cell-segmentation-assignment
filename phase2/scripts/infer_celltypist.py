"""CellTypist (Mouse_Isocortex_Hippocampus) + hier-bridge inference.

Per codelab_13.py section 'CellTypist (pretrained mouse-brain classifier)'.
Uses pretrained logreg on log1p-CPM expression to predict subclass-level
cell types, then bridges those labels to ABCA-4 taxonomy via token
containment, then projects each subclass → modal full-hierarchy.

No training. Inference only — runs on CPU, ~5–10 min for 10 test FOVs.
Used as a diversity voter alongside scANVI / XGB / MLP.

Usage:
    python phase2/scripts/infer_celltypist.py \
        --mode test \
        --fovs FOV_E,FOV_F,FOV_G,FOV_H,FOV_I,FOV_J,FOV_K,FOV_L,FOV_M,FOV_N \
        --masks-dir phase2/runs/sota_test_masks \
        --out-dir phase2/runs/SUBMIT_celltypist_bridge
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.src import io, coords  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, LEVELS,
)


LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}

MODEL_NAME = "Mouse_Isocortex_Hippocampus.pkl"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9/]+", str(s)))


def _build_subclass_to_hier() -> dict[str, dict[str, str]]:
    """Read phase-2 cell_labels_train.csv, build subclass → modal full hierarchy."""
    labels = pd.read_csv(io.ground_truth_dir() / "cell_labels_train.csv")
    labels = labels.dropna(subset=["class_label", "subclass_label", "supertype_label", "cluster_label"])
    out: dict[str, dict[str, str]] = {}
    for sub, g in labels.groupby("subclass_label"):
        out[str(sub)] = {
            "class_label": str(g["class_label"].iloc[0]),
            "subclass_label": str(sub),
            "supertype_label": str(g["supertype_label"].mode().iloc[0]),
            "cluster_label": str(g["cluster_label"].mode().iloc[0]),
        }
    return out


def _bridge_celltypist_labels(ct_labels: set[str], sub_to_hier: dict) -> dict[str, str]:
    """Map CellTypist label string → ABCA-4 subclass via token containment."""
    bridge: dict[str, str] = {}
    for ct_label in ct_labels:
        ct_tok = _tokens(ct_label)
        if not ct_tok:
            continue
        # Match if every CellTypist token appears in the ABCA-4 subclass token set
        matches = [sub for sub in sub_to_hier if ct_tok.issubset(_tokens(sub))]
        if matches:
            # Most specific = shortest matching subclass label
            bridge[ct_label] = min(matches, key=len)
    return bridge


def _build_gt_spot_labels(fov, fov_spots, cells_df, labels_df):
    fov_cells = labels_df[labels_df["fov"] == fov]
    spot_x = fov_spots["global_x"].to_numpy()
    spot_y = fov_spots["global_y"].to_numpy()
    out = fov_spots.copy()
    for lvl in LEVELS:
        out[f"gt_{lvl}"] = "background"
    for cid in fov_cells["cell_id"]:
        if cid not in cells_df.index:
            continue
        row = cells_df.loc[cid]
        poly = coords.parse_boundary_polygon(row.get("boundaryX_z2", ""), row.get("boundaryY_z2", ""))
        if poly is None:
            continue
        inside = coords.spots_in_polygon(spot_x, spot_y, poly)
        if not inside.any():
            continue
        cell_row = fov_cells[fov_cells["cell_id"] == cid].iloc[0]
        for lvl in LEVELS:
            out.loc[fov_spots.index[inside], f"gt_{lvl}"] = cell_row[f"{lvl}_label"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("val", "test"))
    p.add_argument("--fovs", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--confidence-threshold", type=float, default=0.3,
                   help="Below this, label as background")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import anndata as ad
    import scanpy as sc
    import celltypist

    print(f"loading CellTypist model {MODEL_NAME}")
    celltypist.models.download_models(force_update=False, model=[MODEL_NAME])

    # Use phase-2 gene panel (1147 genes) for the count matrix
    genes_h5ad = io.ground_truth_dir() / "counts_train.h5ad"
    adata_train = ad.read_h5ad(genes_h5ad)
    GENE_LIST = list(adata_train.var_names)
    gene_to_idx = {g: i for i, g in enumerate(GENE_LIST)}

    sub_to_hier = _build_subclass_to_hier()
    print(f"subclass→hier lookup: {len(sub_to_hier)} entries")

    fov_list = [f.strip() for f in args.fovs.split(",") if f.strip()]
    masks_dir = Path(args.masks_dir)

    if args.mode == "val":
        spots = pd.read_csv(io.ground_truth_dir() / "spots_train.csv")
        cells_df = pd.read_csv(io.ground_truth_dir() / "cell_boundaries_train.csv", index_col=0)
        labels_df = pd.read_csv(io.ground_truth_dir() / "cell_labels_train.csv")
    else:
        spots = pd.read_csv(io.data_root() / "test_spots.csv")
        cells_df = labels_df = None

    sample_submission_rows = []
    per_fov_ari = {}
    overall_ari = []

    for fov in fov_list:
        print(f"\n=== {fov} ===")
        fov_spots = spots[spots["fov"] == fov].copy().reset_index(drop=True)
        if len(fov_spots) == 0:
            continue

        if args.mode == "val":
            fov_spots_gt = _build_gt_spot_labels(fov, fov_spots, cells_df, labels_df)

        mask_path = masks_dir / f"{fov}.npy"
        if not mask_path.exists():
            print(f"  [skip] {mask_path} missing")
            continue
        masks = np.load(mask_path).astype(np.int32)

        cell_ids, X, spot_label = _featurize_cells_from_mask(masks, fov_spots, gene_to_idx, nn_radius=0)
        if len(cell_ids) == 0:
            cell_predictions = {lvl: np.array([]) for lvl in LEVELS_INTERNAL}
            label_to_idx: dict = {}
        else:
            qry = ad.AnnData(
                X=X.astype(np.float32),
                obs=pd.DataFrame({"fov": [fov] * len(cell_ids)}, index=[str(c) for c in cell_ids]),
                var=pd.DataFrame(index=GENE_LIST),
            )
            sc.pp.normalize_total(qry, target_sum=1e4)
            sc.pp.log1p(qry)
            n_neighbors = max(2, min(10, qry.n_obs - 1))
            try:
                sc.pp.pca(qry, n_comps=min(30, qry.n_vars - 1, qry.n_obs - 1))
                sc.pp.neighbors(qry, n_neighbors=n_neighbors)
                ct_pred = celltypist.annotate(qry, model=MODEL_NAME, majority_voting=True)
            except Exception as e:
                print(f"  [warn] majority_voting failed ({e}); falling back to non-MV")
                ct_pred = celltypist.annotate(qry, model=MODEL_NAME, majority_voting=False)

            pred_labels = ct_pred.predicted_labels.iloc[:, -1].values.astype(str)
            ct_probs = ct_pred.probability_matrix.max(axis=1).values
            unique_ct = set(pred_labels)
            ct_to_subclass = _bridge_celltypist_labels(unique_ct, sub_to_hier)
            print(f"  cells={len(cell_ids)} CT-unique={len(unique_ct)} bridged={len(ct_to_subclass)}")

            cell_predictions = {lvl: np.full(len(cell_ids), "background", dtype=object)
                                for lvl in LEVELS_INTERNAL}
            for i, lbl in enumerate(pred_labels):
                conf = float(ct_probs[i])
                if conf < args.confidence_threshold or lbl not in ct_to_subclass:
                    continue
                hier = sub_to_hier[ct_to_subclass[lbl]]
                for lvl in LEVELS_INTERNAL:
                    cell_predictions[lvl][i] = hier[lvl]

            label_to_idx = {int(c): i for i, c in enumerate(cell_ids)}

        spot_idx = np.array([label_to_idx.get(int(l), -1) for l in spot_label])
        in_cell_mask = spot_idx >= 0

        per_lvl = {}
        spot_preds = {}
        for lvl in LEVELS_INTERNAL:
            preds = cell_predictions[lvl]
            pred_spot = np.full(len(fov_spots), "background", dtype=object)
            if len(preds):
                pred_spot[in_cell_mask] = preds[spot_idx[in_cell_mask]]
            spot_preds[LEVELS_OUT[lvl]] = pred_spot
            if args.mode == "val":
                gt_spot = fov_spots_gt[f"gt_{LEVELS_OUT[lvl]}"].to_numpy().astype(str)
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(gt_spot, pred_spot.astype(str))
                per_lvl[LEVELS_OUT[lvl]] = float(ari)
                overall_ari.append(ari)

        if args.mode == "val":
            per_fov_ari[fov] = per_lvl
            print("  per-level: " + "  ".join(f"{k}={v:+.4f}" for k, v in per_lvl.items()))
        else:
            sub = pd.DataFrame({
                "spot_id": fov_spots["spot_id"].values if "spot_id" in fov_spots.columns else fov_spots.index.astype(str),
                "fov": fov,
                "class": spot_preds["class"].astype(str),
                "subclass": spot_preds["subclass"].astype(str),
                "supertype": spot_preds["supertype"].astype(str),
                "cluster": spot_preds["cluster"].astype(str),
            })
            sample_submission_rows.append(sub)
            print(f"  cells={int(masks.max())} rows={len(sub)}")

    if args.mode == "val":
        mean_ari = float(np.mean(overall_ari)) if overall_ari else 0.0
        print(f"\nMEAN ARI = {mean_ari:.4f}")
        Path(out_dir / "val_metrics.json").write_text(json.dumps({
            "mean_ari": mean_ari, "per_fov": per_fov_ari}, indent=2))
    else:
        if sample_submission_rows:
            our = pd.concat(sample_submission_rows, ignore_index=True)
        else:
            our = pd.DataFrame(columns=["spot_id", "fov", "class", "subclass", "supertype", "cluster"])
        sample = pd.read_csv(io.data_root() / "sample_submission.csv", dtype={"spot_id": str})
        our["spot_id"] = our["spot_id"].astype(str)
        merged = sample.merge(our[["spot_id", "class", "subclass", "supertype", "cluster"]],
                              on="spot_id", how="left", suffixes=("", "_pred"))
        for col in ("class", "subclass", "supertype", "cluster"):
            pc = col + "_pred"
            if pc in merged.columns:
                merged[col] = merged[pc].fillna("background")
                merged.drop(columns=[pc], inplace=True)
            else:
                merged[col] = "background"
        merged = merged[sample.columns.tolist()]
        out_csv = out_dir / "submission.csv"
        merged.to_csv(out_csv, index=False)
        print(f"\n→ {out_csv} rows={len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
