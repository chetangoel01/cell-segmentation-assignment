"""Inference for the Hierarchical MLP bundle (modal_train_mlp_hier.py).

Mirrors infer_xgb.py — uses the same masks-dir / segment-on-the-fly flow,
loads the MLP bundle (PyTorch state dicts + sklearn LabelEncoders), runs
top-down conditional softmax-feature classification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase2.src import io, coords  # noqa: E402
from phase2.tasks.infer_baseline import (  # noqa: E402
    _featurize_cells_from_mask, _pick_device, LEVELS,
)


LEVELS_INTERNAL = ("class_label", "subclass_label", "supertype_label", "cluster_label")
LEVELS_OUT = {"class_label": "class", "subclass_label": "subclass",
              "supertype_label": "supertype", "cluster_label": "cluster"}


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


def _build_torch_model(spec: dict):
    """Match training-side `MLPHead` (Module with .net Sequential)."""
    import torch.nn as nn

    class _MLPHead(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            prev = spec["in_dim"]
            for _ in range(spec["n_hidden"]):
                layers.append(nn.Linear(prev, spec["hidden"]))
                layers.append(nn.BatchNorm1d(spec["hidden"]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(spec["dropout"]))
                prev = spec["hidden"]
            layers.append(nn.Linear(prev, spec["n_classes"]))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return _MLPHead()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, help="mlp_bundle.joblib path")
    p.add_argument("--mode", required=True, choices=("val", "test"))
    p.add_argument("--fovs", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F

    bundle_obj = joblib.load(args.bundle)
    assert bundle_obj.get("type") == "mlp_hier_modal", "expected mlp_hier_modal bundle"
    genes = bundle_obj["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    feat_mean = np.asarray(bundle_obj["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(bundle_obj["feat_std"], dtype=np.float32)
    levels_internal = tuple(bundle_obj["levels"])
    encoders = bundle_obj["encoders"]
    models_spec = bundle_obj["models"]

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}, feature_dim={bundle_obj['feature_dim']}")

    # Materialize PyTorch modules in eval mode
    nets = {}
    for lvl in levels_internal:
        spec = models_spec[lvl]
        m = _build_torch_model(spec).to(device)
        m.load_state_dict(spec["state_dict"])
        m.eval()
        nets[lvl] = m

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
        else:
            # log1p CPM, then z-score with the train feat_mean/std
            X32 = X.astype(np.float32)
            rs = X32.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            Xq = np.log1p(X32 / rs * 1e4)
            Xq = (Xq - feat_mean) / feat_std
            xt = torch.from_numpy(Xq).float().to(device)

            cell_predictions = {}
            prev_logits: torch.Tensor | None = None
            with torch.no_grad():
                for lvl in levels_internal:
                    feats = xt if prev_logits is None else torch.cat([xt, prev_logits], dim=1)
                    logits = nets[lvl](feats)
                    pred_idx = logits.argmax(dim=1).cpu().numpy()
                    cell_predictions[lvl] = encoders[lvl].inverse_transform(pred_idx)
                    prev_logits = F.softmax(logits, dim=1)

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
