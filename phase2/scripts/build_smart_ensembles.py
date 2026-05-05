#!/usr/bin/env python3
"""Build the best ensemble candidates from whichever submissions exist at run time.

Usage: python phase2/scripts/build_smart_ensembles.py
Builds N ensembles in phase2/runs/SUBMIT_FINAL_*/ for whatever's available.
Prints a ranked summary at the end.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("class", "subclass", "supertype", "cluster")

# Path → (label, prior known kaggle score, model class for diversity tagging)
CANDIDATES = {
    "phase2/runs/SUBMIT_aws001_clusfilt_bg40_dist/submission.csv": ("SOTA-AWS-kNN-k5", 0.5593, "knn"),
    "phase2/runs/SUBMIT_hier_aws_kp15/submission.csv":            ("HierAWS-k50",      0.5577, "knn"),
    "phase2/runs/SUBMIT_aws_k15_bg40_dist/submission.csv":        ("AWS-kNN-k15",      0.5571, "knn"),
    "phase2/runs/SUBMIT_sota_erode2/submission.csv":              ("SOTA-erode2",      0.5538, "knn"),
    "phase2/runs/SUBMIT_v7_PQM_cpsam_4way/submission.csv":        ("V7-ensemble",      0.5421, "ensemble"),
    "phase2/runs/SUBMIT_aws_cp0.5/submission.csv":                ("AWS-cp0.5",        0.5483, "knn"),
    "phase2/runs/SUBMIT_celltypist_bridge/submission.csv":        ("CellTypist-bridge", None, "logreg"),
    "phase2/runs/SUBMIT_mlp_hier/submission.csv":                 ("MLP-h512-d3",      None, "nn"),
    "phase2/runs/SUBMIT_mlp_xl/submission.csv":                   ("MLP-XL-h1024-d5",  None, "nn"),
    "phase2/runs/SUBMIT_scanvi_old/submission.csv":               ("scANVI-old-modal", None, "vae"),
    "phase2/runs/SUBMIT_scanvi_xl/submission.csv":                ("scANVI-XL",        None, "vae"),
    "phase2/runs/SUBMIT_xgb_old/submission.csv":                  ("XGB-old-d4n200",   None, "xgb"),
    "phase2/runs/SUBMIT_xgb_xl_d8/submission.csv":                ("XGB-XL-d8n3k",     None, "xgb"),
    "phase2/runs/SUBMIT_xgb_xl_d6/submission.csv":                ("XGB-XL-d6n3k-bg80", None, "xgb"),
    "phase2/runs/SUBMIT_xgb_xl_d5/submission.csv":                ("XGB-XL-d5n5k",     None, "xgb"),
}


def in_cell_frac(path: Path) -> float:
    df = pd.read_csv(path, usecols=["class"])
    return float((df["class"] != "background").mean())


def vote_4(arrays: list[np.ndarray], anchor_idx: int = 0) -> tuple[np.ndarray, dict]:
    n = len(arrays[0])
    out = np.empty(n, dtype=object)
    stats = {"unanimous": 0, "majority": 0, "tie_anchor_wins": 0}
    threshold = len(arrays) // 2 + 1
    stacked = np.vstack(arrays)
    for i in range(n):
        col = stacked[:, i]
        if (col == col[0]).all():
            out[i] = col[0]; stats["unanimous"] += 1; continue
        c = Counter(col.tolist())
        top_val, top_n = c.most_common(1)[0]
        if top_n >= threshold:
            out[i] = top_val; stats["majority"] += 1
        else:
            out[i] = arrays[anchor_idx][i]; stats["tie_anchor_wins"] += 1
    return out, stats


def build_ensemble(name: str, voter_paths: list[str], anchor_idx: int = 0) -> Path | None:
    out_dir = ROOT / "phase2" / "runs" / f"SUBMIT_FINAL_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    available = [p for p in voter_paths if (ROOT / p).exists()]
    if len(available) < 2:
        print(f"  [{name}] only {len(available)} voters present — skip")
        return None

    print(f"\n[{name}] {len(available)} voters:")
    dfs = []
    for i, rel in enumerate(available):
        df = pd.read_csv(ROOT / rel)
        anchor_marker = " ★anchor" if i == anchor_idx else ""
        print(f"    [{i}]{anchor_marker} {rel}  rows={len(df):,}")
        dfs.append(df)

    base = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        if len(df) != len(base) or not (df.spot_id == base.spot_id).all():
            print(f"  [{name}] alignment mismatch at voter {i} — skip")
            return None

    out = base[["spot_id", "fov"]].copy()
    for lvl in LEVELS:
        arrays = [df[lvl].to_numpy().astype(str) for df in dfs]
        voted, stats = vote_4(arrays, anchor_idx=anchor_idx)
        out[lvl] = voted
    out_path = out_dir / "submission.csv"
    out.to_csv(out_path, index=False)

    ic = float((out["class"] != "background").mean())
    print(f"    → {out_path} in_cell={ic:.4f}")
    return out_path


def main() -> int:
    available = {rel: meta for rel, meta in CANDIDATES.items() if (ROOT / rel).exists()}
    print(f"=== {len(available)}/{len(CANDIDATES)} candidate submissions present ===")
    for rel, (label, score, kind) in available.items():
        ic = in_cell_frac(ROOT / rel)
        score_str = f"K={score:.4f}" if score else "K=?"
        print(f"  {kind:<8s} {score_str:<10s} ic={ic:.3f}  {label}")

    # Best known voter as anchor
    proven = [(rel, meta) for rel, meta in available.items() if meta[1] is not None]
    proven.sort(key=lambda kv: kv[1][1] or 0, reverse=True)
    if not proven:
        print("[fatal] no proven Kaggle-scored submission to anchor on")
        return 1
    anchor = proven[0][0]
    print(f"\nAnchor: {proven[0][1][0]} ({anchor})")

    # Group new (unscored) candidates by kind, pick best per kind for diversity
    new_subs = {kind: [] for kind in {"xgb", "nn", "vae", "logreg"}}
    for rel, (label, score, kind) in available.items():
        if score is None and kind in new_subs:
            new_subs[kind].append(rel)

    # Build ensembles
    print("\n=== Building ensembles ===")

    # E1: anchor + 1 best from each new kind (max diversity)
    diverse = [anchor]
    for kind in ("xgb", "nn", "vae", "logreg"):
        if new_subs[kind]:
            diverse.append(new_subs[kind][0])
    if len(diverse) >= 3:
        build_ensemble("E1_diverse_4kind", diverse, anchor_idx=0)

    # E2: anchor + top 3 proven kNN-style
    proven_paths = [r for r, _ in proven[:4]]
    if len(proven_paths) >= 3:
        build_ensemble("E2_proven_top4", proven_paths, anchor_idx=0)

    # E3: anchor + ALL new (heavy new-model presence)
    all_new = []
    for kind in ("xgb", "nn", "vae"):
        all_new.extend(new_subs[kind][:2])  # at most 2 per kind to limit voters
    if len(all_new) >= 2:
        build_ensemble("E3_anchor_plus_all_new", [anchor] + all_new, anchor_idx=0)

    # E4: anchor + top 2 proven + top NN + top XGB (5-way mixed)
    mixed = [anchor]
    if len(proven) > 1: mixed.append(proven[1][0])  # 2nd best proven
    if new_subs["xgb"]: mixed.append(new_subs["xgb"][0])
    if new_subs["nn"]: mixed.append(new_subs["nn"][0])
    if new_subs["vae"]: mixed.append(new_subs["vae"][0])
    if len(mixed) >= 4:
        build_ensemble("E4_mixed_5way", mixed, anchor_idx=0)

    # E5: anchor + top 2 proven (3-way conservative kNN-stack)
    if len(proven) >= 3:
        build_ensemble("E5_top3_proven", [r for r, _ in proven[:3]], anchor_idx=0)

    # E6: best new-NN as anchor + SOTA + top XGB (alternative anchor)
    if new_subs["nn"] and new_subs["xgb"]:
        build_ensemble("E6_nn_anchor", [new_subs["nn"][0], anchor, new_subs["xgb"][0]], anchor_idx=0)

    # E7: best XGB-XL alone vs anchor (pure 2-way)
    xgb_xl = [r for r in new_subs["xgb"] if "xl" in r]
    if xgb_xl:
        build_ensemble("E7_anchor_xgbxl", [anchor, xgb_xl[0]], anchor_idx=0)

    # E8: 7-way mega-ensemble (anchor + everything new)
    mega = [anchor]
    for kind in ("xgb", "nn", "vae", "logreg"):
        mega.extend(new_subs[kind])
    if len(mega) >= 5:
        build_ensemble("E8_mega_7way", mega, anchor_idx=0)

    # Summary
    print("\n=== ALL CANDIDATE SUBMISSIONS ===")
    rows = []
    for rel, (label, score, kind) in available.items():
        ic = in_cell_frac(ROOT / rel)
        rows.append((label, kind, score, ic, str(ROOT / rel)))
    for f in sorted((ROOT / "phase2" / "runs").glob("SUBMIT_FINAL_*/submission.csv")):
        ic = in_cell_frac(f)
        rows.append((f.parent.name, "ensemble", None, ic, str(f)))
    rows.sort(key=lambda r: r[2] or 0, reverse=True)
    print(f"  {'label':<30} {'kind':<10} {'kaggle':<8} {'ic':<6}")
    for label, kind, score, ic, path in rows:
        s = f"{score:.4f}" if score is not None else "?"
        print(f"  {label:<30} {kind:<10} {s:<8} {ic:<6.4f}  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
