"""Compare segmentation models side-by-side on val FOVs 156-160.

Reads validate_local JSON outputs (one per model) and produces a side-by-side
table with mean ARI + per-FOV frac_in_cell {pred, gt} + error metrics.

Usage:
    python phase2/scripts/compare_seg.py \\
        old=path/to/old_val.json \\
        cyto3=path/to/cyto3_val.json \\
        stardist=path/to/stardist_val.json \\
        cpsam_fresh=path/to/cpsam_fresh_val.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    items: list[tuple[str, dict]] = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            print(f"[skip] expected tag=path, got {arg}")
            continue
        tag, p = arg.split("=", 1)
        path = Path(p)
        if not path.exists():
            print(f"[skip] {tag}: {path} not found")
            continue
        items.append((tag, load(path)))

    if not items:
        print("no inputs")
        return 1

    # Collect FOV order from the first entry.
    fovs = sorted(items[0][1]["per_fov"].keys())
    print(f"\nval FOVs: {fovs}")

    # Header.
    print("\n=== mean ARI (Kaggle-equivalent) + frac_in_cell ===")
    width_tag = max(8, max(len(t) for t, _ in items)) + 2
    print(f"{'model':<{width_tag}}{'mean_ARI':>10}{'fic_pred':>10}{'fic_gt':>10}{'fic_err':>10}{'n_cells':>10}")
    for tag, d in items:
        mean_ari = float(d.get("mean_ari", 0.0))
        fic_pred = statistics.mean(f.get("frac_in_cell_pred", 0.0) for f in d["per_fov"].values())
        fic_gt   = statistics.mean(f.get("frac_in_cell_gt", 0.0)   for f in d["per_fov"].values())
        fic_err  = abs(fic_pred - fic_gt)
        n_cells = sum(int(f.get("n_cells", 0)) for f in d["per_fov"].values())
        print(f"{tag:<{width_tag}}{mean_ari:>10.4f}{fic_pred:>10.4f}{fic_gt:>10.4f}{fic_err:>10.4f}{n_cells:>10d}")

    # Per-FOV ARI breakdown.
    print("\n=== per-FOV mean ARI (across 4 levels) ===")
    print(f"{'model':<{width_tag}}", *(f"{fov:>10}" for fov in fovs))
    for tag, d in items:
        cells = []
        for fov in fovs:
            entry = d["per_fov"].get(fov, {})
            ari = entry.get("mean_ari", entry.get("ari_mean", None))
            if ari is None:
                # Fallback: avg the per-level fields if present
                lvls = [entry.get(k) for k in ("ari_class", "ari_subclass", "ari_supertype", "ari_cluster")]
                lvls = [v for v in lvls if v is not None]
                ari = sum(lvls) / len(lvls) if lvls else 0.0
            cells.append(f"{ari:>10.4f}")
        print(f"{tag:<{width_tag}}", *cells)

    # Per-FOV frac_in_cell delta.
    print("\n=== per-FOV frac_in_cell pred (target: drop toward GT ~0.17) ===")
    print(f"{'model':<{width_tag}}", *(f"{fov:>10}" for fov in fovs))
    for tag, d in items:
        cells = [f"{d['per_fov'].get(fov,{}).get('frac_in_cell_pred',0.0):>10.4f}" for fov in fovs]
        print(f"{tag:<{width_tag}}", *cells)

    print(f"\n{'GT':<{width_tag}}",
          *(f"{items[0][1]['per_fov'].get(fov,{}).get('frac_in_cell_gt',0.0):>10.4f}" for fov in fovs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
