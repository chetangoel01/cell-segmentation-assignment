"""Inference-time segmentation matrix for Phase 2.

No training and no TTA. This script coordinates:
  1. Cellpose/CPSAM mask generation from existing checkpoints/backbones.
  2. Optional mask-only postprocessing.
  3. Local downstream ARI scoring via validate_local.py --masks-dir.

Default mode prints commands. Use --execute to run them sequentially.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VAL_FOVS = ",".join(f"FOV_{i:03d}" for i in range(156, 161))
TEST_FOVS = ",".join(f"FOV_{c}" for c in "EFGHIJKLMN")
RF500 = "phase2/runs/baseline-codelab-rf500-log1p-mf01"
P_KNN = "phase2/runs/baseline-codelab-p2only"


@dataclass(frozen=True)
class SegConfig:
    name: str
    checkpoint: str | None
    include_density: bool
    preprocess: str
    cellprob: float
    flow: float = 0.4
    diameter: float = 0.0


@dataclass(frozen=True)
class PostConfig:
    name: str
    morph: str = "none"
    radius: int = 0
    min_area: int = 0
    min_spots: int = 0


@dataclass(frozen=True)
class ExistingMaskConfig:
    name: str
    masks_dir: str


SEG_CONFIGS = [
    # Current anchor, but sweep inference-time thresholds/preprocessing.
    SegConfig("old_ep125_cp-1", "phase2/external_models/cellpose_nuclei_cosine_ep125", True, "none", -1.0),
    SegConfig("old_ep125_cp-05", "phase2/external_models/cellpose_nuclei_cosine_ep125", True, "none", -0.5),
    SegConfig("old_ep125_cp0", "phase2/external_models/cellpose_nuclei_cosine_ep125", True, "none", 0.0),
    SegConfig("old_ep125_clahe_cp-05", "phase2/external_models/cellpose_nuclei_cosine_ep125", True, "clahe", -0.5),
    SegConfig("old_ep125_pclip_cp-05", "phase2/external_models/cellpose_nuclei_cosine_ep125", True, "pclip", -0.5),

    # Existing fine-tuned candidates: not retraining, just scoring them properly.
    SegConfig("cpsam_p1p2_ep025", "phase2/runs/20260503-130529-train-seg-cpsam_phase2_p1stack/models/cpsam_phase2_p1stack_ep025", True, "none", -0.5),
    SegConfig("cpsam_p1p2_ep030", "phase2/runs/20260503-130529-train-seg-cpsam_phase2_p1stack/models/cpsam_phase2_p1stack_ep030", True, "none", -0.5),
    SegConfig("cyto3_p1_ep035", "phase2/runs/20260503-174912-train-seg-cyto3_p1stack/models/cyto3_p1stack_ep035", True, "none", -0.5),
    SegConfig("cyto3_p1_ep035_cp0", "phase2/runs/20260503-174912-train-seg-cyto3_p1stack/models/cyto3_p1stack_ep035", True, "none", 0.0),

    # Off-the-shelf CPSAM as a different backbone anchor.
    SegConfig("cpsam_zero_cp-05", None, True, "none", -0.5),
    SegConfig("cpsam_zero_clahe_cp-05", None, True, "clahe", -0.5),
]

EXISTING_MASK_CONFIGS = [
    ExistingMaskConfig("stardist", "phase2/runs/stardist_val_masks"),
    ExistingMaskConfig("maskens_majority", "phase2/runs/ensemble3_majority_val_masks"),
    ExistingMaskConfig("maskens_intersect", "phase2/runs/ensemble2_intersect_val_masks"),
    ExistingMaskConfig("cyto3_z2_cp-1", "phase2/runs/cyto3_z2_cp-1.0_val_masks"),
    ExistingMaskConfig("cpsam_erode1", "phase2/runs/cpsam_erode1_val_masks"),
    ExistingMaskConfig("cpsam_nospotdensity", "phase2/runs/cpsam_nospotdensity_val_masks"),
]

POST_CONFIGS = [
    PostConfig("raw"),
    PostConfig("erode1", morph="erode", radius=1),
    PostConfig("dilate1", morph="dilate", radius=1),
    PostConfig("open1", morph="open", radius=1),
    PostConfig("minspots3", min_spots=3),
    PostConfig("minarea80", min_area=80),
]


def _q(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run(cmd: list[str], execute: bool) -> int:
    print(_q(cmd))
    if not execute:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _py() -> list[str]:
    return [_python(), "-u"]


def _existing_configs(configs: list[SegConfig]) -> list[SegConfig]:
    kept = []
    for cfg in configs:
        if cfg.checkpoint is None or (ROOT / cfg.checkpoint).exists():
            kept.append(cfg)
        else:
            print(f"[skip] {cfg.name}: missing checkpoint {cfg.checkpoint}", file=sys.stderr)
    return kept


def _existing_mask_configs(configs: list[ExistingMaskConfig]) -> list[ExistingMaskConfig]:
    kept = []
    for cfg in configs:
        if (ROOT / cfg.masks_dir).is_dir():
            kept.append(cfg)
        else:
            print(f"[skip] {cfg.name}: missing masks dir {cfg.masks_dir}", file=sys.stderr)
    return kept


def mask_dir(base: Path, cfg: SegConfig) -> Path:
    return base / cfg.name / "masks"


def existing_mask_dir(cfg: ExistingMaskConfig) -> Path:
    return Path(cfg.masks_dir)


def post_dir(base: Path, cfg: SegConfig, post: PostConfig) -> Path:
    return base / cfg.name / f"masks_{post.name}"


def score_json(base: Path, cfg: SegConfig, post: PostConfig, clf_name: str) -> Path:
    return base / cfg.name / f"score_{post.name}_{clf_name}.json"


def existing_score_json(base: Path, cfg: ExistingMaskConfig, post: PostConfig, clf_name: str) -> Path:
    return base / cfg.name / f"score_{post.name}_{clf_name}.json"


def infer_cmd(base: Path, cfg: SegConfig, fovs: str, split: str = "train") -> list[str]:
    cmd = [
        *_py(), "phase2/scripts/infer_cellpose_masks.py",
        "--fovs", fovs,
        "--split", split,
        "--out-dir", str(mask_dir(base, cfg)),
        "--device", "auto",
        "--cellpose-diameter", str(cfg.diameter),
        "--cellprob-threshold", str(cfg.cellprob),
        "--flow-threshold", str(cfg.flow),
        "--preprocess", cfg.preprocess,
    ]
    if cfg.checkpoint:
        cmd.extend(["--seg-checkpoint", cfg.checkpoint])
    if cfg.include_density:
        cmd.append("--include-spot-density")
    return cmd


def post_cmd(base: Path, cfg: SegConfig, post: PostConfig, fovs: str, split: str = "train") -> list[str] | None:
    if post.name == "raw":
        return None
    cmd = [
        *_py(), "phase2/scripts/postprocess_masks.py",
        "--masks-dir", str(mask_dir(base, cfg)),
        "--out-dir", str(post_dir(base, cfg, post)),
        "--fovs", fovs,
        "--split", split,
        "--morph", post.morph,
        "--radius", str(post.radius),
        "--min-area", str(post.min_area),
        "--min-spots", str(post.min_spots),
    ]
    return cmd


def validate_cmd(base: Path, cfg: SegConfig, post: PostConfig, clf_dir: str, clf_name: str, fovs: str) -> list[str]:
    masks = mask_dir(base, cfg) if post.name == "raw" else post_dir(base, cfg, post)
    score_json(base, cfg, post, clf_name).parent.mkdir(parents=True, exist_ok=True)
    return [
        *_py(), "phase2/scripts/validate_local.py",
        "--models-dir", clf_dir,
        "--masks-dir", str(masks),
        "--val-fovs", fovs,
        "--nn-radius", "0.0",
        "--device", "cpu",
        "--out", str(score_json(base, cfg, post, clf_name)),
    ]


def existing_post_cmd(base: Path, cfg: ExistingMaskConfig, post: PostConfig, fovs: str, split: str = "train") -> list[str] | None:
    if post.name == "raw":
        return None
    return [
        *_py(), "phase2/scripts/postprocess_masks.py",
        "--masks-dir", str(existing_mask_dir(cfg)),
        "--out-dir", str(base / cfg.name / f"masks_{post.name}"),
        "--fovs", fovs,
        "--split", split,
        "--morph", post.morph,
        "--radius", str(post.radius),
        "--min-area", str(post.min_area),
        "--min-spots", str(post.min_spots),
    ]


def existing_validate_cmd(base: Path, cfg: ExistingMaskConfig, post: PostConfig, clf_dir: str, clf_name: str, fovs: str) -> list[str]:
    masks = existing_mask_dir(cfg) if post.name == "raw" else base / cfg.name / f"masks_{post.name}"
    existing_score_json(base, cfg, post, clf_name).parent.mkdir(parents=True, exist_ok=True)
    return [
        *_py(), "phase2/scripts/validate_local.py",
        "--models-dir", clf_dir,
        "--masks-dir", str(masks),
        "--val-fovs", fovs,
        "--nn-radius", "0.0",
        "--device", "cpu",
        "--out", str(existing_score_json(base, cfg, post, clf_name)),
    ]


def summarize(base: Path) -> int:
    rows = []
    for path in sorted(base.glob("*/score_*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        parts = path.stem.removeprefix("score_").rsplit("_", 1)
        post = parts[0] if parts else "?"
        clf = parts[1] if len(parts) > 1 else "?"
        per_fov = d.get("per_fov", {})
        fic_pred = np_mean([v.get("frac_in_cell_pred", 0.0) for v in per_fov.values()])
        fic_gt = np_mean([v.get("frac_in_cell_gt", 0.0) for v in per_fov.values()])
        rows.append((d.get("mean_ari", 0.0), path.parent.name, post, clf, fic_pred, fic_gt, path))
    rows.sort(reverse=True, key=lambda r: r[0])
    print("mean_ari  config                    post       clf    fic_pred  fic_gt    json")
    for ari, cfg, post, clf, fic_pred, fic_gt, path in rows:
        print(f"{ari:8.4f}  {cfg:<25} {post:<10} {clf:<6} {fic_pred:8.4f} {fic_gt:7.4f}  {path}")
    return 0


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="phase2/runs/seg-matrix")
    p.add_argument("--fovs", default=VAL_FOVS)
    p.add_argument("--execute", action="store_true", help="Actually run commands; default only prints.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of segmentation configs.")
    p.add_argument("--classifier", default="rf500", choices=("rf500", "p", "both"))
    p.add_argument("--posts", default=None,
                   help="Comma-separated post configs to run, e.g. raw,minspots3. "
                        "Default: all post configs.")
    p.add_argument("--existing-only", action="store_true",
                   help="Only score existing mask directories; skip fresh Cellpose inference.")
    p.add_argument("--existing-names", default=None,
                   help="Comma-separated existing mask config names to score.")
    p.add_argument("--cellpose-only", action="store_true",
                   help="Only run fresh Cellpose/CPSAM mask configs; skip existing mask dirs.")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    base = Path(args.out_root)
    if args.summarize:
        return summarize(base)

    configs = _existing_configs(SEG_CONFIGS)
    if args.limit:
        configs = configs[: args.limit]
    classifiers = []
    if args.classifier in ("rf500", "both"):
        classifiers.append(("rf500", RF500))
    if args.classifier in ("p", "both"):
        classifiers.append(("p", P_KNN))
    post_configs = POST_CONFIGS
    if args.posts:
        wanted = {p.strip() for p in args.posts.split(",") if p.strip()}
        post_configs = [p for p in POST_CONFIGS if p.name in wanted]
        missing = wanted - {p.name for p in post_configs}
        if missing:
            print(f"[fatal] unknown post configs: {sorted(missing)}")
            return 2

    if not args.existing_only:
        for cfg in configs:
            rv = _run(infer_cmd(base, cfg, args.fovs), args.execute)
            if rv:
                return rv
            for post in post_configs:
                pcmd = post_cmd(base, cfg, post, args.fovs)
                if pcmd is not None:
                    rv = _run(pcmd, args.execute)
                    if rv:
                        return rv
                for clf_name, clf_dir in classifiers:
                    rv = _run(validate_cmd(base, cfg, post, clf_dir, clf_name, args.fovs), args.execute)
                    if rv:
                        return rv
    if args.cellpose_only:
        print(f"\nAfter execution, summarize with:\n  {_python()} phase2/scripts/run_seg_matrix.py --out-root {base} --summarize")
        return 0

    existing_configs = _existing_mask_configs(EXISTING_MASK_CONFIGS)
    if args.existing_names:
        wanted = {n.strip() for n in args.existing_names.split(",") if n.strip()}
        existing_configs = [c for c in existing_configs if c.name in wanted]
        missing = wanted - {c.name for c in existing_configs}
        if missing:
            print(f"[fatal] unknown or unavailable existing configs: {sorted(missing)}")
            return 2

    for cfg in existing_configs:
        for post in post_configs:
            pcmd = existing_post_cmd(base, cfg, post, args.fovs)
            if pcmd is not None:
                rv = _run(pcmd, args.execute)
                if rv:
                    return rv
            for clf_name, clf_dir in classifiers:
                rv = _run(existing_validate_cmd(base, cfg, post, clf_dir, clf_name, args.fovs), args.execute)
                if rv:
                    return rv
    print(f"\nAfter execution, summarize with:\n  {_python()} phase2/scripts/run_seg_matrix.py --out-root {base} --summarize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
