"""End-to-end phase-2 pipeline.

Chains the three stages in order:

    1. train-baseline       → classifier joblibs       (CPU, sklearn)
    2. train-segmentation   → fine-tuned cpsam ckpt    (GPU/MPS, slow)
    3. infer-baseline       → submission.csv           (GPU/MPS)

Stage outputs nest under one timestamped run dir so a single pipeline run
is one folder on disk:

    phase2/runs/<ts>-pipeline-<exp>/
      train-baseline/      model_*.joblib, metrics.json
      train-segmentation/  <exp>_final, train_state.json, summary.json
      infer-baseline/      submission.csv, summary.json

Skip flags let you reuse artifacts from previous runs without rerunning the
slow stages — useful for iterating on the classifier or the test-FOV split
without paying for another segmentation fine-tune.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from phase2.tasks import Task, register


def _add_args(p: argparse.ArgumentParser) -> None:
    # FOV split — same flags as the underlying tasks for consistency.
    p.add_argument("--train-fovs", required=True,
                   help="Comma-separated train FOVs (used by stages 1 and 2).")
    p.add_argument("--val-fovs", default="",
                   help="Comma-separated val FOVs.")
    p.add_argument("--test-fovs", required=True,
                   help="Comma-separated test FOVs for inference.")

    # Output + naming.
    p.add_argument("--out-dir", default=None,
                   help="Pipeline run dir (default: phase2/runs/<ts>-pipeline-<exp>/).")
    p.add_argument("--exp-name", default="pipeline_run",
                   help="Experiment name (used for segmentation checkpoint name and run dir).")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))

    # Stage-skip / reuse flags.
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip stage 1; requires --baseline-dir.")
    p.add_argument("--baseline-dir", default=None,
                   help="Existing classifier dir (with model_*.joblib) to reuse.")
    p.add_argument("--skip-segmentation", action="store_true",
                   help="Skip stage 2; inference uses off-the-shelf cpsam.")
    p.add_argument("--seg-checkpoint", default=None,
                   help="Existing segmentation checkpoint to reuse instead of training.")

    # Stage-1 (train-baseline) pass-through.
    p.add_argument("--baseline-model", default="logreg", choices=("logreg", "knn"),
                   help="Classifier family for stage 1 (default: logreg).")
    p.add_argument("--knn-k", type=int, default=15,
                   help="k for kNN if --baseline-model knn.")

    # Stage-2 (train-segmentation) pass-through.
    p.add_argument("--epochs", type=int, default=300,
                   help="Segmentation training epochs (default: 300).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--resume", action="store_true",
                   help="Resume stage-2 from latest checkpoint in segmentation out-dir.")
    p.add_argument("--time-budget", default=None,
                   help="Wall-clock limit for stage 2 (e.g. '2h30m', '150m', '9000s'). "
                        "Whichever of --epochs or --time-budget hits first wins.")
    p.add_argument("--keep-best", type=int, default=2,
                   help="Stage 2: keep only the N best chunk checkpoints "
                        "(by val loss if --val-fovs is set, else train loss). "
                        "0 = keep all. Default 2.")

    # Stage-3 (infer-baseline) pass-through.
    p.add_argument("--cellpose-diameter", type=float, default=30.0)
    p.add_argument("--cellprob-threshold", type=float, default=0.0,
                   help="Cellpose cellprob_threshold (lower = more permissive). "
                        "Default 0.0; try -1.0 or -2.0 for dense MERFISH FOVs.")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--include-spot-density", action="store_true",
                   help="Add spot density (σ=8) as 3rd input channel for inference. "
                        "Required for phase-1 nuclei models trained on 3 channels.")
    p.add_argument("--spot-density-sigma", type=float, default=8.0)


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def _banner(text: str) -> None:
    bar = "=" * (len(text) + 8)
    print(f"\n{bar}\n=== {text} ===\n{bar}")


def _run(args: argparse.Namespace) -> int:
    from phase2.tasks import train_baseline, train_segmentation, infer_baseline

    wall_t0 = time.time()
    base_out = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "runs" /
        f"{time.strftime('%Y%m%d-%H%M%S')}-pipeline-{args.exp_name}"
    )
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"pipeline run dir: {base_out}")

    stage_t: dict[str, float] = {}

    # ---------- Stage 1: train-baseline ----------
    if args.skip_baseline:
        if not args.baseline_dir:
            print("[fatal] --skip-baseline requires --baseline-dir")
            return 1
        baseline_dir = Path(args.baseline_dir)
        if not baseline_dir.exists():
            print(f"[fatal] --baseline-dir {baseline_dir} does not exist")
            return 1
        _banner(f"STAGE 1/3 train-baseline  SKIPPED — reusing {baseline_dir}")
    else:
        baseline_dir = base_out / "train-baseline"
        _banner(f"STAGE 1/3 train-baseline  →  {baseline_dir}")
        t0 = time.time()
        rv = train_baseline._run(_ns(
            train_fovs=args.train_fovs,
            val_fovs=args.val_fovs,
            model=args.baseline_model,
            knn_k=args.knn_k,
            out_dir=str(baseline_dir),
            max_cells=None,
        ))
        stage_t["train_baseline"] = time.time() - t0
        if rv:
            print(f"[fatal] train-baseline failed (rv={rv})")
            return int(rv)

    # ---------- Stage 2: train-segmentation ----------
    if args.skip_segmentation and args.seg_checkpoint:
        print("[fatal] choose at most one of --skip-segmentation, --seg-checkpoint")
        return 1
    if args.skip_segmentation:
        seg_ckpt: str | None = None
        _banner("STAGE 2/3 train-segmentation  SKIPPED — off-the-shelf cpsam")
    elif args.seg_checkpoint:
        if not Path(args.seg_checkpoint).exists():
            print(f"[fatal] --seg-checkpoint {args.seg_checkpoint} does not exist")
            return 1
        seg_ckpt = args.seg_checkpoint
        _banner(f"STAGE 2/3 train-segmentation  SKIPPED — reusing {seg_ckpt}")
    else:
        seg_dir = base_out / "train-segmentation"
        _banner(f"STAGE 2/3 train-segmentation  →  {seg_dir}")
        t0 = time.time()
        rv = train_segmentation._run(_ns(
            train_fovs=args.train_fovs,
            val_fovs=args.val_fovs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            exp_name=args.exp_name,
            out_dir=str(seg_dir),
            resume=args.resume,
            time_budget=args.time_budget,
            keep_best=args.keep_best,
        ))
        stage_t["train_segmentation"] = time.time() - t0
        if rv:
            print(f"[fatal] train-segmentation failed (rv={rv})")
            return int(rv)
        seg_ckpt = str(seg_dir / f"{args.exp_name}_final")
        if not Path(seg_ckpt).exists():
            print(f"[fatal] expected final checkpoint at {seg_ckpt}, not produced")
            return 1

    # ---------- Stage 3: infer-baseline ----------
    infer_dir = base_out / "infer-baseline"
    _banner(f"STAGE 3/3 infer-baseline  →  {infer_dir}")
    t0 = time.time()
    rv = infer_baseline._run(_ns(
        models_dir=str(baseline_dir),
        test_fovs=args.test_fovs,
        out_dir=str(infer_dir),
        device=args.device,
        cellpose_diameter=args.cellpose_diameter,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        include_spot_density=args.include_spot_density,
        spot_density_sigma=args.spot_density_sigma,
        seg_checkpoint=seg_ckpt,
    ))
    stage_t["infer_baseline"] = time.time() - t0
    if rv:
        print(f"[fatal] infer-baseline failed (rv={rv})")
        return int(rv)

    sub_csv = infer_dir / "submission.csv"
    wall = time.time() - wall_t0
    _banner(f"PIPELINE COMPLETE  ({wall:.1f}s wall)")
    print(f"  baseline dir : {baseline_dir}")
    print(f"  seg ckpt     : {seg_ckpt or '(off-the-shelf cpsam)'}")
    print(f"  submission   : {sub_csv}")
    print("\nstage timings:")
    for k, v in stage_t.items():
        print(f"  {k:<22} {v:8.1f}s")
    return 0


register(Task(
    name="pipeline",
    summary="Run the full phase-2 pipeline: train-baseline → train-segmentation → infer-baseline.",
    add_args=_add_args,
    run=_run,
    requirements={
        "gpu": True,
        "modal_gpu": "A10G",
        "modal_image": "cellpose",
        "modal_volume": "cell-seg-data",
        "modal_timeout": 12 * 3600,
        "hpc_partition": "gpu",
        "hpc_hours": 12.0,
        "hpc_gpus": 1,
    },
))
