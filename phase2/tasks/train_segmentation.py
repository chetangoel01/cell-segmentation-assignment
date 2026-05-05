"""Cellpose-SAM fine-tuning task on MERFISH FOVs (phase 1 or phase 2 data).

Mirrors phase 1's `train.py` shape but uses Cellpose 4 (cpsam) and the phase 2
runner contract. Builds (image, mask) training pairs from:

    image: (3, 2048, 2048) — [polyT_max, DAPI_max, spot_density(σ=8)]
    mask : (2048, 2048) int32 — drawn from cell_boundaries_train.csv polygons

then calls cellpose.train.train_seg in CHUNK_EPOCHS-sized chunks with
checkpoint save/resume so a long HPC run can survive preemption.

Honors MERFISH_DATA_ROOT — point at phase1/data or phase2/data as needed.
The task itself is data-source-agnostic.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.tasks import Task, register

PIXEL_SIZE_UM = 0.109
IMAGE_SIZE = 2048
CHUNK_EPOCHS = 5  # checkpoint every N epochs


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-fovs", required=True,
                   help="Comma-separated train FOVs (e.g. FOV_001,FOV_002,...).")
    p.add_argument("--val-fovs", default="",
                   help="Comma-separated val FOVs (held out — for ARI eval).")
    p.add_argument("--epochs", type=int, default=300,
                   help="Total training epochs (default 300).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--exp-name", default="cpsam_finetune",
                   help="Experiment name (controls model save dir).")
    p.add_argument("--out-dir", default=None,
                   help="Override output dir. Default: phase2/runs/<ts>-train-seg-<exp>/.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --out-dir if present.")
    p.add_argument("--time-budget", default=None,
                   help="Wall-clock limit; stop after the next 5-epoch checkpoint "
                        "once exceeded. Accepts e.g. '2h30m', '150m', '9000s', '2.5h'. "
                        "Whichever of --epochs or --time-budget hits first wins.")
    p.add_argument("--keep-best", type=int, default=2,
                   help="Keep only the N best chunk checkpoints (lowest val loss "
                        "if val FOVs supplied, else lowest train loss). 0 = keep all. "
                        "Default 2. The latest checkpoint is also retained transiently "
                        "for --resume continuity.")
    p.add_argument("--n-channels", type=int, default=3, choices=(2, 3),
                   help="Number of input channels: 2 = [polyT, DAPI] (codelab "
                        "recipe), 3 = [polyT, DAPI, spot_density σ=8] (phase-1 "
                        "recipe). Default 3 for phase-1 compat.")
    p.add_argument("--bsize", type=int, default=256,
                   help="Cellpose train_seg bsize. cpsam REQUIRES 256 per the "
                        "codelab; the default 224 crashes.")
    p.add_argument("--include-phase1", action="store_true",
                   help="Stack phase-1 train FOVs (with their polygon GT) onto "
                        "phase-2 training set. Reads phase1/data/ if available.")
    p.add_argument("--z-planes", default="2",
                   help="Comma-separated z-planes whose polygons to UNION into the "
                        "training mask. Default '2' (single mid z-slice — old behavior). "
                        "Use '0,1,2,3,4' for full 3D-extent UNION mask (matches "
                        "max-projection input). Major training-inference alignment fix.")
    p.add_argument("--base-model", default="cpsam",
                   choices=("cpsam", "cyto3", "cyto2", "nuclei"),
                   help="Pretrained Cellpose backbone to fine-tune from (fresh "
                        "training only — ignored on --resume). cpsam (Cellpose 4 "
                        "default) is SAM-based, 3-channel, big. cyto3/cyto2 are "
                        "membrane-aware (target full cell extent — counters "
                        "over-segmentation from nuclei-trained models). nuclei "
                        "is the DAPI single-channel baseline. Default cpsam.")


def _parse_duration(s: str | None) -> float | None:
    """Parse '2h30m', '150m', '9000s', '2.5h', or a bare float (seconds) → seconds."""
    if s is None:
        return None
    s = s.strip().lower()
    if not s:
        return None
    # Bare float = seconds.
    try:
        return float(s)
    except ValueError:
        pass
    total = 0.0
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch in ("h", "m", "s"):
            if not num:
                raise ValueError(f"bad duration {s!r}")
            mult = {"h": 3600.0, "m": 60.0, "s": 1.0}[ch]
            total += float(num) * mult
            num = ""
        elif ch.isspace():
            continue
        else:
            raise ValueError(f"bad duration {s!r} (unexpected {ch!r})")
    if num:
        # trailing number with no unit — assume seconds
        total += float(num)
    if total <= 0:
        raise ValueError(f"duration must be positive: {s!r}")
    return total


def _fmt_duration(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _pick_device(spec: str):
    import torch
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps"), True
        if torch.cuda.is_available():
            return torch.device("cuda"), True
        return torch.device("cpu"), False
    if spec == "mps":
        return torch.device("mps"), True
    if spec == "cuda":
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _polygons_to_mask(cells_df: pd.DataFrame, fov_x: float, fov_y: float,
                      z_planes=(2,)) -> np.ndarray:
    """Render cell-boundary polygons → integer mask (background=0, cells=1..N)."""
    from skimage.draw import polygon as draw_polygon

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    cell_int = 1
    fov_x_max = fov_x + IMAGE_SIZE * PIXEL_SIZE_UM
    fov_y_max = fov_y + IMAGE_SIZE * PIXEL_SIZE_UM

    def _px(xs_um, ys_um):
        rows = np.clip(np.array([IMAGE_SIZE - 1 - (x - fov_x) / PIXEL_SIZE_UM for x in xs_um]),
                       0, IMAGE_SIZE - 1)
        cols = np.clip(np.array([(y - fov_y) / PIXEL_SIZE_UM for y in ys_um]),
                       0, IMAGE_SIZE - 1)
        return rows, cols

    for _cid, row in cells_df.iterrows():
        # Spatial pre-filter on z=2 centroid to skip cells that aren't in this FOV.
        xs_ref = row.get("boundaryX_z2", "")
        ys_ref = row.get("boundaryY_z2", "")
        if not isinstance(xs_ref, str) or not isinstance(ys_ref, str):
            continue
        try:
            _xs = [float(v) for v in xs_ref.split(",") if v.strip()]
            _ys = [float(v) for v in ys_ref.split(",") if v.strip()]
        except ValueError:
            continue
        if not _xs or not _ys:
            continue
        cx = sum(_xs) / len(_xs)
        cy = sum(_ys) / len(_ys)
        if not (fov_x <= cx < fov_x_max and fov_y <= cy < fov_y_max):
            continue
        drew = False
        for z in z_planes:
            xs = row.get(f"boundaryX_z{z}", "")
            ys = row.get(f"boundaryY_z{z}", "")
            if not isinstance(xs, str) or not isinstance(ys, str):
                continue
            try:
                xs_um = [float(v) for v in xs.split(",") if v.strip()]
                ys_um = [float(v) for v in ys.split(",") if v.strip()]
            except ValueError:
                continue
            if len(xs_um) < 3:
                continue
            rr, cc = draw_polygon(*_px(xs_um, ys_um), shape=(IMAGE_SIZE, IMAGE_SIZE))
            mask[rr, cc] = cell_int
            drew = True
        if drew:
            cell_int += 1
    return mask


def _load_fov_from_dir(fov_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mirror phase2.src.io.load_fov_images but for an arbitrary FOV directory
    (used for phase-1 data which lives under a different root)."""
    from phase2.src.io import load_dax, get_dapi_stack, get_polyt_stack
    candidates = sorted(fov_dir.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))
    if not candidates:
        existing = sorted(p.name for p in fov_dir.glob("*.dax"))[:5]
        raise FileNotFoundError(f"No Epi file in {fov_dir}; saw {existing}")
    raw = load_dax(candidates[0])
    return get_dapi_stack(raw), get_polyt_stack(raw)


def _spot_density(spots_fov: pd.DataFrame, sigma: float = 8.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    d = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    rows = spots_fov["image_row"].to_numpy().astype(int).clip(0, IMAGE_SIZE - 1)
    cols = spots_fov["image_col"].to_numpy().astype(int).clip(0, IMAGE_SIZE - 1)
    np.add.at(d, (rows, cols), 1)
    d = gaussian_filter(d, sigma=sigma)
    if d.max() > 0:
        d = d / d.max() * 65535.0
    return d.astype(np.float32)


def _build_pair(fov: str, meta_row, cells_df, spots_df,
                n_channels: int = 3, data_root_override: Path | None = None,
                z_planes: tuple = (2,),
                ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (image (C, H, W) float32, mask (H, W) int32) or None if skipped.
    C = 2 (polyT, DAPI) for codelab recipe, 3 (+spot_density) for phase-1 recipe."""
    from phase2.src import io
    try:
        if data_root_override is not None:
            # phase-1 data lives under a different root; load images from that path.
            from phase2.src.io import load_fov_images
            # Patch in an alternate path: phase1/data/train/<FOV>/
            fov_dir = data_root_override / "train" / fov
            if not fov_dir.exists():
                print(f"  [skip] {fov}: {fov_dir} not found")
                return None
            # io.load_fov_images is hardcoded; use a manual loader for the override.
            dapi, polyt = _load_fov_from_dir(fov_dir)
        else:
            dapi, polyt = io.load_fov_images(fov, split="train")
    except Exception as exc:
        print(f"  [skip] {fov}: load failed ({exc})")
        return None
    channels = [
        polyt.max(axis=0).astype(np.float32),
        dapi.max(axis=0).astype(np.float32),
    ]
    if n_channels == 3:
        fov_spots = spots_df[spots_df["fov"] == fov]
        channels.append(_spot_density(fov_spots, sigma=8.0))
    img = np.stack(channels, axis=0)
    mask = _polygons_to_mask(cells_df, float(meta_row["fov_x"]), float(meta_row["fov_y"]),
                             z_planes=z_planes)
    if mask.max() == 0:
        print(f"  [skip] {fov}: no cells in mask")
        return None
    print(f"  {fov}: {int(mask.max())} cells, image dtype={img.dtype}")
    return img, mask


def _run(args: argparse.Namespace) -> int:
    import torch
    from cellpose import models as cp_models
    from cellpose import train as cp_train
    from phase2.src import io

    stages: dict[str, float] = {}
    wall_t0 = time.time()

    train_fovs = [f.strip() for f in args.train_fovs.split(",") if f.strip()]
    val_fovs = [f.strip() for f in args.val_fovs.split(",") if f.strip()]
    print(f"data_root: {io.data_root()}")
    print(f"train FOVs: {len(train_fovs)} ({train_fovs[:3]} … {train_fovs[-3:] if len(train_fovs) > 3 else ''})")
    print(f"val FOVs:   {val_fovs}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "runs" /
        f"{time.strftime('%Y%m%d-%H%M%S')}-train-seg-{args.exp_name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    print("\nloading metadata + ground-truth tables …")
    t0 = time.time()
    meta = pd.read_csv(io.reference_dir() / "fov_metadata.csv").set_index("fov")
    cells = pd.read_csv(io.ground_truth_dir() / "cell_boundaries_train.csv", index_col=0)
    spots = pd.read_csv(io.ground_truth_dir() / "spots_train.csv")
    stages["load_tables"] = time.time() - t0
    print(f"  cells={len(cells):,} spots={len(spots):,}  ({stages['load_tables']:.2f}s)")

    print("\nbuilding training pairs (image + GT mask per FOV) …")
    t0 = time.time()
    z_planes_tuple = tuple(int(z.strip()) for z in str(args.z_planes).split(",") if z.strip())
    print(f"\nz-planes for mask UNION: {z_planes_tuple}")
    train_imgs: list[np.ndarray] = []
    train_masks: list[np.ndarray] = []
    for fov in train_fovs:
        if fov not in meta.index:
            print(f"  [skip] {fov}: not in fov_metadata.csv")
            continue
        pair = _build_pair(fov, meta.loc[fov], cells, spots,
                            n_channels=args.n_channels,
                            z_planes=z_planes_tuple)
        if pair is not None:
            train_imgs.append(pair[0])
            train_masks.append(pair[1])

    # Optional phase-1 stack: same Epi file format, separate ground truth tables.
    if getattr(args, "include_phase1", False):
        p1_root = Path(__file__).resolve().parents[2] / "phase1" / "data"
        if not p1_root.exists():
            print(f"  [skip phase1] {p1_root} not found")
        else:
            print(f"\nloading phase-1 ground truth from {p1_root} …")
            p1_meta = meta  # phase-1 FOVs aren't in phase-2 meta; need their own
            try:
                p1_meta = pd.read_csv(p1_root / "reference" / "fov_metadata.csv").set_index("fov")
            except FileNotFoundError:
                print(f"  [warn] phase-1 fov_metadata.csv not found; using phase-2 (likely wrong fov_x/fov_y)")
            p1_cells = pd.read_csv(p1_root / "train" / "ground_truth" / "cell_boundaries_train.csv", index_col=0)
            p1_spots = pd.read_csv(p1_root / "train" / "ground_truth" / "spots_train.csv")
            print(f"  phase-1 cells={len(p1_cells):,} spots={len(p1_spots):,}")
            # Default phase-1 train FOVs: 001-035 (matches phase-1 split)
            p1_train_fovs = [f"FOV_{i:03d}" for i in range(1, 36)]
            for fov in p1_train_fovs:
                if fov not in p1_meta.index:
                    print(f"  [skip phase1] {fov}: not in phase-1 fov_metadata.csv")
                    continue
                pair = _build_pair(fov, p1_meta.loc[fov], p1_cells, p1_spots,
                                    n_channels=args.n_channels,
                                    data_root_override=p1_root,
                                    z_planes=z_planes_tuple)
                if pair is not None:
                    train_imgs.append(pair[0])
                    train_masks.append(pair[1])

    stages["build_train"] = time.time() - t0
    print(f"  built {len(train_imgs)} train pairs  ({stages['build_train']:.2f}s)")
    if not train_imgs:
        print("[fatal] no training pairs built")
        return 1

    val_imgs: list[np.ndarray] = []
    val_masks: list[np.ndarray] = []
    if val_fovs:
        print("\nbuilding validation pairs …")
        t0 = time.time()
        for fov in val_fovs:
            if fov not in meta.index:
                continue
            pair = _build_pair(fov, meta.loc[fov], cells, spots,
                                n_channels=args.n_channels,
                                z_planes=z_planes_tuple)
            if pair is not None:
                val_imgs.append(pair[0])
                val_masks.append(pair[1])
        stages["build_val"] = time.time() - t0
        print(f"  built {len(val_imgs)} val pairs  ({stages['build_val']:.2f}s)")

    device, gpu = _pick_device(args.device)
    print(f"\ndevice: {device}  (gpu={gpu})")

    # Resume detection.
    state_file = out_dir / "train_state.json"
    completed = 0
    latest_ckpt: str | None = None
    # Each entry: {"path": str, "epoch": int, "metric": float, "metric_name": str}
    kept_best: list[dict] = []
    if args.resume and state_file.exists():
        try:
            st = json.loads(state_file.read_text())
            completed = int(st.get("completed_epochs", 0))
            latest_ckpt = st.get("latest_checkpoint")
            if latest_ckpt and not Path(latest_ckpt).exists():
                print(f"  state file references missing ckpt {latest_ckpt}; restarting")
                completed = 0
                latest_ckpt = None
            else:
                print(f"  resuming from epoch {completed}/{args.epochs}: {latest_ckpt}")
                # Restore best-tracking state (drop entries whose files were pruned).
                for b in st.get("kept_best", []):
                    if Path(b.get("path", "")).exists():
                        kept_best.append(b)
                if kept_best:
                    print(f"  restored kept_best ({len(kept_best)}): "
                          + ", ".join(f"{Path(b['path']).name}@{b['metric']:.4f}"
                                      for b in kept_best))
        except Exception as e:
            print(f"  state file corrupt ({e}); restarting")

    if completed >= args.epochs:
        print("training already complete — nothing to do")
        return 0

    print("\ninstantiating Cellpose-SAM …")
    t0 = time.time()
    if latest_ckpt:
        model = cp_models.CellposeModel(gpu=gpu, pretrained_model=latest_ckpt, device=device)
    else:
        base = getattr(args, "base_model", "cpsam")
        print(f"  base model: {base}")
        model = cp_models.CellposeModel(gpu=gpu, pretrained_model=base, device=device)
    stages["load_cellpose"] = time.time() - t0
    print(f"  ({stages['load_cellpose']:.2f}s)")

    time_budget = _parse_duration(args.time_budget)
    train_t0 = time.time()
    if time_budget:
        print(f"\ntime budget: {_fmt_duration(time_budget)} "
              f"(wall-clock, checked at each {CHUNK_EPOCHS}-epoch checkpoint)")

    epoch = completed
    last_ckpt = latest_ckpt
    train_chunks: list[float] = []
    stopped_on_time = False
    while epoch < args.epochs:
        chunk = min(CHUNK_EPOCHS, args.epochs - epoch)
        target = epoch + chunk
        ckpt_name = f"{args.exp_name}_ep{target:03d}"
        print(f"\nepochs {epoch+1}-{target} / {args.epochs}  → {ckpt_name}")
        t0 = time.time()
        cp_returns = cp_train.train_seg(
            model.net,
            train_data=train_imgs,
            train_labels=train_masks,
            test_data=val_imgs or None,
            test_labels=val_masks or None,
            channel_axis=0,
            save_path=str(out_dir),
            n_epochs=chunk,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            bsize=args.bsize,
            model_name=ckpt_name,
        )
        last_ckpt = str(cp_returns[0])
        train_losses = cp_returns[1] if len(cp_returns) > 1 else []
        test_losses = cp_returns[2] if len(cp_returns) > 2 else None
        chunk_t = time.time() - t0
        train_chunks.append(chunk_t)
        epoch = target

        train_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
        if test_losses is not None and len(test_losses):
            val_loss = float(np.mean(test_losses))
            metric, metric_name = val_loss, "val_loss"
        else:
            val_loss = None
            metric, metric_name = train_loss, "train_loss"
        print(f"  chunk {chunk_t:.1f}s  train_loss={train_loss:.4f}"
              + (f"  val_loss={val_loss:.4f}" if val_loss is not None else "")
              + f"  ckpt={last_ckpt}")

        # Update best-N tracking, prune losers.
        kept_best.append({"path": last_ckpt, "epoch": epoch,
                          "metric": metric, "metric_name": metric_name,
                          "train_loss": train_loss, "val_loss": val_loss})
        if args.keep_best > 0:
            kept_best.sort(key=lambda b: b["metric"])
            top = kept_best[: args.keep_best]
            top_paths = {b["path"] for b in top}
            # Always keep `last_ckpt` for resume, even if not in top-N.
            keep_paths = top_paths | {last_ckpt}
            pruned = [b for b in kept_best if b["path"] not in keep_paths]
            for b in pruned:
                p = Path(b["path"])
                if p.exists():
                    try:
                        p.unlink()
                        print(f"  pruned {p.name}  ({b['metric_name']}={b['metric']:.4f})")
                    except OSError as e:
                        print(f"  prune failed for {p}: {e}")
            kept_best = [b for b in kept_best if b["path"] in keep_paths]
            kept_best.sort(key=lambda b: b["metric"])

        state_file.write_text(json.dumps({"completed_epochs": epoch,
                                          "latest_checkpoint": last_ckpt,
                                          "train_loss": train_loss,
                                          "val_loss": val_loss,
                                          "metric_name": metric_name,
                                          "kept_best": kept_best}))
        if time_budget is not None:
            train_elapsed = time.time() - train_t0
            if train_elapsed >= time_budget:
                print(f"  time budget reached "
                      f"({_fmt_duration(train_elapsed)} ≥ {_fmt_duration(time_budget)}) "
                      f"— stopping at epoch {epoch}/{args.epochs}")
                stopped_on_time = True
                break
        # Re-instantiate so the next chunk uses the saved weights (matches phase 1).
        if epoch < args.epochs:
            model = cp_models.CellposeModel(gpu=gpu, pretrained_model=last_ckpt, device=device)

    # End-of-run: pick the actual best, copy to _final, prune the latest if it
    # was kept only for resume continuity.
    best_entry = min(kept_best, key=lambda b: b["metric"]) if kept_best else None
    if best_entry is not None:
        final = out_dir / f"{args.exp_name}_final"
        best_path = Path(best_entry["path"])
        if best_path.resolve() != final.resolve():
            shutil.copy(best_path, final)
        print(f"\nbest model → {final}  "
              f"({best_entry['metric_name']}={best_entry['metric']:.4f}, "
              f"epoch {best_entry['epoch']})")
        # Prune the latest if it wasn't in the top-N (it was retained only for resume).
        if args.keep_best > 0 and last_ckpt and last_ckpt != best_entry["path"]:
            top_paths = {b["path"] for b in sorted(kept_best, key=lambda b: b["metric"])[: args.keep_best]}
            if last_ckpt not in top_paths:
                p = Path(last_ckpt)
                if p.exists():
                    try:
                        p.unlink()
                        print(f"  pruned {p.name}  (latest, not in top-{args.keep_best})")
                        kept_best = [b for b in kept_best if b["path"] != last_ckpt]
                    except OSError as e:
                        print(f"  prune failed for {p}: {e}")

    stages["train_total"] = sum(train_chunks)
    stages["wall_total"] = time.time() - wall_t0
    summary = {
        "exp_name": args.exp_name,
        "device": str(device),
        "n_train_pairs": len(train_imgs),
        "n_val_pairs": len(val_imgs),
        "epochs_completed": epoch,
        "epochs_requested": args.epochs,
        "stopped_on_time_budget": stopped_on_time,
        "time_budget_seconds": time_budget,
        "chunk_seconds": train_chunks,
        "stage_timings_s": stages,
        "keep_best": args.keep_best,
        "kept_best": kept_best,
        "best_checkpoint": best_entry,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== stage timings ===")
    for k, v in stages.items():
        print(f"  {k:<18} {v:8.2f}s")
    if train_chunks:
        print(f"\n  per-chunk ({CHUNK_EPOCHS} ep): mean={np.mean(train_chunks):.1f}s   "
              f"per-epoch ≈ {np.mean(train_chunks)/CHUNK_EPOCHS:.1f}s")
    return 0


register(Task(
    name="train-segmentation",
    summary="Fine-tune Cellpose-SAM on MERFISH FOVs (image+GT-polygon mask pairs).",
    add_args=_add_args,
    run=_run,
    requirements={
        "gpu": True,
        "modal_gpu": "A10G",
        "modal_image": "cellpose",
        "modal_volume": "cell-seg-data",  # phase 1 lives here
        "modal_timeout": 8 * 3600,
        "hpc_partition": "gpu",
        "hpc_hours": 6.0,
        "hpc_gpus": 1,
    },
))
