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


def _build_pair(fov: str, meta_row, cells_df, spots_df) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (image (3, H, W) float32, mask (H, W) int32) or None if skipped."""
    from phase2.src import io
    try:
        dapi, polyt = io.load_fov_images(fov, split="train")
    except Exception as exc:
        print(f"  [skip] {fov}: load failed ({exc})")
        return None
    fov_spots = spots_df[spots_df["fov"] == fov]
    density = _spot_density(fov_spots, sigma=8.0)
    img = np.stack([
        polyt.max(axis=0).astype(np.float32),
        dapi.max(axis=0).astype(np.float32),
        density,
    ], axis=0)
    mask = _polygons_to_mask(cells_df, float(meta_row["fov_x"]), float(meta_row["fov_y"]))
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
    train_imgs: list[np.ndarray] = []
    train_masks: list[np.ndarray] = []
    for fov in train_fovs:
        if fov not in meta.index:
            print(f"  [skip] {fov}: not in fov_metadata.csv")
            continue
        pair = _build_pair(fov, meta.loc[fov], cells, spots)
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
            pair = _build_pair(fov, meta.loc[fov], cells, spots)
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
        model = cp_models.CellposeModel(gpu=gpu, device=device)  # default = cpsam
    stages["load_cellpose"] = time.time() - t0
    print(f"  ({stages['load_cellpose']:.2f}s)")

    epoch = completed
    last_ckpt = latest_ckpt
    train_chunks: list[float] = []
    while epoch < args.epochs:
        chunk = min(CHUNK_EPOCHS, args.epochs - epoch)
        target = epoch + chunk
        ckpt_name = f"{args.exp_name}_ep{target:03d}"
        print(f"\nepochs {epoch+1}-{target} / {args.epochs}  → {ckpt_name}")
        t0 = time.time()
        last_ckpt, train_losses, *_ = cp_train.train_seg(
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
            model_name=ckpt_name,
        )
        chunk_t = time.time() - t0
        train_chunks.append(chunk_t)
        last_ckpt = str(last_ckpt)
        epoch = target
        avg_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
        print(f"  chunk {chunk_t:.1f}s   avg loss={avg_loss:.4f}   ckpt={last_ckpt}")
        state_file.write_text(json.dumps({"completed_epochs": epoch,
                                          "latest_checkpoint": last_ckpt,
                                          "avg_loss": avg_loss}))
        # Re-instantiate so the next chunk uses the saved weights (matches phase 1).
        if epoch < args.epochs:
            model = cp_models.CellposeModel(gpu=gpu, pretrained_model=last_ckpt, device=device)

    if last_ckpt:
        final = out_dir / f"{args.exp_name}_final"
        if Path(last_ckpt).resolve() != final.resolve():
            shutil.copy(last_ckpt, final)
        print(f"\nfinal model → {final}")

    stages["train_total"] = sum(train_chunks)
    stages["wall_total"] = time.time() - wall_t0
    summary = {
        "exp_name": args.exp_name,
        "device": str(device),
        "n_train_pairs": len(train_imgs),
        "n_val_pairs": len(val_imgs),
        "epochs_completed": epoch,
        "chunk_seconds": train_chunks,
        "stage_timings_s": stages,
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
