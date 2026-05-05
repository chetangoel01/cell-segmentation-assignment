"""Fine-tune Cellpose-SAM on phase-2 train FOVs (101-150) using rasterized
z=2 GT polygons. Pattern adapted from `phase2/docs/codelab_13.py:522`.

The codelab fine-tunes on 2 FOVs (FOV_160 + FOV_107). We have all 50 phase-2
train FOVs and the ground-truth polygon CSV — we use them all to address the
"Cellpose-cyto3 over-segments 3× vs Allen GT" issue called out in the codelab
header (line 84).

Run:
    modal run phase2/modal/modal_train_cpsam_ft.py::run

Then download the fine-tuned weights:
    mkdir -p phase2/runs/cpsam_p2_ft_modal
    modal volume get cell-seg-phase2 trained/cpsam_p2_ft \\
        phase2/runs/cpsam_p2_ft_modal/
"""
from __future__ import annotations

import modal

app = modal.App("phase2-train-cpsam-ft")
data_vol = modal.Volume.from_name("cell-seg-phase2", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1")
    .pip_install(
        "numpy<2", "pandas", "scipy",
        "scikit-image", "shapely",
        "cellpose",  # latest, ships cpsam
        "torch",
    )
)

VOLUMES = {"/root/data": data_vol}


@app.function(image=image, gpu="A10G", timeout=10800, volumes=VOLUMES)
def train_cpsam_ft(
    train_fov_start: int = 101,
    train_fov_end: int = 151,        # exclusive — 50 FOVs (101..150)
    n_epochs: int = 100,
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    bsize: int = 256,
    weight_decay: float = 1e-5,
    out_subdir: str = "trained/cpsam_p2_ft",
    exclude_background_cells: bool = False,
) -> str:
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd

    DAPI_FRAMES = [6, 11, 16, 21, 26]
    POLYT_FRAMES = [5, 10, 15, 20, 25]
    IMAGE_SIZE = 2048
    PIXEL = 0.109

    DATA = Path("/root/data")
    OUT = DATA / out_subdir
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Load FOV metadata + GT polygons ──────────────────────────────────────
    fov_meta = pd.read_csv(DATA / "reference" / "fov_metadata.csv").set_index("fov")
    boundaries = pd.read_csv(DATA / "train" / "ground_truth" / "cell_boundaries_train.csv",
                              index_col=0)
    labels = pd.read_csv(DATA / "train" / "ground_truth" / "cell_labels_train.csv")

    def load_fov_images(fov: str):
        fov_id = fov.split("_", 1)[1].zfill(3) if fov.split("_", 1)[1].isdigit() else fov.split("_", 1)[1]
        epi = DATA / "train" / fov / f"Epi-750s5-635s5-545s1-473s5-408s5_{fov_id}.dax"
        raw = np.fromfile(str(epi), dtype=np.uint16)
        n_frames = raw.size // (IMAGE_SIZE * IMAGE_SIZE)
        raw = raw.reshape(n_frames, IMAGE_SIZE, IMAGE_SIZE)
        return raw[DAPI_FRAMES], raw[POLYT_FRAMES]

    def rasterize_gt_mask(fov: str) -> np.ndarray:
        """z=2 polygon rasterization, matching codelab_13.py:rasterize_gt_mask."""
        from skimage.draw import polygon as sk_polygon
        ox = float(fov_meta.loc[fov, "fov_x"])
        oy = float(fov_meta.loc[fov, "fov_y"])
        ps = float(fov_meta.loc[fov, "pixel_size"])

        labels_fov = labels[labels.fov == fov].set_index("cell_id")
        if exclude_background_cells and "class_label" in labels_fov.columns:
            labels_fov = labels_fov[labels_fov["class_label"] != "background"]
        cells_here = boundaries.loc[boundaries.index.astype(str).isin(labels_fov.index.astype(str))]

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
        next_id = 1
        for cid, row in cells_here.iterrows():
            xs = row.get("boundaryX_z2", "")
            ys = row.get("boundaryY_z2", "")
            if not isinstance(xs, str) or not isinstance(ys, str):
                continue
            try:
                xs_arr = np.array([float(v) for v in xs.split(",") if v.strip()])
                ys_arr = np.array([float(v) for v in ys.split(",") if v.strip()])
            except ValueError:
                continue
            if len(xs_arr) < 3:
                continue
            # MERFISH coord convention (CLAUDE.md):
            #   image_row = 2048 - (global_x - fov_x) / pixel_size
            #   image_col =        (global_y - fov_y) / pixel_size
            r = IMAGE_SIZE - 1 - (xs_arr - ox) / ps
            c = (ys_arr - oy) / ps
            r = np.clip(r, 0, IMAGE_SIZE - 1)
            c = np.clip(c, 0, IMAGE_SIZE - 1)
            rr, cc = sk_polygon(r, c, shape=mask.shape)
            mask[rr, cc] = next_id
            next_id += 1
        return mask

    train_fovs = [f"FOV_{i:03d}" for i in range(train_fov_start, train_fov_end)]
    print(f"Building train pairs for {len(train_fovs)} FOVs")

    train_imgs, train_masks = [], []
    for fov in train_fovs:
        try:
            dapi, polyt = load_fov_images(fov)
            dapi2d = dapi.max(axis=0).astype(np.float32)
            polyt2d = polyt.max(axis=0).astype(np.float32)
            # Cellpose channel ordering: (C, H, W) with [polyT, DAPI] for cyto/nuc
            img = np.stack([polyt2d, dapi2d], axis=0)
            gt_mask = rasterize_gt_mask(fov)
            n_cells = int(gt_mask.max())
            if n_cells < 5:
                print(f"  {fov}: only {n_cells} cells in GT — skip")
                continue
            train_imgs.append(img)
            train_masks.append(gt_mask)
            print(f"  {fov}: {n_cells} cells")
        except Exception as e:
            print(f"  {fov}: skip ({e})")
            continue

    print(f"\nTrain set: {len(train_imgs)} FOVs")

    # ── Cellpose-SAM fine-tune ──────────────────────────────────────────────
    import torch
    from cellpose import train as cp_train
    from cellpose.models import CellposeModel

    cp_ft = CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")
    print(f"  device: cuda? {torch.cuda.is_available()}")

    t0 = time.time()
    cp_train.train_seg(
        cp_ft.net,
        train_data=train_imgs,
        train_labels=train_masks,
        channel_axis=0,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        bsize=bsize,                # cpsam REQUIRES 256
        weight_decay=weight_decay,
        save_path=str(OUT),
        model_name="cpsam_p2_ft",
    )
    elapsed = time.time() - t0
    print(f"\nFine-tune done in {elapsed:.1f}s")

    # Persist on volume
    data_vol.commit()

    # Find the saved weights file
    saved = list(OUT.glob("**/cpsam_p2_ft*"))
    print("Saved files:")
    for p in saved:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return str(OUT)


@app.local_entrypoint()
def run(n_epochs: int = 100, exclude_bg: bool = False, out_subdir: str = "trained/cpsam_p2_ft"):
    out_path = train_cpsam_ft.remote(
        n_epochs=n_epochs,
        exclude_background_cells=exclude_bg,
        out_subdir=out_subdir,
    )
    print(f"Trained: {out_path}")
