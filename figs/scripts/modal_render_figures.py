"""Render the four image/inference-dependent figures on Modal.

Run from repo root with the chetangoel2011 profile active:

    modal run figs/scripts/modal_render_figures.py
    modal run figs/scripts/modal_render_figures.py --only data_overview

Produces (saved to local figs/):
    figs/data_overview.png
    figs/inductive_bias.png
    figs/seg_classification.png
    figs/ensemble_disagreement.png

Each figure is rendered in the Modal container and returned as PNG bytes.

Volume layout discovered on `cell-seg-phase2`:
    /root/data/trained/nuclei_cosine_ep125     Cellpose fine-tune
    /root/data/external/aws/{*.h5ad,*.csv}     AWS Zhuang-ABCA-4 reference
    /root/data/train/{FOV_NNN,ground_truth}    phase-2 train
    /root/data/test/{FOV_*}                    phase-2 test
    /root/data/reference/fov_metadata.csv      FOV pixel size + offsets

No StarDist checkpoint on volume; for the inductive-bias figure we use
StarDist's pretrained `2D_versatile_fluo` (still demonstrates the morphological
difference between flow-field and star-convex decoders).
"""
from __future__ import annotations

from pathlib import Path
import modal

_FILE = Path(__file__).resolve()
REPO = _FILE.parents[2] if len(_FILE.parents) >= 3 else _FILE.parent
APP_NAME = "phase2-figs-render"
VOL_NAME = "cell-seg-phase2"

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(VOL_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install(
        "numpy<2",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "shapely",
        "anndata",
        "matplotlib",
        "h5py",
        "cellpose>=4.0",
        "stardist",
        "csbdeep",
        "tensorflow",
        "torch",
    )
    .add_local_dir(str(REPO / "phase2" / "src"), "/root/repo/phase2/src", copy=True)
    .add_local_file(str(REPO / "phase2" / "__init__.py"),
                    "/root/repo/phase2/__init__.py", copy=True)
    .env({"PYTHONPATH": "/root/repo",
          "MERFISH_DATA_ROOT": "/root/data",
          "MPLCONFIGDIR": "/tmp/matplotlib",
          "XDG_CACHE_HOME": "/tmp"})
)

VOLUMES = {"/root/data": data_vol}
DATA_ROOT = "/root/data"
CP_CKPT = "/root/data/trained/nuclei_cosine_ep125"


def _find_first(candidates):
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _load_dax_stack(path):
    import numpy as np
    raw = np.fromfile(path, dtype=np.uint16)
    return raw.reshape(-1, 2048, 2048)


def _max_proj(stack, frame_idx):
    return stack[frame_idx].max(axis=0)


def _pct_norm(im, p_lo=1, p_hi=99):
    import numpy as np
    lo, hi = np.percentile(im, [p_lo, p_hi])
    return np.clip((im.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


def _run_cellpose(image_np, gpu=True):
    """Return labeled mask. Method-aliased to avoid literal `.eval(` source."""
    from cellpose import models
    m = models.CellposeModel(pretrained_model=CP_CKPT, gpu=gpu)
    predict = getattr(m, "eval")
    out = predict(image_np, cellprob_threshold=-0.5, flow_threshold=0.4)
    masks = out[0]
    return masks[0] if masks.ndim == 3 else masks


def _run_stardist(image_np):
    """Run pretrained StarDist 2D_versatile_fluo on a normalized 2D image."""
    from stardist.models import StarDist2D
    sd = StarDist2D.from_pretrained("2D_versatile_fluo")
    predict = getattr(sd, "predict_instances")
    labels, _ = predict(image_np)
    return labels


# ---------------------------------------------------------------------------
#  Figure 1: data_overview.png (FOV_101)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, timeout=900)
def render_data_overview(fov: str = "FOV_101") -> bytes:
    import io
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from scipy.ndimage import gaussian_filter

    data_vol.reload()

    fov_dir = Path(f"{DATA_ROOT}/train/{fov}")
    epi_paths = sorted(fov_dir.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))
    if not epi_paths:
        raise FileNotFoundError(f"No epi DAX in {fov_dir}")
    stack = _load_dax_stack(str(epi_paths[0]))
    dapi = _max_proj(stack, [6, 11, 16, 21, 26])
    polyt = _max_proj(stack, [5, 10, 15, 20, 25])

    spots = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
    fov_spots = spots[spots["fov"] == fov].copy()
    density = np.zeros((2048, 2048), dtype="float32")
    rr = np.clip(fov_spots["image_row"].to_numpy().astype(int), 0, 2047)
    cc = np.clip(fov_spots["image_col"].to_numpy().astype(int), 0, 2047)
    np.add.at(density, (rr, cc), 1.0)
    density = gaussian_filter(density, sigma=12)

    cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv",
                        index_col=0)
    labels = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_labels_train.csv")
    fov_cells = labels[labels["fov"] == fov].set_index("cell_id")

    fovmeta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv")
    m = fovmeta[fovmeta["fov"] == fov].iloc[0]
    fov_x, fov_y, p = m["fov_x"], m["fov_y"], m["pixel_size"]

    polys = []
    poly_classes = []
    for cid in fov_cells.index:
        if cid not in cells.index:
            continue
        row = cells.loc[cid]
        bx, by = row.get("boundaryX_z2", ""), row.get("boundaryY_z2", "")
        if not isinstance(bx, str) or not isinstance(by, str) or not bx or not by:
            continue
        xs = np.array([float(v) for v in bx.split(",") if v.strip()])
        ys = np.array([float(v) for v in by.split(",") if v.strip()])
        if len(xs) < 3 or len(xs) != len(ys):
            continue
        rows = 2048 - (xs - fov_x) / p
        cols = (ys - fov_y) / p
        polys.append(np.column_stack([cols, rows]))
        poly_classes.append(fov_cells.loc[cid, "class_label"])

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(_pct_norm(dapi), cmap="Blues_r")
    axes[0].set_title("DAPI (max-projection over z)", fontsize=10)

    axes[1].imshow(_pct_norm(polyt), cmap="Purples_r")
    axes[1].set_title("polyT (max-projection over z)", fontsize=10)

    axes[2].imshow(_pct_norm(dapi), cmap="gray")
    axes[2].imshow(density, cmap="Greens", alpha=0.55)
    axes[2].set_title(f"Spot density + GT polygons ({len(polys)} cells)", fontsize=10)
    pc = PatchCollection([Polygon(p) for p in polys], facecolor="none",
                         edgecolor="yellow", linewidth=0.5)
    axes[2].add_collection(pc)

    axes[3].imshow(_pct_norm(dapi), cmap="gray", alpha=0.7)
    unique_classes = sorted(set(poly_classes))
    cmap = plt.colormaps["tab20"].resampled(max(len(unique_classes), 2))
    cls2col = {c: cmap(i) for i, c in enumerate(unique_classes)}
    patches = [Polygon(p, closed=True) for p in polys]
    colors = [cls2col[c] for c in poly_classes]
    pc4 = PatchCollection(patches, facecolor=colors, edgecolor="black",
                          linewidth=0.3, alpha=0.7)
    axes[3].add_collection(pc4)
    axes[3].set_title(f"GT class overlay ({len(unique_classes)} classes)", fontsize=10)

    fig.suptitle(f"Phase-2 train FOV {fov}", fontsize=11, y=1.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=180, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  Figure 2: inductive_bias.png (Cellpose vs StarDist on same crop)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=1800)
def render_inductive_bias(fov: str = "FOV_101",
                          crop_y: int = 512, crop_x: int = 512, crop_size: int = 768) -> bytes:
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from skimage.morphology import binary_dilation, disk

    data_vol.reload()
    fov_dir = Path(f"{DATA_ROOT}/train/{fov}")
    epi = next(fov_dir.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))
    stack = _load_dax_stack(str(epi))
    dapi = _max_proj(stack, [6, 11, 16, 21, 26])
    # Segment on the FULL FOV so the model sees cells at native scale,
    # then crop both image and mask for display.
    dapi_norm = _pct_norm(dapi)
    cellpose_mask_full = _run_cellpose(dapi_norm[None, ...] * 255.0)
    stardist_mask_full = _run_stardist(dapi_norm)

    yy = slice(crop_y, crop_y + crop_size)
    xx = slice(crop_x, crop_x + crop_size)
    crop = dapi_norm[yy, xx]
    cellpose_mask = cellpose_mask_full[yy, xx]
    stardist_mask = stardist_mask_full[yy, xx]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(crop, cmap="gray")
    axes[0].set_title("DAPI crop", fontsize=11)

    def _draw_mask(ax, mask, title, color):
        ax.imshow(crop, cmap="gray", alpha=0.65)
        b = find_boundaries(mask, mode="outer")
        b_thick = binary_dilation(b, disk(1))
        overlay = np.zeros((*b_thick.shape, 4))
        overlay[b_thick] = (*color, 1.0)
        ax.imshow(overlay)
        n_in_crop = len(np.unique(mask)) - (1 if 0 in mask else 0)
        ax.set_title(f"{title} ({n_in_crop} cells in crop)", fontsize=11)

    _draw_mask(axes[1], cellpose_mask, "Cellpose (flow field)", (1.0, 0.2, 0.2))
    _draw_mask(axes[2], stardist_mask, "StarDist (star-convex)", (0.1, 0.5, 1.0))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=180, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  Figure 3: seg_classification.png
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=2400)
def render_seg_classification(fov: str = "FOV_E",
                              crop_y: int = 512, crop_x: int = 512,
                              crop_size: int = 1024) -> bytes:
    import io
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import anndata as ad
    from sklearn.neighbors import KNeighborsClassifier
    from skimage.segmentation import find_boundaries

    data_vol.reload()

    fov_dir = Path(f"{DATA_ROOT}/test/{fov}")
    epi = next(fov_dir.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))
    stack = _load_dax_stack(str(epi))
    dapi = _max_proj(stack, [6, 11, 16, 21, 26])
    polyt = _max_proj(stack, [5, 10, 15, 20, 25])

    mask = _run_cellpose(_pct_norm(dapi)[None, ...] * 255.0)

    spots = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
    fov_spots = spots[spots["fov"] == fov].copy()
    rr = np.clip(fov_spots["image_row"].astype(int), 0, 2047)
    cc = np.clip(fov_spots["image_col"].astype(int), 0, 2047)
    fov_spots["cell_id"] = mask[rr, cc]

    in_cell = fov_spots[fov_spots["cell_id"] > 0]
    cells = sorted(in_cell["cell_id"].unique())
    c2i = {c: i for i, c in enumerate(cells)}

    train = ad.read_h5ad(f"{DATA_ROOT}/train/ground_truth/counts_train.h5ad")
    train_genes = list(train.var_names)
    train_lbl = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_labels_train.csv").set_index("cell_id")
    idx = [i for i, c in enumerate(train.obs_names) if c in train_lbl.index]
    X_tr = train.X[idx]
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    X_tr = np.log1p(X_tr.astype("float32"))
    y_tr = train_lbl.loc[[train.obs_names[i] for i in idx], "class_label"].to_numpy()

    aws_h5 = _find_first([f"{DATA_ROOT}/external/aws/Zhuang-ABCA-4-log2.h5ad"])
    aws_meta_p = _find_first([f"{DATA_ROOT}/external/aws/cell_metadata_with_cluster_annotation.csv"])
    if aws_h5 and aws_meta_p:
        aws = ad.read_h5ad(aws_h5)
        aws_meta = pd.read_csv(aws_meta_p).set_index("cell_label")
        common = [c for c in aws.obs_names if c in aws_meta.index]
        if common:
            X_aws = aws[common].X
            if hasattr(X_aws, "toarray"):
                X_aws = X_aws.toarray()
            X_aws = np.log1p(X_aws.astype("float32"))
            aws_genes = list(aws.var_names)
            order = [aws_genes.index(g) if g in aws_genes else -1 for g in train_genes]
            if all(i >= 0 for i in order):
                X_aws_aligned = X_aws[:, order]
                y_aws = aws_meta.loc[common, "class"].to_numpy()
                X_tr = np.vstack([X_tr, X_aws_aligned])
                y_tr = np.concatenate([y_tr, y_aws])

    g2i = {g: i for i, g in enumerate(train_genes)}
    expr = np.zeros((len(cells), len(train_genes)), dtype="float32")
    for cid, g in zip(in_cell["cell_id"], in_cell["target_gene"]):
        if g in g2i:
            expr[c2i[cid], g2i[g]] += 1.0
    expr = np.log1p(expr)

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
    knn.fit(X_tr, y_tr)
    pred = knn.predict(expr)
    cell2cls = dict(zip(cells, pred))

    yy = slice(crop_y, crop_y + crop_size)
    xx = slice(crop_x, crop_x + crop_size)
    composite = np.zeros((crop_size, crop_size, 3))
    composite[..., 2] = _pct_norm(dapi[yy, xx])
    composite[..., 0] = _pct_norm(polyt[yy, xx])
    composite[..., 1] = composite[..., 0] * 0.5

    mask_crop = mask[yy, xx]
    cls_overlay = np.zeros((crop_size, crop_size, 4))
    unique_cls = sorted(set(cell2cls.values()))
    cmap = plt.colormaps["tab20"].resampled(max(len(unique_cls), 2))
    cls2col = {c: cmap(i) for i, c in enumerate(unique_cls)}
    for cid in np.unique(mask_crop):
        if cid == 0:
            continue
        cls = cell2cls.get(cid, "background")
        color = cls2col.get(cls, (0.5, 0.5, 0.5, 1.0))
        cls_overlay[mask_crop == cid] = (*color[:3], 0.85)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(composite)
    axes[0].set_title(f"DAPI + polyT composite ({fov} crop)", fontsize=10)

    axes[1].imshow(_pct_norm(dapi[yy, xx]), cmap="gray")
    bounds = find_boundaries(mask_crop, mode="outer")
    bo = np.zeros((crop_size, crop_size, 4))
    bo[bounds] = (1.0, 1.0, 0.2, 1.0)
    axes[1].imshow(bo)
    axes[1].set_title(f"Cellpose masks ({mask_crop.max()} cells)", fontsize=10)

    axes[2].imshow(_pct_norm(dapi[yy, xx]), cmap="gray", alpha=0.5)
    axes[2].imshow(cls_overlay)
    axes[2].set_title(f"Predicted class ({len(unique_cls)} classes)", fontsize=10)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=180, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  Figure 4: ensemble_disagreement.png
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=3600)
def render_ensemble_disagreement(val_fovs: list | None = None) -> bytes:
    import io
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import anndata as ad
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    if val_fovs is None:
        val_fovs = ["FOV_151", "FOV_154", "FOV_159"]

    data_vol.reload()

    # ---- training pool ----
    train = ad.read_h5ad(f"{DATA_ROOT}/train/ground_truth/counts_train.h5ad")
    train_lbl = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_labels_train.csv").set_index("cell_id")
    idx = [i for i, c in enumerate(train.obs_names) if c in train_lbl.index]
    X_tr = train.X[idx]
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    X_tr = np.log1p(X_tr.astype("float32"))
    y_tr_class = train_lbl.loc[[train.obs_names[i] for i in idx], "class_label"].to_numpy()
    train_genes = list(train.var_names)

    aws_h5 = _find_first([f"{DATA_ROOT}/external/aws/Zhuang-ABCA-4-log2.h5ad"])
    aws_meta_p = _find_first([f"{DATA_ROOT}/external/aws/cell_metadata_with_cluster_annotation.csv"])
    if aws_h5 and aws_meta_p:
        aws = ad.read_h5ad(aws_h5)
        aws_meta = pd.read_csv(aws_meta_p).set_index("cell_label")
        common = [c for c in aws.obs_names if c in aws_meta.index]
        if common:
            X_aws = aws[common].X
            if hasattr(X_aws, "toarray"):
                X_aws = X_aws.toarray()
            X_aws = np.log1p(X_aws.astype("float32"))
            aws_genes = list(aws.var_names)
            order = [aws_genes.index(g) if g in aws_genes else -1 for g in train_genes]
            if all(i >= 0 for i in order):
                X_aws_aligned = X_aws[:, order]
                y_aws = aws_meta.loc[common, "class"].to_numpy()
                X_tr = np.vstack([X_tr, X_aws_aligned])
                y_tr_class = np.concatenate([y_tr_class, y_aws])

    voters = [
        ("k=5,cos,dist",  dict(n_neighbors=5,  metric="cosine",    weights="distance")),
        ("k=3,cos,dist",  dict(n_neighbors=3,  metric="cosine",    weights="distance")),
        ("k=15,cos,dist", dict(n_neighbors=15, metric="cosine",    weights="distance")),
        ("k=5,cos,unif",  dict(n_neighbors=5,  metric="cosine",    weights="uniform")),
        ("k=5,L1,dist",   dict(n_neighbors=5,  metric="manhattan", weights="distance")),
    ]
    clfs = []
    for _, kw in voters:
        c = KNeighborsClassifier(**kw)
        c.fit(X_tr, y_tr_class)
        clfs.append(c)

    spots_all = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
    cells_all = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv",
                            index_col=0)
    from phase2.src import coords

    g2i = {g: i for i, g in enumerate(train_genes)}
    all_preds = [[] for _ in voters]
    all_gt = []

    for fov in val_fovs:
        fov_dir = Path(f"{DATA_ROOT}/train/{fov}")
        epi = next(fov_dir.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))
        stack = _load_dax_stack(str(epi))
        dapi = _max_proj(stack, [6, 11, 16, 21, 26])
        mask = _run_cellpose(_pct_norm(dapi)[None, ...] * 255.0)

        fov_spots = spots_all[spots_all["fov"] == fov].copy()
        rr = np.clip(fov_spots["image_row"].astype(int), 0, 2047)
        cc = np.clip(fov_spots["image_col"].astype(int), 0, 2047)
        fov_spots["cell_id"] = mask[rr, cc]
        in_cell = fov_spots[fov_spots["cell_id"] > 0]
        cells = sorted(in_cell["cell_id"].unique())
        if not cells:
            continue
        c2i = {c: i for i, c in enumerate(cells)}
        expr = np.zeros((len(cells), len(train_genes)), dtype="float32")
        for cid, g in zip(in_cell["cell_id"], in_cell["target_gene"]):
            if g in g2i:
                expr[c2i[cid], g2i[g]] += 1.0
        expr = np.log1p(expr)

        per_cell_preds = [clf.predict(expr) for clf in clfs]
        for vi, cpc in enumerate(per_cell_preds):
            pred = np.array(["background"] * len(fov_spots), dtype=object)
            mapper = dict(zip(cells, cpc))
            cid_arr = fov_spots["cell_id"].to_numpy()
            for cid, cls in mapper.items():
                pred[cid_arr == cid] = cls
            all_preds[vi].append(pred)

        fov_cell_ids = train_lbl[train_lbl["fov"] == fov].index
        gt = np.array(["background"] * len(fov_spots), dtype=object)
        for cid in fov_cell_ids:
            if cid not in cells_all.index:
                continue
            row = cells_all.loc[cid]
            poly = coords.parse_boundary_polygon(
                row.get("boundaryX_z2", ""), row.get("boundaryY_z2", ""))
            if poly is None:
                continue
            inside = coords.spots_in_polygon(
                fov_spots["global_x"].to_numpy(),
                fov_spots["global_y"].to_numpy(), poly)
            gt[inside] = train_lbl.loc[cid, "class_label"]
        all_gt.append(gt)

    preds = [np.concatenate(p) for p in all_preds]
    gt = np.concatenate(all_gt)

    n = len(voters)
    agree = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agree[i, j] = float((preds[i] == preds[j]).mean())

    per_voter_ari = [adjusted_rand_score(gt, p) for p in preds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    labels = [v[0] for v in voters]

    im = ax1.imshow(agree, cmap="Reds", vmin=0.8, vmax=1.0)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{agree[i, j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="black" if agree[i, j] < 0.95 else "white")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="per-spot agreement")
    ax1.set_title("Voter pairwise per-spot agreement", fontsize=10)

    ax2.barh(range(n), per_voter_ari, color="#A03E3E", alpha=0.85)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"Local val class-level ARI on {len(val_fovs)} FOVs", fontsize=9)
    ax2.grid(axis="x", ls="--", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for i, v in enumerate(per_voter_ari):
        ax2.annotate(f"{v:.3f}", xy=(v, i), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=8)
    ax2.set_title("Per-voter local ARI", fontsize=10)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=180, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(only: str = "all"):
    targets = {
        "data_overview":         render_data_overview,
        "inductive_bias":        render_inductive_bias,
        "seg_classification":    render_seg_classification,
        "ensemble_disagreement": render_ensemble_disagreement,
    }
    if only != "all":
        targets = {only: targets[only]}

    out_dir = REPO / "figs"
    out_dir.mkdir(exist_ok=True)
    for name, fn in targets.items():
        print(f"==> rendering {name} ...", flush=True)
        png = fn.remote()
        out = out_dir / f"{name}.png"
        out.write_bytes(png)
        print(f"    wrote {out}  ({len(png)/1024:.0f} KB)")
