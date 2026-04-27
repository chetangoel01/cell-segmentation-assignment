# Phase 1: Cell Segmentation & Spot Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a MERFISH cell segmentation pipeline that assigns ~224K mRNA spots to cells (or background) across 4 test FOVs, targeting ARI well above the 0.632 Cellpose pretrained baseline.

**Architecture:** A Jupyter notebook (`pipeline.ipynb`) drives the full pipeline end-to-end, backed by utility modules in `src/`. The pipeline loads raw `.dax` images → runs Cellpose segmentation → assigns spots via point-in-polygon → labels extracellular spots as background → generates `submission.csv`. Fine-tuning Cellpose on all 40 training FOVs is the primary improvement over baseline.

**Tech Stack:** Python, NumPy, Pandas, Cellpose 2.x, Shapely, AnnData, scikit-image, Matplotlib

**Data path (HPC):** `/scratch/pl2820/competition/`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pipeline.ipynb` | Main deliverable — full end-to-end pipeline with Markdown explanations |
| `src/io.py` | Load `.dax` files, extract DAPI/polyT frames by z-plane |
| `src/coords.py` | Pixel ↔ µm coordinate conversion; parse ground truth boundaries to Shapely polygons |
| `src/assign.py` | Point-in-polygon spot-to-cell assignment with background labeling |
| `src/evaluate.py` | Local ARI evaluation using sklearn |
| `src/train_cellpose.py` | Prepare Cellpose training data from GT boundaries; run fine-tuning |
| `submission.csv` | Final output (not committed — in .gitignore) |

---

## Task 1: Environment Setup & Data Access

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`

- [ ] **Step 1: Install core dependencies**

```bash
pip install cellpose==2.4.2 shapely anndata scikit-image matplotlib tifffile
pip install "numpy<2.0"  # cellpose 2.x compatibility
```

- [ ] **Step 2: Verify data path is accessible**

```bash
ls /scratch/pl2820/competition/
# Expected: train/  test/  reference/  test_spots.csv  sample_submission.csv  metric.py
ls /scratch/pl2820/competition/train/ | head -5
# Expected: FOV_001  FOV_002  ...
```

- [ ] **Step 3: Write requirements.txt**

```
cellpose==2.4.2
shapely>=2.0
anndata>=0.10
scikit-image>=0.21
matplotlib>=3.7
numpy<2.0
pandas>=2.0
tifffile
```

- [ ] **Step 4: Create src/__init__.py (empty)**

```bash
touch src/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py
git commit -m "feat: add project structure and requirements"
```

---

## Task 2: Data Loading Utilities (`src/io.py`)

**Files:**
- Create: `src/io.py`

The `.dax` file is raw binary uint16. The multichannel Epi file has 27 frames total. Frame layout (from `dataorganization.csv`):
- DAPI (405 nm): frames 6, 11, 16, 21, 26 → z-planes 0–4
- polyT (488 nm): frames 5, 10, 15, 20, 25 → z-planes 0–4

- [ ] **Step 1: Write the failing test for dax loading**

```python
# tests/test_io.py
import numpy as np
from src.io import load_dax, get_dapi_stack, get_polyt_stack

DATA_ROOT = "/scratch/pl2820/competition"

def test_load_dax_shape():
    """Epi file has 27 frames, each 2048x2048."""
    raw = load_dax(f"{DATA_ROOT}/train/FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax", n_pixels=2048)
    assert raw.shape == (27, 2048, 2048)
    assert raw.dtype == np.uint16

def test_get_dapi_stack_shape():
    raw = load_dax(f"{DATA_ROOT}/train/FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax", n_pixels=2048)
    dapi = get_dapi_stack(raw)
    assert dapi.shape == (5, 2048, 2048)  # 5 z-planes

def test_get_polyt_stack_shape():
    raw = load_dax(f"{DATA_ROOT}/train/FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax", n_pixels=2048)
    polyt = get_polyt_stack(raw)
    assert polyt.shape == (5, 2048, 2048)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_io.py -v
# Expected: FAIL — src.io not found
```

- [ ] **Step 3: Implement src/io.py**

```python
import numpy as np
from pathlib import Path

# Frame indices within the 27-frame Epi file
DAPI_FRAMES  = [6, 11, 16, 21, 26]   # z0..z4, 405 nm
POLYT_FRAMES = [5, 10, 15, 20, 25]   # z0..z4, 488 nm


def load_dax(path: str, n_pixels: int = 2048) -> np.ndarray:
    """Load a raw .dax file as a (n_frames, n_pixels, n_pixels) uint16 array."""
    raw = np.fromfile(path, dtype=np.uint16)
    n_frames = raw.size // (n_pixels * n_pixels)
    return raw.reshape(n_frames, n_pixels, n_pixels)


def get_dapi_stack(raw: np.ndarray) -> np.ndarray:
    """Extract 5 DAPI z-plane images from the Epi raw array. Returns (5, H, W)."""
    return raw[DAPI_FRAMES]


def get_polyt_stack(raw: np.ndarray) -> np.ndarray:
    """Extract 5 polyT z-plane images from the Epi raw array. Returns (5, H, W)."""
    return raw[POLYT_FRAMES]


def load_fov_images(fov_dir: str, n_pixels: int = 2048):
    """Load DAPI and polyT stacks for a single FOV directory.

    Args:
        fov_dir: path to e.g. /scratch/pl2820/competition/train/FOV_001/
    Returns:
        dapi_stack: (5, 2048, 2048) uint16
        polyt_stack: (5, 2048, 2048) uint16
    """
    fov_dir = Path(fov_dir)
    epi_files = sorted(fov_dir.glob("Epi-750s5-635s5-545s1-473s5-408s5_*.dax"))
    if not epi_files:
        raise FileNotFoundError(f"No Epi file found in {fov_dir}")
    raw = load_dax(str(epi_files[0]), n_pixels=n_pixels)
    return get_dapi_stack(raw), get_polyt_stack(raw)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_io.py -v
# Expected: 3 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add src/io.py tests/test_io.py
git commit -m "feat: add dax file loading utilities"
```

---

## Task 3: Coordinate Utilities (`src/coords.py`)

**Files:**
- Create: `src/coords.py`

Spots are stored in global µm coordinates. Cell boundaries are also in µm. The FOV origin (`fov_x`, `fov_y`) and `pixel_size` (0.109 µm/px) let us convert between the two. For point-in-polygon testing, both spots and boundaries must be in the same coordinate system — we use µm throughout.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_coords.py
import numpy as np
from shapely.geometry import Polygon
from src.coords import pixel_to_um, parse_boundary_polygon, spots_in_polygon

def test_pixel_to_um_origin():
    x_um, y_um = pixel_to_um(0, 0, fov_x=100.0, fov_y=200.0, pixel_size=0.109)
    assert abs(x_um - 100.0) < 1e-6
    assert abs(y_um - 200.0) < 1e-6

def test_pixel_to_um_offset():
    x_um, y_um = pixel_to_um(10, 20, fov_x=0.0, fov_y=0.0, pixel_size=0.109)
    assert abs(x_um - 1.09) < 1e-6
    assert abs(y_um - 2.18) < 1e-6

def test_parse_boundary_polygon_valid():
    xs = "1.0,2.0,2.0,1.0"
    ys = "1.0,1.0,2.0,2.0"
    poly = parse_boundary_polygon(xs, ys)
    assert poly is not None
    assert poly.is_valid

def test_parse_boundary_polygon_empty():
    poly = parse_boundary_polygon("", "")
    assert poly is None

def test_spots_in_polygon():
    xs = "0.0,4.0,4.0,0.0"
    ys = "0.0,0.0,4.0,4.0"
    poly = parse_boundary_polygon(xs, ys)
    spot_x = np.array([2.0, 10.0])  # inside, outside
    spot_y = np.array([2.0, 10.0])
    inside = spots_in_polygon(spot_x, spot_y, poly)
    assert inside.tolist() == [True, False]
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_coords.py -v
# Expected: FAIL
```

- [ ] **Step 3: Implement src/coords.py**

```python
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from typing import Optional


def pixel_to_um(px: float, py: float, fov_x: float, fov_y: float,
                pixel_size: float = 0.109):
    """Convert pixel (px, py) within a FOV to global µm coordinates."""
    return fov_x + px * pixel_size, fov_y + py * pixel_size


def parse_boundary_polygon(xs_str: str, ys_str: str) -> Optional[Polygon]:
    """Parse comma-separated boundary coordinate strings into a Shapely Polygon.

    Args:
        xs_str: e.g. "100.5,101.2,102.0,..."
        ys_str: e.g. "200.1,200.8,201.3,..."
    Returns:
        Shapely Polygon, or None if empty/degenerate.
    """
    if not xs_str or not ys_str:
        return None
    xs = [float(v) for v in xs_str.split(",")]
    ys = [float(v) for v in ys_str.split(",")]
    if len(xs) < 3:
        return None
    poly = Polygon(zip(xs, ys))
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area == 0:
        return None
    return poly


def spots_in_polygon(spot_x: np.ndarray, spot_y: np.ndarray,
                     polygon: Polygon) -> np.ndarray:
    """Return boolean array: True if spot (x, y) is inside polygon."""
    from shapely.vectorized import contains
    return contains(polygon, spot_x, spot_y)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_coords.py -v
# Expected: 5 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add src/coords.py tests/test_coords.py
git commit -m "feat: add coordinate conversion and polygon utilities"
```

---

## Task 4: Spot Assignment (`src/assign.py`)

**Files:**
- Create: `src/assign.py`

Given a dict of cell polygons (from Cellpose or GT) and a spots DataFrame with global µm coordinates, assign each spot to its containing cell. Spots outside all cells → `background`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_assign.py
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from src.assign import assign_spots_to_cells

def test_assign_spots_basic():
    polygons = {
        "cell_A": Polygon([(0,0),(4,0),(4,4),(0,4)]),
        "cell_B": Polygon([(10,10),(14,10),(14,14),(10,14)]),
    }
    spots = pd.DataFrame({
        "spot_id": ["s0", "s1", "s2"],
        "global_x": [2.0, 12.0, 50.0],
        "global_y": [2.0, 12.0, 50.0],
    })
    result = assign_spots_to_cells(spots, polygons)
    assert result["s0"] == "cell_A"
    assert result["s1"] == "cell_B"
    assert result["s2"] == "background"

def test_assign_all_background_when_no_cells():
    spots = pd.DataFrame({
        "spot_id": ["s0"],
        "global_x": [999.0],
        "global_y": [999.0],
    })
    result = assign_spots_to_cells(spots, {})
    assert result["s0"] == "background"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_assign.py -v
# Expected: FAIL
```

- [ ] **Step 3: Implement src/assign.py**

```python
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from typing import Dict
from src.coords import spots_in_polygon


def assign_spots_to_cells(
    spots: pd.DataFrame,
    cell_polygons: Dict[str, Polygon],
) -> Dict[str, str]:
    """Assign each spot to a cell by point-in-polygon test.

    Args:
        spots: DataFrame with columns spot_id, global_x, global_y (µm)
        cell_polygons: dict mapping cell_id -> Shapely Polygon in µm coordinates
    Returns:
        dict mapping spot_id -> cell_id (or "background")
    """
    assignments = {sid: "background" for sid in spots["spot_id"]}
    spot_x = spots["global_x"].to_numpy()
    spot_y = spots["global_y"].to_numpy()
    spot_ids = spots["spot_id"].to_numpy()

    for cell_id, polygon in cell_polygons.items():
        if polygon is None:
            continue
        inside = spots_in_polygon(spot_x, spot_y, polygon)
        for sid in spot_ids[inside]:
            assignments[sid] = cell_id  # last cell wins on overlap (rare)

    return assignments
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_assign.py -v
# Expected: 2 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add src/assign.py tests/test_assign.py
git commit -m "feat: add spot-to-cell assignment with background labeling"
```

---

## Task 5: Local ARI Evaluation (`src/evaluate.py`)

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_evaluate.py
import pandas as pd
from src.evaluate import compute_ari

def test_perfect_ari():
    gt  = pd.DataFrame({"spot_id": ["s0","s1","s2"], "fov": ["A","A","A"],
                         "cluster_id": ["c1","c1","c2"]})
    pred = pd.DataFrame({"spot_id": ["s0","s1","s2"], "fov": ["A","A","A"],
                          "cluster_id": ["c1","c1","c2"]})
    assert abs(compute_ari(gt, pred) - 1.0) < 1e-6

def test_all_background_ari():
    gt  = pd.DataFrame({"spot_id": ["s0","s1"], "fov": ["A","A"],
                         "cluster_id": ["c1","c2"]})
    pred = pd.DataFrame({"spot_id": ["s0","s1"], "fov": ["A","A"],
                          "cluster_id": ["background","background"]})
    ari = compute_ari(gt, pred)
    assert ari == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_evaluate.py -v
# Expected: FAIL
```

- [ ] **Step 3: Implement src/evaluate.py**

```python
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def compute_ari(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Compute mean ARI across all FOVs present in solution.

    Args:
        solution: DataFrame with spot_id, fov, cluster_id (ground truth)
        submission: DataFrame with spot_id, fov, cluster_id (predictions)
    Returns:
        Mean ARI across FOVs.
    """
    merged = solution.merge(submission, on="spot_id", suffixes=("_gt", "_pred"))
    fov_scores = []
    for _fov, group in merged.groupby("fov_gt"):
        score = adjusted_rand_score(group["cluster_id_gt"], group["cluster_id_pred"])
        fov_scores.append(score)
    return float(sum(fov_scores) / len(fov_scores)) if fov_scores else 0.0
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_evaluate.py -v
# Expected: 2 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add local ARI evaluation wrapper"
```

---

## Task 6: EDA Notebook (`notebooks/01_eda.ipynb`)

**Files:**
- Create: `notebooks/01_eda.ipynb`

- [ ] **Step 1: Load FOV_001 and visualize DAPI + polyT z2**

```python
# Cell 1
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from src.io import load_fov_images

DATA_ROOT = "/scratch/pl2820/competition"
dapi, polyt = load_fov_images(f"{DATA_ROOT}/train/FOV_001")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(dapi[2], cmap="gray", vmin=0, vmax=np.percentile(dapi[2], 99))
axes[0].set_title("DAPI z2 — FOV_001")
axes[1].imshow(polyt[2], cmap="gray", vmin=0, vmax=np.percentile(polyt[2], 99))
axes[1].set_title("polyT z2 — FOV_001")
plt.tight_layout(); plt.savefig("eda_dapi_polyt.png", dpi=100); plt.show()
```

- [ ] **Step 2: Overlay ground truth cell boundaries on DAPI**

```python
# Cell 2
import pandas as pd
from src.coords import parse_boundary_polygon

meta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
fov_x = meta.loc["FOV_001", "fov_x"]
fov_y = meta.loc["FOV_001", "fov_y"]
pixel_size = meta.loc["FOV_001", "pixel_size"]

cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
fov_cells = cells[cells.index.str.startswith("FOV_001")]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(dapi[2], cmap="gray", vmin=0, vmax=np.percentile(dapi[2], 99))

for _cell_id, row in fov_cells.iterrows():
    poly = parse_boundary_polygon(row.get("boundaryX_z2", ""), row.get("boundaryY_z2", ""))
    if poly is None:
        continue
    xs_px = [(x - fov_x) / pixel_size for x in poly.exterior.xy[0]]
    ys_px = [(y - fov_y) / pixel_size for y in poly.exterior.xy[1]]
    ax.plot(xs_px, ys_px, 'c-', linewidth=0.5, alpha=0.8)

ax.set_title(f"FOV_001 — {len(fov_cells)} ground truth cells")
plt.savefig("eda_gt_boundaries.png", dpi=100); plt.show()
```

- [ ] **Step 3: Print spot and cell count statistics**

```python
# Cell 3
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
print("Spots per FOV:")
print(spots_train.groupby("fov").size().describe())
print(f"\nFOV_001 spots: {len(spots_train[spots_train['fov'] == 'FOV_001'])}")
print(f"FOV_001 GT cells: {len(fov_cells)}")
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_eda.ipynb
git commit -m "feat: add EDA notebook with DAPI and GT boundary visualization"
```

---

## Task 7: Cellpose Baseline Notebook (`notebooks/02_baseline.ipynb`)

**Files:**
- Create: `notebooks/02_baseline.ipynb`

Run pretrained Cellpose on FOV_001 to confirm we can replicate the ~0.632 baseline locally.

- [ ] **Step 1: Run pretrained Cellpose on FOV_001 DAPI z2**

```python
# Cell 1
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from cellpose import models
from src.io import load_fov_images

DATA_ROOT = "/scratch/pl2820/competition"
dapi, polyt = load_fov_images(f"{DATA_ROOT}/train/FOV_001")

cp_model = models.Cellpose(gpu=True, model_type="cyto2")
masks, flows, styles, diams = cp_model.eval(
    dapi[2],
    diameter=None,
    channels=[0, 0],
    flow_threshold=0.4,
    cellprob_threshold=0.0,
)
print(f"Cells detected: {masks.max()}")
```

- [ ] **Step 2: Convert Cellpose integer masks to µm polygons**

```python
# Cell 2
from shapely.geometry import Polygon
from skimage import measure

meta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
fov_x = meta.loc["FOV_001", "fov_x"]
fov_y = meta.loc["FOV_001", "fov_y"]
pixel_size = 0.109


def masks_to_polygons(masks, fov_x, fov_y, pixel_size=0.109):
    """Convert Cellpose integer mask array to dict of cell_id -> Shapely Polygon in µm."""
    polygons = {}
    for cell_int in range(1, masks.max() + 1):
        cell_mask = (masks == cell_int).astype(np.uint8)
        contours = measure.find_contours(cell_mask, 0.5)
        if not contours:
            continue
        # find_contours returns (row, col) = (y, x)
        contour = contours[0]
        xs_um = fov_x + contour[:, 1] * pixel_size
        ys_um = fov_y + contour[:, 0] * pixel_size
        poly = Polygon(zip(xs_um, ys_um))
        if poly.is_valid and poly.area > 0:
            polygons[f"cellpose_{cell_int}"] = poly
    return polygons


cell_polygons = masks_to_polygons(masks, fov_x, fov_y)
print(f"Valid cell polygons: {len(cell_polygons)}")
```

- [ ] **Step 3: Assign spots and compute local ARI vs GT boundaries**

```python
# Cell 3
from src.assign import assign_spots_to_cells
from src.evaluate import compute_ari
from src.coords import parse_boundary_polygon

spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")
fov001_spots = spots_train[spots_train["fov"] == "FOV_001"].copy()
fov001_spots["spot_id"] = [f"FOV_001_{i}" for i in range(len(fov001_spots))]

# Predicted assignments
pred_assignments = assign_spots_to_cells(fov001_spots, cell_polygons)
fov001_spots["pred_cluster"] = fov001_spots["spot_id"].map(pred_assignments)

# GT assignments (infer from cell boundary polygons)
cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)
fov_cells = cells[cells.index.str.startswith("FOV_001")]
gt_polygons = {}
for cell_id, row in fov_cells.iterrows():
    poly = parse_boundary_polygon(row.get("boundaryX_z2", ""), row.get("boundaryY_z2", ""))
    if poly:
        gt_polygons[cell_id] = poly
gt_assignments = assign_spots_to_cells(fov001_spots, gt_polygons)
fov001_spots["gt_cluster"] = fov001_spots["spot_id"].map(gt_assignments)

solution   = fov001_spots[["spot_id", "fov", "gt_cluster"]].rename(columns={"gt_cluster": "cluster_id"})
submission = fov001_spots[["spot_id", "fov", "pred_cluster"]].rename(columns={"pred_cluster": "cluster_id"})

ari = compute_ari(solution, submission)
print(f"Local ARI on FOV_001 (pretrained Cellpose): {ari:.4f}")
# Expected: ~0.40–0.65
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_baseline.ipynb
git commit -m "feat: add Cellpose pretrained baseline notebook with local ARI"
```

---

## Task 8: Fine-Tune Cellpose (`src/train_cellpose.py` + `notebooks/03_finetune.ipynb`)

**Files:**
- Create: `src/train_cellpose.py`
- Create: `notebooks/03_finetune.ipynb`

Cellpose fine-tuning takes (image, mask) pairs. Convert GT boundary polygons → integer mask images, then fine-tune. Train on FOVs 001–035, validate on FOVs 036–040.

- [ ] **Step 1: Implement boundaries_to_mask in src/train_cellpose.py**

```python
# src/train_cellpose.py
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from src.coords import parse_boundary_polygon


def boundaries_to_mask(cell_boundaries_df: pd.DataFrame, fov_name: str,
                        fov_x: float, fov_y: float,
                        pixel_size: float = 0.109, image_size: int = 2048,
                        z_plane: int = 2) -> np.ndarray:
    """Convert GT cell boundary polygons for one FOV/z-plane to an integer mask.

    Args:
        cell_boundaries_df: cell_boundaries_train.csv loaded with index_col=0
        fov_name: e.g. "FOV_001"
        fov_x, fov_y: FOV origin in µm
        pixel_size: µm per pixel
        image_size: 2048
        z_plane: which z-plane to use (0–4)
    Returns:
        (2048, 2048) int32 array, 0=background, 1..N=cell integer IDs
    """
    from skimage.draw import polygon as draw_polygon

    fov_cells = cell_boundaries_df[cell_boundaries_df.index.str.startswith(fov_name)]
    mask = np.zeros((image_size, image_size), dtype=np.int32)
    cell_int = 1

    for _cell_id, row in fov_cells.iterrows():
        xs_str = row.get(f"boundaryX_z{z_plane}", "")
        ys_str = row.get(f"boundaryY_z{z_plane}", "")
        poly = parse_boundary_polygon(xs_str, ys_str)
        if poly is None:
            continue
        xs_um, ys_um = poly.exterior.xy
        col_px = np.array([(x - fov_x) / pixel_size for x in xs_um])
        row_px = np.array([(y - fov_y) / pixel_size for y in ys_um])
        rr, cc = draw_polygon(row_px, col_px, shape=(image_size, image_size))
        mask[rr, cc] = cell_int
        cell_int += 1

    return mask
```

- [ ] **Step 2: Verify mask on FOV_001 — cell count must match GT**

```python
# In notebooks/03_finetune.ipynb — Cell 1
import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.io import load_fov_images
from src.train_cellpose import boundaries_to_mask

DATA_ROOT = "/scratch/pl2820/competition"
meta  = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")
cells = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/cell_boundaries_train.csv", index_col=0)

fov_x = meta.loc["FOV_001", "fov_x"]
fov_y = meta.loc["FOV_001", "fov_y"]
dapi, _ = load_fov_images(f"{DATA_ROOT}/train/FOV_001")
mask = boundaries_to_mask(cells, "FOV_001", fov_x, fov_y)

print(f"Cells in mask: {mask.max()}")
gt_count = len(cells[cells.index.str.startswith("FOV_001")])
print(f"GT cells in csv: {gt_count}")
# mask.max() should be close to gt_count (some may be degenerate)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(dapi[2], cmap="gray", vmin=0, vmax=np.percentile(dapi[2], 99))
axes[1].imshow(mask > 0, cmap="gray")
axes[0].set_title("DAPI z2"); axes[1].set_title("GT mask (any cell = white)")
plt.savefig("mask_check_fov001.png", dpi=80); plt.show()
```

- [ ] **Step 3: Build training set (35 FOVs) and run Cellpose fine-tuning**

```python
# In notebooks/03_finetune.ipynb — Cell 2
import os
from cellpose import models as cp_models, train as cp_train

train_images = []
train_masks  = []
train_fovs   = [f"FOV_{i:03d}" for i in range(1, 36)]

for fov_name in train_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        continue
    try:
        dapi, _ = load_fov_images(fov_dir)
        fov_x = meta.loc[fov_name, "fov_x"]
        fov_y = meta.loc[fov_name, "fov_y"]
        m = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
        if m.max() == 0:
            continue
        train_images.append(dapi[2])
        train_masks.append(m)
    except Exception as exc:
        print(f"Skipping {fov_name}: {exc}")

print(f"Training on {len(train_images)} FOVs")

base_model = cp_models.CellposeModel(gpu=True, model_type="cyto2")
os.makedirs("models", exist_ok=True)
model_path = cp_train.train_seg(
    base_model.net,
    train_data=train_images,
    train_labels=train_masks,
    channels=[0, 0],
    save_path="models/",
    n_epochs=100,
    learning_rate=0.005,
    weight_decay=1e-5,
    batch_size=8,
    model_name="cellpose_finetuned",
)
print(f"Saved: {model_path}")
```

- [ ] **Step 4: Evaluate fine-tuned model on validation FOVs 036–040**

```python
# In notebooks/03_finetune.ipynb — Cell 3
from skimage import measure
from shapely.geometry import Polygon
from src.assign import assign_spots_to_cells
from src.evaluate import compute_ari

finetuned = cp_models.CellposeModel(gpu=True, pretrained_model="models/cellpose_finetuned")
val_fovs = [f"FOV_{i:03d}" for i in range(36, 41)]
spots_train = pd.read_csv(f"{DATA_ROOT}/train/ground_truth/spots_train.csv")

ari_scores = {}
for fov_name in val_fovs:
    fov_dir = f"{DATA_ROOT}/train/{fov_name}"
    if not os.path.exists(fov_dir):
        continue
    dapi, _ = load_fov_images(fov_dir)
    fov_x = meta.loc[fov_name, "fov_x"]
    fov_y = meta.loc[fov_name, "fov_y"]

    # Run finetuned model
    pred_masks, _, _ = finetuned.eval(dapi[2], channels=[0, 0])
    pred_polys = masks_to_polygons(pred_masks, fov_x, fov_y)

    # Build GT polygons from boundaries
    gt_mask = boundaries_to_mask(cells, fov_name, fov_x, fov_y)
    gt_polys = masks_to_polygons(gt_mask, fov_x, fov_y)

    fov_spots = spots_train[spots_train["fov"] == fov_name].copy()
    fov_spots["spot_id"] = [f"{fov_name}_{i}" for i in range(len(fov_spots))]

    pred_assign = assign_spots_to_cells(fov_spots, pred_polys)
    gt_assign   = assign_spots_to_cells(fov_spots, gt_polys)
    fov_spots["pred_cluster"] = fov_spots["spot_id"].map(pred_assign)
    fov_spots["gt_cluster"]   = fov_spots["spot_id"].map(gt_assign)

    solution   = fov_spots[["spot_id","fov","gt_cluster"]].rename(columns={"gt_cluster":"cluster_id"})
    submission = fov_spots[["spot_id","fov","pred_cluster"]].rename(columns={"pred_cluster":"cluster_id"})
    ari = compute_ari(solution, submission)
    ari_scores[fov_name] = ari
    print(f"{fov_name}: ARI = {ari:.4f}")

print(f"\nMean val ARI: {np.mean(list(ari_scores.values())):.4f}")
# Target: > 0.632 (pretrained baseline)
```

- [ ] **Step 5: Commit**

```bash
git add src/train_cellpose.py notebooks/03_finetune.ipynb
git commit -m "feat: add Cellpose fine-tuning on 35 training FOVs with validation"
```

---

## Task 9: Test FOV Inference & Submission (`notebooks/04_submission.ipynb`)

**Files:**
- Create: `notebooks/04_submission.ipynb`

- [ ] **Step 1: Load test spots and sample submission template**

```python
# Cell 1
import sys; sys.path.insert(0, '..')
import pandas as pd
import numpy as np

DATA_ROOT = "/scratch/pl2820/competition"
test_spots   = pd.read_csv(f"{DATA_ROOT}/test_spots.csv")
sub_template = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

print(f"Test spots: {test_spots.shape}")
print(f"Template rows: {len(sub_template)}")
print(test_spots["fov"].value_counts())
```

- [ ] **Step 2: Run segmentation on each test FOV**

```python
# Cell 2
import os
from cellpose import models as cp_models
from src.io import load_fov_images
from src.assign import assign_spots_to_cells
from skimage import measure
from shapely.geometry import Polygon

meta = pd.read_csv(f"{DATA_ROOT}/reference/fov_metadata.csv").set_index("fov")

# Use fine-tuned model if available, otherwise pretrained
model_path = "models/cellpose_finetuned"
if os.path.exists(model_path):
    seg_model = cp_models.CellposeModel(gpu=True, pretrained_model=model_path)
    def run_segmentation(image):
        masks, _, _ = seg_model.eval(image, channels=[0, 0])
        return masks
else:
    seg_model = cp_models.Cellpose(gpu=True, model_type="cyto2")
    def run_segmentation(image):
        masks, _, _, _ = seg_model.eval(image, channels=[0, 0], diameter=None)
        return masks

all_assignments = {}

for fov_name in ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]:
    fov_dir = f"{DATA_ROOT}/test/{fov_name}"
    dapi, _ = load_fov_images(fov_dir)
    fov_x = meta.loc[fov_name, "fov_x"]
    fov_y = meta.loc[fov_name, "fov_y"]

    masks = run_segmentation(dapi[2])
    cell_polygons = masks_to_polygons(masks, fov_x, fov_y, pixel_size=0.109)
    print(f"{fov_name}: {len(cell_polygons)} cells detected")

    fov_spots = test_spots[test_spots["fov"] == fov_name].copy()
    assignments = assign_spots_to_cells(fov_spots, cell_polygons)
    all_assignments.update(assignments)

bg_frac = sum(1 for v in all_assignments.values() if v == "background") / len(all_assignments)
print(f"\nBackground fraction: {bg_frac:.1%}")
```

- [ ] **Step 3: Build and validate submission.csv**

```python
# Cell 3
sub = sub_template.copy()
sub["cluster_id"] = sub["spot_id"].map(all_assignments)

# Validation
assert sub["cluster_id"].isna().sum() == 0, "Null cluster_ids found!"
assert len(sub) == len(sub_template),       "Row count mismatch!"
assert list(sub.columns) == ["spot_id", "fov", "cluster_id"], "Wrong columns!"
assert (sub["cluster_id"] != "").all(),     "Empty string cluster_ids!"

sub.to_csv("submission.csv", index=False)
print(f"Saved submission.csv — {len(sub)} rows")
print(sub.head())
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_submission.ipynb
git commit -m "feat: add test FOV inference and submission generation"
```

---

## Task 10: Consolidated Pipeline Notebook (`pipeline.ipynb`)

**Files:**
- Create: `pipeline.ipynb`

This is the primary submission artifact for Brightspace. Consolidate all steps into one notebook with Markdown explanations. Structure:

```
## 1. Setup & Imports
## 2. Data Loading
## 3. Exploratory Analysis (one FOV visualization)
## 4. Segmentation Model (fine-tuned Cellpose)
## 5. Spot Assignment & Background Labeling
## 6. Local Validation (held-out FOVs)
## 7. Test FOV Inference
## 8. Submission Generation
## 9. Methods Summary
```

- [ ] **Step 1: Add Methods Summary Markdown cell at top**

```markdown
## Methods Summary

**Phase 1 Goal:** Segment cells from raw MERFISH DAPI images and assign ~224K mRNA spots 
to cells (or background), measured by Adjusted Rand Index (ARI).

**Approach:**
1. Load raw `.dax` images and extract DAPI z2 (middle z-plane) for each FOV
2. Fine-tune Cellpose (cyto2) on 35 training FOVs using provided GT boundaries as masks
3. Run fine-tuned Cellpose on each FOV to produce cell boundary polygons
4. Assign spots to cells by point-in-polygon testing in global µm coordinates
5. Label spots outside all cell boundaries as `background`

**Baseline:** Pretrained Cellpose (no fine-tuning) ARI = 0.632  
**Improvement:** Fine-tuning on domain-specific tissue data should push ARI > 0.7
```

- [ ] **Step 2: Execute notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute pipeline.ipynb \
  --output pipeline.ipynb \
  --ExecutePreprocessor.timeout=7200
```

- [ ] **Step 3: Zip for Brightspace**

```bash
zip -r submission_code.zip pipeline.ipynb src/ requirements.txt
echo "Zip size: $(du -sh submission_code.zip)"
```

- [ ] **Step 4: Final commit**

```bash
git add pipeline.ipynb
git commit -m "feat: add consolidated pipeline notebook for submission"
```

---

## Spec Coverage Check

| Requirement | Task |
|-------------|------|
| Load raw .dax DAPI/polyT images | Task 2 (`src/io.py`) |
| Coordinate conversion pixel ↔ µm | Task 3 (`src/coords.py`) |
| Point-in-polygon spot assignment | Task 4 (`src/assign.py`) |
| Background labeling for extracellular spots | Task 4 |
| Local ARI evaluation | Task 5 (`src/evaluate.py`) |
| EDA with visualizations | Task 6 |
| Cellpose baseline | Task 7 |
| Fine-tuned segmentation on 35 training FOVs | Task 8 |
| Validation on held-out FOVs 036–040 | Task 8 |
| Test FOV inference (FOV_A–D) | Task 9 |
| submission.csv with exact format | Task 9 |
| pipeline.ipynb with Markdown explanations | Task 10 |
| Deadline: April 24, 2026 at 11:55 PM EST | — |

## Final Submission Checklist

- [ ] `submission.csv` uploaded to Kaggle (verify row count = 224,500)
- [ ] `submission_code.zip` uploaded to Brightspace (pipeline.ipynb + src/ + requirements.txt)
- [ ] Methods Markdown present in `pipeline.ipynb`
- [ ] No null cluster_id values in submission
- [ ] Private leaderboard ARI > 0.632
