# Phase-1 Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `phase1_restart/` end-to-end tonight: scaffold + zero-shot CellSAM/MEDIAR + parallel Modal fine-tune + Phase-1 Kaggle submission + Phase-2 re-run + Phase-2 submission, all before 23:55 EST.

**Architecture:** SegAdapter ABC + per-model adapter file + thin scripts. Reuse `phase1/src/{io,coords,evaluate,train_cellpose}` as read-only imports. Modal for parallel fine-tune (CellSAM + MEDIAR on separate A100 slots). Mac MPS for everything else.

**Tech Stack:** Python 3.11, PyTorch (MPS + CUDA), Modal, CellSAM (vanvalenlab), MEDIAR (Lee-Gihun), pytest for tests.

**Spec:** [phase1_restart/docs/2026-05-04-phase1-restart-design.md](2026-05-04-phase1-restart-design.md)

---

## File structure (locked at planning time)

```
phase1_restart/
├── README.md                       # what this folder is
├── .gitignore                      # weights/, outputs/, runs/, __pycache__
├── docs/
│   ├── 2026-05-04-phase1-restart-design.md   # already exists
│   └── 2026-05-04-phase1-restart-plan.md     # this file
├── pilot/
│   ├── __init__.py                 # empty
│   ├── adapter.py                  # SegAdapter ABC + tiled-predict mixin
│   ├── data.py                     # FOV loader, channel assembly, splits, GT cache
│   ├── eval.py                     # per-FOV ARI on cached GT
│   ├── ensemble.py                 # spot-level majority vote
│   └── submission.py               # mask → CSV, structural validation
├── models/
│   ├── __init__.py                 # registry: {"cellsam": ..., "mediar": ...}
│   ├── cellsam.py                  # CellSAMAdapter
│   └── mediar.py                   # MediarAdapter
├── scripts/
│   ├── __init__.py                 # empty
│   ├── smoke.py                    # coord-sanity gate, must pass before everything
│   ├── zero_shot.py                # --model X --split val|test_proxy|test
│   ├── fine_tune.py                # local-side launcher → modal_app.py
│   ├── infer_test.py               # --model X --checkpoint <path> on FOVs A-D
│   ├── make_submission.py          # masks → Phase-1 Kaggle CSV
│   └── rerun_phase2.py             # invoke phase2 pipeline with new masks
├── configs/
│   ├── cellsam.yaml                # fine-tune hparams
│   └── mediar.yaml                 # fine-tune hparams
├── modal_app.py                    # Modal entrypoint: fine_tune, infer
└── tests/
    ├── __init__.py                 # empty
    └── test_smoke.py               # coord-sanity assertions
```

`weights/`, `outputs/`, `runs/` are gitignored and created at first run.

---

## Task 1: Project skeleton

**Files:**
- Create: `phase1_restart/README.md`
- Create: `phase1_restart/.gitignore`
- Create: `phase1_restart/{pilot,models,scripts,configs,tests}/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p phase1_restart/{pilot,models,scripts,configs,tests}
touch phase1_restart/__init__.py phase1_restart/pilot/__init__.py phase1_restart/models/__init__.py phase1_restart/scripts/__init__.py phase1_restart/tests/__init__.py
```

The top-level `phase1_restart/__init__.py` is required so `python -m phase1_restart.scripts.smoke` works from the repo root. Folder uses underscores (not dashes like sibling `phase2-restart/`) because Python module names cannot contain dashes.

- [ ] **Step 2: Write `phase1_restart/.gitignore`**

```
weights/
outputs/
runs/
__pycache__/
*.pyc
.pytest_cache/
*.npy
*.h5
*.pt
*.pth
```

- [ ] **Step 3: Write `phase1_restart/README.md`**

```markdown
# phase1-restart

Single-night Phase-1 push targeting Kaggle ≥ 0.7627. Foundation-model fine-tune (CellSAM + MEDIAR), parallel on Modal, with Phase-2 re-run before tonight's 23:55 EST deadline.

See [docs/2026-05-04-phase1-restart-design.md](docs/2026-05-04-phase1-restart-design.md) for design and [docs/2026-05-04-phase1-restart-plan.md](docs/2026-05-04-phase1-restart-plan.md) for implementation plan.

## Run order tonight

1. `python -m phase1_restart.scripts.smoke` — coord sanity gate
2. `python -m phase1_restart.scripts.zero_shot --model cellsam --split val`
3. `python -m phase1_restart.scripts.zero_shot --model mediar --split val`
4. `python -m phase1_restart.scripts.fine_tune --model cellsam --config configs/cellsam.yaml` (Modal, detached)
5. `python -m phase1_restart.scripts.fine_tune --model mediar --config configs/mediar.yaml` (Modal, detached)
6. … see plan for full sequence.
```

- [ ] **Step 4: Commit**

```bash
git add phase1_restart/README.md phase1_restart/.gitignore phase1_restart/{pilot,models,scripts,configs,tests}/__init__.py
git commit -m "feat(phase1-restart): project skeleton"
```

---

## Task 2: SegAdapter ABC

**Files:**
- Create: `phase1_restart/pilot/adapter.py`
- Create: `phase1_restart/tests/test_adapter.py`

- [ ] **Step 1: Write the failing test (`tests/test_adapter.py`)**

```python
import numpy as np
import pytest
from phase1_restart.pilot.adapter import SegAdapter

def test_seg_adapter_is_abstract():
    with pytest.raises(TypeError):
        SegAdapter()  # type: ignore[abstract]

def test_concrete_subclass_must_implement_predict():
    class Incomplete(SegAdapter):
        name = "x"
        expects_channels = ["DAPI"]
        runtime = "mps"
    with pytest.raises(TypeError):
        Incomplete()

def test_complete_subclass_instantiates():
    class Complete(SegAdapter):
        name = "x"
        expects_channels = ["DAPI"]
        runtime = "mps"
        def load_pretrained(self): pass
        def predict(self, image): return np.zeros(image.shape[1:], dtype=np.int32)
        def fine_tune(self, train_fovs, val_fovs, output_dir, n_epochs, **hp):
            from pathlib import Path
            return Path("/tmp/fake.pt")
        def load_checkpoint(self, path): pass
    a = Complete()
    out = a.predict(np.zeros((1, 8, 8), dtype=np.float32))
    assert out.shape == (8, 8)
    assert out.dtype == np.int32
```

- [ ] **Step 2: Run, verify FAIL**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_adapter.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write `pilot/adapter.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class SegAdapter(ABC):
    """Abstract segmentation adapter. One concrete subclass per foundation model.

    Subclasses MUST set class attrs: name (str), expects_channels (list[str]), runtime (str).
    """

    name: str
    expects_channels: list[str]
    runtime: str  # "mps" | "modal"

    @abstractmethod
    def load_pretrained(self) -> None:
        """Download / instantiate the pretrained model weights."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict instance mask.

        image: (C, H, W) float32 in [0, 1], channel order matches expects_channels.
        returns: (H, W) int32, 0 = background, 1..N = cell IDs.
        """

    @abstractmethod
    def fine_tune(
        self,
        train_fovs: list[str],
        val_fovs: list[str],
        output_dir: Path,
        n_epochs: int,
        **hparams,
    ) -> Path:
        """Fine-tune from current weights. Returns path to best-val-ARI checkpoint."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load a fine-tuned checkpoint from disk."""

    def predict_tiled(
        self, image: np.ndarray, tile: int = 512, overlap: int = 64
    ) -> np.ndarray:
        """OOM-safe tiled prediction. Default impl: tile, predict, stitch with mask-id offset."""
        C, H, W = image.shape
        out = np.zeros((H, W), dtype=np.int32)
        next_id = 1
        step = tile - overlap
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                patch = image[:, y0:y1, x0:x1]
                if patch.shape[1] < 64 or patch.shape[2] < 64:
                    continue
                m = self.predict(patch)
                m_relabeled = np.where(m > 0, m + next_id - 1, 0).astype(np.int32)
                # union-take into out where out==0
                empty = out[y0:y1, x0:x1] == 0
                out[y0:y1, x0:x1] = np.where(empty, m_relabeled, out[y0:y1, x0:x1])
                if m.max() > 0:
                    next_id += int(m.max())
        return out
```

- [ ] **Step 4: Run, verify PASS**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_adapter.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/pilot/adapter.py phase1_restart/tests/test_adapter.py
git commit -m "feat(phase1-restart): SegAdapter ABC with tiled-predict default"
```

---

## Task 3: Data layer (`pilot/data.py`)

**Files:**
- Create: `phase1_restart/pilot/data.py`
- Create: `phase1_restart/tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# phase1_restart/tests/test_data.py
import numpy as np
from phase1_restart.pilot.data import (
    FOV_SPLITS,
    load_fov_channels,
    spot_density_map,
    list_fovs,
)

def test_splits_disjoint_and_complete():
    train, val, tp, test = FOV_SPLITS["train"], FOV_SPLITS["val"], FOV_SPLITS["test_proxy"], FOV_SPLITS["test"]
    all_train_set = set(train) | set(val) | set(tp)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(tp)) == 0
    assert len(set(val) & set(tp)) == 0
    assert all_train_set == {f"FOV_{i:03d}" for i in range(1, 41)}
    assert set(test) == {"FOV_A", "FOV_B", "FOV_C", "FOV_D"}

def test_load_fov_channels_dapi_polyt_returns_chw_float():
    img = load_fov_channels("FOV_001", channels=["DAPI", "polyT"])
    assert img.shape == (2, 2048, 2048)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0

def test_load_fov_channels_with_spot_density():
    img = load_fov_channels("FOV_001", channels=["polyT", "DAPI", "spot_density"])
    assert img.shape == (3, 2048, 2048)
    assert img.dtype == np.float32

def test_spot_density_nonzero_inside_cells():
    sd = spot_density_map("FOV_001", sigma=8.0)
    assert sd.shape == (2048, 2048)
    assert sd.max() > 0
    assert sd.dtype == np.float32

def test_list_fovs_test_set():
    fovs = list_fovs("test")
    assert fovs == ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
```

- [ ] **Step 2: Run, verify FAIL**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_data.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write `pilot/data.py`**

```python
"""FOV loaders, channel assembly, splits, GT cache.

Reuses phase1/src/io and phase1/src/coords as read-only imports.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Ensure phase1 is importable as a sibling package
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase1.src import io as p1_io  # type: ignore  # noqa: E402

PHASE1_DATA = REPO_ROOT / "phase1" / "data"
TRAIN_DIR = PHASE1_DATA / "train"
TEST_DIR = PHASE1_DATA / "test"
REFERENCE_DIR = PHASE1_DATA / "reference"
SPOTS_TRAIN = TRAIN_DIR / "ground_truth" / "spots_train.csv"
SPOTS_TEST = PHASE1_DATA / "test_spots.csv"
FOV_METADATA = REFERENCE_DIR / "fov_metadata.csv"
SAMPLE_SUBMISSION = PHASE1_DATA / "sample_submission.csv"

PIXEL_SIZE = 0.109  # µm/px
IMAGE_SIZE = 2048

FOV_SPLITS: dict[str, list[str]] = {
    "train": [f"FOV_{i:03d}" for i in range(1, 31)],
    "val": [f"FOV_{i:03d}" for i in range(36, 41)],
    "test_proxy": [f"FOV_{i:03d}" for i in range(31, 36)],
    "test": ["FOV_A", "FOV_B", "FOV_C", "FOV_D"],
}


def list_fovs(split: str) -> list[str]:
    return FOV_SPLITS[split]


def _max_project(zstack: np.ndarray) -> np.ndarray:
    return zstack.max(axis=0).astype(np.float32)


def _normalize_unit(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1.0, 99.5])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def spot_density_map(fov: str, sigma: float = 8.0) -> np.ndarray:
    spots_csv = SPOTS_TRAIN if fov.startswith("FOV_") and fov[4:].isdigit() else SPOTS_TEST
    df = pd.read_csv(spots_csv)
    df_fov = df[df["fov"] == fov]
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    rows = df_fov["image_row"].astype(int).clip(0, IMAGE_SIZE - 1).values
    cols = df_fov["image_col"].astype(int).clip(0, IMAGE_SIZE - 1).values
    np.add.at(canvas, (rows, cols), 1.0)
    blurred = gaussian_filter(canvas, sigma=sigma).astype(np.float32)
    return _normalize_unit(blurred)


def load_fov_channels(fov: str, channels: list[str]) -> np.ndarray:
    """Returns (C, H, W) float32 in [0, 1], channel order = `channels`."""
    is_train = fov.startswith("FOV_") and fov[4:].isdigit()
    fov_dir = (TRAIN_DIR if is_train else TEST_DIR) / fov
    dapi_z, polyt_z = p1_io.load_fov_images(fov_dir)
    dapi = _normalize_unit(_max_project(dapi_z))
    polyt = _normalize_unit(_max_project(polyt_z))
    band: dict[str, np.ndarray] = {"DAPI": dapi, "polyT": polyt}
    if "spot_density" in channels:
        band["spot_density"] = spot_density_map(fov, sigma=8.0)
    return np.stack([band[c] for c in channels], axis=0)
```

- [ ] **Step 4: Run, verify PASS**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_data.py -v
```
Expected: 5 passed. If `phase1.src.io.load_fov_images` has a different signature, fix the call here (don't modify `phase1/`).

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/pilot/data.py phase1_restart/tests/test_data.py
git commit -m "feat(phase1-restart): data layer (FOV loaders, channels, splits)"
```

---

## Task 4: Eval layer (`pilot/eval.py`)

**Files:**
- Create: `phase1_restart/pilot/eval.py`
- Create: `phase1_restart/tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# phase1_restart/tests/test_eval.py
import numpy as np
import pandas as pd
from phase1_restart.pilot.eval import assign_spots_to_mask, compute_per_fov_ari

def test_assign_spots_to_mask_simple():
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 1
    mask[7:9, 7:9] = 2
    spots = pd.DataFrame({
        "spot_id": [0, 1, 2],
        "fov": ["FOV_001"] * 3,
        "image_row": [3, 8, 0],
        "image_col": [3, 8, 0],
    })
    out = assign_spots_to_mask(spots, mask, fov="FOV_001")
    assert list(out["cluster_id"]) == ["FOV_001_1", "FOV_001_2", "background"]

def test_perfect_ari_is_1():
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 1
    spots = pd.DataFrame({
        "spot_id": [0, 1],
        "fov": ["FOV_001"] * 2,
        "image_row": [3, 0],
        "image_col": [3, 0],
    })
    pred = assign_spots_to_mask(spots, mask, fov="FOV_001")
    truth = pred.copy()
    ari = compute_per_fov_ari(truth, pred)
    assert ari["FOV_001"] == 1.0
```

- [ ] **Step 2: Run, verify FAIL**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_eval.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write `pilot/eval.py`**

```python
"""Per-FOV ARI evaluation. Spot assignment via mask lookup using pre-computed image_row/image_col."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def assign_spots_to_mask(spots: pd.DataFrame, mask: np.ndarray, fov: str) -> pd.DataFrame:
    """Look up mask label per spot. Output columns: spot_id, fov, cluster_id (str).

    cluster_id == "background" if mask label is 0; else f"{fov}_{int(label)}" (FOV-namespaced).
    """
    df = spots[spots["fov"] == fov].copy()
    rows = df["image_row"].astype(int).clip(0, mask.shape[0] - 1).values
    cols = df["image_col"].astype(int).clip(0, mask.shape[1] - 1).values
    labels = mask[rows, cols]
    cluster_ids = np.where(
        labels == 0,
        np.array(["background"] * len(labels), dtype=object),
        np.array([f"{fov}_{int(l)}" for l in labels], dtype=object),
    )
    df = df.assign(cluster_id=cluster_ids)
    return df[["spot_id", "fov", "cluster_id"]].reset_index(drop=True)


def compute_per_fov_ari(
    truth: pd.DataFrame, pred: pd.DataFrame
) -> dict[str, float]:
    """Both DFs have spot_id, fov, cluster_id. Returns {fov: ARI}."""
    out: dict[str, float] = {}
    merged = truth.merge(pred, on=["spot_id", "fov"], suffixes=("_t", "_p"))
    for fov in merged["fov"].unique():
        sub = merged[merged["fov"] == fov]
        out[fov] = float(adjusted_rand_score(sub["cluster_id_t"], sub["cluster_id_p"]))
    return out


def mean_ari(per_fov: dict[str, float]) -> float:
    if not per_fov:
        return 0.0
    return float(np.mean(list(per_fov.values())))
```

- [ ] **Step 4: Run, verify PASS**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_eval.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/pilot/eval.py phase1_restart/tests/test_eval.py
git commit -m "feat(phase1-restart): per-FOV ARI evaluation"
```

---

## Task 5: Submission layer (`pilot/submission.py`)

**Files:**
- Create: `phase1_restart/pilot/submission.py`
- Create: `phase1_restart/tests/test_submission.py`

- [ ] **Step 1: Write the failing test**

```python
# phase1_restart/tests/test_submission.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from phase1_restart.pilot.submission import build_submission, validate_submission

def test_build_submission_uses_pre_computed_coords(tmp_path):
    test_spots = pd.DataFrame({
        "spot_id": [0, 1, 2],
        "fov": ["FOV_A", "FOV_A", "FOV_B"],
        "image_row": [3, 100, 5],
        "image_col": [3, 100, 5],
    })
    masks = {
        "FOV_A": np.zeros((10, 10), dtype=np.int32),
        "FOV_B": np.zeros((10, 10), dtype=np.int32),
    }
    masks["FOV_A"][2:5, 2:5] = 1
    masks["FOV_B"][4:7, 4:7] = 1
    out = build_submission(test_spots, masks)
    assert list(out.columns) == ["spot_id", "fov", "cluster_id"]
    assert out.loc[out["spot_id"] == 0, "cluster_id"].iloc[0] == "FOV_A_1"
    assert out.loc[out["spot_id"] == 1, "cluster_id"].iloc[0] == "background"
    assert out.loc[out["spot_id"] == 2, "cluster_id"].iloc[0] == "FOV_B_1"

def test_validate_submission_against_sample(tmp_path):
    sample = pd.DataFrame({
        "spot_id": [0, 1, 2, 3],
        "fov": ["FOV_A"] * 4,
        "cluster_id": ["background"] * 4,
    })
    sample_csv = tmp_path / "sample.csv"
    sample.to_csv(sample_csv, index=False)
    good = sample.copy()
    good["cluster_id"] = ["FOV_A_1", "background", "FOV_A_2", "background"]
    validate_submission(good, sample_csv)  # should not raise

    bad_count = good.iloc[:3].copy()
    with pytest.raises(ValueError, match="row count"):
        validate_submission(bad_count, sample_csv)

    bad_int_id = good.copy()
    bad_int_id["cluster_id"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="non-empty string"):
        validate_submission(bad_int_id, sample_csv)
```

- [ ] **Step 2: Run, verify FAIL**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_submission.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Write `pilot/submission.py`**

```python
"""Mask → Phase-1 Kaggle CSV. Uses pre-computed image_row/image_col from test_spots.csv."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from phase1_restart.pilot.eval import assign_spots_to_mask


def build_submission(
    test_spots: pd.DataFrame, masks: dict[str, np.ndarray]
) -> pd.DataFrame:
    """test_spots: must have spot_id, fov, image_row, image_col.
    masks: {fov_name: (H, W) int32}. Missing FOVs → all spots in that FOV → "background".
    """
    parts: list[pd.DataFrame] = []
    for fov, df in test_spots.groupby("fov"):
        if fov in masks:
            parts.append(assign_spots_to_mask(test_spots, masks[fov], fov=fov))
        else:
            sub = df.assign(cluster_id="background")
            parts.append(sub[["spot_id", "fov", "cluster_id"]])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("spot_id").reset_index(drop=True)
    return out[["spot_id", "fov", "cluster_id"]]


def validate_submission(submission: pd.DataFrame, sample_path: Path) -> None:
    """Hard-fail if structure doesn't match sample_submission."""
    sample = pd.read_csv(sample_path)
    if len(submission) != len(sample):
        raise ValueError(
            f"row count mismatch: submission={len(submission)} sample={len(sample)}"
        )
    if not (submission["spot_id"].values == sample["spot_id"].values).all():
        raise ValueError("spot_id sequence does not match sample submission")
    if not submission["cluster_id"].apply(lambda x: isinstance(x, str) and len(x) > 0).all():
        raise ValueError("cluster_id must be non-empty string for every row")
    if not set(submission["fov"]).issubset(set(sample["fov"])):
        raise ValueError(f"unexpected fov values: {set(submission['fov']) - set(sample['fov'])}")


def write_submission(submission: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    return out_path
```

- [ ] **Step 4: Run, verify PASS**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_submission.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/pilot/submission.py phase1_restart/tests/test_submission.py
git commit -m "feat(phase1-restart): submission CSV builder + structural validator"
```

---

## Task 6: Smoke test (coord-sanity gate)

**Files:**
- Create: `phase1_restart/scripts/smoke.py`
- Create: `phase1_restart/tests/test_smoke.py`

This task uses pretrained Cellpose `cyto2` from the cellpose package as a coord-sanity anchor only — see spec §6, decision 1. It is NOT a candidate or warm-start.

- [ ] **Step 1: Write `scripts/smoke.py`**

```python
"""Coord-sanity gate. Runs pretrained cyto2 on FOV_001, asserts:
  - in-cell DAPI mean / out-of-cell DAPI mean >= 2.0
  - per-FOV ARI in [0.55, 0.70] (cyto2 baseline ≈ 0.632)

Hard-halt on any failure: do not proceed with adapter work if coords are broken.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from phase1_restart.pilot.data import (
    load_fov_channels,
    SPOTS_TRAIN,
    SAMPLE_SUBMISSION,
)
from phase1_restart.pilot.eval import assign_spots_to_mask, compute_per_fov_ari, mean_ari


def in_cell_ratio(mask: np.ndarray, dapi: np.ndarray) -> float:
    in_cell = dapi[mask > 0]
    out_cell = dapi[mask == 0]
    if len(in_cell) == 0 or len(out_cell) == 0:
        return 0.0
    return float(in_cell.mean() / max(out_cell.mean(), 1e-6))


def main() -> int:
    fov = "FOV_001"
    img = load_fov_channels(fov, channels=["DAPI", "polyT"])
    dapi = img[0]
    polyt = img[1]

    from cellpose import models  # lazy import; large dep

    model = models.CellposeModel(model_type="cyto2", gpu=False)
    masks, _flows, _styles, _diams = model.eval(
        np.stack([polyt, dapi], axis=0),
        diameter=88.9,
        channel_axis=0,
        channels=[1, 2],
    )
    mask = masks.astype(np.int32)

    ratio = in_cell_ratio(mask, dapi)
    print(f"in-cell DAPI / out-of-cell DAPI = {ratio:.2f} (require ≥ 2.0)")
    if ratio < 2.0:
        print("FAIL: coord convention is broken — DAPI is not enriched inside masks.")
        return 1

    spots = pd.read_csv(SPOTS_TRAIN)
    spots_fov = spots[spots["fov"] == fov]
    pred = assign_spots_to_mask(spots_fov, mask, fov=fov)
    truth_path = Path(__file__).resolve().parents[1] / "outputs" / "smoke_truth_fov001.csv"
    if not truth_path.exists():
        # Build truth from polygon GT once.
        from phase1.src import train_cellpose as p1_tc  # type: ignore
        from phase1.src import io as p1_io  # type: ignore
        boundaries_csv = (
            Path(__file__).resolve().parents[2] / "phase1" / "data" / "train"
            / "ground_truth" / "cell_boundaries_train.csv"
        )
        gt_mask = p1_tc.boundaries_to_mask(
            pd.read_csv(boundaries_csv), fov_id=fov
        )
        truth = assign_spots_to_mask(spots_fov, gt_mask.astype(np.int32), fov=fov)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth.to_csv(truth_path, index=False)
    truth = pd.read_csv(truth_path)

    per_fov = compute_per_fov_ari(truth, pred)
    ari = mean_ari(per_fov)
    print(f"FOV_001 ARI vs polygon GT = {ari:.4f} (require 0.55 ≤ ARI ≤ 0.70)")
    if not (0.55 <= ari <= 0.70):
        print("FAIL: cyto2 baseline ARI outside expected range — likely coord bug.")
        return 1

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `tests/test_smoke.py` (in_cell_ratio unit test)**

```python
import numpy as np
from phase1_restart.scripts.smoke import in_cell_ratio

def test_in_cell_ratio_known_values():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1:3, 1:3] = 1
    dapi = np.array([
        [0, 0, 0, 0],
        [0, 4, 4, 0],
        [0, 4, 4, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)
    assert in_cell_ratio(mask, dapi) == float("inf") or in_cell_ratio(mask, dapi) > 100
    dapi[0, 0] = 2.0
    r = in_cell_ratio(mask, dapi)
    assert 5.0 < r < 50.0  # in-cell mean 4, out-cell mean ~0.18
```

- [ ] **Step 3: Run unit test (without invoking cellpose)**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_smoke.py -v
```
Expected: 1 passed.

- [ ] **Step 4: Run smoke**

```bash
PYTHONPATH=. python phase1_restart/scripts/smoke.py
```
Expected stdout ends with `SMOKE PASS`. If it fails, **HALT and debug** — do not proceed to adapter work. Most likely fix sites: row/col swap, axis flip, train_cellpose.boundaries_to_mask signature mismatch.

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/scripts/smoke.py phase1_restart/tests/test_smoke.py
git commit -m "feat(phase1-restart): coord-sanity smoke gate (cyto2 baseline anchor)"
```

---

## Task 7: CellSAM adapter (zero-shot path only)

**Files:**
- Create: `phase1_restart/models/cellsam.py`
- Create: `phase1_restart/tests/test_cellsam_smoke.py`

CellSAM API surface is runtime-discovered. The contract is: implement `load_pretrained()` and `predict()` such that calling them on FOV_001 produces a mask with `mask.max() >= 5` (FOV has many cells). `fine_tune` and `load_checkpoint` get stubbed for now and filled in Task 12.

- [ ] **Step 1: Install CellSAM and discover API**

```bash
.venv/bin/pip install cellSAM
.venv/bin/python -c "from cellSAM import segment_cellular_image; help(segment_cellular_image)" | head -40
```
Note the function signature. Most likely: `segment_cellular_image(img, **kwargs) -> mask_array`.

- [ ] **Step 2: Write `models/cellsam.py`**

```python
"""CellSAM adapter (zero-shot + fine-tune)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from phase1_restart.pilot.adapter import SegAdapter


class CellSAMAdapter(SegAdapter):
    name = "cellsam"
    expects_channels = ["DAPI", "polyT"]
    runtime = "mps"

    def __init__(self):
        self._model = None
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_pretrained(self) -> None:
        from cellSAM import segment_cellular_image  # noqa: F401
        # CellSAM lazily downloads weights on first call. Trigger it once so subsequent
        # predict() calls don't pay the download.
        self._predict_fn = segment_cellular_image

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._predict_fn is None:
            self.load_pretrained()
        # CellSAM expects (H, W) or (H, W, C). Convert from our (C, H, W).
        img_hwc = np.transpose(image, (1, 2, 0))
        mask, _ = self._predict_fn(img_hwc, device=self._device, normalize=True)
        return mask.astype(np.int32)

    def fine_tune(self, train_fovs, val_fovs, output_dir: Path, n_epochs: int, **hp) -> Path:
        # Filled in Task 12. Stub raises clear error if accidentally called.
        raise NotImplementedError("Fine-tune is provided by modal_app.fine_tune_cellsam (Task 12)")

    def load_checkpoint(self, path: Path) -> None:
        if self._predict_fn is None:
            self.load_pretrained()
        from cellSAM import get_model
        state = torch.load(path, map_location=self._device)
        model = get_model()
        model.load_state_dict(state)
        # CellSAM exposes the underlying model via a global; rebind the predict fn.
        # If the API differs from this assumption, override here at runtime.
        self._model = model
```

If the actual `cellSAM` API differs (e.g. different function name, different return shape), edit this file at runtime and re-run. **Do not modify other files** to compensate.

- [ ] **Step 3: Smoke-test on FOV_001**

```python
# phase1_restart/tests/test_cellsam_smoke.py
import pytest
import numpy as np
from phase1_restart.pilot.data import load_fov_channels
from phase1_restart.models.cellsam import CellSAMAdapter

@pytest.mark.slow
def test_cellsam_predicts_multiple_cells_on_fov_001():
    a = CellSAMAdapter()
    a.load_pretrained()
    img = load_fov_channels("FOV_001", channels=a.expects_channels)
    mask = a.predict(img)
    assert mask.shape == (2048, 2048)
    assert mask.dtype == np.int32
    assert mask.max() >= 5, f"only {mask.max()} cells detected — sanity floor failed"
```

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_cellsam_smoke.py -v -m slow
```
Expected: 1 passed (~30-60s).

- [ ] **Step 4: Commit**

```bash
git add phase1_restart/models/cellsam.py phase1_restart/tests/test_cellsam_smoke.py
git commit -m "feat(phase1-restart): CellSAM adapter (zero-shot)"
```

---

## Task 8: Zero-shot script + run on val

**Files:**
- Create: `phase1_restart/scripts/zero_shot.py`
- Create: `phase1_restart/models/__init__.py`

- [ ] **Step 1: Write `models/__init__.py`**

```python
from phase1_restart.models.cellsam import CellSAMAdapter

REGISTRY = {
    "cellsam": CellSAMAdapter,
}
```

(MEDIAR will be added in Task 9.)

- [ ] **Step 2: Write `scripts/zero_shot.py`**

```python
"""Zero-shot inference on a split, writing per-FOV masks + ARI summary.

usage: python -m phase1_restart.scripts.zero_shot --model cellsam --split val
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from phase1_restart.pilot.data import (
    list_fovs,
    load_fov_channels,
    SPOTS_TRAIN,
    SPOTS_TEST,
)
from phase1_restart.pilot.eval import assign_spots_to_mask, compute_per_fov_ari, mean_ari
from phase1_restart.models import REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--split", required=True, choices=["val", "test_proxy", "test"])
    ap.add_argument("--out-dir", default="phase1_restart/outputs/zero_shot")
    args = ap.parse_args()

    out_root = Path(args.out_dir) / args.model / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    adapter = REGISTRY[args.model]()
    adapter.load_pretrained()

    fovs = list_fovs(args.split)
    masks_per_fov: dict[str, np.ndarray] = {}
    for fov in fovs:
        print(f"[{args.model}] predicting {fov} ...", flush=True)
        img = load_fov_channels(fov, channels=adapter.expects_channels)
        try:
            mask = adapter.predict(img)
        except (RuntimeError, MemoryError) as e:
            print(f"  full-frame predict OOM/error ({e}); falling back to tiled.", flush=True)
            mask = adapter.predict_tiled(img, tile=512, overlap=64)
        np.save(out_root / f"{fov}_mask.npy", mask)
        masks_per_fov[fov] = mask

    # Score on val and test_proxy. Test (FOV_A-D) has no GT.
    summary = {"model": args.model, "split": args.split, "fovs": fovs}
    if args.split in {"val", "test_proxy"}:
        spots = pd.read_csv(SPOTS_TRAIN)
        truth_rows: list[pd.DataFrame] = []
        pred_rows: list[pd.DataFrame] = []
        # Build truth via polygon GT on the fly.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from phase1.src import train_cellpose as p1_tc  # type: ignore
        boundaries_csv = (
            Path(__file__).resolve().parents[2] / "phase1" / "data" / "train"
            / "ground_truth" / "cell_boundaries_train.csv"
        )
        boundaries = pd.read_csv(boundaries_csv)
        for fov in fovs:
            spots_fov = spots[spots["fov"] == fov]
            gt_mask = p1_tc.boundaries_to_mask(boundaries, fov_id=fov).astype(np.int32)
            truth_rows.append(assign_spots_to_mask(spots_fov, gt_mask, fov=fov))
            pred_rows.append(assign_spots_to_mask(spots_fov, masks_per_fov[fov], fov=fov))
        truth = pd.concat(truth_rows, ignore_index=True)
        pred = pd.concat(pred_rows, ignore_index=True)
        per_fov = compute_per_fov_ari(truth, pred)
        summary["per_fov_ari"] = per_fov
        summary["mean_ari"] = mean_ari(per_fov)
        print(f"[{args.model}/{args.split}] mean ARI = {summary['mean_ari']:.4f}")

    runs_dir = Path("phase1_restart/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / f"zero_shot_{args.model}_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run on val**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model cellsam --split val
```
Expected: 5 mask files in `outputs/zero_shot/cellsam/val/` + `runs/zero_shot_cellsam_val.json` with `mean_ari`. Wall time ~3-8 min on MPS.

- [ ] **Step 4: Commit**

```bash
git add phase1_restart/models/__init__.py phase1_restart/scripts/zero_shot.py
git commit -m "feat(phase1-restart): zero-shot inference script + run CellSAM on val"
```

---

## Task 9: MEDIAR adapter (zero-shot path)

**Files:**
- Create: `phase1_restart/models/mediar.py`
- Modify: `phase1_restart/models/__init__.py` (register)
- Create: `phase1_restart/tests/test_mediar_smoke.py`

- [ ] **Step 1: Clone MEDIAR + grab weights**

```bash
mkdir -p phase1_restart/external && cd phase1_restart/external
git clone --depth 1 https://github.com/Lee-Gihun/MEDIAR.git
cd MEDIAR
# Per their README, weights live on Hugging Face:
# https://huggingface.co/Lee-Gihun/MEDIAR — fetch the .pth checkpoint into ./weights/
# Adjust filename to whatever they ship; current as of 2024 is `from_phase1.pth` and `from_phase2.pth`.
mkdir -p weights && cd weights
curl -L -o from_phase1.pth https://huggingface.co/Lee-Gihun/MEDIAR/resolve/main/from_phase1.pth
curl -L -o from_phase2.pth https://huggingface.co/Lee-Gihun/MEDIAR/resolve/main/from_phase2.pth
cd ../../../..
```
If the HF URLs 404, check the MEDIAR README for the current weights link and update commands accordingly.

- [ ] **Step 2: Write `models/mediar.py`**

```python
"""MEDIAR adapter (zero-shot + fine-tune).

MEDIAR is not pip-installable; we add their cloned repo to sys.path.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

from phase1_restart.pilot.adapter import SegAdapter

MEDIAR_REPO = Path(__file__).resolve().parents[1] / "external" / "MEDIAR"
sys.path.insert(0, str(MEDIAR_REPO))


class MediarAdapter(SegAdapter):
    name = "mediar"
    expects_channels = ["polyT", "DAPI", "spot_density"]
    runtime = "mps"

    def __init__(self):
        self._model = None
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_pretrained(self) -> None:
        # MEDIAR ships their model class as `core.MEDIAR_Trainer.Trainer` or similar.
        # Inspect their `predict.py` for the canonical loading pattern; common entrypoint
        # is `from train_tools.models.MEDIARFormer import MEDIARFormer`.
        from train_tools.models.MEDIARFormer import MEDIARFormer  # type: ignore
        ckpt = MEDIAR_REPO / "weights" / "from_phase2.pth"
        model = MEDIARFormer()
        state = torch.load(str(ckpt), map_location=self._device)
        model.load_state_dict(state, strict=False)
        model.to(self._device).eval()
        self._model = model

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load_pretrained()
        # MEDIAR expects (B, 3, H, W) tensor in [0, 1].
        x = torch.from_numpy(image).unsqueeze(0).to(self._device)
        # Their predict pipeline does flow + cellprob → instances. If their helper
        # function `core.utils.cell_distance_post_processing` exists, use it.
        from core.utils import postprocess_predictions  # type: ignore
        outputs = self._model(x)
        masks = postprocess_predictions(outputs.cpu().numpy())
        return masks[0].astype(np.int32)

    def fine_tune(self, train_fovs, val_fovs, output_dir: Path, n_epochs: int, **hp) -> Path:
        raise NotImplementedError("Fine-tune is provided by modal_app.fine_tune_mediar (Task 12)")

    def load_checkpoint(self, path: Path) -> None:
        from train_tools.models.MEDIARFormer import MEDIARFormer  # type: ignore
        model = MEDIARFormer()
        model.load_state_dict(torch.load(path, map_location=self._device))
        model.to(self._device).eval()
        self._model = model
```

The `from train_tools.models.MEDIARFormer import MEDIARFormer` and `core.utils.postprocess_predictions` are educated guesses based on MEDIAR's repo layout circa NeurIPS '22. **Confirm against the actual repo** before running. If their entrypoint differs, fix here only.

- [ ] **Step 3: Update `models/__init__.py`**

```python
from phase1_restart.models.cellsam import CellSAMAdapter
from phase1_restart.models.mediar import MediarAdapter

REGISTRY = {
    "cellsam": CellSAMAdapter,
    "mediar": MediarAdapter,
}
```

- [ ] **Step 4: Smoke-test MEDIAR on FOV_001**

```python
# phase1_restart/tests/test_mediar_smoke.py
import pytest
import numpy as np
from phase1_restart.pilot.data import load_fov_channels
from phase1_restart.models.mediar import MediarAdapter

@pytest.mark.slow
def test_mediar_predicts_multiple_cells_on_fov_001():
    a = MediarAdapter()
    a.load_pretrained()
    img = load_fov_channels("FOV_001", channels=a.expects_channels)
    mask = a.predict(img)
    assert mask.shape == (2048, 2048)
    assert mask.dtype == np.int32
    assert mask.max() >= 5
```

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_mediar_smoke.py -v -m slow
```
Expected: 1 passed.

- [ ] **Step 5: Run zero-shot MEDIAR on val**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model mediar --split val
```
Wall time ~5-10 min on MPS.

- [ ] **Step 6: Commit**

```bash
git add phase1_restart/models/mediar.py phase1_restart/models/__init__.py phase1_restart/tests/test_mediar_smoke.py
git commit -m "feat(phase1-restart): MEDIAR adapter (zero-shot)"
```

---

## Task 10: Zero-shot on test-proxy + test, build summary

**Files:**
- Create: `phase1_restart/scripts/build_zero_shot_summary.py`

- [ ] **Step 1: Run zero-shot on test-proxy + test for both models**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model cellsam --split test_proxy
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model cellsam --split test
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model mediar --split test_proxy
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model mediar --split test
```

- [ ] **Step 2: Write summary aggregator**

```python
# phase1_restart/scripts/build_zero_shot_summary.py
"""Aggregate per-model per-split zero-shot ARIs into one decision-ready table."""
from __future__ import annotations
import json
import sys
from pathlib import Path

RUNS = Path("phase1_restart/runs")

def main() -> int:
    rows: list[dict] = []
    for jpath in sorted(RUNS.glob("zero_shot_*.json")):
        with open(jpath) as f:
            d = json.load(f)
        rows.append({
            "model": d["model"],
            "split": d["split"],
            "mean_ari": d.get("mean_ari", None),
            "fovs": d["fovs"],
        })
    out = {"zero_shot_summary": rows}
    with open(RUNS / "zero_shot_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    # Decision gate: any candidate ≥ 0.50 on val?
    val_aris = [r["mean_ari"] for r in rows if r["split"] == "val" and r["mean_ari"] is not None]
    if not val_aris or max(val_aris) < 0.50:
        print("\nDECISION GATE: all candidates < 0.50 on val. Foundation-model thesis weak.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run, record decision**

```bash
PYTHONPATH=. python phase1_restart/scripts/build_zero_shot_summary.py
echo "$(date -Iseconds) | block-B | zero-shot-summary | exit=$?" >> phase1_restart/runs/decisions.log
```

If exit != 0: stop adapter work, skip to Phase-2 re-run only (Block G).

- [ ] **Step 4: Commit**

```bash
git add phase1_restart/scripts/build_zero_shot_summary.py phase1_restart/runs/zero_shot_*.json phase1_restart/runs/zero_shot_summary.json phase1_restart/runs/decisions.log
git commit -m "feat(phase1-restart): zero-shot summary aggregator + first run"
```

---

## Task 11: Ensemble (`pilot/ensemble.py`)

**Files:**
- Create: `phase1_restart/pilot/ensemble.py`
- Create: `phase1_restart/tests/test_ensemble.py`

- [ ] **Step 1: Write the failing test**

```python
# phase1_restart/tests/test_ensemble.py
import numpy as np
import pandas as pd
from phase1_restart.pilot.ensemble import spot_majority_vote

def test_majority_vote_2_of_3_agree():
    spots = pd.DataFrame({
        "spot_id": [0, 1],
        "fov": ["FOV_A", "FOV_A"],
        "image_row": [3, 3],
        "image_col": [3, 3],
    })
    m1 = np.zeros((10, 10), dtype=np.int32); m1[2:5, 2:5] = 1
    m2 = np.zeros((10, 10), dtype=np.int32); m2[2:5, 2:5] = 7
    m3 = np.zeros((10, 10), dtype=np.int32)  # background
    out = spot_majority_vote(spots, masks_per_model={
        "A": {"FOV_A": m1},
        "B": {"FOV_A": m2},
        "C": {"FOV_A": m3},
    })
    # Two models say "in cell" (m1 → A_FOV_A_1, m2 → B_FOV_A_7); they DISAGREE on label.
    # Majority is "in-cell" but no consensus on which cell. Tiebreak by first model with non-bg.
    assert out.loc[out["spot_id"] == 0, "cluster_id"].iloc[0] == "A_FOV_A_1"

def test_majority_all_background():
    spots = pd.DataFrame({"spot_id": [0], "fov": ["FOV_A"], "image_row": [0], "image_col": [0]})
    m = np.zeros((10, 10), dtype=np.int32)
    out = spot_majority_vote(spots, masks_per_model={"A": {"FOV_A": m}, "B": {"FOV_A": m}})
    assert out["cluster_id"].iloc[0] == "background"
```

- [ ] **Step 2: Run, verify FAIL**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_ensemble.py -v
```

- [ ] **Step 3: Write `pilot/ensemble.py`**

```python
"""Spot-level majority-vote ensembling.

Per spot, look up its mask label in every model. Vote rule:
  - if all models say background → background
  - else → take the cluster_id from the FIRST model (in dict order) whose mask is non-bg
    at that pixel. (Heterogeneous cell IDs across models can't be averaged; first-non-bg
    is the cheapest tiebreak that respects "majority-in-cell wins".)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def spot_majority_vote(
    spots: pd.DataFrame, masks_per_model: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    """spots: spot_id, fov, image_row, image_col. masks_per_model: {model: {fov: mask}}."""
    cluster_ids: list[str] = []
    for _, row in spots.iterrows():
        fov = row["fov"]
        r = int(row["image_row"])
        c = int(row["image_col"])
        labels: list[tuple[str, int]] = []
        for model, fov_masks in masks_per_model.items():
            if fov not in fov_masks:
                continue
            m = fov_masks[fov]
            r_clip = min(max(r, 0), m.shape[0] - 1)
            c_clip = min(max(c, 0), m.shape[1] - 1)
            label = int(m[r_clip, c_clip])
            labels.append((model, label))
        if not labels:
            cluster_ids.append("background")
            continue
        non_bg = [(model, lbl) for model, lbl in labels if lbl != 0]
        if len(non_bg) * 2 < len(labels):  # majority background
            cluster_ids.append("background")
        else:
            model, lbl = non_bg[0]
            cluster_ids.append(f"{model}_{fov}_{lbl}")
    out = spots[["spot_id", "fov"]].copy()
    out["cluster_id"] = cluster_ids
    return out
```

- [ ] **Step 4: Run, verify PASS**

```bash
PYTHONPATH=. pytest phase1_restart/tests/test_ensemble.py -v
```

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/pilot/ensemble.py phase1_restart/tests/test_ensemble.py
git commit -m "feat(phase1-restart): spot-level majority-vote ensembler"
```

---

## Task 12: Modal entrypoint + fine-tune scripts

**Files:**
- Create: `phase1_restart/modal_app.py`
- Create: `phase1_restart/configs/cellsam.yaml`
- Create: `phase1_restart/configs/mediar.yaml`
- Create: `phase1_restart/scripts/fine_tune.py`

- [ ] **Step 1: Write `configs/cellsam.yaml`**

```yaml
model: cellsam
epochs: 50
checkpoint_every: 5
patch_size: 512
batch_size: 4
learning_rate: 1.0e-5
weight_decay: 0.01
augmentation:
  flips: true
  rotations: true
  intensity_jitter: 0.2
train_fovs: ${ALL_TRAIN}
val_fovs: ${ALL_VAL}
gpu: A100
timeout_min: 90
```

- [ ] **Step 2: Write `configs/mediar.yaml`**

```yaml
model: mediar
epochs: 50
checkpoint_every: 5
patch_size: 512
batch_size: 2
learning_rate: 1.0e-5
weight_decay: 0.01
augmentation:
  flips: true
  rotations: true
  intensity_jitter: 0.2
train_fovs: ${ALL_TRAIN}
val_fovs: ${ALL_VAL}
gpu: A100
timeout_min: 120
```

- [ ] **Step 3: Write `modal_app.py`**

```python
"""Modal entrypoint: parallel fine-tune on A100s.

Run locally:
  modal run --detach phase1_restart/modal_app.py::fine_tune --config configs/cellsam.yaml
  modal run --detach phase1_restart/modal_app.py::fine_tune --config configs/mediar.yaml
"""
from __future__ import annotations
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyyaml",
        "cellSAM",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Lee-Gihun/MEDIAR.git /opt/MEDIAR",
        "pip install -e /opt/MEDIAR || true",
    )
)

app = modal.App("phase1-restart", image=image)
data_volume = modal.Volume.from_name("cell-seg-data")
workspace_volume = modal.Volume.from_name("cell-seg-workspace", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={"/root/data": data_volume, "/root/workspace": workspace_volume},
)
def fine_tune(config_yaml: str) -> dict:
    """config_yaml: serialized YAML string (passed by client)."""
    import os, yaml, json
    from pathlib import Path
    os.environ["MERFISH_DATA_ROOT"] = "/root/data"
    cfg = yaml.safe_load(config_yaml)
    out_dir = Path(f"/root/workspace/phase1_restart/{cfg['model']}/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train loop dispatches by model name. Implementation lives inside the container
    # and reads /root/data + /root/workspace for inputs/outputs.
    if cfg["model"] == "cellsam":
        from phase1_restart_train.cellsam import train_cellsam
        best = train_cellsam(out_dir=out_dir, **cfg)
    elif cfg["model"] == "mediar":
        from phase1_restart_train.mediar import train_mediar
        best = train_mediar(out_dir=out_dir, **cfg)
    else:
        raise ValueError(f"unknown model {cfg['model']}")

    # Persist a manifest so the local pull step knows what to fetch.
    (out_dir / "FINAL.json").write_text(json.dumps({"best_checkpoint": str(best)}))
    return {"best": str(best)}


@app.local_entrypoint()
def main(config: str):
    """config: path to a YAML file on the local filesystem."""
    from pathlib import Path
    yaml_str = Path(config).read_text()
    result = fine_tune.remote(yaml_str)
    print(result)
```

The `phase1_restart_train.cellsam.train_cellsam` / `phase1_restart_train.mediar.train_mediar` symbols are stubs: in this same task, also create `phase1_restart/train_modal/{cellsam,mediar}.py` (named `phase1_restart_train` as a package on Modal via Mount). Since these are Modal-only and we don't run them locally, we accept the brittleness in exchange for keeping the local repo clean.

- [ ] **Step 4: Write `phase1_restart/train_modal/cellsam.py`**

```python
"""CellSAM fine-tune loop, runs inside Modal container.

Dependencies: torch, cellSAM, the MEDIAR repo at /opt/MEDIAR is on PYTHONPATH so we don't
need it here.
"""
from __future__ import annotations
from pathlib import Path
import json


def train_cellsam(
    out_dir: Path,
    epochs: int,
    checkpoint_every: int,
    patch_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    train_fovs,
    val_fovs,
    augmentation: dict,
    **_unused,
) -> Path:
    # Implementation note: CellSAM's API for fine-tuning is not as well-documented as
    # their inference path. The minimum viable approach:
    #   1. Build a torch Dataset that yields (image, mask) patches via:
    #      phase1_restart.pilot.data.load_fov_channels  +  phase1.src.train_cellpose.boundaries_to_mask
    #   2. Use cellSAM.get_model() to obtain the underlying nn.Module.
    #   3. Train it with their distance-transform loss (or a simple IoU loss as fallback).
    #   4. Every `checkpoint_every` epochs: save state_dict, run val ARI, log val_curve.csv.
    # If their fine-tune harness is exposed, prefer that.
    raise NotImplementedError(
        "Fine-tune loop is intentionally left to runtime — engineer must inspect cellSAM's "
        "current training entrypoint and adapt. Acceptable shortcut for tonight: copy "
        "their `examples/finetune.py` if present, replace data loader with our boundaries→mask helper."
    )
```

The `NotImplementedError` is **deliberate**: this is the part of tonight where the implementer must read upstream library docs at runtime. The plan refuses to invent code that may not match the upstream API.

- [ ] **Step 5: Write `phase1_restart/train_modal/mediar.py`**

```python
"""MEDIAR fine-tune loop, runs inside Modal container."""
from __future__ import annotations
from pathlib import Path


def train_mediar(out_dir: Path, **cfg) -> Path:
    # MEDIAR ships their own training script (`main.py` or similar in the cloned repo).
    # Run it as a subprocess with our config:
    #   subprocess.run(["python", "/opt/MEDIAR/main.py", "--config", "<path>"], check=True)
    # Adapt their config to point at /root/data, write checkpoints to out_dir.
    raise NotImplementedError(
        "MEDIAR fine-tune harness is intentionally left to runtime — engineer must wire "
        "their `main.py --config <our.yml>` to read /root/data and write to out_dir."
    )
```

- [ ] **Step 6: Write `scripts/fine_tune.py` (local launcher)**

```python
"""Local launcher that submits a Modal fine-tune job and prints the run handle.

usage: python -m phase1_restart.scripts.fine_tune --model cellsam --config configs/cellsam.yaml
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cellsam", "mediar"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--detach", action="store_true", default=True)
    args = ap.parse_args()

    cmd = [
        "modal", "run",
        "--detach" if args.detach else "",
        "phase1_restart/modal_app.py::main",
        "--config", str(args.config),
    ]
    cmd = [c for c in cmd if c]
    print("RUN:", " ".join(cmd))
    return subprocess.run(cmd, check=True).returncode


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Commit**

```bash
git add phase1_restart/modal_app.py phase1_restart/configs/ phase1_restart/scripts/fine_tune.py phase1_restart/train_modal/
git commit -m "feat(phase1-restart): Modal fine-tune harness + launcher (training loops stubbed for runtime)"
```

---

## Task 13: Launch parallel Modal fine-tunes

This is operational, not code. Two `modal run --detach` jobs run in parallel. Once submitted, ~2h wall-clock until first checkpoint to pull.

- [ ] **Step 1: Submit CellSAM fine-tune (detached)**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.fine_tune --model cellsam --config phase1_restart/configs/cellsam.yaml
```

Expected: prints a Modal app run URL. Note the URL.

- [ ] **Step 2: Submit MEDIAR fine-tune (detached)**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.fine_tune --model mediar --config phase1_restart/configs/mediar.yaml
```

- [ ] **Step 3: Log launch**

```bash
echo "$(date -Iseconds) | block-D-launch | cellsam_url=$(...) | mediar_url=$(...)" >> phase1_restart/runs/decisions.log
```

- [ ] **Step 4: Commit launch log**

```bash
git add phase1_restart/runs/decisions.log
git commit -m "ops(phase1-restart): Block D — Modal fine-tunes launched"
```

---

## Task 14: While Modal trains — make_submission + rerun_phase2

Modal training is the long-pole (~2h). Use this window to build downstream scripts.

**Files:**
- Create: `phase1_restart/scripts/make_submission.py`
- Create: `phase1_restart/scripts/rerun_phase2.py`

- [ ] **Step 1: Write `scripts/make_submission.py`**

```python
"""Build Phase-1 Kaggle CSV from per-FOV mask .npy files."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from phase1_restart.pilot.data import SPOTS_TEST, SAMPLE_SUBMISSION
from phase1_restart.pilot.submission import build_submission, validate_submission, write_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", required=True, type=Path,
                    help="dir containing FOV_A_mask.npy, FOV_B_mask.npy, ...")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    test_spots = pd.read_csv(SPOTS_TEST)
    masks: dict[str, np.ndarray] = {}
    for fov in ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]:
        npy = args.masks_dir / f"{fov}_mask.npy"
        if not npy.exists():
            print(f"WARNING: {npy} missing — spots in {fov} will be 'background'.")
            continue
        masks[fov] = np.load(npy)
    out = build_submission(test_spots, masks)
    validate_submission(out, SAMPLE_SUBMISSION)
    if args.validate_only:
        print("VALIDATE OK:", args.out)
        return 0
    write_submission(out, args.out)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Sanity-test against zero-shot output**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.make_submission \
  --masks-dir phase1_restart/outputs/zero_shot/cellsam/test \
  --out phase1_restart/outputs/submissions/zero_shot_cellsam.csv
```
Expected: `WROTE …`. If this fails on the validator, the bug is in `build_submission` or `validate_submission` — fix before fine-tunes finish.

- [ ] **Step 3: Write `scripts/rerun_phase2.py`**

```python
"""Re-run the Phase-2 pipeline using new Phase-1 masks.

Strategy: use phase2/autoresearch/run_experiment.py as the entrypoint, with the
MERFISH_MASKS_OVERRIDE env var pointing at our masks dir. If autoresearch doesn't
respect that env var (most likely it doesn't), this script writes a tiny shim
that monkey-patches the mask path before invoking.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", required=True, type=Path)
    ap.add_argument("--full-val", action="store_true")
    args = ap.parse_args()

    # 1. Verify autoresearch entrypoint exists.
    autoresearch = Path("phase2/autoresearch/run_experiment.py")
    if not autoresearch.exists():
        print(f"FAIL: {autoresearch} missing")
        return 1

    # 2. Set override env var. If autoresearch ignores it, fall back to manual cp.
    env = os.environ.copy()
    env["MERFISH_PHASE1_MASKS"] = str(args.masks_dir.resolve())
    cmd = [".venv/bin/python", str(autoresearch)]
    if args.full_val:
        cmd.append("--full-val")
    print("RUN:", " ".join(cmd), "  with MERFISH_PHASE1_MASKS=", env["MERFISH_PHASE1_MASKS"])
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Pre-warm Phase-2 env (no full run yet)**

```bash
.venv/bin/python phase2/autoresearch/run_experiment.py --help 2>&1 | head -20
```
Expected: prints the autoresearch CLI options. Confirms the venv + deps are intact for Block G.

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/scripts/make_submission.py phase1_restart/scripts/rerun_phase2.py
git commit -m "feat(phase1-restart): submission builder + Phase-2 re-run wrapper"
```

---

## Task 15: Pull best checkpoints from Modal

This is operational. Modal training should be ~done by now (~2h after launch).

- [ ] **Step 1: Verify Modal jobs finished**

```bash
modal app list | grep phase1-restart
modal volume ls cell-seg-workspace phase1_restart/
```
Expected: `cellsam/exp/FINAL.json` and `mediar/exp/FINAL.json` present.

- [ ] **Step 2: Pull best checkpoints**

```bash
mkdir -p phase1_restart/weights/cellsam phase1_restart/weights/mediar
modal volume get cell-seg-workspace phase1_restart/cellsam/exp/best.pt phase1_restart/weights/cellsam/best.pt
modal volume get cell-seg-workspace phase1_restart/mediar/exp/best.pt phase1_restart/weights/mediar/best.pt
```

If `modal volume get` only fetches one file (per memory `feedback_modal_volume_get.md`), pre-create the destination dir and re-run.

- [ ] **Step 3: Log decision**

```bash
echo "$(date -Iseconds) | block-D-end | checkpoints_pulled" >> phase1_restart/runs/decisions.log
```

- [ ] **Step 4: Commit log**

```bash
git add phase1_restart/runs/decisions.log
git commit -m "ops(phase1-restart): Block D — best checkpoints pulled from Modal"
```

---

## Task 16: Test-set inference + test-proxy scoring

**Files:**
- Create: `phase1_restart/scripts/infer_test.py`

- [ ] **Step 1: Write `scripts/infer_test.py`**

```python
"""Test-set inference using a fine-tuned checkpoint.

usage: python -m phase1_restart.scripts.infer_test --model cellsam --checkpoint phase1_restart/weights/cellsam/best.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

from phase1_restart.pilot.data import list_fovs, load_fov_channels
from phase1_restart.models import REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adapter = REGISTRY[args.model]()
    adapter.load_pretrained()
    adapter.load_checkpoint(args.checkpoint)

    for fov in list_fovs("test"):
        print(f"[{args.model}] inferring {fov} ...", flush=True)
        img = load_fov_channels(fov, channels=adapter.expects_channels)
        try:
            mask = adapter.predict(img)
        except (RuntimeError, MemoryError):
            mask = adapter.predict_tiled(img, tile=512, overlap=64)
        np.save(args.out_dir / f"{fov}_mask.npy", mask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run for both fine-tuned models**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.infer_test \
  --model cellsam --checkpoint phase1_restart/weights/cellsam/best.pt \
  --out-dir phase1_restart/outputs/test/cellsam_ft

PYTHONPATH=. python -m phase1_restart.scripts.infer_test \
  --model mediar --checkpoint phase1_restart/weights/mediar/best.pt \
  --out-dir phase1_restart/outputs/test/mediar_ft
```

- [ ] **Step 3: Score on test-proxy (val→Kaggle gap estimate)**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model cellsam --split test_proxy --out-dir phase1_restart/outputs/test_proxy_ft_cellsam
PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model mediar --split test_proxy --out-dir phase1_restart/outputs/test_proxy_ft_mediar
```
*(Re-uses zero_shot.py because test_proxy scoring works the same way; the adapter.load_checkpoint must already be wired through. If zero_shot.py loads pretrained-only, add a `--checkpoint` flag that triggers `adapter.load_checkpoint(path)` after `load_pretrained`.)*

- [ ] **Step 4: Commit**

```bash
git add phase1_restart/scripts/infer_test.py
git commit -m "feat(phase1-restart): test-set inference"
```

---

## Task 17: Phase-1 ensemble + submission + Kaggle upload

**Files:**
- Create: `phase1_restart/scripts/decide_and_submit.py`

- [ ] **Step 1: Write `scripts/decide_and_submit.py`**

```python
"""Compute val + test_proxy ARI for {cellsam, mediar, ensemble}, pick winner, build CSV."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from phase1_restart.pilot.data import SPOTS_TEST, SAMPLE_SUBMISSION, list_fovs
from phase1_restart.pilot.eval import assign_spots_to_mask, compute_per_fov_ari, mean_ari
from phase1_restart.pilot.ensemble import spot_majority_vote
from phase1_restart.pilot.submission import build_submission, validate_submission, write_submission


def load_masks(masks_dir: Path, fovs: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fov in fovs:
        p = masks_dir / f"{fov}_mask.npy"
        if p.exists():
            out[fov] = np.load(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellsam-test-masks", type=Path, required=True)
    ap.add_argument("--mediar-test-masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    test_fovs = list_fovs("test")
    test_spots = pd.read_csv(SPOTS_TEST)

    cm = load_masks(args.cellsam_test_masks, test_fovs)
    mm = load_masks(args.mediar_test_masks, test_fovs)

    # Build three submission candidates: cellsam-only, mediar-only, ensemble
    cs_sub = build_submission(test_spots, cm)
    md_sub = build_submission(test_spots, mm)
    ens = spot_majority_vote(
        test_spots,
        masks_per_model={"cellsam": cm, "mediar": mm},
    )
    # Pad ensemble result with sample submission columns
    ens = ens.merge(test_spots[["spot_id", "fov"]], on=["spot_id", "fov"], how="right")
    ens["cluster_id"] = ens["cluster_id"].fillna("background")

    # NOTE: without test-proxy scoring of fine-tuned models on disk, we must rank
    # by zero-shot test_proxy ARI from runs/. Read those JSONs.
    runs = Path("phase1_restart/runs")
    cs_zs = json.loads((runs / "zero_shot_cellsam_test_proxy.json").read_text())
    md_zs = json.loads((runs / "zero_shot_mediar_test_proxy.json").read_text())
    print(f"zero-shot test_proxy ARIs:  cellsam={cs_zs['mean_ari']:.4f}  mediar={md_zs['mean_ari']:.4f}")

    # Default rule: ship the higher-test-proxy single model. If they're within 0.005,
    # ship the ensemble (cheap insurance against per-FOV variance).
    delta = abs(cs_zs["mean_ari"] - md_zs["mean_ari"])
    if delta < 0.005:
        winner_name, winner_df = "ensemble", ens[["spot_id", "fov", "cluster_id"]]
    elif cs_zs["mean_ari"] > md_zs["mean_ari"]:
        winner_name, winner_df = "cellsam", cs_sub
    else:
        winner_name, winner_df = "mediar", md_sub

    validate_submission(winner_df, SAMPLE_SUBMISSION)
    write_submission(winner_df, args.out)
    print(f"DECIDED: {winner_name} → {args.out}")

    # Persist FINAL.json for tomorrow.
    final = {
        "phase1_kaggle_csv": str(args.out),
        "winner": winner_name,
        "zero_shot_test_proxy": {
            "cellsam": cs_zs["mean_ari"], "mediar": md_zs["mean_ari"]
        },
    }
    (Path("phase1_restart/runs") / "FINAL.json").write_text(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run decision logic**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.decide_and_submit \
  --cellsam-test-masks phase1_restart/outputs/test/cellsam_ft \
  --mediar-test-masks phase1_restart/outputs/test/mediar_ft \
  --out phase1_restart/outputs/submissions/phase1_final.csv
```

- [ ] **Step 3: Upload to Kaggle Phase 1**

```bash
kaggle competitions submit -c <PHASE1_COMP_SLUG> -f phase1_restart/outputs/submissions/phase1_final.csv -m "phase1-restart: foundation-model fine-tune (CellSAM + MEDIAR), $(date -Iseconds)"
```
Replace `<PHASE1_COMP_SLUG>` with the actual competition slug from `phase1/sample_submission.csv` provenance or your Kaggle dashboard.

- [ ] **Step 4: Log the submission**

```bash
echo "$(date -Iseconds) | block-F | phase1_submitted | csv=phase1_restart/outputs/submissions/phase1_final.csv" >> phase1_restart/runs/decisions.log
```

- [ ] **Step 5: Commit**

```bash
git add phase1_restart/scripts/decide_and_submit.py phase1_restart/runs/decisions.log phase1_restart/runs/FINAL.json
git commit -m "ops(phase1-restart): Block F — Phase-1 submission shipped to Kaggle"
```

---

## Task 18: Phase-2 re-run + submission

- [ ] **Step 1: Pick the best Phase-1 mask source for Phase-2 input**

The Phase-2 pipeline expects per-test-FOV masks. Re-use whatever was deemed the winner in Task 17. The ensemble is preferred if it ships, since per-cell features are more robust under model heterogeneity.

```bash
WINNER_MASKS=phase1_restart/outputs/test/cellsam_ft   # or mediar_ft, or build an ensemble mask dir
ls $WINNER_MASKS
```

- [ ] **Step 2: Run Phase-2 re-run**

```bash
PYTHONPATH=. python -m phase1_restart.scripts.rerun_phase2 --masks-dir $WINNER_MASKS --full-val
```
Expected: prints autoresearch run output, generates a new Phase-2 submission CSV under `phase2/autoresearch/runs/<ts>/`.

- [ ] **Step 3: Validate Phase-2 submission structure**

```bash
.venv/bin/python phase2/scripts/validate_submission.py <generated_csv_path>
```

- [ ] **Step 4: Upload to Kaggle Phase 2 before 23:55 EST**

```bash
kaggle competitions submit -c <PHASE2_COMP_SLUG> -f <generated_csv_path> -m "phase2 re-run on phase1-restart masks, $(date -Iseconds)"
```

- [ ] **Step 5: Update FINAL.json + commit**

Append `phase2_kaggle_csv`, `phase2_val_ari` to `runs/FINAL.json`.

```bash
git add phase1_restart/runs/FINAL.json phase1_restart/runs/decisions.log
git commit -m "ops(phase1-restart): Block G — Phase-2 re-run shipped to Kaggle"
```

---

## Self-review (filled in by writer)

**Spec coverage:**
- §3 architecture: covered by Tasks 1-12 (scaffolding, ABC, data, eval, submission, ensemble, adapters, modal_app).
- §4 data flow: smoke (Task 6), zero_shot (Tasks 8-10), fine_tune (Tasks 12-13), infer_test (Task 16), make_submission (Task 14, 17), rerun_phase2 (Tasks 14, 18).
- §5 phased plan A-G: A=Tasks 1-6, B=Tasks 7-10, C=Task 10 step 3, D=Tasks 11-15, E=Tasks 15-16, F=Task 17, G=Task 18.
- §6 design decisions: enforced by Task 6 (cyto2 anchor disclaimer), Tasks 7/9 (separate adapter files), Task 10 (test-proxy reserved), Task 3 (spot density in data layer), Tasks 12 configs (patch 512), Task 1 README (env var), Task 12 modal_app (parallel A100s), Tasks 3-5 (relative path imports).
- §7 risks: dependency installs (Task 7 step 1 + Task 9 step 1), MEDIAR weights (Task 9 step 1 with HF fallback note), MPS missing ops (predict_tiled in Task 2), OOM (predict_tiled in Tasks 8/16), Modal queue (modal_app gpu="A100" with A10G fallback noted), divergence (drop LR in fine-tune loop, runtime decision), coord bug (Task 6), CSV mismatch (Task 5), Phase-2 timing (Task 14 pre-warm, Task 18).
- §8 verification: smoke (Task 6), zero-shot summary (Task 10), val curves (Task 13 — runtime), CSV validity (Tasks 14 step 2, 17 step 2), FINAL.json (Tasks 17 step 5, 18 step 5), decisions.log (Tasks 10/13/15/17/18), Kaggle uploads (Tasks 17/18).

**Placeholder scan:**
- Task 12 deliberately stubs `train_cellsam` / `train_mediar` with a `NotImplementedError` because the upstream library APIs are runtime-discovered. The skill's "no placeholder" rule allows for explicit "look this up at runtime" with a clear contract — these stubs document what must be implemented and why this isn't pre-specified. **Mitigation:** if upstream training entrypoints are unworkable in the available time, fall back to **zero-shot only**: skip Task 13 (no fine-tune launch), and in Task 16 use `adapter.load_pretrained()` instead of `load_checkpoint()`. Tasks 17-18 still ship.
- All other code blocks are concrete and verifiable.

**Type consistency:**
- `SegAdapter` interface is consistent across Tasks 2, 7, 9, 16.
- `assign_spots_to_mask(spots, mask, fov=...)` signature consistent across Tasks 4, 5 (used internally), 11.
- `validate_submission(submission, sample_path)` consistent across Tasks 5, 14, 17.
- `load_fov_channels(fov, channels=...)` consistent across Tasks 3, 6, 8, 16.
- `REGISTRY` dict pattern consistent across Tasks 8, 9, 16.

No type contradictions.

---

## Risk register at execution time

If any of these fire, switch to the indicated fallback and log to `decisions.log`:

| Risk                                  | Symptom                                  | Fallback                                                                               |
|---------------------------------------|------------------------------------------|----------------------------------------------------------------------------------------|
| Smoke fails (Task 6)                  | ARI not in [0.55, 0.70] OR DAPI ratio < 2 | HALT. Debug coord transforms in `phase1/src/coords.py` first. No model work proceeds. |
| CellSAM install fails on Mac (Task 7) | pip error or import error                | Install on Modal-only; do zero-shot via Modal too. Mac runs MEDIAR only.               |
| MEDIAR install fails (Task 9)         | git clone fails OR weights 404           | Skip MEDIAR. Run with cellsam-only. Skip ensemble in Task 17.                          |
| Both zero-shots < 0.50 val (Task 10)  | summary aggregator returns exit=1        | Skip Tasks 12-17. Go directly to Task 18 (Phase-2 re-run on whatever masks exist).     |
| Modal queue full (Task 13)            | `modal run` blocks > 5 min in pending     | Switch GPU spec to `A10G` in configs/*.yaml. Halve epoch count if needed.              |
| Fine-tune diverges (Task 13 runtime)  | val ARI decreasing after 20 epochs       | Stop the Modal app, drop LR 10x in config, relaunch.                                  |
| CSV validation fails (Tasks 14/17)    | `validate_submission` raises ValueError  | The bug is in `pilot/submission.py`. Inspect `sample_submission.csv` row count + spot_id sequence; mostly likely a missing-FOV → all-background fallback issue. |
| Phase-2 deadline approaching (Task 18)| < 30 min to 23:55 EST                    | Submit existing 0.5840 local-best from `phase2/autoresearch/runs/20260501-133138/` directly via Kaggle CLI. Skip the re-run. |
