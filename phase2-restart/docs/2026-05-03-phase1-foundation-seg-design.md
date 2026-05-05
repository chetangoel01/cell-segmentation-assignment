---
date: 2026-05-03
topic: phase-1 segmentation via foundation-model adaptation
status: draft, awaiting user review
---

# Phase-1 Segmentation via Foundation-Model Adaptation

## Context

Phase-1 of the MERFISH cell-segmentation project assigns each mRNA spot in 4 test FOVs to a cell (or `"background"` if extracellular). Score is per-FOV ARI averaged across the 4 test FOVs. Current Kaggle leader: **0.8285**. Best in-house: **0.7627** (StarDist 2D, 28-epoch fine-tune).

Every model tried so far in phase-1 is in the U-Net + radial-polygon family (Cellpose cyto2/cyto3/nuclei, multiscale Cellpose, U-Net, StarDist, InstanSeg). Foundation models with biology-specific pretraining (Mesmer, MEDIAR, DeepCell, μSAM, CellSAM) are explicitly listed as "not tried" in `phase1/experiments.md`. The thesis: **pretraining-distribution match is the lever that buys us 7 ARI points**, because the existing models are limited by what cyto2/3 saw during training, not by architecture.

Tonight (2026-05-03 → morning of 05-04) is one all-night session on Mac MPS. Goal: at least one foundation model fine-tuned to val ARI ≥ 0.87 on phase-1 FOVs 036–040, structurally-valid Kaggle submission CSV produced.

## Goal & success criterion

**Primary**: foundation-model adapter for phase-1 segmentation, fine-tuned to **val ARI ≥ 0.87** on FOVs 036–040 (proxy for Kaggle ≥ 0.8285 assuming ~5% domain-shift gap of a robust architecture; Cellpose's 9-pt gap suggests ~0.92 needed if architecture is fragile, so 0.87 is the floor).

**Success ladder**:
- **Hit (target)**: val ARI ≥ 0.87 with at least one candidate post-fine-tune.
- **Partial win**: val ARI in [0.7627, 0.87) — beats StarDist baseline, validates the foundation-model thesis, sets up a follow-up night for hyperparameter tuning.
- **Reset signal**: all candidates < 0.7627 post-fine-tune. The foundation-model thesis is wrong for this data; pivot in next session to spot-clustering bypass.

**Out of scope tonight**:
- Submitting to Kaggle (validate CSV, don't submit — no rush).
- Phase-2 classifier work.
- 3D volumetric segmentation (stay 2D max-projection).
- BIL or external dataset augmentation (off-limits for phase-1 — test data is from BIL).
- TTA (proven to hurt by 0.0003 in phase-1 and ~0.01 in phase-2).

## Architecture

All work lives in `phase2-restart/` (sibling to `phase1/` and `phase2/` in the repo). Layout:

```
phase2-restart/
├── docs/
│   └── 2026-05-03-phase1-foundation-seg-design.md   # this file
├── pilot/                       # library code
│   ├── __init__.py
│   ├── adapter.py               # ABC: SegAdapter
│   ├── data.py                  # FOV loaders, splits, spot-density channel
│   ├── eval.py                  # per-FOV ARI scoring
│   └── submission.py            # mask → CSV (validates against sample_submission.csv)
├── models/
│   ├── __init__.py              # registry: {"mesmer": ..., "mediar": ..., "cellsam": ...}
│   ├── mesmer.py                # MesmerAdapter — 2-channel TF backend
│   ├── mediar.py                # MediarAdapter — 3-channel ConvNeXt PyTorch
│   └── cellsam.py               # CellSAMAdapter — held in reserve as Mesmer fallback
├── scripts/                     # entry points (one per mode)
│   ├── smoke.py                 # end-to-end smoke + coord-convention sanity check
│   ├── zero_shot.py             # python scripts/zero_shot.py --model X --fovs 036-040
│   ├── fine_tune.py             # python scripts/fine_tune.py --model X --config configs/X.yaml
│   ├── infer_test.py            # python scripts/infer_test.py --model X --checkpoint <path>
│   └── make_submission.py       # mask dir → submission CSV + structural validate
├── configs/
│   ├── mesmer.yaml
│   └── mediar.yaml
├── weights/                     # .gitignored — pretrained + fine-tuned checkpoints
├── outputs/                     # .gitignored — masks, submission CSVs
└── runs/                        # .gitignored — per-experiment logs, val curves, hparams
```

### `SegAdapter` interface (the central abstraction)

```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class SegAdapter(ABC):
    name: str                          # "mesmer", "mediar", "cellsam"
    expects_channels: list[str]        # ["polyT", "DAPI"] or ["polyT", "DAPI", "spot_density"]

    @abstractmethod
    def load_pretrained(self) -> None: ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: (C, H, W) float32 in [0,1]. Returns (H, W) int32 instance labels (0 = background)."""

    @abstractmethod
    def fine_tune(self, train_fovs: list[str], val_fovs: list[str],
                  output_dir: Path, n_epochs: int, **hparams) -> Path:
        """Returns path to best-val-ARI checkpoint."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None: ...
```

### Reused modules (imported from existing phase-1 code, no copying)

- `phase1/src/io.py` — DAX loading, channel extraction, frame indices `DAPI=[6,11,16,21,26]` `polyT=[5,10,15,20,25]`
- `phase1/src/coords.py` — x-flip-aware coords (`image_row = 2048 − (global_x − fov_x) / pixel_size`); polygon parsing
- `phase1/src/evaluate.py` — `compute_ari(solution, submission)` per-FOV-then-averaged
- `phase1/src/train_cellpose.py` — `boundaries_to_mask(...)` polygon-to-integer-mask helper (reuse for GT mask construction during fine-tune)
- `phase1/data/sample_submission.csv` — submission schema source of truth

### Spot-density channel (per phase-1 experiments)

Multi-scale spot-density heatmap was a key input channel in the best Cellpose runs. For each FOV:

1. Read `phase1/data/train/ground_truth/spots_train.csv` (or `phase1/data/test_spots.csv` for test).
2. Filter to spots in this FOV.
3. Rasterize `(image_row, image_col)` to a 2048×2048 zero array, += 1 per spot.
4. Gaussian-blur at σ ∈ {4, 8, 16} px → 3 channels (or sum → 1 channel).
5. Normalize per-FOV to [0, 1].

For Mesmer (2-channel only), spot density is **not** ingested. For MEDIAR (3-channel), use `[polyT_max, DAPI_max, spot_density_σ8]`. (Multi-scale via 3 channels would consume the entire input; pick σ=8 as the single best per phase-1 sweep.)

## Data flow

**Inputs (per FOV)**: a stack of multi-channel max-projected images plus spot density:
- DAPI max-projection: max over `DAPI = [6,11,16,21,26]` z-frames → (2048, 2048) float32, normalized per-FOV.
- polyT max-projection: max over `polyT = [5,10,15,20,25]` → (2048, 2048) float32, normalized per-FOV.
- Spot density σ=8: rasterize spots → blur → (2048, 2048) float32, normalized per-FOV.

**Splits** (phase-1 specific):
- **Train**: FOVs 001–030 (30 FOVs) — used for fine-tuning.
- **Val** (the canonical hold-out matching phase-1 experiments): FOVs 036–040 (5 FOVs) — never seen during training; ARI here is the primary signal.
- **Test-proxy** (NEW): FOVs 031–035 (5 FOVs) — never seen during training, also never used for hparam selection. Used at end-of-night to estimate val→Kaggle gap (since we can't see Kaggle test GT).
- **Test**: FOVs A, B, C, D — final submission target.

**Modes** (one script each):

```
zero_shot.py:
  for fov in --fovs:
    image = load_fov(fov, channels=adapter.expects_channels)
    mask = adapter.predict(image)
    save_mask(mask, outputs/zero_shot/<model>/<fov>_mask.npy)
  ari = compute_ari(spots_train, predicted_assignment)
  log to runs/zero_shot_<model>_<ts>.json

fine_tune.py:
  config = yaml(configs/<model>.yaml)
  for fov in train_fovs (001-030):
    image, gt_mask = load_fov_with_polygon_gt(fov)
    add_to_train_set(image, gt_mask)
  adapter.fine_tune(train_set, val_fovs=036-040, ...)
    → every 5 epochs: checkpoint to weights/<model>/<exp>/ep<NNN>.pt
                     + compute val ARI, append to runs/<exp>/val_curve.csv
    → return best-checkpoint path
  print best val ARI + checkpoint path

infer_test.py:
  adapter.load_checkpoint(--checkpoint)
  for fov in [A, B, C, D]:
    image = load_fov(fov)
    mask = adapter.predict(image)
    save_mask(mask, outputs/test/<model>/<fov>_mask.npy)

make_submission.py:
  load test masks
  load test_spots.csv → for each spot: lookup mask[image_row, image_col]
    → integer label or "background" (label==0 → "background" string)
  validate against sample_submission.csv (row count, spot_id sequence, schema)
  write outputs/submissions/submission_<model>_<exp>_<ts>.csv
```

## Phased execution plan (~8h MPS budget)

| Block | Time | What | Decision gate |
|---|---|---|---|
| **A. Harness** | 0:00–1:30 | Build `pilot/{adapter,data,eval,submission}.py` + `scripts/smoke.py`. Smoke test runs Cellpose cyto2 zero-shot on FOV_001, confirms ARI ≈ 0.632 baseline + DAPI in-cell intensity ≥ 2× outside (coord sanity). | Smoke fails → halt, debug. Do NOT proceed to bake-off if coord convention is broken. |
| **B. Mesmer zero-shot** | 1:30–2:30 | Install `deepcell`, write `MesmerAdapter`, run on FOVs 036-040. Record val ARI. | If `deepcell-tf` won't install on Apple Silicon or model load takes >15 min → swap to CellSAM (PyTorch, same lab lineage). |
| **C. MEDIAR zero-shot** | 2:30–3:30 | Clone MEDIAR repo (Lee et al. NeurIPS 2022), wrap their `Predictor` class in `MediarAdapter`, run on FOVs 036-040 with `[polyT, DAPI, spot_density]`. Record val ARI. | If MEDIAR can't ingest the third channel without retraining → fall back to `[DAPI, polyT, polyT]` (channel duplication). |
| **D. Compare + commit** | 3:30–4:00 | Compare zero-shot ARIs side by side. Commit fine-tune budget to highest-ceiling candidate (often the *lower* zero-shot one with the better-matched pretraining, e.g., Mesmer/TissueNet). | All candidates < 0.50 zero-shot → pivot from foundation-model approach to spot-clustering bypass for remaining hours. |
| **E. Fine-tune winner** | 4:00–7:00 | Patch-crop to 512×512, augment (8× flips/rotations + intensity jitter per phase-1 best practice). Train 50–100 epochs, checkpoint every 5, val ARI per checkpoint. Early-stop after 3 plateau checkpoints. | Val ARI not increasing after 1h → drop LR 10× and restart. Still not improving after 2h → ship best-so-far. |
| **F. Test inference + submission** | 7:00–8:00 | Best checkpoint → masks for FOVs A–D → submission CSV → structural validate against `phase1/data/sample_submission.csv`. Compare val ARI on FOVs 031–035 (test-proxy) to estimate val→Kaggle gap. Write final summary to `runs/FINAL.json`. | None — this block must complete. |

## Key design decisions

1. **Adapter pattern + registry**: each candidate is one file (`models/<name>.py`) implementing `SegAdapter`. Adding a 4th candidate (e.g., CellSAM mid-night) is a one-file change + one line in `models/__init__.py`. No `if model == "..."` ladders in scripts.
2. **Separate scripts per mode** (not single CLI dispatch): better for an all-night session — shell history reruns, cleaner stack traces, parallel execution in two terminals if needed.
3. **Test-proxy split (FOVs 031–035)**: never used for training or hparam selection. End-of-night ARI on this set estimates the val→Kaggle gap (the 9-pt gap that hurt Cellpose).
4. **Spot density baked into data layer**, not adapter: `pilot/data.py` produces channels per `adapter.expects_channels`. Adapters declare what they want; data layer fulfills.
5. **Patch crop 512×512 for fine-tune**: 2048×2048 OOMs on MPS at any meaningful batch size. 512×512 with batch=2 should fit.
6. **`PYTORCH_ENABLE_MPS_FALLBACK=1`** set globally: some PyTorch ops (especially newer ConvNeXt layers in MEDIAR) lack MPS kernels. Accept 2-3× slowdown when fallback fires.
7. **Reuse phase-1 modules unchanged**: `io.py`, `coords.py`, `evaluate.py`, `train_cellpose.py`. Don't copy or rewrite. Import via relative path.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mesmer/DeepCell TF stack broken on Apple Silicon | Medium-High | CellSAM (PyTorch) ready as drop-in. ~15 min swap. Decision gate at end of Block B. |
| MPS missing PyTorch ops | Medium | `PYTORCH_ENABLE_MPS_FALLBACK=1` env var globally. Surface as one-line check in adapter `__init__`. |
| OOM at 2048×2048 inference | Medium | Add `predict_tiled(image, tile=512, overlap=64)` to base adapter. Use only if `predict()` OOMs. |
| Fine-tune diverges or val ARI worsens vs zero-shot | Medium | Drop LR 10× (1e-5 → 1e-6), freeze backbone for first 10 epochs. Worst case: revert to best zero-shot model and ship. |
| Coord convention bug (silent ~4× ARI loss) | Low (using `phase1/src/coords.py`) | Smoke test at end of Block A asserts in-cell DAPI intensity ≥ 2× outside. Hard halt if violated. |
| Submission CSV format mismatch | Low (validator catches) | `pilot/submission.py` validates row count + spot_id sequence + cluster_id type before writing. Hard error on mismatch. |
| Val→Kaggle gap is huge (Cellpose-style 9-pt) | Medium | Test-proxy (FOVs 031–035) ARI estimates the gap end-of-night. If gap is huge, downgrade target announcement from "ready to submit" to "needs another night of robustness work". |

## Verification (end-of-night checks)

1. **Smoke** (after Block A): `python scripts/smoke.py` runs end-to-end, exits 0, prints baseline ARI ≈ 0.632 and in-cell DAPI ratio ≥ 2.
2. **Zero-shot recorded** (after Block D): three rows in `runs/zero_shot_summary.json`, comparable.
3. **Fine-tune curve healthy** (during Block E): val ARI in `runs/<exp>/val_curve.csv` is increasing-or-plateau over first 20 epochs, not noisy/diverging.
4. **Submission structural validity** (after Block F):
   ```
   python scripts/make_submission.py --validate-only outputs/submissions/<csv>
   ```
   exits 0 against `phase1/data/sample_submission.csv` (same row count, identical spot_id sequence, `cluster_id` ∈ `int | "background"`, `fov` ∈ `{A, B, C, D}`, UTF-8 encoded).
5. **Final reportable number** (end of Block F): `runs/FINAL.json` contains `{val_ari, test_proxy_ari, model, checkpoint, hparams, kaggle_csv_path}`.
6. **Decision-gate honesty**: every gate decision (swap Mesmer→CellSAM, drop LR, ship-best-so-far, pivot to bypass) is logged with timestamp + reason in `runs/decisions.log`. Tomorrow-morning-readable.

## Reference paths (for implementation)

Reused files (read-only imports):
- `../phase1/src/io.py` — DAX loaders, frame-index constants
- `../phase1/src/coords.py` — `parse_boundary_polygon`, `spots_in_polygon`, x-flip-aware transforms
- `../phase1/src/evaluate.py` — `compute_ari(solution, submission)`
- `../phase1/src/assign.py` — polygon-based spot assignment (alternative to mask lookup)
- `../phase1/src/train_cellpose.py` — `boundaries_to_mask(...)` polygon→integer-mask
- `../phase1/data/sample_submission.csv` — schema source of truth
- `../phase1/data/train/FOV_NNN/Epi-750s5-635s5-545s1-473s5-408s5_FOV_NNN.dax` — train images
- `../phase1/data/train/ground_truth/{cell_boundaries_train,spots_train}.csv` — GT polygons + spots
- `../phase1/data/test/FOV_X/*.dax` + `../phase1/data/test_spots.csv` — test
- `../phase1/data/reference/fov_metadata.csv` — `fov_x`, `fov_y`, `pixel_size`

Project-wide constants (don't re-derive):
- 2048 × 2048 px, 0.109 µm/px
- DAPI frames `[6, 11, 16, 21, 26]`, polyT frames `[5, 10, 15, 20, 25]`
- `image_row = 2048 − (global_x − fov_x) / pixel_size`, `image_col = (global_y − fov_y) / pixel_size`
- Phase-1 z-handling: max-project all 5 planes (NOT z=2-only — that's phase-2's convention)
- Extracellular spots → literal string `"background"`

External candidate model sources:
- Mesmer / DeepCell: `pip install deepcell`, model auto-downloads on first call
- MEDIAR: github.com/Lee-Gihun/MEDIAR (clone + add to PYTHONPATH; weights via their HF hub link)
- CellSAM (reserve): github.com/vanvalenlab/cellSAM (PyTorch, pip-installable)
