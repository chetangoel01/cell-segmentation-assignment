---
date: 2026-05-04
topic: phase-1 segmentation restart, foundation-model fine-tune, single-night build
status: spec, awaiting implementation
supersedes: phase2-restart/docs/2026-05-03-phase1-foundation-seg-design.md (intended-for-last-night, never executed)
---

# Phase-1 Restart — Foundation-Model Fine-Tune (CellSAM + MEDIAR)

## 1. Context

Phase-1 of the MERFISH cell-segmentation project assigns each mRNA spot in the 4 test FOVs (A, B, C, D) to a cell ID or `"background"`. Score is per-FOV ARI averaged across the 4 FOVs.

- **Current Kaggle leader:** 0.8285 (someone else).
- **Best in-house Phase-1:** 0.7627 (StarDist 28-ep). All 15 prior experiments are in `phase1/experiments.md`. Pushing beyond 0.7627 inside the Cellpose+StarDist family has hit a ceiling — val ARI rises but Kaggle regresses (val→Kaggle gap of up to 9 points on Cellpose).
- **Phase-2 dependency:** Phase-2 (cell-type classification, deadline 2026-05-04 23:55 EST) consumes Phase-1 masks directly. Better masks change the per-cell feature distribution that the Phase-2 classifier sees. A Phase-1 win cascades into a Phase-2 win **iff** we can re-run the Phase-2 inference path before deadline.
- **Untried lever:** Every prior experiment is in the U-Net + radial-polygon family (Cellpose `cyto2/cyto3/nuclei`, multiscale Cellpose, U-Net, StarDist, InstanSeg). Foundation models with biology-specific pretraining (CellSAM, MEDIAR, Mesmer, μSAM, DeepCell) have **never been tried**. The thesis: **pretraining-distribution match is the lever that buys ≥ 5 ARI points**, because the existing models are limited by what cyto2/3 saw during training, not by architecture.

This restart is a clean-slate `phase1_restart/` folder, parallel to existing work. No warm-start from any prior experiment checkpoint. Pretrained *foundation-model* weights (TissueNet, LIVECell) are used as the fine-tune starting point — that's the entire point of the approach.

## 2. Goal & success criterion

**Primary:** beat 0.7627 on Phase-1 Kaggle public leaderboard tonight (2026-05-04). Stretch: beat 0.8285 (leader).

**Secondary:** re-run the Phase-2 pipeline (`phase2/autoresearch/run_experiment.py`) on the new masks and submit Phase-2 before 23:55 EST. Phase-2 currently sits at Kaggle 0.5375 (PQM ensemble) and 0.5840 local-best. Better masks should push at least the local-best.

**Success ladder (Phase 1):**
- **Hit:** Kaggle ≥ 0.79 — beats StarDist baseline.
- **Stretch:** Kaggle ≥ 0.83 — beats current leader 0.8285.
- **Reset:** all candidates < 0.76 — foundation-model thesis falsified for this data. Do **not** ship a new Phase-1 submission (the existing 0.7627 StarDist leaderboard entry stands; the user's "no existing trained models" rule applies to *what we ship*, not what's already on Kaggle). Pivot to Phase-2 re-run with the best available foundation-model masks anyway — even sub-StarDist Phase-1 masks may surface a different cell-size distribution that helps Phase-2's classifier.

**Out of scope tonight:**
- Mesmer (TF-on-Apple-Silicon fragility — explicitly skipped, not a fallback this round).
- Test-time augmentation (proven to hurt by 0.0003 in phase 1, ~0.01 in phase 2).
- 3D volumetric segmentation (max-project 5 z-planes, 2D per the phase-1 convention).
- BIL/Xenium external augmentation (Xenium rejected, BIL throttled).
- Fine-tuning on test FOVs A–D (data leak; only 001–030 train, 031–035 test-proxy, 036–040 val).
- Touching `phase1/`, `phase2/`, or `phase2-restart/` code (read-only imports only).

## 3. Architecture

```
phase1_restart/
├── README.md
├── docs/
│   └── 2026-05-04-phase1-restart-design.md       # this file
├── pilot/                                  # library code (importable)
│   ├── __init__.py
│   ├── adapter.py                          # SegAdapter ABC
│   ├── data.py                             # FOV loaders, channel assembly, splits
│   ├── eval.py                             # per-FOV ARI on cached GT
│   ├── ensemble.py                         # spot-level majority vote
│   └── submission.py                       # mask → CSV, structurally validates
├── models/
│   ├── __init__.py                         # registry: {"cellsam": ..., "mediar": ...}
│   ├── cellsam.py                          # CellSAMAdapter
│   └── mediar.py                           # MediarAdapter
├── scripts/                                # entry points, one per mode
│   ├── smoke.py                            # coord/IO sanity, hard gate before anything else
│   ├── zero_shot.py                        # --model X --split val|test_proxy|test
│   ├── fine_tune.py                        # --model X --config configs/X.yaml (Modal target)
│   ├── infer_test.py                       # --checkpoint <path> on test FOVs A-D
│   ├── make_submission.py                  # masks → Phase-1 CSV
│   └── rerun_phase2.py                     # invoke phase2 pipeline with new masks → Phase-2 CSV
├── configs/
│   ├── cellsam.yaml
│   └── mediar.yaml
├── modal_app.py                            # Modal entrypoint: fine_tune, infer
├── weights/                                # .gitignored — pretrained DLs + fine-tuned ckpts
├── outputs/                                # .gitignored — masks per FOV per model
├── runs/                                   # .gitignored — val curves, decisions.log, FINAL.json
└── tests/
    └── test_coord_smoke.py                 # in-cell DAPI ≥ 2× outside on FOV_001
```

### `SegAdapter` interface

```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class SegAdapter(ABC):
    name: str                            # "cellsam", "mediar"
    expects_channels: list[str]          # e.g. ["DAPI", "polyT"] or ["polyT", "DAPI", "spot_density"]
    runtime: str                         # "mps" or "modal"

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

### Adapters

| Adapter   | Source                                     | Pretraining           | Channels                            | MPS inference | Modal fine-tune |
|-----------|--------------------------------------------|-----------------------|-------------------------------------|---------------|-----------------|
| CellSAM   | `pip install cellSAM` (vanvalenlab)        | TissueNet (multi-modality cell imaging) | `[DAPI]` or `[DAPI, polyT]` (per their docs) | yes | yes             |
| MEDIAR    | git clone github.com/Lee-Gihun/MEDIAR + their HF weights | NeurIPS '22 cell-seg leader | `[polyT, DAPI, spot_density(σ=8)]` | yes | yes             |

Both are PyTorch. No TF dependency tonight.

### Reused from `phase1/` (imported, never copied)

- `phase1/src/io.py` — `.dax` loaders, DAPI/polyT frame indices.
- `phase1/src/coords.py` — x-flip-aware coordinate transforms. **Do not re-derive.** Wrong axis is the dominant silent ARI killer.
- `phase1/src/evaluate.py` — `compute_ari(solution, submission)` per-FOV-then-averaged.
- `phase1/src/train_cellpose.py` — `boundaries_to_mask(...)` polygon→integer-mask, used for building val/test-proxy GT masks during evaluation and fine-tune supervision.
- `phase1/data/sample_submission.csv` — schema source of truth.
- `phase1/data/{train,test,reference}/` — full image + spot data.

### Spot-density channel (for MEDIAR's 3-ch input)

Per the prior phase-1 best practice (3-channel `[polyT_max, DAPI_max, spot_density(σ=8)]`):

1. Read `phase1/data/train/ground_truth/spots_train.csv` (or `phase1/data/test_spots.csv` for test/test-proxy/test).
2. Filter to spots in this FOV.
3. Rasterize `(image_row, image_col)` to a 2048×2048 zero array, += 1 per spot.
4. Gaussian-blur σ=8 px.
5. Normalize per-FOV to [0, 1] for input.

CellSAM doesn't expect 3 channels — for CellSAM, send `[DAPI_max, polyT_max]` only.

## 4. Data flow

**Inputs (per FOV):** max-projected DAPI (5 z-planes), max-projected polyT (5 z-planes), spot-density σ=8.

**Splits (matching phase-1 convention):**
- **Train:** FOVs 001–030 (30 FOVs).
- **Val:** FOVs 036–040 (5 FOVs). Hparam selection + checkpoint promotion.
- **Test-proxy:** FOVs 031–035 (5 FOVs). **Never used for training or hparam selection.** End-of-night ARI estimates the val→Kaggle gap.
- **Test:** FOVs A, B, C, D. Final submission target.

**Modes (one script each):**

```
smoke.py:
  # uses pretrained Cellpose cyto2 from the cellpose package as a COORD-SANITY ANCHOR ONLY.
  # this is not a "trained model" we shipped — it's the off-the-shelf baseline that defines
  # the 0.632 reference number. it does NOT count as warm-start under the "no existing
  # trained models" rule because it never enters any candidate or submission path.
  for fov in [FOV_001]:
    image = load_fov(fov, channels=["DAPI", "polyT"])
    mask = cyto2_pretrained.predict(image)
    assert in_cell_dapi_intensity(mask, image) / out_of_cell_dapi(mask, image) >= 2.0
    ari = compute_ari(spots_train_fov_001, mask_assignment)
    assert 0.55 <= ari <= 0.70   # baseline ARI ~0.632 ± noise
  → if any assert fails: HALT. coord convention is broken. do not proceed.

zero_shot.py --model {cellsam,mediar} --split {val,test_proxy,test}:
  adapter = registry[model]()
  adapter.load_pretrained()
  for fov in split_fovs:
    image = load_fov(fov, channels=adapter.expects_channels)
    mask = adapter.predict(image)
    save_mask(mask, outputs/zero_shot/<model>/<fov>_mask.npy)
  if split in {val, test_proxy}:
    ari_per_fov = [compute_ari(gt[fov], mask_assignment[fov]) for fov in split_fovs]
    log_to_runs(model, split, mean(ari_per_fov), per_fov=ari_per_fov)

fine_tune.py --model {cellsam,mediar} --config configs/<model>.yaml:
  config = yaml(...)
  adapter = registry[model]()
  adapter.load_pretrained()
  best_ckpt = adapter.fine_tune(
    train_fovs=[FOV_001..030], val_fovs=[FOV_036..040],
    output_dir=weights/<model>/<exp>, n_epochs=config.epochs,
    **config.hparams,
  )
  → every checkpoint_interval epochs: save ckpt, compute val ARI, append to runs/<exp>/val_curve.csv
  → return path to best-val-ARI checkpoint
  print {best_val_ari, best_ckpt_path}

infer_test.py --model X --checkpoint <path>:
  adapter.load_checkpoint(path)
  for fov in [A, B, C, D]:
    image = load_fov(fov)
    mask = adapter.predict(image)
    save_mask(mask, outputs/test/<model>/<fov>_mask.npy)

make_submission.py --masks-dir <path> --out <csv>:
  for spot in test_spots.csv:
    cluster_id = mask[spot.image_row, spot.image_col]   # using pre-computed columns
    if cluster_id == 0: cluster_id = "background"
    else: cluster_id = f"{spot.fov}_{int(cluster_id)}"  # namespace per FOV
  validate against sample_submission.csv:
    - row count matches
    - spot_id sequence matches exactly
    - cluster_id is non-empty string
  write CSV.

rerun_phase2.py --masks-dir <path>:
  set MERFISH_MASKS_OVERRIDE=<path>
  run phase2/autoresearch/run_experiment.py --full-val with new masks
  → output Phase-2 submission CSV
```

## 5. Phased execution plan (~6h)

| Block | Time   | Mac MPS                                                              | Modal (parallel)                                          | Decision gate                                                                                                      |
|-------|--------|----------------------------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **A** | 0:00–0:45 | Scaffold `phase1_restart/`. Build `pilot/{adapter,data,eval,submission,ensemble}.py` + `scripts/smoke.py`. Run smoke. | —                                                         | smoke fails → HALT, debug coords. No bake-off until smoke passes.                                                   |
| **B** | 0:45–1:45 | Implement `models/cellsam.py` + `models/mediar.py` (load_pretrained + predict only). Run zero-shot on val + test-proxy for both. Build `runs/zero_shot_summary.json`. | —                                                         | both candidates < 0.50 zero-shot → HALT. Foundation-model thesis is falsified. Ship StarDist + Phase-2 re-run only. |
| **C** | 1:45–2:00 | Compare zero-shot val ARIs side by side. Commit fine-tune slot to **both** (parallel on Modal). | Spin up `modal_app.py`, push fine-tune jobs.              | If one zero-shot is ≥ 0.78 and clearly best, *also* fine-tune just that one (single Modal slot, save GPU credit).    |
| **D** | 2:00–4:00 | Build `pilot/ensemble.py`, `scripts/make_submission.py`, `scripts/rerun_phase2.py`. Validate submission CSV against `sample_submission.csv` using zero-shot output as scaffold. Pre-stage Phase-2 re-run env. | **CellSAM fine-tune** (Modal A100, 50 epochs, patch 512, batch 4). **MEDIAR fine-tune** (Modal A100, 50 epochs, patch 512, batch 2). Checkpoint every 5 epochs, val ARI logged. | Modal job dies → switch to Mac MPS fine-tune for whichever survives. Val ARI plateaus by epoch 20 → early-stop, drop LR 10×, restart from best ckpt. |
| **E** | 4:00–4:45 | Pull best checkpoints from Modal. Run `infer_test.py` on FOVs A-D for each fine-tuned model. Score on test-proxy (031–035) — this estimates val→Kaggle gap. | —                                                         | Fine-tuned val < zero-shot val → revert to zero-shot weights (fine-tune broke things).                              |
| **F** | 4:45–5:15 | Spot-level ensemble across {best fine-tuned CellSAM ckpt, best fine-tuned MEDIAR ckpt} on val + test-proxy. Pick winner: best single OR ensemble (whichever has higher test-proxy ARI). Build Phase-1 submission CSV, structural-validate, **submit to Kaggle Phase-1** (1 slot). | —                                                         | Test-proxy ARI < 0.75 → submit anyway (informational; best ckpt still goes to Phase-2 re-run regardless). Do not burn a 2nd Phase-1 Kaggle slot tonight. |
| **G** | 5:15–6:15 | `rerun_phase2.py` with new test masks → Phase-2 submission CSV → structural-validate → **submit to Kaggle Phase-2** (before 23:55 EST). | —                                                         | New Phase-2 val < 0.55 → also submit existing 0.5840 local-best as backup (different Kaggle slot).                  |

**End-of-night artifact:** `phase1_restart/runs/FINAL.json` with `{phase1_kaggle_csv, phase1_val_ari, phase1_test_proxy_ari, phase2_kaggle_csv, phase2_val_ari, model_used, ckpt_used, decisions_log}`.

## 6. Key design decisions

1. **No warm-start from prior experiment weights.** Per the user's "no using existing trained models" constraint. Foundation-model pretrained weights (CellSAM TissueNet, MEDIAR HF release) ARE used as the fine-tune init — that's intrinsic to the approach.
2. **Adapter pattern + registry.** Each candidate is one file (`models/<name>.py`) implementing `SegAdapter`. Adding a 3rd candidate (e.g., μSAM if we have time) is one file + one registry line.
3. **Separate scripts per mode** (no single CLI dispatch). All-night session: shell history reruns, cleaner stack traces, parallel terminals.
4. **Test-proxy split (FOVs 031–035) reserved.** Never used for training or hparam selection. End-of-night ARI on this set estimates val→Kaggle gap. The 9-point Cellpose val→Kaggle gap was the most expensive lesson of phase-1; spending 5 FOVs on this sentinel pays for itself.
5. **Spot density baked into data layer**, not adapter. `pilot/data.py` produces channels per `adapter.expects_channels`.
6. **Patch crop 512×512 for fine-tune.** 2048×2048 OOMs on MPS at any meaningful batch size and is heavy even on A100. 512×512 with batch=4 is the operating point.
7. **`PYTORCH_ENABLE_MPS_FALLBACK=1`** set globally on Mac. CellSAM and MEDIAR may have ops missing MPS kernels; accept ~2-3× slowdown on those when fallback fires.
8. **Modal parallel fine-tune.** Two A100 slots, one per model. ~2h wall-clock for what would be ~6h serial. Mac is free during this window for ensembler/submission/Phase-2-prep work.
9. **Reuse phase-1 modules via relative path import**, not `cp -r`. Don't fork the coord transform.
10. **Phase-2 re-run is a first-class block (G).** Many "Phase 1 wins" papers stop at "we shipped a Phase-1 submission." Phase-2 deadline is the binding constraint tonight; (G) must complete.

## 7. Risks & mitigations

| Risk                                                               | Likelihood | Mitigation                                                                                                                                              |
|--------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| CellSAM `pip install` breaks on Mac (Apple Silicon wheel issues)   | Medium     | Install on Modal first (Linux x86) and test there; only port to Mac if zero-shot inference can run on CPU as fallback. CellSAM weights are not large.    |
| MEDIAR weights download (HF) is rate-limited or blocked            | Low–Medium | Start the download inside Block A and let it run in the background. If HF is blocked, mirror the weights to Modal volume `cell-seg-workspace`.            |
| MPS missing PyTorch ops in CellSAM/MEDIAR                           | Medium     | `PYTORCH_ENABLE_MPS_FALLBACK=1` set globally. Surface as one-line check in adapter `__init__`. Tile-mode if memory is the issue, not ops.                 |
| OOM at 2048×2048 inference on MPS                                  | Medium     | `pilot/adapter.py` provides `predict_tiled(image, tile=512, overlap=64)`. Adapters call tiled mode if `predict()` raises OOM.                              |
| Modal queue full / both A100 slots unavailable                      | Low        | Fall back to A10G (~2× slower but available). If even A10G is full, fine-tune sequentially on whichever is free — drop epoch budget 50 → 25.              |
| Fine-tune diverges or val ARI worsens vs zero-shot                  | Medium     | Drop LR 10× (1e-5 → 1e-6), freeze backbone for first 10 epochs. Worst case: revert to best zero-shot model and ship.                                     |
| Coord convention bug (silent 4× ARI loss)                           | Low (using `phase1/src/coords.py`) | Smoke (Block A) asserts in-cell DAPI ≥ 2× outside; `compute_ari` baseline ≈ 0.632. Hard halt if either fails.                          |
| Submission CSV row count or spot_id sequence mismatch               | Low (validator catches) | `pilot/submission.py` validates row count + spot_id sequence + cluster_id type before writing. Hard error on mismatch.                       |
| Phase-2 re-run takes >1h (autoresearch needs longer than expected)  | Medium     | Run on `--full-val` only after `--smoke` confirms wiring; pre-warm classifier on Mac during Block D. If wall-clock blows past 23:00, ship existing 0.5840 local-best as Phase-2 fallback. |
| Val→Kaggle gap is huge (Cellpose-style 9pt)                         | Medium     | Test-proxy (031–035) ARI in Block E estimates the gap. If gap looks > 5pt, **don't** stretch toward 0.83 — ship the safer ensemble.                       |
| Network outage or NYU VPN drop kills Modal mid-fine-tune            | Low        | Modal jobs are detached (`modal run --detach`); checkpoints persist on volume `cell-seg-workspace`. Re-pull at Block E start.                              |

## 8. Verification (end-of-night checks)

1. **Smoke** (after Block A): `python -m phase1_restart.scripts.smoke` exits 0, prints baseline cyto2 ARI ≈ 0.632 ± 0.05 and in-cell DAPI ratio ≥ 2.
2. **Zero-shot recorded** (after Block B): `runs/zero_shot_summary.json` has rows for `{cellsam, mediar} × {val, test_proxy}`.
3. **Fine-tune curves healthy** (during Block D): `runs/<exp>/val_curve.csv` shows val ARI increasing-or-plateau over first 20 epochs, not noisy/diverging.
4. **Submission structural validity** (after Block F):
   ```
   python -m phase1_restart.scripts.make_submission --validate-only outputs/submissions/<csv>
   ```
   exits 0 against `phase1/data/sample_submission.csv` (same row count, identical spot_id sequence, `cluster_id` is non-empty string, `fov` ∈ `{A, B, C, D}`, UTF-8).
5. **Final reportable number** (end of Block G): `runs/FINAL.json` populated.
6. **Decision-gate honesty**: every gate decision (CellSAM install path, fine-tune LR change, fallback to zero-shot, Phase-2 backup ship) logged with timestamp + reason in `runs/decisions.log`. Tomorrow-morning-readable.
7. **Kaggle submissions visible**: both Phase-1 and Phase-2 CSVs uploaded and scored before 23:55 EST.

## 9. Reference paths (for implementation)

Reused (read-only imports):
- `../phase1/src/io.py` — DAX loaders, frame-index constants
- `../phase1/src/coords.py` — `parse_boundary_polygon`, `spots_in_polygon`, x-flip-aware transforms
- `../phase1/src/evaluate.py` — `compute_ari(solution, submission)`
- `../phase1/src/train_cellpose.py` — `boundaries_to_mask(...)` polygon→integer-mask
- `../phase1/data/sample_submission.csv`
- `../phase1/data/train/FOV_NNN/Epi-750s5-635s5-545s1-473s5-408s5_FOV_NNN.dax`
- `../phase1/data/train/ground_truth/{cell_boundaries_train,spots_train}.csv`
- `../phase1/data/test/FOV_X/*.dax` + `../phase1/data/test_spots.csv`
- `../phase1/data/reference/fov_metadata.csv`

Project-wide constants (do NOT re-derive):
- 2048 × 2048 px, 0.109 µm/px
- DAPI frames `[6, 11, 16, 21, 26]`, polyT frames `[5, 10, 15, 20, 25]`
- `image_row = 2048 − (global_x − fov_x) / pixel_size`, `image_col = (global_y − fov_y) / pixel_size`
- Phase-1 z-handling: max-project all 5 planes
- Extracellular spots → literal string `"background"`

Foundation-model sources:
- CellSAM: `pip install cellSAM` ; github.com/vanvalenlab/cellSAM
- MEDIAR: `git clone github.com/Lee-Gihun/MEDIAR` + their HF Hub weights link

Modal:
- Volumes: `cell-seg-data` (data, mounted at `/root/data`), `cell-seg-workspace` (weights/outputs, mounted at `/root/workspace`)
- GPUs: A100 first, A10G fallback
- Profile: `chetangoel2011`
- Existing precedent: `phase2/modal/modal_fetch_data.py`, root `modal_app.py`

## 10. What this spec is NOT

- A plan for implementing this. That's the writing-plans output, separate file.
- A guarantee Kaggle ≥ 0.7627 will be hit. That's the foundation-model thesis at risk.
- A recipe to also push Phase-1 past tonight. This restart is single-night-scoped; multi-night work would need a fresh design.
