# Modal setup

Run training + inference on Modal A100s. Replaces the NYU HPC pipeline that was getting killed at xx:01:01 every hour by the `c12m85-a100-1` partition admin.

## Volumes

| Volume | Purpose | Mount |
|---|---|---|
| `cell-seg-data` | Competition dataset (read-only) — already uploaded 2026-04-23 | `/data` |
| `cell-seg-models` | Model checkpoints and `best_params_*.json` | `/models` |
| `cell-seg-submissions` | Kaggle submission CSVs | `/submissions` |

`cell-seg-data` mirrors `/scratch/cg4652/competition/` on HPC: `train/FOV_001..040/`, `train/ground_truth/`, `test/FOV_A..D/`, `reference/`, root CSVs.

The existing scripts (`train.py`, `train_stardist.py`, `infer.py`, `infer_stardist.py`, `ensemble_infer.py`, `sweep_thresholds.py`, `eval_best_checkpoint.py`) honour `MERFISH_DATA_ROOT`, so they run unchanged both on HPC and on Modal.

## Entrypoints

| File | What it does |
|---|---|
| `modal_cellpose.py` | Cellpose train + infer + sweep + eval_best_checkpoint + ensemble_infer |
| `modal_stardist.py` | StarDist train + infer (separate image because of TF vs Torch dep conflict) |

Both expose a `main` local entrypoint driven by `modal run`; individual functions (`train`, `infer`, `sweep_thresholds`, `ensemble_infer`, …) can be called via `modal run modal_cellpose.py::<fn>`.

## Cellpose

```bash
# Fresh run, train + infer (augment is on by default; use --no-augment to disable)
modal run --detach modal_cellpose.py \
    --exp-name cyto2_modal_long \
    --base-model cyto2 \
    --epochs 500 \
    --lr-schedule warmup_cosine \
    --augment

# Train only (infer separately later with tuned thresholds)
modal run --detach modal_cellpose.py \
    --exp-name nuclei_modal \
    --base-model nuclei \
    --epochs 500 \
    --augment \
    --no-run-infer

# Run the threshold sweep, then re-infer with the tuned params
modal run modal_cellpose.py::sweep_thresholds --exp-name cyto2_modal_long
modal run modal_cellpose.py::infer \
    --exp-name cyto2_modal_long \
    --params-json /models/best_params_cyto2_modal_long.json

# Ensemble multiple trained Cellpose models (spot-level majority vote).
# exp-names is comma-separated; spot-sigmas-map is semicolon-separated entries
# where each entry's sigmas are still comma-separated.
modal run modal_cellpose.py::ensemble_infer \
    --exp-names "cyto2_modal_long,nuclei_modal,multiscale_modal" \
    --spot-sigmas-map "cyto2_modal_long:8;nuclei_modal:8;multiscale_modal:4,8,16"
# Note: ensemble_infer currently only runs Cellpose-backbone models.
# Mixed Cellpose+StarDist ensembles need spot-CSV merging locally (see below).
```

## StarDist

```bash
# Current best on Kaggle (0.7627) is StarDist @ 28 ep.  Replicate:
modal run --detach modal_stardist.py \
    --exp-name stardist_v3 \
    --epochs 28 \
    --channel dapi

# Longer training to study the val vs Kaggle divergence
modal run --detach modal_stardist.py \
    --exp-name stardist_100 \
    --epochs 100 \
    --channel dapi
```

## Pulling artefacts locally

```bash
# Download a submission
modal volume get cell-seg-submissions submission_cyto2_modal_long.csv .

# Download the full model dir (for archival)
modal volume get cell-seg-models cyto2_modal_long/ ./models/

# List what's there
modal volume ls cell-seg-submissions
modal volume ls cell-seg-models
```

## How Cellpose + StarDist ensembles work today

`ensemble_infer.py` runs each Cellpose model, looks up a cell ID per test spot, and majority-votes. It doesn't know about StarDist because StarDist needs a different CUDA runtime.

To ensemble StarDist with Cellpose, run the two inference jobs independently, download both submission CSVs, and majority-vote the `cluster_id` columns row-by-row locally. This is a 15-line pandas script — worth adding before the final Kaggle push.

## Cost / runtime sanity

- `cell-seg-data` is ~100 GB of `.dax` raw uint16 files; read-only mount is cheap.
- An A100 at current Modal pricing is ~$3-4/hr. 500-epoch Cellpose run on 35 FOVs: expect ~4-6 hr.
- StarDist 100 epochs: ~1-2 hr.
- `--detach` lets you close your laptop while jobs run.

## Going back to HPC

HPC paths are the default when `MERFISH_DATA_ROOT` is unset. Nothing else changed — the SLURM `.sbatch` scripts still work.

## What's NOT on Modal yet

- `train_unet.py` / `infer_unet.py` (U-Net semantic + watershed). Scaffolded but never trained — adding a `modal_unet.py` would be ~30 lines when you're ready.
- `eval_best_checkpoint.py` for StarDist — StarDist handles best-checkpoint selection internally via Keras `ModelCheckpoint`, so not needed.
- Kaggle submission automation (`submit_best_to_kaggle.py`). Submissions land in the `cell-seg-submissions` volume; you upload via the Kaggle CLI locally.
