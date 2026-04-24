# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

MERFISH cell segmentation pipeline: fine-tune Cellpose on microscope images (DAPI + polyT channels) to segment cells, then assign mRNA spots to cells and generate a Kaggle submission CSV scored by Adjusted Rand Index (ARI). Runs on NYU HPC (Torch cluster) with GPU nodes via SLURM.

## Commands

```bash
# Local environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests
PYTHONPATH=. pytest                   # all tests
PYTHONPATH=. pytest tests/test_coords.py  # single test file

# Training (HPC, GPU required)
python train.py --base-model cyto2 --exp-name my_run
sbatch train.sbatch                   # SLURM submission

# Inference (generates submission_<exp-name>.csv)
python infer.py --exp-name my_run
sbatch infer.sbatch

# End-to-end notebook
jupyter nbconvert --to notebook --execute pipeline.ipynb
```

## Data Layout (HPC only)

All large data lives at `/scratch/cg4652/competition/` (not in this repo):
- `train/FOV_001/` … `train/FOV_040/` — training FOVs (`.dax` images)
- `train/ground_truth/cell_boundaries_train.csv` — polygon boundaries in µm
- `train/ground_truth/spots_train.csv` — mRNA spots with `fov`, `image_row`, `image_col`
- `reference/fov_metadata.csv` — per-FOV origin offsets (`fov_x`, `fov_y`) in µm
- `test/FOV_A/` … `test/FOV_D/` — test FOVs
- `test_spots.csv` — test mRNA spots

## Architecture

### Pipeline flow
1. `src/io.py` — load `.dax` binary images → extract DAPI (frames 6,11,16,21,26) and polyT (frames 5,10,15,20,25) z-stacks
2. `src/train_cellpose.py` — convert ground truth polygon boundaries to integer masks; compute mRNA spot density heatmap (σ=8px)
3. `train.py` — fine-tune Cellpose on FOVs 001–035; validate on 036–040; checkpoint every 5 epochs for HPC preemption safety
4. `infer.py` — run fine-tuned model on test FOVs; collect 2D masks
5. `generate_submission.py` (`build_submission()`) — map each spot pixel coordinate to its mask label → `submission.csv`
6. `src/assign.py` — alternative polygon-based assignment (point-in-polygon via Shapely)
7. `metric.py` / `src/evaluate.py` — local ARI evaluation

### Model input
Three-channel stack `[polyT_max, DAPI_max, spot_density]` (channel_axis=0), where max-projection collapses 5 z-planes.

### Training config (train.py)
- Base model: `cyto2` (default), also `cyto3`, `nuclei`
- 300 total epochs, chunked in 5-epoch segments with checkpoint save/resume
- `learning_rate=1e-5`, `weight_decay=0.1`, `batch_size=8`
- Checkpoints: `models/<exp-name>/cellpose_<exp-name>_ep<NNN>`
- State file: `models/<exp-name>/train_state.json` (tracks `completed_epochs` + `latest_checkpoint`)

## Critical Coordinate Convention

MERFISH x-axis is **flipped** relative to image row:
```
image_row = 2048 − (global_x − fov_x) / pixel_size   # x is flipped
image_col = (global_y − fov_y) / pixel_size
```
`pixel_size = 0.109 µm/px`, image size = 2048×2048. This is pre-computed into `image_row`/`image_col` columns in the spots CSVs. Getting this wrong silently produces ~4× worse ARI.

## Kaggle Compliance

- Extracellular spots (mask label 0) must be assigned the string `"background"` (not `0` or `NaN`) in `submission.csv`.
- Baseline ARI with pretrained cyto2 (no fine-tuning): **0.632**.

## HPC Notes

- Keep code in `/home/cg4652/` and data/overlays in `/scratch/cg4652/` (scratch is purged; home is not).
- SIGUSR1 sent 120s before SLURM time limit triggers graceful stop after the current checkpoint chunk.
- Singularity overlay (`overlay-15GB-500K.ext3`) provides the Python environment on compute nodes.
