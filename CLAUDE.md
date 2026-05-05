# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

MERFISH cell-type classification on the NeuroinfoClass Kaggle competition. Source dataset: Zhuang lab `220912_wb3_sa2_2_5z18R_merfish5` — mouse brain sagittal sections, ~1,200 genes, 5 z-planes × 15 imaging rounds, ~220 × 220 µm per FOV at 0.109 µm/px. Two phases:

- **Phase 1 (frozen, `phase1/`)** — *Cell Segmentation*. Cluster decoded mRNA spots into cells (or `background`) per FOV. 40 train / 4 test FOVs (`FOV_001..040` train, `FOV_A..D` test). Submission: `spot_id, fov, cluster_id` (cluster_id is any string; metric is cluster-ID independent). Pretrained Cellpose baseline = 0.632 ARI. Our best Kaggle ARI: **0.7627** (StarDist).
- **Phase 2 (active, `phase2/`)** — *Cell Type Classification*. Per-spot 4-level Allen Brain Cell Atlas taxonomy prediction (`class`, `subclass`, `supertype`, `cluster`). 60 train / 10 test FOVs (`FOV_101..160` train, `FOV_E..N` test, ~439K test spots). Each level has 10 named labels + `background`; **~83% of test spots are `background`** (extracellular OR cells the GT pipeline couldn't confidently label). Metric: mean ARI over 40 (FOV, level) pairs (5 public + 5 private FOVs × 4 levels). Cellpose+kNN baseline = 0.351, perfect = 1.000. Deadline: **2026-05-04 23:55 EST**.

Cell-type counts in phase-2 train: 5,230 cells across 11 class values, 37 subclass, 62 supertype, 83 cluster — long-tailed (some classes have ~15 cells).

Two newer experimental scaffolds also live at the root: `phase1_restart/` and `phase2-restart/` — clean reimplementations used for late-stage experiments (foundation models, AWS-volume training, etc.).

## Commands

```bash
# Local environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests (only phase1_restart has a test suite)
PYTHONPATH=. pytest phase1_restart/tests
PYTHONPATH=. pytest phase1_restart/tests/test_smoke.py    # single file

# Phase 2 unified runner — same task, three backends
python -m phase2 --list                                    # registered tasks
python -m phase2 <task> --help                             # task-specific args
python -m phase2 smoke FOV_101                             # local sanity
python -m phase2 fetch-data --target aws --backend modal
python -m phase2 train-baseline --train-fovs FOV_101,FOV_105 \
    --val-fovs FOV_110 --backend hpc

# Phase 1 (frozen — invoke directly, not via runner)
python phase1/train.py --base-model cyto2 --exp-name my_run
sbatch phase1/train.sbatch
python phase1/infer.py --exp-name my_run                   # writes submission_<exp>.csv
```

`MERFISH_DATA_ROOT` env var points all data-loading code at the right root. Set it before running anything that touches FOVs:
- HPC (phase 1): `/scratch/cg4652/competition/`
- HPC (phase 2): `/scratch/pl2820/competition_phase2/` (read-only course staff scratch)
- Modal: `/root/data/` (mounted from the `cell-seg-phase2` volume)

## Architecture: phase2 task/backend split

The whole point of `phase2/` is that a task body is written once and runs on local, HPC, or Modal without changes.

- **`phase2/__main__.py`** — single CLI entry. Parses `<task>` then `--backend {local,hpc,modal}`. Backends are imported **lazily** so `--backend local` doesn't pull in `modal` or SLURM deps.
- **`phase2/tasks/__init__.py`** — registry. Each task is a `Task(name, summary, add_args, run, requirements)` dataclass. `requirements` carries advisory hints the backends consume (`gpu`, `modal_image`, `modal_gpu`, `modal_volume`, `hpc_partition`, `hpc_hours`, `hpc_gpus`).
- **`phase2/tasks/<name>.py`** — exports a top-level `TASK = Task(...)`. The `add_args` callback registers task argparse flags; `run` does the work given a `Namespace`. Tasks must be backend-agnostic — no `import modal`, no SLURM-specific paths.
- **`phase2/backends/{local,hpc,modal}.py`** — `launch(task, args)` for each. The Modal backend wraps the task in a generic `_remote_runner(task_name, args_dict)` and re-imports the task module inside the container, so the task body is identical to local execution.

**Three-backend parity is a hard rule** (see `~/.claude/projects/.../memory/feedback_three_backend_parity.md`): when you change a phase 2 task, the `--backend local`, `--backend hpc`, and `--backend modal` paths must all still work. Argv must round-trip through argparse — pass `--flag value`, never positionals that depend on order.

To add a task: create `phase2/tasks/<name>.py` exporting `TASK`, then add the import to `_register_builtins()` in `phase2/tasks/__init__.py`.

`phase2/src/` holds shared library code (`io.py` for DAX loading, `coords.py` for spot/polygon math). Tasks import from here; backends do not.

## Critical: MERFISH coordinate convention

The MERFISH x-axis is **flipped** relative to image rows:

```
image_row = 2048 - (global_x - fov_x) / pixel_size
image_col =        (global_y - fov_y) / pixel_size
pixel_size = 0.109 µm/px,  image size = 2048×2048
DAPI z-frames  = [6, 11, 16, 21, 26]
polyT z-frames = [5, 10, 15, 20, 25]
```

Pre-computed `image_row` / `image_col` columns exist in `spots_train.csv` and `test_spots.csv` — use them directly with `mask[row, col]`. Getting this wrong silently produces ~4× worse ARI (was the dominant phase-1 bug).

## Data layout

Three canonical data stores. The repo's `phase{1,2}/data/` directories are gitignored but **fully populated locally** (see "Local data inventory" below) — you can run end-to-end pipelines on the laptop without HPC.

**HPC scratch** (course staff sources, read-only):
- `/scratch/cg4652/competition/` — phase 1 (40 train FOVs, 4 test FOVs A–D), ~101 GB
- `/scratch/pl2820/competition_phase2/` — phase 2 (60 train FOVs 101–160, 10 test FOVs E–N), ~170 GB
- `/scratch/cg4652/phase2/` — your writable scratch for checkpoints

**Modal volume `cell-seg-phase2`** (mounted at `/root/data`): mirrors phase-2 layout. Sync procedure for HPC → Modal is documented in `phase2/HPC_SYNC.md`. BIL is throttled and blocked from Modal egress, hence the HPC bridge for raw `.dax` extras.

**Common subpaths** under any root:
- `train/FOV_NNN/` — 18 raw `.dax` files per FOV (uint16, 2048×2048 frames):
  - `Epi-750s5-635s5-545s1-473s5-408s5_{fov}.dax` — 27 frames, multichannel preimage (DAPI+polyT+fiducial+2 gene channels × 5 z)
  - `Epi-750s5-635s5-545s1_{fov}_{00..14}.dax` — 17 frames each, 2 gene channels × 5 z (15 rounds)
  - `Epi-750s1-635s1-545s1_{fov}_{0,1}.dax` — 7 frames each, fiducials for registration
- `train/ground_truth/`:
  - `cell_boundaries_train.csv` — polygons in µm; `boundaryX_z0..z4` and `boundaryY_z0..z4` are comma-separated coords per z-plane (Cellpose 2.0 output)
  - `spots_train.csv` — decoded mRNA: `barcode_id, fov, image_row, image_col, global_x, global_y, global_z, x, y, target_gene`
  - `cell_labels_train.csv` (phase 2 only) — `cell_id, fov, center_x, center_y, class_label, subclass_label, supertype_label, cluster_label, ccf_x, ccf_y, ccf_z` (CCF coords are 3D Allen registration in mm; absent for cells Allen couldn't register)
  - `counts_train.h5ad` — AnnData cell × gene matrix. Phase 1: (4082 × 1147). Phase 2: (5230 × 1147).
- `reference/codebook.csv` — 1,240 genes × 32-bit binary barcodes (4 bits set each)
- `reference/dataorganization.csv` — frame ↔ channel ↔ z mapping for `.dax` files
- `reference/fov_metadata.csv` — `fov, fov_x, fov_y, pixel_size` (pixel_size = 0.109 for all)
- `test/FOV_*/`, `test_spots.csv`, `sample_submission.csv`

**Submission format:**
- Phase 1: `spot_id, fov, cluster_id` — cluster_id is any string (or `background`); metric is cluster-ID independent.
- Phase 2: `spot_id, fov, class, subclass, supertype, cluster` — Allen taxonomy strings or `background`. Row order **must match `sample_submission.csv` exactly** (Kaggle joins on `spot_id` only, but rely on the template).

`metric.py` (alongside the data on HPC) is the official ARI scorer — copy it locally for offline cross-val on held-out training FOVs.

## Local data inventory (laptop, gitignored)

Both phases' competition data is fully present locally. Use `MERFISH_DATA_ROOT` to switch between them.

| Path | Size | Contents |
|---|---|---|
| `phase1/data/train/FOV_001..040/` | 92 GB | 40 phase-1 train FOVs |
| `phase1/data/test/FOV_A..D/` | 9.1 GB | 4 phase-1 test FOVs |
| `phase2/data/train/FOV_101..160/` | 137 GB | 60 phase-2 train FOVs |
| `phase2/data/test/FOV_E..N/` | 23 GB | 10 phase-2 test FOVs |
| `phase2/data/train/ground_truth/cell_boundaries_train.csv` | 305 MB | polygons |
| `phase2/data/train/ground_truth/spots_train.csv` | 201 MB | decoded train spots |
| `phase2/data/train/ground_truth/counts_train.h5ad` | 24 MB | cell × gene |
| `phase2/data/train/ground_truth/cell_labels_train.csv` | 888 KB | 4-level labels + CCF |
| `phase2/data/test_spots.csv` | 24 MB | what you classify |
| `phase2/data/sample_submission.csv` | 26 MB | submission template |

Phase-2 model weights (`phase2/models/cellpose_nuclei_cosine_ep125`, `phase2/models/stardist_p12_v1`) are also on disk locally. Phase-1 weights live on HPC / Modal — `phase1/models/` is empty in this checkout.

Phase-2 historical run outputs are under `phase2/runs/` (timestamped dirs + `SUBMIT_*` directories with the actual submission CSVs).

## Augmented data sources (`phase2/data/external/`)

Same animal/session/panel as the competition source — usable as extra labeled training without distribution shift. Rationale and verification in `phase2/docs/extra_training_data.md`.

| Local path | Size | What it is | Use for |
|---|---|---|---|
| `external/aws/` (`Zhuang-ABCA-4-log2.h5ad`, `cell_metadata_with_cluster_annotation.csv`, `gene.csv`) | 150 MB | AWS Allen Brain Cell Atlas. Section `.001` = competition source, **32,528 labeled cells** with all 4 taxonomy levels populated | classifier path (★ start here) |
| `external/aws_mouse3/` | 467 MB | AWS section for mouse3 (sagittal_1, same panel) — cross-animal labeled cells | extra train data; expect mild distribution shift |
| `external/competition_support/` | empty | placeholder for BIL `cell_boundaries_updated/` (3.62 GB) + `decoded_spots/` (1.81 GB) + counts h5ad — full polygon/spot data for the whole 783-tile session | segmentation path; would let you train on 60 extra adjacent tiles |
| `external/competition_extras/` | empty | placeholder for raw `.dax` from non-train/non-test tiles in the competition source session (~133 GB if 60 tiles, rounds-only) | fine-tune segmentation on more in-distribution data |
| `external/sa1_sample1/`, `external/sa2_sample3/`, `external/bil_sa1_sample1/` | placeholders | matched-pool sessions from the same animal (sa2_sample3 = section `.003`, same panel) and cross-animal (sa1) | future data expansion |

**Source-of-truth identification:** competition is a 60+10 tile crop of a 783-tile session. AWS section `.001` ⇄ BIL session `220912_wb3_sa2_2_5z18R_merfish5` ⇄ course `FOV_101..160`. `cell_label` joins are byte-equal across BIL and AWS (no normalization).

**Throughput:** AWS S3 ≈ 18 MB/s (full Zhuang-ABCA-4 = ~157 MB in ~10 s). BIL ≈ 35 KB/s on residential and **blocked from Modal egress**; pull big BIL files from HPC. The single `download.brainimagelibrary.org` host has no failover (memory: `bil_download_spof.md`) — plan extras pulls days ahead if you need them.

**Extras puller:** `phase2/scripts/fetch_bil.py` has `verify`, `list-pool`, `inspect`, `fetch`, `fetch-source`, `fetch-support`, `fetch-aws`, `fetch-counts`. Run `python3 phase2/scripts/inventory.py` for a status report across all five external groups.

**Leakage rule:** when pulling extras from the competition source session, always `--exclude-tiles` for both train (101–160) and the 10 test tile indices (`fetch_bil.py verify FOV_E..N --split test` discovers them). `fetch-source` refuses to run without `--exclude-tiles`.

## Strategy hints from the docs (phase 2)

The Kaggle overview spells out the headroom directions; phase-1 results constrain which are worth GPU-hours:

1. **Hierarchy-aware classifier** (predict class → condition subclass on class → …). The kNN baseline produces flat per-level ARI (0.347 / 0.352 / 0.352 / 0.351); a hierarchy-aware model wins specifically at the finer levels.
2. **Spatial context** (CCF coordinates + neighborhood expression). A cortical L2/3 cell looks different from L5 even with overlapping marker genes.
3. **Background calibration** — 83% of test spots are `background`. The "is this spot inside a confidently-labeled cell?" decision dominates the score; tune the threshold separately from the classifier.
4. **Fine-tune segmentation** — only worth doing if you can verifiably exceed the phase-1 StarDist 0.7627. Not the bottleneck given the kNN classifier already plateaus at 0.35.

## Operational gotchas

- **HPC partition `c12m85-a100-1` kills jobs at xx:01:01 every hour.** Avoid it, or chain ≤55-min runs with checkpoint/resume. This is why Modal became the primary phase-2 runtime.
- **`modal volume get` silently downloads only one file when the local destination doesn't exist as a directory.** Always `mkdir -p` the parent and verify file counts after.
- **Cellpose v4 quirks** baked into phase-1 train scripts: `train_seg()` returns a tuple, `channels` arg dropped, `learning_rate=1e-5` (not the documented 0.005). Don't "fix" these.
- **Val ARI is not a Kaggle proxy across architectures.** Within one architecture it's reliable; across architectures only Kaggle scores count (StarDist had lower val ARI than Cellpose but won Kaggle).

## Models & configs tried (digest — full logs in `phase1/experiments.md` and `phase2/runs/SUBMISSIONS.md`)

### Phase 1 — segmentation (all on val FOVs 036–040)

Eighteen `phase1/best_params_*.json` files persist the best `(cellprob_threshold, flow_threshold)` from a 30-cell sweep per Cellpose variant. Canonical training scripts: `phase1/train.py` (Cellpose), `phase1/train_stardist.py`, `phase1/train_unet.py`. Inference: `phase1/infer.py`, `infer_stardist.py`, `infer_instanseg.py`, `infer_unet.py`.

| Family | Variant | Channels | Val ARI | Kaggle | Notes |
|---|---|---|---|---|---|
| Cellpose | cyto2 baseline (3-ch) | polyt_max + dapi_max + spot_density σ=8 | 0.8147 | 0.7464 | post-coord-bug-fix baseline |
| Cellpose | cyto3 | same | 0.8137 | — | initial run collapsed (broken-coord masks); retrain ok |
| Cellpose | nuclei | same | 0.8051 | — | DAPI-specialist base |
| Cellpose | multiscale (5-ch) | + density σ=4, σ=16 | 0.8087 | — | adds sub-cell + cyto extent scales |
| Cellpose | cyto2_aug | +flips/rot/intensity jitter | 0.8311 | — | aug helps every variant |
| Cellpose | nuclei_aug | same aug | 0.8344 | — | |
| Cellpose | multiscale_aug | same aug | 0.8311 | — | |
| Cellpose | cyto2_cosine | LR 1e-5 → 1e-7 cosine, 300 ep | 0.8267 | — | |
| Cellpose | cyto2_warmup | 20-ep linear warmup → 1e-5 | 0.8324 | — | |
| Cellpose | **cyto2_warmup_long** | warmup + 500 ep | **0.8361** | — | best Cellpose val (cellprob=-1.0, flow=0.4) |
| Cellpose | cyto2_warmup_lowwd | warmup + WD 0.01 | 0.8245 | — | lower WD hurts |
| Cellpose | nuclei_warmup | warmup | 0.8240 | — | |
| Cellpose | nuclei_warmup_long | warmup + 500 ep | 0.8337 | — | |
| Cellpose | nuclei_cosine | cosine | 0.8174 | 0.7588 | only nuclei variant submitted |
| Cellpose | cyto2_long | 500 ep, no warmup | 0.8232 | — | |
| Cellpose | cyto2_multiz | per-z plane as separate sample | 0.7960 | — | **regressed** vs max-proj |
| Cellpose | cyto2/nuclei_zstats (7-ch) | + polyt/dapi mean & std across z | — | — | trained, did not beat 3-ch, not submitted |
| StarDist2D | `2D_versatile_fluo` ft | DAPI max only, 1-ch, percentile norm | 0.8039 | **0.7627** ✅ | 28 ep, our best Kaggle |
| StarDist2D | longer chain (~98 ep) | same | 0.8269 | — | val rose, weights overwritten before submission |
| StarDist2D | `stardist_v2` (~180 ep) | same | 0.8247 | 0.7421 | val ↑ but Kaggle ↓ → past sweet spot |
| InstanSeg | `fluorescence_nuclei_and_cells` zero-shot | — | <baseline | — | not pursued |
| U-Net | 3-class semantic + watershed | — | — | — | scaffolded, never trained |
| MEDIAR | zero-shot (`phase1_restart/`) | DAPI + polyT | 0.6898 (0.7073 w/ min_area=4000) | — | foundation-model thesis, did **not** beat StarDist baseline → not submitted |
| CellSAM | adapter scaffolded | — | — | — | blocked on `DEEPCELL_ACCESS_TOKEN` (vendor signup) |

**Phase-1 lessons baked into the choices:** TTA hurts (0.7461 vs 0.7464), NN-radius spot assignment is harmful (0.5063), Cellpose v4 wants `learning_rate=1e-5` (500× lower than v3), and val ARI is **not** a Kaggle proxy across architectures (StarDist had lower val ARI than top Cellpose but won Kaggle).

### Phase 2 — segmentation × classifier × ensemble (`phase2/runs/`, ~140 directories)

The current SOTA was assembled from three carryover phase-1 segmentations and a long classifier sweep, then ensembled.

**Segmentations tried in phase 2:**
- `nuclei_cosine_ep125` — phase-1 Cellpose nuclei_cosine carried over; the **anchor segmentation** for P/Q/M and most ensembles. cellprob=-0.5, flow=0.4, 3-channel input, dilate=0.
- `stardist_p2_v1` — phase-1 StarDist applied to phase-2 (used in early submissions 1–4, dropped after correlation issues).
- `codelab_v1_final` — CPSAM fine-tune from the codelab notebook (10/220 ep on phase-1 stack), 2-channel.
- `cpsam-phase1-fresh-…`, `cpsam_phase2_p1stack` — fresh CPSAM fine-tunes on the phase-1 stack (`run_seg_matrix.py`).
- `cyto2_p1stack`, `cyto3_p1stack`, `cyto3_p1stack_multiZ` — re-trained Cellpose variants on the phase-1 stack.
- Erosion sweep on CPSAM masks (`cpsam_erode{1..4}_val_masks`) and dilate=1 variants — post-process probes.
- `stardist_p2test_masks` — StarDist run directly on phase-2 test FOVs.

**Classifiers tried (`baseline-codelab-*` and `cls-*` dirs):** kNN with k ∈ {3, 5, 15, 25} (cosine + L1), kNN with `wdist`, logistic regression on log1p, RandomForest at 300/500/600/1000 trees with min_samples_leaf 1/2 and max_features 0.1/0.15/0.2, ExtraTrees at 300/500, HGB at 200, MLP-128, PCA-100→RF-300, voting ensembles (RF+ET, RF+ET+kNN), spatial kNN with AWS-augmented neighborhoods, "drop background from training" variants (`*-nobg`).

**Best singles:**
- **P** = `nuclei_cosine_ep125 (cp=-0.5, fl=0.4, 3ch) + codelab kNN(k=5, cosine, L1)` — local 0.5677, **Kaggle 0.5346**.
- **M** = same seg + logreg-log1p — local 0.5701, Kaggle 0.4859.
- **Q** = `codelab_v1_final cpsam (10ep/220) + codelab kNN + cp=-0.5` — Kaggle 0.5031.
- **RF500-log1p-mf01** on nuclei_cosine_ep125 — local **0.5840** but Kaggle only 0.4881 (−0.10 gap; the autoresearch best).

**Best ensembles:**
- **PQM** plurality vote of P+Q+M = **0.5375** (May 1).
- **PQR-V7** = P + Q + RF500-log1p-mf01 (drops M) = **0.5419** (May 3, on disk at `phase2/runs/ensemble-v7-P-Q-RF500/submission.csv`).
- **V7 + PQM + PQM-cp3 + cpsam_floor 4-way (anchor=V7)** = **0.5421** ✅ **CURRENT SOTA** (May 4 05:25, on disk at `phase2/runs/SUBMIT_v7_PQM_cpsam_4way/submission.csv`). Marginal +0.0002 over PQR-V7 — cpsam_floor adds 85% disagreement diversity vs the 99%-similar PQM/PQM-cp3 voters.
- **PQMRF-V6** = PQM + RF500 = 0.5408.
- 30+ ensemble variants v1..v33 on disk (`ensemble-v*` dirs); SUBMIT_* dirs hold the actual submitted CSVs.

**Phase-2 lessons:**
- **Local val → Kaggle correlation is weak/inverted past May 1.** Higher local val (0.5840 RF500) → lower Kaggle (0.4881); cpsam-zeroshot+RF500 had local 0.5959 → Kaggle **0.3378** (May 4 slot 1, even worse). Stop trusting local val on FOVs 156–160 as a Kaggle proxy.
- **Classifier diversity > segmentation diversity** in ensemble lift (RF on old seg helped; new-seg variants of the same classifier as P didn't).
- **logreg voters drag PQM-style ensembles down** (M reduces ensemble; dropping M for RF500 won).
- **HGB unviable at 1000+ classes** (cluster level) — local 0.2431.
- **CPSAM fine-tunes regressed** vs the carryover phase-1 nuclei_cosine_ep125 segmentation (Kaggle 0.4268 / 0.5031 vs 0.5346); fresh CPSAM zero-shot also regressed (0.3378). Multiple May-4 attempts to add cpsam-segmentation voters (slot 1, slot 3 = 0.4979) confirmed: **cpsam-anything as a voter hurts**.
- **Autoresearch was last run 2026-05-01 16:31** (last result: RF500-log1p-mf01 local 0.5840 → Kaggle 0.4881). Subsequent work was hand-tuned ensembling — do NOT resume the agent loop without changing its eval (FOVs 156–160 is now misleading).

### `phase2-restart/` — separate StarDist+rescue probe

Cleaner reimplementation testing rescue-radius and cleanup-min-spots post-processing on a StarDist baseline. Best: rungA val ARI **0.7712** (no rescue, no cleanup). All `rungB_R*` (rescue_radius ∈ {3,5,8,12,20,30}) and `rungC_M*` (cleanup_min_spots ∈ {3,5,10,20,50,100}) regressed — rescue_radius=20 dropped to 0.4304. Conclusion: post-processing knobs hurt. Used as a sanity baseline, not the main pipeline.

### `phase1_restart/` — single-night foundation-model push (2026-05-04)

Goal was Kaggle ≥ 0.7627 via foundation-model adaptation; coord-sanity gate passed (DAPI in-cell ratio 5.23×); MEDIAR zero-shot val 0.6898 → did not beat the existing 0.7627 leader → no submission made (`FINAL.json` records "do not submit"). CellSAM blocked on DeepCell signup.

### `phase2/external_models/` — local segmentation weights

Two anchor segmentations are checked into the repo for direct local inference (no fine-tune needed):
- `cellpose_nuclei_cosine_ep125/` — the phase-1 nuclei_cosine carryover used by every winning ensemble (P, M, V7, the May-4 SOTA, etc.). Pair with `cellprob_threshold=-0.5, flow_threshold=0.4, 3-channel input` to reproduce P's segmentation.
- `stardist_p12_v1/` — phase-1 StarDist (`weights_best.h5`, `weights_now.h5`, `config.json`, `thresholds.json`). The 0.7627 phase-1 winner; useful in phase 2 as a diversity voter (one of the components of the May-4 slot-3 ensemble).

### Phase-2 task definitions (`phase2/tasks/*.py`)

The runner registers six tasks. Notable defaults and flags worth knowing before iterating:

- **`pipeline`** — chains `train-baseline → train-segmentation → infer-baseline` into one timestamped `phase2/runs/<ts>-pipeline-<exp>/` dir. Has `--skip-baseline`/`--baseline-dir` and `--skip-segmentation`/`--seg-checkpoint` so you can iterate on just the classifier or just the inference without paying for re-training.
- **`train-segmentation`** — Cellpose-SAM fine-tune. Defaults: `epochs=300`, `lr=1e-5`, `wd=0.1`, `bsize=256` (cpsam *requires* 256; default 224 crashes), 5-epoch checkpoint chunks, `keep-best=2`. `--include-phase1` stacks phase-1 train FOVs onto the phase-2 set. `--z-planes "0,1,2,3,4"` UNIONs polygons across all z (the source code labels this *"Major training-inference alignment fix"* vs the old z=2-only mask).
- **`infer-baseline`** — Cellpose seg + classifier → submission CSV. Defaults: `cellpose_diameter=30`, `cellprob_threshold=0.0`, `flow_threshold=0.4` — but the winning recipe needs `--cellprob-threshold -0.5 --flow-threshold 0.4 --include-spot-density --cellpose-diameter 0`. Has `--masks-dir <FOV>.npy` to plug in pre-computed masks (StarDist, etc.) and bypass Cellpose entirely. Also has `--seg-checkpoint` to use a fine-tuned cpsam, and `--nn-radius` for spot-space cell dilation (phase 1 used 5–15 px; phase 2 leaves it at 0).
- **`train-baseline`**, **`infer-baseline`**, **`fetch-data`**, **`smoke`** — supporting tasks; see `python -m phase2 <task> --help`.

### `phase2/autoresearch/` — Karpathy-style agent loop (idle since 2026-05-01)

Self-iterating sweep over `experiment.py` config. Layout: `program.md` (human-edited research direction), `experiment.py` (agent-edited config), `run_experiment.py` (fixed harness — do not edit), `results.md` (append-only log, 28 entries, last on 2026-05-01 16:31), `results/<ts>.json` (per-iteration full per-FOV-per-level ARI). All 23 entries are from a single 2026-05-01 session that ended on `RF500-log1p-mf01` local 0.5840 → Kaggle 0.4881. Resuming the loop without re-anchoring its eval (the local val FOVs are no longer a Kaggle proxy) will burn iterations chasing the wrong objective. Originally driven by `/loop` per the README.

## Today's Kaggle activity (2026-05-04)

`SUBMISSIONS.md` is dated 2026-05-03. Verified live via `kaggle competitions submissions cell-type-classification-phase-2-cs-gy-9223`:

| Slot | Time (EDT) | Composition | Public score |
|---|---|---|---|
| 1 | 05:21 | cpsam zero-shot + RF500-log1p-mf01 (local 0.5959) | **0.3378** ❌ huge gap |
| 2 | 05:25 | **V7 + PQM + PQM-cp3 + cpsam_floor 4-way (anchor=V7)** | **0.5421** ✅ NEW SOTA |
| 3 | 15:37 | V7 + StarDist (phase-2) + cpsam ep20 (phase-1 ft) 3-way (anchor=V7) | 0.4979 |

**Quota remaining today:** 2 slots (assuming 5/day). Deadline 2026-05-04 23:55 EDT.

## Pointers

- `phase2/RESUME.md` — full phase-2 context handoff (deadline, 60-GPU-hr budget, plan, open questions).
- `phase2/HPC_SYNC.md` — exact commands for HPC → Modal volume sync.
- `phase2/docs/kaggle_overview.md`, `phase2/docs/kaggle_data.md` — full task / data spec (cribbed from Kaggle).
- `phase2/docs/extra_training_data.md` — BIL/AWS source identification, throughput, fetch workflows, leakage rules.
- `phase1/docs/kaggle_overview.md`, `phase1/docs/kaggle_data.md` — phase-1 task / data spec.
- `phase1/experiments.md` — phase-1 experiment log and best-checkpoint history.
- `~/.claude/skills/nyu-hpc/SKILL.md` — Cloud Burst access, Singularity, sbatch patterns.
