# Phase 2 Resume — context handoff

You are picking up Phase 2 (cell-type classification) of the MERFISH project.
Phase 1 (segmentation) is in `phase1/` and frozen — read but don't modify.

## Deadline
**Monday 2026-05-04, 23:55 EST.** Started 2026-04-27 (≤7 days).

## Compute
- **NYU Cloud Burst HPC**, account `cs_gy_9223-2026sp`.
- **Budget: 60 GPU-hours total.** Tight.
- Login: `https://ood-burst-001.hpc.nyu.edu/` (browser MFA + Duo).
- Data transfer node: `cg4652@log-burst.hpc.nyu.edu`.
- See `~/.claude/skills/nyu-hpc/SKILL.md` for partition / Singularity / sbatch patterns.

## Paths (on Cloud Burst)
- Code: `~/cell-segmentation-assignment/` (mirror of this repo).
- Phase-2 data (read-only, course staff scratch): `/scratch/pl2820/competition_phase2/`.
- Writable scratch (your checkpoints, intermediate files): `/scratch/cg4652/phase2/`.
- Phase-1 artifacts (StarDist weights etc.): `~/cell-segmentation-assignment/models/`
  — but see "Open question" below; some 0.7627-leader weights may be overwritten.

## What Phase 2 actually is
Per-spot cell-type classification at 4 hierarchy levels (`class`, `subclass`,
`supertype`, `cluster`) using the Allen Brain Cell Atlas taxonomy. ~439K test
spots across 10 test FOVs. Vocabulary at each level: 10 named cell types +
`background`. **~83% of test spots are `background`** (extracellular OR cells
the GT pipeline couldn't confidently label). Metric: **mean ARI over 40
(FOV, level) pairs**, partition-based. Provided baseline (Cellpose+kNN) = **0.351**.

Submission columns (must match `sample_submission.csv` row order):
`spot_id, fov, class, subclass, supertype, cluster`.

Read `phase2/docs/Project_3_Phase_2_Neuroinformatics.pdf`,
`phase2/docs/kaggle_overview.md`, `phase2/docs/kaggle_data.md` for full task
spec and data layout.

## Code already in place
- `phase2/src/io.py` — DAX loading, `data_root()` honors `MERFISH_DATA_ROOT` env
  var (set it to `/scratch/pl2820/competition_phase2` on HPC). DAPI frames
  `[6,11,16,21,26]`, polyT `[5,10,15,20,25]`, pixel size `0.109 µm/px`,
  image size `2048`. Helpers: `load_fov_images(fov, split)`, `find_epi_file`,
  `train_dir/test_dir/reference_dir/ground_truth_dir`.
- `phase2/src/coords.py` — `parse_boundary_polygon`, `spots_in_polygon` (Shapely
  vectorized fast path with fallbacks). Verbatim from phase 1.
- `phase2/scripts/fov101_smoke_test.py` — confirms image+spots+boundaries load
  and the (flipped-x) coord convention isn't broken. Run with
  `python -m phase2.scripts.fov101_smoke_test` from repo root.

## Critical coordinate convention (don't get this wrong)
MERFISH x-axis is **flipped** relative to image_row:
```
image_row = 2048 - (global_x - fov_x) / pixel_size
image_col = (global_y - fov_y) / pixel_size
```
Pre-computed `image_row`/`image_col` columns exist in `spots_train.csv` and
`test_spots.csv` — use them directly with `mask[row, col]`. Getting this
wrong silently produces ~4× worse ARI (this was the dominant phase-1 bug).

## Open question (resolve first)
Phase 1's `experiments.md` says the 0.7627 Kaggle leader was `stardist_v1` at
epoch 28, but **subsequent resumes overwrote `weights_best.h5`**. Before
relying on phase-1 weights as warm start:

```bash
ls -lah ~/cell-segmentation-assignment/models/
find ~/cell-segmentation-assignment/models -name weights_best.h5 \
  -printf "%T@ %p %s bytes\n" | sort -n
```

If the on-disk weights post-date the v1 submission and are the worse
~180-epoch overfit version: don't bother warm-starting from them. Train
StarDist fresh on the 60 phase-2 train FOVs (FOV_101–FOV_160) for ~25–30 ep.
The phase-1 finding still holds — Kaggle ARI regresses past ~28 ep on the
val FOVs, so don't push past 30.

## Strategic plan & 60 GPU-hr budget

| # | Phase                                                               | Hours |
|---|---------------------------------------------------------------------|-------|
| 1 | Reproduce kNN baseline on phase-2 data (sanity ≥ 0.351)             | 6     |
| 2 | StarDist fine-tune on 60 train FOVs (warm start if possible)        | 8     |
| 3 | Hierarchy-aware classifier (XGBoost or MLP, conditional on parent)  | 6     |
| 4 | Spatial-context features (CCF coords + neighborhood expression)     | 6     |
| 5 | Background calibration (confidence threshold τ + "inside-cell" gate)| 4     |
| 6 | Submission iteration (3–4 Kaggle cycles)                            | 15    |
| 7 | Buffer for preemption / failed runs                                 | 15    |
|   | **Total**                                                           | 60    |

**Why this allocation:** 83% of spots are `background`, so the segmentation
interior/exterior decision dominates the score, but the kNN baseline already
plateaus at ARI=0.35 across all 4 levels — meaning *classifier sophistication*
is the unblocked lever. Spend GPU-hrs on (3)+(4)+(5), not on more StarDist
training.

## Important phase-1 learnings to carry over
- **Val ARI is not a Kaggle proxy across architectures.** StarDist had lower val
  ARI than Cellpose but won Kaggle. Within one architecture val is reliable;
  across architectures, only Kaggle counts.
- **HPC partition `c12m85-a100-1` killed jobs at xx:01:01 every hour** during
  phase 1. Either avoid it, or chain ≤55-min runs with checkpoint/resume.
- **Cellpose v4 quirks fixed in commits b97364a and earlier:** `train_seg()`
  returns a tuple, `channels` arg dropped, `learning_rate=1e-5` (not 0.005).
  Phase-1 train scripts already encode the right config.

## What to do first (suggested first session on HPC)
1. Resolve the "open question" above — figure out what StarDist weights exist.
2. `cd ~/cell-segmentation-assignment && git pull` to get this scaffold.
3. `export MERFISH_DATA_ROOT=/scratch/pl2820/competition_phase2`, then
   `python -m phase2.scripts.fov101_smoke_test FOV_101` — confirms the data
   path + coord convention work end-to-end on HPC.
4. Write `phase2/src/segment.py` (StarDist inference wrapper) and
   `phase2/src/expression.py` (per-cell × gene matrix from spots+mask), then
   the kNN baseline reproduction script.
5. Submit your first sbatch: a 1-A100 baseline run, before touching any
   modeling.

## Memory references
- `~/.claude/projects/-Users-chetangoel-Desktop-Repositories-cell-segmentation-assignment/memory/MEMORY.md` —
  index of relevant memories (phase split, modal setup, HPC partition killer,
  phase 2 plan).
- `~/.claude/skills/nyu-hpc/SKILL.md` — Cloud Burst access, sbatch patterns,
  Singularity overlay setup.
