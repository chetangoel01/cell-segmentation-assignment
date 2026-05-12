# MERFISH Cell-Type Classification — Project 3 Submission

NYU Neuroinformatics (CS-GY 9223), Chetan Goel.

Two-phase Kaggle competition on the Zhuang lab `220912_wb3_sa2_2_5z18R_merfish5`
mouse-brain MERFISH dataset (5 z-planes × 15 imaging rounds, ~1.2 K genes).

| Phase | Task | Final public Kaggle ARI | Baseline |
|---|---|---|---|
| 1 | Cell segmentation (cluster mRNA spots into cells) | **0.7627** | 0.632 (pretrained Cellpose) |
| 2 | 4-level Allen-Atlas cell-type classification | **0.5421** | 0.351 (Cellpose + kNN) |

## Where to start

1. **`report.tex`** — the writeup. Compile with `pdflatex report.tex` (no
   external `.bib`; references are inline). All figures it depends on live in
   `figs/`.
2. **`CLAUDE.md`** — exhaustive project map: directory layout, the MERFISH
   coordinate convention (a phase-1 footgun), every experiment that was run,
   and the lessons baked into the final pipeline.
3. **`phase1/experiments.md`** and **`phase2/runs/SUBMISSIONS.md`** — full
   experiment logs per phase.

## Layout

```
report.tex                  writeup (compile to PDF)
figs/                       8 figures referenced from report.tex
  scripts/                  generators (panel-by-panel provenance noted in script docstrings)

phase1/                     Phase 1 — segmentation (frozen post-deadline)
  train.py, train_stardist.py, train_unet.py       training entry points
  infer*.py                                        inference / submission generation
  experiments.md                                   per-run log
  best_params_*.json                               post-hoc threshold sweep winners
  final_submissions/comparisons/                   diagnostic plots from final variants
  src/, reference/, notebooks/, plans/             support code, codebook, EDA, design notes

phase2/                     Phase 2 — classification (active codebase)
  __main__.py, tasks/, backends/                   task/backend split (local | HPC | Modal)
  scripts/                                         classifier sweeps, ensembling, BIL fetcher
  autoresearch/                                    Karpathy-style agent loop (idle since 05-01)
  modal/                                           Modal entry points for HPC-bound jobs
  src/                                             shared library (io.py, coords.py)
  RESUME.md, HPC_SYNC.md                           operational notes

phase1_restart/             Single-night foundation-model push (MEDIAR / CellSAM).
                            MEDIAR zero-shot val=0.6898 — below the StarDist 0.7627
                            baseline, so not submitted. Documents the negative result.

phase2-restart/             StarDist + rescue-radius / cleanup post-processing probe.
                            All post-processing variants regressed; kept as a
                            sanity-baseline reference.

docs/                       Assignment specification (Phase 1, exported markdown).
scripts/make_figs.py        Original 3-panel figure generator. Superseded by
                            per-figure scripts in figs/scripts/ for the current
                            8-figure set; still generates figs/data_overview.png.
requirements.txt            Python deps (cellpose, shapely, anndata, scikit-image, ...).
```

## Reproducing

Data and model weights are not checked in — they total ~360 GB raw and are
distributed through HPC scratch (NYU Torch) and the BIL / AWS Allen Brain Cell
Atlas mirrors. `CLAUDE.md` documents the exact paths under "Data layout" and
"Local data inventory". To re-run any experiment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MERFISH_DATA_ROOT=/path/to/competition_data    # see CLAUDE.md for path conventions

# Phase 1 — segmentation
python phase1/train.py --base-model cyto2 --exp-name my_run
python phase1/infer.py --exp-name my_run              # writes submission_my_run.csv

# Phase 2 — unified runner, three backends
python -m phase2 --list                               # list registered tasks
python -m phase2 smoke FOV_101                        # local sanity
python -m phase2 train-baseline --backend hpc ...     # or --backend modal / local
```

The phase-2 runner is designed so a task body runs unchanged on a laptop, on
NYU Torch via SLURM, or on Modal — see `phase2/__main__.py` and the
`Task` dataclass in `phase2/tasks/__init__.py`.

## Final pipeline (what produced the 0.5421 Phase-2 score)

A four-way plurality vote at each taxonomy level, anchored on a phase-1
segmentation carried into phase 2:

- **Segmentation**: Cellpose `nuclei_cosine_ep125` from phase 1
  (`cellprob_threshold=-0.5`, `flow_threshold=0.4`, 3-channel input
  [polyT_max, DAPI_max, spot_density σ=8]).
- **Voters**: P = `nuclei_cosine + codelab kNN(k=5, cosine, L1)` (Kaggle 0.5346);
  PQM = P + Q + M plurality vote (Kaggle 0.5375); PQR-V7 = P + Q + RF500-log1p
  (Kaggle 0.5419); cpsam_floor = background-only floor from a CPSAM variant.
- **Ensemble**: 4-way plurality with V7 as the disagreement-breaking anchor.

The actual submitted CSV is at
`phase2/runs/SUBMIT_v7_PQM_cpsam_4way/submission.csv` — but that runs directory
is gitignored (~140 sub-runs, several GB) so the CSV is not in this submission
bundle. Composition is logged in `phase2/runs/SUBMISSIONS.md` and the report.
