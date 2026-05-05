# Autoresearch — MERFISH cell-type pipeline

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
for the Phase 2 cell-type classification problem.

## Files

- **`program.md`** — agent instructions. The HUMAN edits this to refine
  the research direction. Includes calibration anchors and known dead ends.
- **`experiment.py`** — the AGENT edits this. One-file config covering
  segmentation, classifier, and post-processing knobs.
- **`run_experiment.py`** — fixed harness. Runs the current `experiment.py`,
  validates locally, appends to log. **Do not edit.**
- **`results.md`** — append-only log of every iteration. Auto-maintained.
- **`results/`** — per-iteration JSON with full per-FOV-per-level ARI.

## Run one iteration manually

```bash
# Fast (3 val FOVs, ~3 min)
.venv/bin/python phase2/autoresearch/run_experiment.py

# Full (10 val FOVs, ~10 min) — only after a fast pass shows promise
.venv/bin/python phase2/autoresearch/run_experiment.py --full-val
```

## Run autonomously via Claude Code `/loop`

```
/loop iterate the autoresearch agent at phase2/autoresearch/ — read program.md and results.md, edit experiment.py with one new hypothesis, run run_experiment.py, decide keep-or-revert, repeat
```

The loop will pace itself; each iteration is ~3-5 min for fast val, ~10 min
for full val. Overnight gives ~30-60 fast iterations and ~5-10 full validations.

## Run as a scheduled morning planner

```
/schedule daily at 08:00 — analyze last night's results.md from phase2/autoresearch/, identify the top 1-2 configs, run --full-val on each, post a summary
```

## What to do with results

The harness only computes **local mean ARI** (on phase-2 val FOVs 151-160).
**It does NOT submit to Kaggle.** Once a config beats the current Kaggle best
locally by >0.01, the human pulls the corresponding `experiment.py`, runs
inference on test FOVs, and submits manually.

To convert a local-best config to a Kaggle submission:

```bash
# Look up the winning iteration's config from results.md, mirror it in a manual command:
.venv/bin/python -m phase2 infer-baseline \
  --backend local \
  --models-dir phase2/runs/baseline-codelab-p2only \
  --seg-checkpoint phase2/external_models/cellpose_nuclei_cosine_ep125 \
  --include-spot-density \
  --cellprob-threshold <value from experiment.py> \
  --flow-threshold     <value from experiment.py> \
  --nn-radius          <value from experiment.py> \
  --test-fovs FOV_E,FOV_F,FOV_G,FOV_H,FOV_I,FOV_J,FOV_K,FOV_L,FOV_M,FOV_N \
  --out-dir phase2/runs/<descriptive-name>

kaggle competitions submit \
  -c cell-type-classification-phase-2-cs-gy-9223 \
  -f phase2/runs/<descriptive-name>/submission.csv \
  -m "<descriptive-name>: local val X.XXXX, Y FOVs"
```
