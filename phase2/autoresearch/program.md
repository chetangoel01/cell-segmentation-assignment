# Autoresearch — MERFISH cell-type classification

You are an autonomous research agent iterating on a Kaggle pipeline for
MERFISH cell-type classification (Phase 2 CS-GY 9223). The competition metric
is mean ARI across 40 (FOV, level) pairs (10 test FOVs × 4 hierarchy levels).
Kaggle daily submission cap is 5; you do **not** use it directly. You optimize
on the **local proxy**: mean ARI on phase-2 val FOVs 151-160.

## How you iterate

Every iteration:

1. **Read the state.** Open `phase2/autoresearch/results.md`. Look at the past
   N runs and identify which knobs have been tested. Skim
   `phase2/autoresearch/experiment.py` to see the *current* config.

2. **Form a hypothesis.** Pick ONE thing to change. Examples:
     - A different `CELLPROB_THRESHOLD` value
     - Add `NN_RADIUS=10` (post-proc spot dilation)
     - Switch to `INCLUDE_SPOT_DENSITY=False`
     - Try a different `CLASSIFIER_DIR` (e.g. retrain a kNN with different k)
     - More creative: add a new helper in run_experiment.py to compute
       spatial-neighbor features, then use those in the classifier.
   Update `experiment.py` with the new value AND update `HYPOTHESIS` to a
   1-sentence rationale.

3. **Run a fast iteration.**
   ```
   .venv/bin/python phase2/autoresearch/run_experiment.py
   ```
   This validates on 3 FOVs (~3 min). The result is appended to results.md
   automatically.

4. **Decide.**
   - If ARI improved by >0.005 over fast-val best, run a FULL validation
     (10 FOVs) to confirm:
     ```
     .venv/bin/python phase2/autoresearch/run_experiment.py --full-val
     ```
     If FULL also improves: keep `experiment.py` as-is (this is the new best).
   - If ARI dropped: revert `experiment.py` to the previous best config
     (you can re-derive it from the results.md best line + the corresponding
     results/<id>.json).
   - If ARI is within ±0.005: leave it, try a different knob next iteration.

5. **Repeat.**

## Constraints

- **Don't edit:** `phase2/scripts/validate_local.py`, `phase2/autoresearch/run_experiment.py`,
  `phase2/data/*`, `phase2/external_models/*`. These define the metric, the
  harness, and the data — touching them invalidates the competition.
- **Stay within ~10 min wall-clock per fast iteration.** If a config blows that
  budget, abort and try something cheaper.
- **One change per iteration.** Don't change cellprob AND classifier in the
  same step — you won't know which moved the metric.
- **Don't submit to Kaggle from inside the loop.** Final Kaggle submissions
  are gated by the human.

## What's been tried (calibration anchors as of 2026-05-01)

Best Kaggle so far: **PQM ensemble = 0.5375** (P + Q + M plurality vote).
Best single submission: **P = 0.5346** (nuclei_cosine_ep125 + codelab kNN +
cp=-0.5 + 3ch + nn=0). Local val for P = 0.5677.

Local-val deltas to keep in mind:
- nn_radius > 0 → strictly worse (0.5613 → 0.4372 → 0.3761 → 0.3342 for nn=0/10/15/20)
- cellprob: -0.5 wins vs -1.0/-1.5/-2.0/0.0 (within 0.013 spread)
- classifier: codelab kNN (k=5, cosine, L1-norm on counts.h5ad) marginally beats logreg locally

## Promising directions worth exploring (not yet tested)

1. **Different epoch checkpoints** of nuclei_cosine — only ep125 has been pulled
   from HPC. ep100 / ep150 / ep200 might generalize differently. (Requires
   the human to rsync from HPC; not autoresearch-runnable.)
2. **Spatial-neighbor features** for the classifier — for each cell, append
   gene counts of its k-nearest cells. New feature engineering inside the
   classifier path.
3. **Hierarchy-aware classifier** — predict class first, then condition
   subclass on class, etc. Codelab section 7 has the recipe.
4. **CLAHE pre-processing** before segmentation — codelab uses this for SAM
   variants; might help cpsam too.
5. **Per-FOV cellpose diameter override** — auto-estimate is current; some
   FOVs may need fixed diameter to handle dense regions.
6. **Background-gate calibration** — ~83% of GT spots are background. A
   confidence threshold on the classifier (predict 'background' if the top-k
   neighbors disagree) might reduce false in-cell calls.

## What's already known to NOT work

- nn_radius > 0
- 25-NN classifier (worse than 5-NN with our setup)
- Off-the-shelf cpsam (worse than nuclei_cosine_ep125)
- Our 2h30m fine-tune from yesterday (over-regularized; weight_decay=0.1 was
  10000x too high)
- codelab_v1_final (the codelab-recipe retrain we did) — peaked at ep10 with
  same val_loss as ep125; on Kaggle scored 0.5031 < 0.5346.
