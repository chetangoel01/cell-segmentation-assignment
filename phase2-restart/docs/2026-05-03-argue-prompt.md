# Adversarial Review: Phase-1 Foundation-Model Segmentation Plan

You are two senior ML engineers arguing about a one-night plan to attack a cell-segmentation Kaggle competition. Your job is to adversarially review the proposed design below and converge on the **best possible plan** to execute tonight.

## The situation (immutable facts — do not argue these)

- **Task**: Phase-1 of a MERFISH cell-segmentation project. Each mRNA spot in 4 test FOVs (A, B, C, D) must be assigned to a cell or labeled `"background"` if extracellular. Eval = per-FOV Adjusted Rand Index (ARI), averaged across the 4 test FOVs.
- **Current Kaggle leader**: **0.8285**. Current in-house best: **0.7627** (StarDist 2D, 28 epochs).
- **Compute tonight**: Mac M-series MPS only. ~8 hours.
- **Data**: 40 train FOVs at `phase1/data/train/FOV_001..FOV_040/`, with GT polygon boundaries at z=0..4 and ground-truth spot assignments. 4 test FOVs (A-D), no GT. Each FOV is 2048×2048 px, 5 z-planes, channels: DAPI (frames 6,11,16,21,26) + polyT (frames 5,10,15,20,25). Pixel size 0.109 µm/px.
- **Coordinate convention** (silent ~4× ARI loss if wrong): `image_row = 2048 − (global_x − fov_x) / pixel_size`. Already pre-computed into spots CSVs.
- **What's been tried** (per `phase1/experiments.md`): Cellpose cyto2/cyto3/nuclei (best val 0.8361, Kaggle 0.7464), StarDist 2D (val 0.8039, Kaggle 0.7627), U-Net, InstanSeg, multiscale Cellpose. **NOT tried**: Mesmer, MEDIAR, DeepCell, μSAM, CellSAM, TissueNet-pretrained models, embedding-based seg, spot-clustering bypass.
- **Val→Kaggle gap** is architecture-dependent: Cellpose ~9 pts, StarDist ~4 pts. Implication: to hit Kaggle 0.8285, val must be ≥ 0.87 with a robust architecture, ≥ 0.92 with a fragile one.
- **Phase-1 z-handling**: max-projection over all 5 z-planes (NOT z=2-only — that's phase-2's convention).
- **Spot density was a key input channel** for the best phase-1 Cellpose runs: `[polyT_max, DAPI_max, spot_density_σ8]`.
- **TTA proven to hurt** in both phase-1 (-0.0003) and phase-2 (-0.01). Don't propose it.
- **BIL external data is off-limits** (test data is from BIL — would be leakage).
- **External pretrained models are explicitly allowed** by competition rules.

## The proposed plan (argue about this)

Folder: `phase2-restart/` inside the existing repo.

**Candidates** (after a longer brainstorm):
- **Mesmer (DeepCell)** — TissueNet-pretrained, 2-channel (membrane=polyT, nuclear=DAPI). Closest pretraining-distribution match. TF-based — Mac compat is a concern.
- **MEDIAR** — NeurIPS 2022 cell-seg challenge winner. ConvNeXt + UperNet, 3-channel. Can ingest spot-density as 3rd channel. PyTorch-native.
- **CellSAM** held in reserve as PyTorch drop-in if Mesmer's TF stack breaks on Apple Silicon.
- Dropped: **EmbedSeg** (no pretraining advantage, would need >3h fine-tune), **μSAM**, **CellViT++**, **SAM2-as-video**, **HoVer-NeXt**.

**Architecture**: single `SegAdapter` ABC (`predict`, `fine_tune`, `load_checkpoint`); separate scripts per mode (`smoke.py`, `zero_shot.py`, `fine_tune.py`, `infer_test.py`, `make_submission.py`); reuse phase-1's `io.py`, `coords.py`, `evaluate.py`, `train_cellpose.py`.

**Splits**: train 001–030, val 036–040 (canonical hold-out), test-proxy 031–035 (never used for train or hparam selection — used end-of-night to estimate val→Kaggle gap), test A–D.

**8-hour plan**:
- 0:00–1:30 — Build pilot harness + smoke test (coord-convention check via in-cell DAPI ratio ≥ 2×)
- 1:30–2:30 — Mesmer zero-shot, decision gate (swap to CellSAM if TF broken)
- 2:30–3:30 — MEDIAR zero-shot
- 3:30–4:00 — Compare, pick winner (highest ceiling, not necessarily highest zero-shot)
- 4:00–7:00 — Fine-tune winner (patches 512×512, batch=2 for MPS, 50–100 epochs, checkpoint every 5)
- 7:00–8:00 — Test inference + structural-validated submission CSV

**Augmentation**: 8× flips/rotations + intensity jitter (per phase-1 best practice).

**Target**: val ARI ≥ 0.87 → Kaggle ≥ 0.8285.

## What to argue about (concretely)

Pick at least 3 of these to dispute or defend with specific evidence. Don't be polite.

1. **Is foundation-model adaptation the right thesis at all?** The "not tried" list also includes spot-clustering bypass — i.e., skip segmentation entirely and cluster spots directly via density/graph methods (DBSCAN, HDBSCAN, mean-shift on spot coords + spatial graphs). ARI scores a partition over spots, not pixel masks. Argue: would 8h be better spent on a non-segmentation approach? What's the strongest version of that argument and the strongest counter?

2. **Is Mesmer + MEDIAR the right candidate pair?** Specifically: did we drop EmbedSeg too quickly? It's the only paradigm-different candidate (predicts pixel embeddings rather than masks; the inference-time clustering step is closer to what ARI rewards). Counter: pretraining matters more than paradigm match, and EmbedSeg has no good pretrain. Defend or attack.

3. **Mac MPS constraint**: should we instead spend 30 min standing up Modal (the user already has credits per project memory), and run a 4-way fine-tune in parallel on real GPUs? The "all-night Mac-only" framing might be self-imposed. The 30-min cost might pay for itself 4× over.

4. **Time budget realism**: 1.5h to build the harness from scratch. Is that enough? Too much? Should we cannibalize phase-2's existing `validate_local.py` and `validate_submission.py` to halve harness time?

5. **Decision gates**: are the gates actually decision-making, or are they "let's see what happens" with vague triggers? Tighten or loosen.

6. **Augmentation**: 8× flips/rotations is the phase-1 default, but did anyone try elastic deformations, mixup, or cell-paste augmentation (mix nuclei from two FOVs)? The latter is known to help cell-seg in low-data regimes (40 FOVs is low-data).

7. **Val→Kaggle gap insurance**: the test-proxy split (031-035) is novel. Is it actually informative, or is the in-distribution gap (036-040 vs 031-035) too small to predict the out-of-distribution gap (train vs Kaggle test)?

8. **What's missing entirely?** Anything obvious not on the candidate list? Anything obvious not in the plan? Be specific.

## Format of your output

After arguing, produce a **single revised plan** in the same structure as above (Candidates / Architecture / Splits / 8-hour plan / Augmentation / Target). Include a short "Changes from original" section listing what you adversarially decided to revise and why.

## Full original spec (for reference)

[See the full design at `phase2-restart/docs/2026-05-03-phase1-foundation-seg-design.md` — read it before arguing if not already loaded into your context.]
