# HPC → Modal Sync (handoff)

You are on NYU Cloud Burst HPC. Job: push phase-2 competition data and a couple
of BIL files into the Modal volume `cell-seg-phase2` so a parallel session can
train a cell-type classifier there.

## Context

- Source (read-only): `/scratch/pl2820/competition_phase2/`
- Destination: Modal volume `cell-seg-phase2` (profile `chetangoel2011`)
- Already on Modal under `/external/aws/`: `Zhuang-ABCA-4-log2.h5ad`,
  `cell_metadata_with_cluster_annotation.csv`, `gene.csv`. Don't re-upload these.
- BIL is blocked from Modal egress IPs but **works from HPC** — that's why
  we're using HPC as the bridge.

## Prereqs

```bash
# Modal CLI (per-user install).
pip install --user modal
export PATH="$HOME/.local/bin:$PATH"

# Auth — opens a browser-flow link. Print the link and have the user paste the
# resulting token back. Profile must be `chetangoel2011`.
modal token new
modal profile current   # verify
```

If browser auth is awkward on HPC, generate a token id/secret pair on a laptop
(`modal token new` there) and copy `~/.modal.toml` over.

## 1. Phase-2 raw data sync (priority — required for any submission)

Mirror the course layout into the volume root. Test set first; train can wait.

```bash
SRC=/scratch/pl2820/competition_phase2

# Test FOVs (~25 GB, 10 dirs FOV_E..FOV_N) — required.
modal volume put cell-seg-phase2 "$SRC/test" /test

# Spots + submission template — required.
modal volume put cell-seg-phase2 "$SRC/test_spots.csv"        /test_spots.csv
modal volume put cell-seg-phase2 "$SRC/sample_submission.csv" /sample_submission.csv

# Reference (codebook, fov_metadata, dataorganization) — required.
modal volume put cell-seg-phase2 "$SRC/reference" /reference
```

If `modal volume put` complains about size or stalls on a single big upload,
fall back to per-file:

```bash
for fov in FOV_E FOV_F FOV_G FOV_H FOV_I FOV_J FOV_K FOV_L FOV_M FOV_N; do
    modal volume put cell-seg-phase2 "$SRC/test/$fov" "/test/$fov"
done
```

## 2. Ground-truth + train (recommended — for local validation)

Ground truth files are small enough to take immediately (a few GB total):

```bash
modal volume put cell-seg-phase2 "$SRC/train/ground_truth" /train/ground_truth
```

Skip the 60 train FOV `.dax` directories unless segmentation retraining is on
the menu — that's ~150 GB and Stage-1 of the classifier doesn't need them.

## 3. BIL extras (optional but valuable)

BIL works from HPC. The repo already has `phase2/scripts/fetch_bil.py` which
knows the right URLs. Pull these to a local HPC scratch dir, then push:

```bash
cd ~/cell-segmentation-assignment   # or wherever the repo is
export MERFISH_DATA_ROOT=/scratch/cg4652/phase2_bil_stage

# ~6.5 GB total: cell_boundaries (3.6 GB), counts h5ad (1 GB), spots (1.8 GB),
# tiled_positions, codebook, dataorganization.
python3 phase2/scripts/fetch_bil.py fetch-support

# Push the whole bundle to the Modal volume under /external/competition_support/.
modal volume put cell-seg-phase2 \
    "$MERFISH_DATA_ROOT/external/competition_support" \
    /external/competition_support
```

If `fetch-support` is slow (BIL throttles to ~35 KB/s for big files; the 3.6 GB
cell_boundaries file is the long pole at ~30 hr worst case), kick it off in a
background tmux session and move on. The classifier path can start without it.

## 4. Verify

```bash
# Should show: external/, reference/, test/, train/ (if step 2 ran),
# test_spots.csv, sample_submission.csv
modal volume ls cell-seg-phase2 /

# Spot-check a test FOV.
modal volume ls cell-seg-phase2 /test/FOV_E | head
```

Expected layout when done:

```
/
├── external/
│   ├── aws/                       (already there)
│   └── competition_support/       (step 3, optional)
├── reference/
│   ├── codebook.csv
│   ├── dataorganization.csv
│   └── fov_metadata.csv
├── test/
│   ├── FOV_E/  …  FOV_N/
├── train/
│   └── ground_truth/              (step 2)
├── test_spots.csv
└── sample_submission.csv
```

## 5. Report back

Reply with:
- `modal volume ls cell-seg-phase2 /` output (whole tree, top-level + key subdirs).
- Total bytes synced (`du -sh` on whatever you staged into HPC scratch before push).
- Whether step 3 (BIL extras) ran or was skipped.
- Any errors or partial uploads.

Do NOT modify code in the repo from this session — sync only.
