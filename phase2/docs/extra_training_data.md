# Extra Training Data — BIL + AWS sources

Reference doc for pulling additional matched-distribution training data for Phase 2.
Discovered and verified 2026-04-28.

## TL;DR

The course's 60 train FOVs (`FOV_101`–`FOV_160`) are a 60-tile crop of a **783-tile imaging session** publicly deposited at:

- **Brain Image Library (BIL)** — full raw `.dax` + cell boundaries. Throttled at ~35 KB/s on residential.
- **Allen Brain Cell Atlas (AWS S3)** — same dataset, expression matrices + taxonomy labels, ~18 MB/s.

You can **double or 12× expand** the labeled training pool by pulling tiles outside the course's 60+10 train/test set, all from the same animal/session/panel.

| Path | What you get | Cost | Time |
|---|---|---|---|
| **Classifier path (recommended)** | 32,528 labeled cells from competition source via AWS | 157 MB | ~10 sec on AWS |
| **Segmentation path** | Raw `.dax` for 60 extra tiles + cell boundaries | ~133 GB + 3.6 GB | hours on HPC |

## Source identification

The course's experiment is `220912_wb3_sa2_2_5z18R_merfish5`:
- `220912` = Sept 12 2022
- `wb3` = whole-brain pipeline v3 (lab protocol prefix — **NOT** "animal 3")
- `sa2_2` = sagittal-2 sample 2 = mouse 4
- `5z18R` = 5 z-planes × 18 imaging rounds
- `merfish5` = panel/codebook v2

This corresponds to:
- **BIL deposit DOI 10.35077/act-bag**, dataset ID `293cc39ceea87f6d`, path `sagittal_2/220912_wb3_sa2_2_5z18R_merfish5/`
- **AWS Allen Brain Cell Atlas**, dataset `Zhuang-ABCA-4`, section `Zhuang-ABCA-4.001`

## Section ↔ session ↔ animal mapping

`mouse4_sagittal` (BIL README) = `Zhuang-ABCA-4` (AWS) has 3 sagittal sections:

| AWS section | BIL session | sample_id | Panel | Course role |
|---|---|---|---|---|
| `.001` | `220912_wb3_sa2_2_5z18R_merfish5` | sa2_sample1 | merfish5 / codebook v2 | **competition source** |
| `.002` | `220908_sa2_4_merfish4_adaptor` | sa2_sample2 | merfish4 (older panel) | matched-pool (panel-shifted) |
| `.003` | `220908_wb3_sa2_12_5z18R_merfish5` | sa2_sample3 | merfish5 / codebook v2 | matched-pool (same panel) |

Verified by sampling cell_ids from BIL `cell_boundaries_updated/.../<session>.csv` and grepping in AWS `cell_metadata_with_cluster_annotation.csv` — 3/3 hits for `.001 ↔ 220912`, 1/2 for `.003 ↔ 220908_sa2_12`.

**Cross-animal pool** (mouse3, same protocol): 7 sessions in `sagittal_1/220609..220717_wb3_sa1_*_5z18R_merfish5/`. Different animal but same panel/codebook — 79K labeled cells in AWS section `Zhuang-ABCA-3` (separate dataset, not covered here).

## Tile-level structure (competition source session)

- **783 tiles total**, contiguous indices 0–782 (3-digit padded in this session — other sessions use 4-digit).
- Course took:
  - **60 train tiles**: indices 101–160 (preserved verbatim as `FOV_101`–`FOV_160` directories).
  - **10 test tiles**: unknown indices, renamed to lettered directories `FOV_E`–`FOV_N`.
- **713 unused tiles** are extras candidates — same animal, same section, same panel, same protocol.

Per tile in BIL: 18 `.dax` files = 2 fiducials (5z) + 15 round files (17z) + 1 multichannel preimage (27z) = ~2.45 GB raw.

**Test FOV → tile index discovery:** the course renamed only the directory wrapper; inner `.dax` filenames preserve the lab's numeric tile index. On HPC, run:
```bash
for f in FOV_E FOV_F FOV_G FOV_H FOV_I FOV_J FOV_K FOV_L FOV_M FOV_N; do
    python3 phase2/scripts/fetch_bil.py verify "$f" --split test 2>/dev/null | grep "discovered"
done
```
Output gives the 10 lab tile indices (e.g., 050, 200, 300, …). Pass to `--exclude-tiles` when fetching extras to prevent leakage.

## Support file catalog

| Course local file | BIL equivalent | AWS equivalent | Sliceable by FOV |
|---|---|---|---|
| `reference/fov_metadata.csv` | `settings/tiled_positions.txt` (40 KB) | — | yes (line N = tile N) |
| `spots_train.csv` | `decoded_spots/.../spots_220912_*.csv` (1.81 GB) | — | yes (sorted ascending by `fov`) |
| `cell_boundaries_train.csv` | `cell_boundaries_updated/.../220912_*.csv` (3.62 GB) | — | ⚠️ no `fov` column — needs cell_label join |
| `cell_labels_train.csv` | embedded in `counts_*.h5ad.obs` (1.01 GB) | `cell_metadata_with_cluster_annotation.csv` (51 MB) | yes (filter by `brain_section_label`) |
| `counts_train.h5ad` | `counts_updated/counts_mouse4_sagittal.h5ad` (1.01 GB) | `Zhuang-ABCA-4-log2.h5ad` (107 MB) | yes |
| codebook | `additional_files/codebook_32bit_v2.csv` (93 KB) | — | n/a |

AWS h5ad is 16× smaller than BIL h5ad because it strips out cell boundaries. Use BIL only when you need the polygon-level segmentation outputs.

## Verified findings (2026-04-28)

End-to-end loaded and cross-referenced locally:

```
=== AnnData (Zhuang-ABCA-4-log2.h5ad) ===
shape: (215,278 cells, 1,122 genes)
obs.index.name: 'cell_label' (dtype=object/string)
section coverage:
  .001 = 53,701 cells (competition source)
  .002 = 66,473 cells
  .003 = 95,104 cells

=== Annotation CSV ===
shape: (162,578 cells, 23 cols)
all 4 taxonomy levels 100% populated:
  class, subclass, supertype, cluster

=== Cross-reference ===
labels ↔ h5ad index: 162,578/162,578 match (0 mismatches)
in h5ad but unlabeled: 52,700 (lab quality-filtered out — ignore for training)

=== Section .001 (competition source) ===
labeled cells: 32,528
top classes:
  01 IT-ET Glut       7,298 (22%)
  29 CB Glut          6,587
  33 Vascular         5,115
  30 Astro-Epen       4,377
  31 OPC-Oligo        3,706
```

The cell_label join is **byte-equal** between BIL `cell_boundaries[0]` and AWS `cell_label` — no string normalization needed.

## Throughput observations

- **BIL CDN**: ~35 KB/s per IP for files >100 MB (residential connections). 1 GB takes ~7 hours; 6.45 GB support package takes ~50 hours. Throttling applies to Range requests too.
- **AWS S3** (us-west-2): ~18 MB/s on residential. 1 GB takes ~1 minute; full Zhuang-ABCA-4 dataset (~157 MB default) takes ~10 sec.
- **HPC**: untested but campus gigabit usually outpaces both.

→ Use AWS for big files; BIL only for raw `.dax` and per-session cell_boundaries (which AWS doesn't have).

## Scripts

### `phase2/scripts/fetch_bil.py`

| Subcommand | Purpose |
|---|---|
| `verify FOV_X [--split test]` | Confirm BIL has matching files for a local FOV; reports the discovered lab tile index (works for letter test FOVs too — extracts tile from inner filenames). |
| `list-pool` | Print the safe-list of matched-pool sessions (excludes competition source). |
| `inspect <sample_id>` | List `.dax` files in a matched-pool session with per-tile sizes. |
| `fetch <sample_id> [--rounds-only] [--tile-prefix REGEX] [--limit N]` | Download a matched-pool session's `.dax` files. |
| `fetch-source [--exclude-tiles ...] [--adjacent] [--limit N] [--rounds-only]` | Pull extras from the competition source session. Auto-excludes train tiles 101–160; refuses to fetch without `--exclude-tiles` to prevent leakage. `--adjacent` picks tiles closest in index to train range. |
| `fetch-support` | Pull positions, spots, cell_boundaries, counts h5ad, codebook for the source session (~6.45 GB; throttled). |
| `fetch-aws [--with-raw] [--raw-only]` | Pull Zhuang-ABCA-4 files from AWS S3 (default: annotation + log2 + gene = 157 MB). ★ Use this for the classifier path. |
| `fetch-counts <animal>` | Pull cell-by-gene `.h5ad` from BIL (specific animal). Same data as AWS but slower. |

### `phase2/scripts/inventory.py`

Status report across five data groups: `course/local`, `aws/Zhuang-ABCA-4`, `bil/extras`, `bil/support`, `bil/matched-pool`. Validates byte sizes against expected; flags partial downloads. Generates next-step recommendations based on what's missing.

```bash
python3 phase2/scripts/inventory.py          # text mode
python3 phase2/scripts/inventory.py --json   # machine-readable
```

## Recommended workflows

### Classifier path (start here — fast, simple)

```bash
# 1. Pull AWS files (157 MB, ~10 sec)
python3 phase2/scripts/fetch_bil.py fetch-aws

# 2. Train classifier on h5ad → cell labels
python3 -c "
import anndata, pandas as pd
ad = anndata.read_h5ad('phase2/data/external/aws/Zhuang-ABCA-4-log2.h5ad', backed='r')
labels = pd.read_csv('phase2/data/external/aws/cell_metadata_with_cluster_annotation.csv', dtype={'cell_label': str})

# Filter to competition source section
sec1_labels = labels[labels['brain_section_label'] == 'Zhuang-ABCA-4.001']
sec1_ids = set(sec1_labels['cell_label'])

# 32,528 labeled cells from competition source — train hierarchy classifier
"
```

### Segmentation path (heavy — HPC required)

```bash
# 1. On HPC, discover test tile indices (one-time)
for f in FOV_E FOV_F FOV_G FOV_H FOV_I FOV_J FOV_K FOV_L FOV_M FOV_N; do
    python3 phase2/scripts/fetch_bil.py verify "$f" --split test 2>/dev/null | grep "discovered"
done
# Save the 10 indices.

# 2. Pull 60 adjacent extras (133 GB rounds-only)
python3 phase2/scripts/fetch_bil.py fetch-source \
    --exclude-tiles "<comma-sep test indices>" \
    --adjacent --rounds-only --limit 60 --workers 8

# 3. Pull support files (6.45 GB, throttled — leave overnight)
python3 phase2/scripts/fetch_bil.py fetch-support
```

### Cell_boundaries → fov join (when needed)

BIL `cell_boundaries.csv` has no `fov` column. To slice by tile, join via `obs` from the h5ad:

```python
import anndata, pandas as pd, csv
ad = anndata.read_h5ad('counts_mouse4_sagittal.h5ad', backed='r')
target_tiles = set(range(101, 161))  # or extras
target_cells = set(ad.obs.index[ad.obs.fov.isin(target_tiles)].astype(str))
with open('cell_boundaries_220912_*.csv') as f, open('out.csv', 'w') as out:
    r = csv.reader(f); w = csv.writer(out)
    w.writerow(next(r))
    for row in r:
        if row[0] in target_cells:
            w.writerow(row)
```

(Note: BIL counts h5ad has the `fov` column in obs; AWS h5ad does not — AWS uses `brain_section_label` and CCF coords.)

## Leakage warning

**Always exclude the competition source from the matched pool** when training. The 60 train + 10 test FOVs are tiles within `220912_wb3_sa2_2_5z18R_merfish5`. Expanding within this session is fine for *extras* (just exclude train + test indices), but **don't pull test tiles' `.dax` files unless you want test labels to leak into training**.

The script's `fetch-source` enforces this — it refuses to run without `--exclude-tiles` (or an explicit empty override).

## Open items

- AWS section `.002` mapping is inferred (0/2 cell_id hits in our sample, likely because `merfish4_adaptor` is panel-shifted and AWS quality-filters more aggressively). Doesn't affect main path.
- Final ground-truth file sync from HPC `/scratch/pl2820/competition_phase2/` for local dev — needed for end-to-end pipeline tests.

## Useful URLs

- BIL deposit root: <https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/>
- BIL DOI landing: <https://doi.brainimagelibrary.org/doi/10.35077/act-bag>
- AWS bucket: <https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/>
- AWS manifest: <https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/releases/20241130/manifest.json>
- ABCA tutorial: <https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html>
