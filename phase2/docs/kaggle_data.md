# Cell Type Classification Phase 2 CS-GY 9223 | Kaggle

[

](/pingjunglawrencelu)Ping-Jung (Lawrence) Lu · Community Prediction Competition · 8 days to go

Submit Prediction

more\_horiz

# Cell Type Classification Phase 2 CS-GY 9223

Competition Track for Project 3 Part 2 - Neuroinformatics (CS-GY 9923) at NYU Tandon Spring 2026.

![](/competitions/138545/images/header)

## Cell Type Classification Phase 2 CS-GY 9223

[Overview](/competitions/cell-type-classification-phase-2-cs-gy-9223/overview)[Data](/competitions/cell-type-classification-phase-2-cs-gy-9223/data)[Code](/competitions/cell-type-classification-phase-2-cs-gy-9223/code)[Models](/competitions/cell-type-classification-phase-2-cs-gy-9223/models)[Discussion](/competitions/cell-type-classification-phase-2-cs-gy-9223/discussion)[Leaderboard](/competitions/cell-type-classification-phase-2-cs-gy-9223/leaderboard)[Rules](/competitions/cell-type-classification-phase-2-cs-gy-9223/rules)[Team](/competitions/cell-type-classification-phase-2-cs-gy-9223/team)[Submissions](/competitions/cell-type-classification-phase-2-cs-gy-9223/submissions)

## Dataset Description

### Downloading the Data

The full dataset (~170 GB) is hosted on NYU Cloud Burst HPC at `/scratch/pl2820/competition_phase2/`. If you have a Cloud Burst account, you can read the data directly — no download needed:

```bash
cd /scratch/pl2820/competition_phase2/
ls
# train/  test/  reference/  test_spots.csv  sample_submission.csv  metric.py
```

Otherwise, copy it to your local machine using rsync (strongly recommended over scp because rsync can resume on failure):

```gradle
rsync -avP --exclude='.ipynb_checkpoints' \<netid>@log-burst.hpc.nyu.edu:/scratch/pl2820/competition_phase2/ ./competition_phase2/
```

We strongly recommend working directly on Cloud Burst via SSH rather than pulling 170 GB to your laptop.

---

File Structure

```nix
competition_phase2/
|── train/
│   ├── FOV_101/                          # 18 raw .dax image files per FOV
│   ├── FOV_102/
│   ├── ...  (60 training FOVs)
│   └── ground_truth/
│       ├── spots_train.csv               # decoded spots + gene identity + positions
│       ├── cell_boundaries_train.csv     # cell segmentation polygons (µm per z-plane)
│       ├── cell_labels_train.csv         # per-cell labels at 4 hierarchy levels + CCF
│       └── counts_train.h5ad             # cell-by-gene expression matrix (AnnData)
├── test/
│   ├── FOV_E/                            # 18 raw .dax image files — no ground truth
│   ├── FOV_F/
│   ├── ...  (10 test FOVs)
│   └── FOV_N/
├── reference/
│   ├── codebook.csv                      # 1,240 genes × 32-bit binary barcodes
│   ├── dataorganization.csv              # channel/frame/z-plane layout for .dax files
│   └── fov_metadata.csv                  # FOV origin (fov_x, fov_y in µm) + pixel size
|── test_spots.csv                        # decoded spots for test FOVs (you classify them)
├── sample_submission.csv                 # submission template
└── metric.py                             # official scoring script (ARI at 4 levels)
```

## 60 training FOVs and 10 test FOVs (5 public, 5 private) are provided.

test\_spots.csv

Each row is one decoded mRNA spot in a test FOV that you must classify. This defines the expected rows in your submission:

Column

Description

`spot_id`

Unique string identifying this spot (e.g. `spot_0`)

`fov`

Which test FOV (`FOV_E` through `FOV_N`)

`image_row`

Pixel row in the 2048×2048 DAPI image (use directly with `mask[row, col]`)

`image_col`

Pixel column in the 2048×2048 DAPI image

`global_x`

Spot x-coordinate in µm (global frame)

`global_y`

Spot y-coordinate in µm (global frame)

`global_z`

Z-plane index (0–4)

`target_gene`

Decoded gene name (e.g. `Slc17a7`)

---

sample\_submission.csv

A template submission with every spot set to background. Replace the label columns with your predictions:

Column

Description

`spot_id`

Unique string (matches `test_spots.csv`)

`fov`

Test FOV

`class`

Predicted class label (Allen taxonomy string, `background`)

`subclass`

Predicted subclass label

`supertype`

Predicted supertype label

`cluster`

Predicted cluster label

---

Raw Image Files (.dax)

Each FOV folder contains 18 raw .dax files - uint16, 2048 × 2048 pixels per frame. Frame layout is specified in reference/dataorganization.csv.

File

Frames

Contents

`Epi-750s5-635s5-545s1-473s5-408s5_{fov}.dax`

27

DAPI, polyT, fiducial + 2 gene channels × 5 z-planes

`Epi-750s5-635s5-545s1_{fov}_{round}.dax`

17

2 gene channels × 5 z-planes (rounds 00–14)

`Epi-750s1-635s1-545s1_{fov}_{0,1}.dax`

7

Fiducial-only files for image registration

---

Reference Files

codebook.csv - Maps each of 1,240 genes to a unique 32-bit binary barcode. Each gene has exactly 4 bits set.

dataorganization.csv - Tells you which frame in each .dax file contains which channel (DAPI, polyT, gene bits) and z-plane.

fov\_metadata.csv — FOV origin and pixel size, used to convert pixel <-> global µm coordinates:

Column

Description

`fov`

FOV name (`FOV_101` … `FOV_160`, `FOV_E` … `FOV_N`)

`fov_x`

X-origin of the FOV in µm (global frame)

`fov_y`

Y-origin of the FOV in µm (global frame)

`pixel_size`

µm per pixel (0.109 for all FOVs)

---

Training Ground Truth

spots\_train.csv - Decoded spots from the original Zhuang lab pipeline:

Column

Description

`barcode_id`

Barcode index in the codebook

`fov`

FOV name (`FOV_101` … `FOV_160`)

`image_row`, `image_col`

Pixel coordinates (use directly with `mask[row, col]`)

`global_x`, `global_y`

Spot position in µm (global frame)

`global_z`

Z-plane index (0–4)

`x`, `y`

Spot position in pixels (FOV-local, raw Zhuang convention)

`target_gene`

Decoded gene name

cell\_boundaries\_train.csv - Cell segmentation polygons:

Column

Description

Index

Cell ID (string)

`boundaryX_z0` … `boundaryX_z4`

Comma-separated x-coordinates of boundary polygon (µm) per z-plane

`boundaryY_z0` … `boundaryY_z4`

Comma-separated y-coordinates of boundary polygon (µm) per z-plane

cell\_labels\_train.csv - 4-level cell type labels + CCF coordinates for each training cell:

Column

Description

`cell_id`

Cell ID (matches `cell_boundaries_train.csv` index)

`fov`

FOV name

`center_x`, `center_y`

Cell centroid in µm (global frame)

`class_label`

Allen taxonomy class (level 1) — one of 11 values: 10 named cell types + background\`

`subclass_label`

Allen taxonomy subclass (level 2)

`supertype_label`

Allen taxonomy supertype (level 3)

`cluster_label`

Allen taxonomy cluster (level 4)

`ccf_x`, `ccf_y`, `ccf_z`

Allen CCF 3D coordinates (mm) — only for cells Allen could register

Note: Cells whose class\_label is background are cells that the original GT pipeline could not confidently label (Allen QC drops + rare cell types that were merged out). All four hierarchy levels are background for these cells. They appear in both the training data and the test ground truth, and they are scored normally (not excluded). You can choose whether to use them during training; predicting background correctly for the corresponding spots will count toward your score.

counts\_train.h5ad - Cell-by-gene expression matrix in AnnData format:

```clean
import anndata as ad
adata = ad.read_h5ad('train/ground_truth/counts_train.h5ad')
# adata.X           : (5230 cells × 1147 genes) count matrix
# adata.obs         : cell metadata (fov, volume, center_x, center_y,
#                     class_label, subclass_label, supertype_label, cluster_label,
#                     ccf_x, ccf_y, ccf_z)
# adata.var_names   : gene names
```

---

Loading Data (Python)

```clean
import numpy as np
import pandas as pd
import anndata as ad

# ---- Load a raw DAPI / polyT image for segmentation ----
# Multichannel Epi file: 27 frames total
# DAPI  (405 nm): frames [6, 11, 16, 21, 26]  — z0 to z4
# polyT (488 nm): frames [5, 10, 15, 20, 25]  — z0 to z4
raw = np.fromfile('train/FOV_101/Epi-750s5-635s5-545s1-473s5-408s5_101.dax', dtype=np.uint16).reshape(-1, 2048, 2048)
dapi_z2  = raw[16]   # DAPI at middle z-plane
polyt_z2 = raw[15]   # polyT at middle z-plane

# ---- Load training ground truth ----
spots_train  = pd.read_csv('train/ground_truth/spots_train.csv')
cells_train  = pd.read_csv('train/ground_truth/cell_boundaries_train.csv', index_col=0)
labels_train = pd.read_csv('train/ground_truth/cell_labels_train.csv', index_col='cell_id')
adata        = ad.read_h5ad('train/ground_truth/counts_train.h5ad')

# ---- Load reference files ----
meta = pd.read_csv('reference/fov_metadata.csv').set_index('fov')
fov_x = meta.loc['FOV_101', 'fov_x']
fov_y = meta.loc['FOV_101', 'fov_y']
pixel_size = meta.loc['FOV_101', 'pixel_size']   # 0.109 µm per pixel

# ---- Load test spots ----
test_spots = pd.read_csv('test_spots.csv')
# fov column: 'FOV_E', 'FOV_F', ..., 'FOV_N'

# ---- Example: once you have a segmentation mask for FOV_E ----
# mask = your_segmentation(load_dapi('test/FOV_E/...'))   # (2048, 2048) int
# fov_e_spots = test_spots[test_spots['fov'] == 'FOV_E']
# rows = fov_e_spots['image_row'].values
# cols = fov_e_spots['image_col'].values
# cell_ids = mask[rows, cols]        # 0 = background, >0 = your cell ID
# # then classify each detected cell based on its aggregated expression profile

# ---- Load submission template ----
sub = pd.read_csv('sample_submission.csv')
```

## Files

18 files

## Size

2.45 GB

## Type

dax

## License

[MIT](https://www.mit.edu/~amini/LICENSE.md)

### FOV\_101(18 files)

fullscreen

chevron\_right

insert\_drive\_file

Epi-750s1-635s1-545s1\_101\_0.dax

41.94 MB

insert\_drive\_file

Epi-750s1-635s1-545s1\_101\_1.dax

41.94 MB

insert\_drive\_file

Epi-750s5-635s5-545s1-473s5-408s5\_101.dax

226.49 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_00.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_01.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_02.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_03.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_04.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_05.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_06.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_07.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_08.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_09.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_10.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_11.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_12.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_13.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_101\_14.dax

142.61 MB

## Data Explorer

2.45 GB

-   arrow\_drop\_down
    
    folder
    
    FOV\_101
    
    -   insert\_drive\_file
        
        Epi-750s1-635s1-545s1\_101\_0.dax
        
    -   insert\_drive\_file
        
        Epi-750s1-635s1-545s1\_101\_1.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1-473s5-408s5\_101.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_00.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_01.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_02.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_03.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_04.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_05.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_06.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_07.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_08.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_09.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_10.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_11.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_12.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_13.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_101\_14.dax
        

## Summary

arrow\_right

folder

18 files

get\_appDownload All

text\_snippet

## Metadata

### License

[MIT](https://www.mit.edu/~amini/LICENSE.md)
