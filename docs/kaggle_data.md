# Cell Type Classification CS-GY 9223 | Kaggle

[

](/pingjunglawrencelu)Ping-Jung (Lawrence) Lu · Community Prediction Competition · 10 days to go

Submit Prediction

more\_horiz

# Cell Type Classification CS-GY 9223

Competition Track for Project 3 Part 1 - Neuroinformatics (CS-GY 9923) at NYU Tandon Spring 2026.

![](/competitions/137645/images/header)

## Cell Type Classification CS-GY 9223

[Overview](/competitions/cell-type-classification-cs-gy-9223/overview)[Data](/competitions/cell-type-classification-cs-gy-9223/data)[Code](/competitions/cell-type-classification-cs-gy-9223/code)[Models](/competitions/cell-type-classification-cs-gy-9223/models)[Discussion](/competitions/cell-type-classification-cs-gy-9223/discussion)[Leaderboard](/competitions/cell-type-classification-cs-gy-9223/leaderboard)[Rules](/competitions/cell-type-classification-cs-gy-9223/rules)[Team](/competitions/cell-type-classification-cs-gy-9223/team)[Submissions](/competitions/cell-type-classification-cs-gy-9223/submissions)

## Dataset Description

### Downloading the Data

The full dataset (~101 GB) is hosted on NYU Cloud Burst HPC at `/scratch/pl2820/competition/`. If you have a Cloud Burst account, you can read the data directly - no download needed:

```bash
cd /scratch/pl2820/competition/    # on cloud burst
ls
# train/  test/  reference/  test_spots.csv  sample_submission.csv  metric.py ...
```

Otherwise, copy it to your local machine using rsync:

```elixir
rsync -avP <user>@<hpc-host>:/scratch/pl2820/competition/ ./competition/                                                        
```

After downloading, your local competition/ directory should match the structure below. You only need to upload the small submission CSV (~5 MB) to Kaggle for scoring - the pipeline runs locally on your machine.

---

### File Structure

```nix
  competition/
  ├── train/
  │   ├── FOV_001/                          # 18 raw .dax image files per FOV
  │   ├── FOV_002/
  │   ├── ...  (40 training FOVs)
  │   └── ground_truth/
  │       ├── spots_train.csv               # decoded spots with gene identity + positions
  │       ├── cell_boundaries_train.csv     # cell segmentation polygons (µm per z-plane)
  │       └── counts_train.h5ad             # cell-by-gene expression matrix (AnnData)
  ├── test/
  │   ├── FOV_A/                            # 18 raw .dax image files — no ground truth
  │   ├── FOV_B/
  │   ├── FOV_C/
  │   └── FOV_D/
  ├── reference/
  │   ├── codebook.csv                      # 1,240 genes × 32-bit binary barcodes
  │   ├── dataorganization.csv              # channel/frame/z-plane layout for .dax files
  │   └── fov_metadata.csv                  # FOV origin (fov_x, fov_y in µm) + pixel size
  ├── test_spots.csv                        # decoded spots for test FOVs (you assign to cells)
  ├── sample_submission.csv                 # submission template
  ├── metric.py                             # ARI scoring script (for local evaluation)
  └── generate_submission.py                # helper to build submission from your pipeline outputs
```

40 training FOVs and 4 test FOVs (2 public, 2 private) are provided.

---

test\_spots.csv

Each row is one decoded mRNA spot in a test FOV that you must assign to a cell. This defines the expected rows in your submission:

Column

Description

`spot_id`

Unique string identifying this spot (e.g. `spot_0`)

`fov`

Which test FOV (`FOV_A`, `FOV_B`, `FOV_C`, or `FOV_D`)

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

A template submission with every spot set to background. Replace the cluster\_id column with your cell assignments:

Column

Description

`spot_id`

Unique string (matches `test_spots.csv`)

`fov`

Test FOV (`FOV_A`, `FOV_B`, `FOV_C`, or `FOV_D`)

`cluster_id`

Your cluster assignment — any string ID (`background` for extracellular spots)

---

Raw Image Files (.dax) Each FOV folder contains 18 raw .dax files - uint16, 2048 × 2048 pixels per frame, multiple frames per file covering 5 z-planes:

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

codebook.csv - Maps each of 1,240 genes to a unique 32-bit binary barcode. Each gene has exactly 4 bits set dataorganization.csv - Tells you which frame in each .dax file contains which channel (DAPI, polyT, gene bits) and z-plane. fov\_metadata.csv - FOV origin and pixel size, used to convert pixel <->global µm coordinates:

Column

Description

`fov`

FOV name with `FOV_` prefix (e.g. `FOV_001`, `FOV_A`)

`fov_x`

X-origin of the FOV in µm (global frame)

`fov_y`

Y-origin of the FOV in µm (global frame)

`pixel_size`

µm per pixel (0.109 for all FOVs)

---

Ground Truth (training only) spots\_train.csv - Decoded spots from the original Zhuang lab pipeline:

Column

Description

`barcode_id`

Barcode index in the codebook

`global_x`, `global_y`

Spot position in µm

`global_z`

Z-plane index (0–4)

`x`, `y`

Spot position in pixels (FOV-local)

`fov`

FOV name

`target_gene`

Decoded gene name

cell\_boundaries\_train.csv - Cell segmentation polygons:

Column

Description

Index

Cell ID (string)

`boundaryX_z0` … `boundaryX_z4`

Comma-separated x-coordinates (µm) per z-plane

`boundaryY_z0` … `boundaryY_z4`

Comma-separated y-coordinates (µm) per z-plane

`counts_train.h5ad` - Cell-by-gene expression matrix in AnnData format:

```clean
import anndata as ad
adata = ad.read_h5ad('train/ground_truth/counts_train.h5ad')
# adata.X           : (4082 cells × 1147 genes) count matrix
# adata.obs         : cell metadata (fov, center_x, center_y, volume)
# adata.var_names   : gene names
```

---

metric.py

The ARI metric used for scoring is included. You can use it for local evaluation (e.g. on held-out training FOVs):

```python
import pandas as pd
from metric import score

# Solution is hidden on Kaggle — use a held-out training FOV for local cross-validation
solution = pd.read_csv('local_solution.csv')
submission = pd.read_csv('my_submission.csv')
print(f"ARI: {score(solution, submission, 'spot_id'):.4f}")
```

---

generate\_submission.py

A helper script to build your submission CSV from your pipeline outputs (decoded spots + cell masks):

python generate\_submission.py --spots test\_spots.csv --cells my\_segmentation.csv --output submission.csv

---

Loading Data (Python)

```pgsql
import numpy as np
import pandas as pd
import anndata as ad

# ---- Load a raw DAPI image for segmentation ----                                                                                                                                                                     
raw = np.fromfile('train/FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax', dtype=np.uint16).reshape(-1, 2048, 2048)                                                                                                                                                             
dapi_z2  = raw[16]   # DAPI at middle z-plane                                                                                                                                                                          
polyt_z2 = raw[15]   # polyT at middle z-plane                                                                                                                                                                         

# ---- Load decoded spots (training) ----                                                                                                                                                                              
spots_train = pd.read_csv('train/ground_truth/spots_train.csv')                                                                                                                                                        
# fov column values are strings like 'FOV_001', 'FOV_019', etc.                                                                                                                                                        

# ---- Load ground truth cell boundaries (training) ----                                                                                                                                                               
cells_train = pd.read_csv('train/ground_truth/cell_boundaries_train.csv', index_col=0)                                                                                                                                 

# ---- Load training expression matrix ----                                                                                                                                                                            
adata = ad.read_h5ad('train/ground_truth/counts_train.h5ad')                                                                                                                                                           
# adata.obs['fov'] values are 'FOV_001', 'FOV_002', etc.                                                                                                                                                               

# ---- Load FOV metadata to convert pixel <-> µm ----                                                                                                                                                                  
meta = pd.read_csv('reference/fov_metadata.csv').set_index('fov')                                                                                                                                                      
fov_x = meta.loc['FOV_001', 'fov_x']             # FOV origin in µm                                                                                                                                                    
fov_y = meta.loc['FOV_001', 'fov_y']                                                                                                                                                                                   
pixel_size = meta.loc['FOV_001', 'pixel_size']   # 0.109 µm per pixel                                                                                                                                                  

# ---- Load test spots (what you assign to cells) ----                                                                                                                                                                 
test_spots = pd.read_csv('test_spots.csv')                                                                                                                                                                             
# fov column values: 'FOV_A', 'FOV_B', 'FOV_C', 'FOV_D'                                                                                                                                                                

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

### FOV\_001(18 files)

fullscreen

chevron\_right

insert\_drive\_file

Epi-750s1-635s1-545s1\_001\_0.dax

41.94 MB

insert\_drive\_file

Epi-750s1-635s1-545s1\_001\_1.dax

41.94 MB

insert\_drive\_file

Epi-750s5-635s5-545s1-473s5-408s5\_001.dax

226.49 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_00.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_01.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_02.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_03.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_04.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_05.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_06.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_07.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_08.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_09.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_10.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_11.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_12.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_13.dax

142.61 MB

insert\_drive\_file

Epi-750s5-635s5-545s1\_001\_14.dax

142.61 MB

## Data Explorer

2.45 GB

-   arrow\_drop\_down
    
    folder
    
    FOV\_001
    
    -   insert\_drive\_file
        
        Epi-750s1-635s1-545s1\_001\_0.dax
        
    -   insert\_drive\_file
        
        Epi-750s1-635s1-545s1\_001\_1.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1-473s5-408s5\_001.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_00.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_01.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_02.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_03.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_04.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_05.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_06.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_07.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_08.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_09.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_10.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_11.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_12.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_13.dax
        
    -   insert\_drive\_file
        
        Epi-750s5-635s5-545s1\_001\_14.dax
        

## Summary

arrow\_right

folder

18 files

get\_appDownload All

Download using Kaggle CLI

navigate\_nextminimize

content\_copyhelp

Download using Kaggle CLI

kagglehub

navigate\_nextminimize

content\_copyhelp

kagglehub

MCP

navigate\_nextminimize

content\_copyhelp

MCP

text\_snippet

## Metadata

### License

[MIT](https://www.mit.edu/~amini/LICENSE.md)
