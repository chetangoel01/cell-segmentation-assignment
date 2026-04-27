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

## Overview

In Phase 1, the task is a preprocessing challenge at the heart of spatial transcriptomics: **segment cells from raw microscope images and assign thousands of mRNA molecules to the cells they belong to**.

Spatial transcriptomics maps gene expression at single-cell resolution across intact tissue. **MERFISH** (Multiplexed Error-Robust Fluorescence In Situ Hybridization) encodes each gene as a unique binary barcode read across multiple rounds of fluorescence imaging. After decoding, researchers are left with millions of mRNA dots scattered in space - but to analyze gene expression per cell, you must first determine **which dots belong to which cell**. This is the **cell segmentation and assignment** problem, and it is the bottleneck of every spatial transcriptomics pipeline.

In this competition, you will work with **raw MERFISH imaging data** from mouse brain tissue (sagittal sections, ~1,200 genes). You are given raw DAPI and polyT microscope images along with a list of decoded mRNA spots (position + gene identity) for each field of view. Your goal is to **segment the cells and assign each spot to a cell** (or to background), producing a clustering that matches the ground truth across **4 held-out test fields of view**.

**Key challenges:**

-   **Cell segmentation**: Cells vary widely in size and shape across brain regions, and nuclei often touch or overlap in densely packed tissue. Pretrained models like Cellpose provide a baseline, but the best results will come from models tuned to this tissue type.
-   **Extracellular spots**: Not every mRNA molecule lies inside a cell. A substantial fraction sits in extracellular space and must be correctly labeled as background - mistaking these for real cells hurts your score.

Start

2 days ago

Close

10 days to go

### Description

link

keyboard\_arrow\_up

### The Data

The dataset consists of MERFISH imaging data from mouse brain sagittal sections, collected from experiment `220912_wb3_sa2_2_5z18R_merfish5` (Zhuang lab, hosted on Brain Image Library). Each field of view (FOV) captures a ~220 × 220 µm region of tissue at 2048 × 2048 pixel resolution across 5 z-planes, imaging ~1,200 genes through 15 rounds of fluorescence. Each FOV provides:

-   **Raw DAPI images**: 405 nm fluorescence channel staining cell nuclei. Used as the primary input for cell segmentation. 5 z-planes per FOV.
-   **Raw polyT images**: 488 nm fluorescence channel staining mRNA-rich cytoplasm. Used together with DAPI to define cell body extent. 5 z-planes per FOV.
-   **Decoded spots**: A list of mRNA molecules already decoded from the raw barcode images. Each spot comes with its 3D global coordinates and gene identity - your pipeline does not need to perform spot decoding.
-   **Ground Truth** (training only): Cell boundaries (Cellpose 2.0 segmentation in µm coordinates per z-plane) and the cell-by-gene expression matrix.

**Note:** Spot decoding (matching fluorescence barcodes to the codebook) has already been performed for you. You only need to **segment cells** and **assign spots to cells** - this is what the evaluation measures.

### Target Variables

Column

Description

`cluster_id`

The cell each spot belongs to - an arbitrary string identifier you choose for each cell you segment. Use `background` for spots that lie outside any cell.

### Extracellular Spots

A substantial fraction of mRNA molecules lie in the extracellular space - they are real spots but do not belong to any cell. These must be labeled as `background` in your submission. Marking extracellular spots as cells (or vice versa) hurts your score.

You can identify which spots are extracellular in the training data by checking the ground truth assignment:

```clean
import pandas as pd
spots = pd.read_csv('train/ground_truth/spots_train.csv')
```

Your pipeline must handle this: predicting every spot as belonging to some cell will lower your score.

Approaches

There are several natural strategies for cell segmentation and spot assignment:

1.  Pretrained segmentation model: Run an off-the-shelf model like Cellpose, StarDist, or SAM on the DAPI/polyT images, then assign spots by point-in-polygon testing against the predicted masks. Scores ~0.63 out-of-the-box. 2. Fine-tuned segmentation: Fine-tune Cellpose (or train a custom U-Net) on the 40 training FOVs using the provided ground truth boundaries. This should outperform the pretrained baseline.
2.  Spot-aware segmentation: Use the provided decoded spots as an additional signal - regions with high spot density are more likely to be cells. This can help where DAPI/polyT signal is weak.
3.  3D segmentation: Cells span multiple z-planes. Using 3D segmentation (instead of processing each z-plane independently) may improve boundary quality

The Task

For each of the 4 test FOVs, you are given raw DAPI/polyT images and a list of decoded spots - no cell boundaries or assignments are provided. Your goal is to cluster every spot into a cell (or background) so that your clustering matches the hidden ground truth. Your goal: cluster each decoded mRNA spot into the cell it belongs to.

This is a segmentation and assignment task. You have 40 training FOVs with full ground truth (raw images, decoded spots, cell boundaries, and the cell-by-gene expression matrix) to build and validate your pipeline, then must apply it to 4 test FOVs where only raw images and spot positions are available. The test FOVs are drawn from the same brain region as training (mouse cortex), but are held out for evaluation. Your segmentation model must generalize across tissue regions without access to the ground truth cell boundaries for the test images.

### Evaluation

link

keyboard\_arrow\_up

### Adjusted Rand Index (ARI)

ARI measures how similar your clustering is to the ground truth clustering. It counts the fraction of **pairs of spots** that are grouped consistently between the two clusterings, corrected for chance. Crucially, it is **cluster-ID independent** - you can name your cells anything you want, and the metric only cares whether spots that belong together in the ground truth are also grouped together in your prediction.

For each pair of spots `(i, j)`, the two clusterings agree if either:

-   Both spots are in the **same** cluster in both the prediction and the ground truth
    
-   Both spots are in **different** clusters in both the prediction and the ground truth
    
    ARI is computed per FOV:
    
    ARI(f)\=RI(f)−E\[RI\]max(RI)−E\[RI\]
    
    Where:
    
-   **f** = FOV
    
-   **RI** = raw Rand Index (fraction of pairs that agree)
    
-   **E\[RI\]** = expected Rand Index under random labeling (the chance correction)
    
    **Final Score:**
    
    ARI = mean of ARI(f) across all 4 test FOVs
    
    This ensures all FOVs contribute equally to the final score.
    
    **Range:** -1.0 to 1.0 (higher is better). On the Kaggle leaderboard, a **higher ARI** corresponds to a **higher rank**.
    
    ARI
    
    Meaning
    
    **1.0**
    
    Perfect clustering - all spots grouped exactly as in the ground truth
    
    **\> 0.0**
    
    Clustering is better than random
    
    **0.0**
    
    Equivalent to random clustering (trivial baseline)
    
    **< 0.0**
    
    Worse than random
    

### Public/Private Split

-   **Public leaderboard:** 2 test FOVs (~115,000 spots)
    
-   **Private leaderboard:** 2 test FOVs (~109,000 spots)
    
    The split assigns whole FOVs to each side. The final ranking uses the **private** portion. This prevents overfitting to the leaderboard.
    

**Baseline model scores:**

Model

ARI

All spots -> background

0.000

All spots -> one cell (undersegmentation)

0.000

Each spot -> own cell (oversegmentation)

0.000

Random clusters

0.000

Cellpose (pretrained, out-of-the-box)

**0.632**

Perfect prediction

1.000

Pretrained Cellpose segmentation provides a strong baseline. All trivial / adversarial strategies score ~0 - the metric rewards clusterings that actually match the tissue structure. There is meaningful room to  
improve from 0.63 toward 1.0 through fine-tuning, 3D segmentation, or custom models.

## Submission Format

Submit a CSV file with **three columns** matching the format of `sample_submission.csv`:

spot\_id

fov

cluster\_id

spot\_0

B

my\_cell\_1

spot\_1

B

my\_cell\_1

spot\_2

B

background

…

…

…

-   **spot\_id**: String identifier matching the rows in `sample_submission.csv` and `test_spots.csv`.
    
-   **fov**: Which test FOV this spot belongs to (A, B, C, or D).
    
-   **cluster\_id**: Your cluster assignment for this spot. Use any string identifier for cells you detect (e.g. `my_cell_1`, `cellpose_42`). Use `background` for spots that lie outside any cell.
    
    The file `sample_submission.csv` provides a template with all spot IDs in the correct order. Replace the `cluster_id` column with your predictions.
    
    **IMPORTANT:** Your submission must have the **exact same `spot_id` ordering** as `sample_submission.csv`. Kaggle matches predictions to ground truth using `spot_id` only. The easiest approach is to load  
    `sample_submission.csv` (or `test_spots.csv`), fill in your cluster assignments for each row, and save - this guarantees correct alignment.
    
    Cluster IDs are arbitrary strings - the metric is cluster-ID independent, so naming is entirely up to you. What matters is **which spots are grouped together**, not what the groups are called.
    

**Total rows:** ~224,500 (one per provided spot across all 4 test FOVs)

Check

Requirement

**Column names**

Exactly `spot_id,fov,cluster_id`

**Row count**

Must match `sample_submission.csv`

**No nulls**

All `cluster_id` values must be non-empty strings

**Encoding**

UTF-8 CSV

### Citation

link

keyboard\_arrow\_up

Ping-Jung (Lawrence) Lu, Sirish Visweswar, and Subhrajitnyu. Cell Type Classification CS-GY 9223. https://kaggle.com/competitions/cell-type-classification-cs-gy-9223, 2026. Kaggle.

Cite

## Competition Host

Ping-Jung (Lawrence) Lu

[

](/pingjunglawrencelu)

## Prizes & Awards

Kudos

Does not award Points or Medals

## Participation

23 Entrants

3 Participants

3 Teams

5 Submissions

## Tags

Custom Metric
