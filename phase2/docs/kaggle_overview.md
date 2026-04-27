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

## Overview

In Phase 1, you built a preprocessing pipeline: segmenting cells from raw microscope images and clustering decoded mRNA spots into cells. In Phase 2, the task builds on top of that: **classify every cell at multiple levels of biological specificity**.

Spatial transcriptomics only becomes useful when every cell is assigned a biologically meaningful identity. The Allen Brain Cell Atlas organizes mouse brain cells into a **5-level taxonomy** - from broad neurotransmitter classes down to fine-grained clusters - capturing how cell types are nested within each other. Automatically mapping new cells onto this taxonomy is what turns raw spatial data into a scientific resource.

In this competition, you will work with **raw MERFISH imaging data** from mouse brain tissue (sagittal sections, ~1,200 genes) and must **predict, for each decoded mRNA spot, the cell type of the cell it belongs to - at four hierarchy levels** (class, subclass, supertype, cluster) across 10 held-out test fields of view.

Label vocabulary: at every hierarchy level there are 10 named cell-type labels plus a single background label. background covers spots that aren't assignable to a labeled cell - either spots outside any cell, or spots in cells the original GT pipeline could not confidently label. You are not expected to predict any other labels.

Key challenges

-   Multi-level classification: your model must predict at 4 levels of granularity simultaneously. A reference Cellpose+kNN baseline (segment cells, then classify each by k-nearest-neighbours on its expression vector) scores 0.35 out of 1.0 - there is large headroom for better segmentation, better classifiers, and use of spatial context.
-   Long-tailed cell type distribution: some classes have hundreds of cells in training, others only ~15. Rare cell types are easy to ignore - and easy to get wrong.
-   Spatial context matters: a neuron in cortical layer 5 looks different from one in layer 2/3 even when expression profiles partially overlap. Leveraging tissue neighborhood (CCF coordinates are provided) can push well past the naive baseline.

Start

16 hours ago

Close

8 days to go

### Description

link

keyboard\_arrow\_up

### The Data

The dataset consists of raw MERFISH imaging data from mouse brain sagittal sections (experiment `220912_wb3_sa2_2_5z18R_merfish5`, Zhuang lab, Brain Image Library). Each field of view (FOV) captures a ~220 × 220 µm region of tissue at 2048 × 2048 pixel resolution across 5 z-planes, imaging ~1,200 genes through 15 rounds of fluorescence microscopy.

For Phase 2, cell type labels come from the **Allen Brain Cell Atlas taxonomy** (Zhuang-ABCA-4), which organizes cells into a 5-level nested hierarchy. We provide the 4 lower levels in ground truth; you predict all four.

Split

FOVs

Cells

Spots

Classes

Train

60

5,230

2.9M

11

Test

10

846

439K

11

### Cell Type Hierarchy

For each labeled cell, the Allen taxonomy provides:

-   class (11 values) - broad category: e.g., 01 IT-ET Glut, 29 CB Glut, 30 Astro-Epen, 33 Vascular
-   subclass (37 values) - mid-level: e.g., 006 L4/5 IT CTX Glut
-   supertype (62 values) - fine-level variant: e.g., 0030 L4/5 IT CTX Glut\_2
-   cluster (83 values) - finest: individual molecular clusters

### Special Labels

-   background - the only special label. Covers (a) spots that don't fall inside any cell (extracellular), and (b) spots in cells that the original GT pipeline could not confidently label, including rare cell types that were merged out. Together this is ~83% of test spots. Predict background whenever you don't believe a spot belongs to one of the 10 named classes.

### Segmentation and Classification Approaches

Phase 2 is an end-to-end pipeline task. Natural strategies include:

1.  **Pretrained segmentation + gene-expression classifier.** Run Cellpose / StarDist / SAM on DAPI and polyT, build per-cell expression profiles, and train a classifier (logistic regression, Random Forest, XGBoost, MLP) to predict cell type from gene expression. This is roughly the baseline we provide.
    
2.  **Fine-tuned segmentation.** Use the provided training boundaries to fine-tune a segmentation model specifically for this tissue type, then classify.
    
3.  **Hierarchy-aware classifiers.** Predict the coarse class first, then predict subclass conditional on class, then supertype conditional on subclass, etc. Our naive kNN baseline produces ~the same ARI at every level; a hierarchy-aware model should score higher at finer levels.
    
4.  **Spatial context models.** Use the CCF coordinates and/or local neighborhood expression to inform classification. A cortical cell in L2/3 differs from one in L5 even if their individual expression profiles partially overlap.
    

### The Task

For each of the 10 test FOVs, you are given raw DAPI, polyT, and gene-channel images plus a list of decoded mRNA spots. **You predict, for each spot, the cell type at 4 hierarchy levels** (class, subclass, supertype, cluster). You also submit the label `background` for spots not inside any cell.

Your training data (60 FOVs) includes full ground truth: raw images, decoded spots, cell boundaries, expression matrix, and the 4-level cell type labels plus CCF coordinates for each cell. The test FOVs contain raw images and a `test_spots.csv` of decoded spots - cell boundaries, cell identities, and labels are hidden.

### Evaluation

link

keyboard\_arrow\_up

Submissions are evaluated by comparing your per-spot predictions to the hidden ground truth across **10 test FOVs** (~439,000 total spots).

### Adjusted Rand Index (ARI) at 4 Hierarchy Levels

For each of the 4 cell-type hierarchy levels (`class`, `subclass`, `supertype`, `cluster`) and each of the 10 test FOVs, we compute the **Adjusted Rand Index (ARI)** between your predicted labels and the ground truth labels, treating labels as cluster assignments.

ARI measures how similar two clusterings are — it counts the fraction of **pairs of spots** that are grouped consistently between the two clusterings, corrected for chance. Crucially, ARI is **cluster-ID independent**: for a given level, what matters is which spots share a label, not what the label is called.

ARI is computed per (FOV, level):

ARI(f,ℓ)\=RI(f,ℓ)−E\[RI\]max(RI)−E\[RI\]

where

f

is a FOV,

ℓ

is a hierarchy level,

RI

is the raw Rand Index (fraction of agreeing pairs) and

E\[RI\]

is the expected Rand Index under random labeling.

### Final Score

Final Score = mean of ARI(f, ℓ) across all 10 test FOVs × 4 levels = 40 values

This is equivalent to computing the mean ARI per level across FOVs, then averaging those 4 numbers - all hierarchy levels contribute equally.

**Range:** -1.0 to 1.0 (higher is better). On the Kaggle leaderboard, a **higher mean ARI** corresponds to a **higher rank**.

Score

Meaning

**1.0**

Perfect - predictions match GT clustering at every level

**\> 0.0**

Clustering is better than random

**0.0**

Equivalent to random labeling (trivial baseline)

**< 0.0**

Worse than random

### Scoring includes every spot

All ~439,000 test spots contribute to the score, including spots labeled background in the GT. There are no excluded spots and no special labels. Predict background for spots you don't believe belong to one of the 10 named cell types.

### Public/Private Leaderboard Split

Leaderboard

Test FOVs

Spots

Public

5 FOVs

~221,000

Private

5 FOVs

~218,000

The split assigns whole FOVs to each side. The final ranking uses the **private** portion.

### Baseline Scores

Model

Mean ARI

All `background`

0.000

Majority class (constant)

0.000

Random draw from GT distribution

0.001

**Cellpose (pretrained) + kNN classifier**

**0.351**

Perfect prediction (upper bound)

1.000

**Baseline model per-level ARI (Cellpose + kNN)**:

Level

ARI

class

0.347

subclass

0.352

supertype

0.352

cluster

0.351

The baseline's per-level ARI is essentially flat - the kNN classifier treats the four hierarchy levels independently and its errors correlate across all levels. This is a competition design feature, not a bug: **hierarchy-aware models** (predicting coarse -> fine conditionally) should be able to outperform the baseline specifically at the finer levels, where the baseline's correlated-error pattern is most visible.

**Goal:** achieve mean ARI well above 0.351 by building pipelines that exploit the hierarchy, leverage spatial context (CCF coordinates), or fine-tune segmentation models to this tissue type.

### Citation

link

keyboard\_arrow\_up

Ping-Jung (Lawrence) Lu. Cell Type Classification Phase 2 CS-GY 9223. https://kaggle.com/competitions/cell-type-classification-phase-2-cs-gy-9223, 2026. Kaggle.

Cite

## Competition Host

Ping-Jung (Lawrence) Lu

[

](/pingjunglawrencelu)

## Prizes & Awards

Kudos

Does not award Points or Medals

## Participation

12 Entrants

2 Participants

2 Teams

3 Submissions

## Tags

Custom Metric
