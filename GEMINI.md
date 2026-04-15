# Project Overview: MERFISH Cell Segmentation Assignment

This project implements a pipeline for segmenting cells from raw MERFISH microscope images (DAPI and polyT channels) and assigning mRNA spots to these cells. The ultimate goal is to generate a `submission.csv` for the Cell Type Classification competition on Kaggle, targeting high Adjusted Rand Index (ARI) scores.

## Key Technologies
- **Python 3.14+**
- **Cellpose 4.x**: For deep learning-based cell segmentation (specifically the `cyto2` model).
- **Shapely**: For point-in-polygon assignment of mRNA spots.
- **Scikit-image**: For image processing and contour extraction.
- **Pandas / NumPy**: For large-scale data manipulation and spot-to-mask mapping.
- **Pytest**: For unit and integration testing.

## Architecture
- **`src/`**: Core utility modules.
    - `io.py`: Raw `.dax` image loading and frame extraction (DAPI/polyT).
    - `coords.py`: Pixel-to-micron coordinate conversions and polygon parsing.
    - `assign.py`: Logic for assigning spots to cell polygons.
    - `train_cellpose.py`: Utilities for fine-tuning Cellpose on ground truth boundaries.
    - `evaluate.py`: Local ARI score calculation.
- **`notebooks/`**: Step-by-step development and analysis.
    - `01_eda.ipynb`: Exploratory data analysis.
    - `02_baseline.ipynb`: Pretrained Cellpose baseline.
    - `03_finetune.ipynb`: Fine-tuning Cellpose on training FOVs.
    - `04_submission.ipynb`: Test FOV inference and CSV generation.
- **`pipeline.ipynb`**: The consolidated, end-to-end pipeline driving the submission.

## Building and Running

### 1. Environment Setup
The project uses a virtual environment and a Singularity/Apptainer overlay for HPC compatibility.
```bash
# Local setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Tests
Tests are configured via `pytest.ini` and cover I/O, coordinate logic, and assignment.
```bash
PYTHONPATH=. pytest
```

### 3. Pipeline Execution
The pipeline is designed to run on the **NYU HPC (Torch)** cluster using GPU nodes.
- Refer to `docs/hpc/hpc-project-guide.md` for the full HPC workflow.
- Execute the notebook end-to-end:
```bash
jupyter nbconvert --to notebook --execute pipeline.ipynb
```

## Development Conventions
- **Two-Channel Segmentation**: Prefer using both DAPI (nuclei) and polyT (cytoplasm) for segmentation.
- **Fine-Tuning**: Always fine-tune the `cyto2` model on the 40 training FOVs before test inference.
- **Kaggle Compliance**: Use the literal string `"background"` for extracellular spots (cluster ID 0).
- **Testing**: Add a test in `tests/` for any new utility added to `src/`.
- **HPC Safety**: Keep code in `/home` and large data/overlays in `/scratch` to avoid purge/quota issues.
