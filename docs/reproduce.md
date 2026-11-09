# Reproducibility Guide

**1. Environment Setup**
Recreate the exact execution environment using the root configuration file:
`bash
conda env create -f environment.yml
conda activate citypulse
`

**2. Data Handling**
*   Place all raw datasets into `data/raw/`.
*   Processed data will automatically generate in `data/processed/audio/`, `data/processed/image/`, and `data/processed/text/`. Do not track heavy `.npy`, `.csv`, or `.pt` files in version control.

**3. Pipeline Execution**
Execute the modular pipelines from the repository root:
*   **Image Pipeline:** `python src/image/run_all.py`
*   **Audio & Text Pipelines:** Execute relevant sequential scripts from `src/preprocessing/`, `src/clustering/`, and `src/models/`.
*   **Notebooks:** Experimental workflows and EDA are isolated in `notebooks/` and should be ported to `src/` for production execution.