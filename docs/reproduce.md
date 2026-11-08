# CityPulse Pipeline Reproduction Guide

This document outlines the end-to-end execution order, input/output artifacts, and execution commands for all preprocessing pipelines and diagnostic notebooks in the CityPulse project (Part 3).

---

## Data Artifacts & Gitignore Specification

Large data artifacts and intermediate extracts are excluded from version control via `.gitignore`.

### Gitignored Data Files
- `data/raw/*`: Original municipal 311 service request dumps.
- `data/processed/*.csv.gz`: Compressed window extractions and stage checkpoints:
  - `data/processed/complaints_window_all.csv.gz`
  - `data/processed/brooklyn_stage1.csv.gz`
  - `data/processed/brooklyn_stage2.csv.gz`
- `data/processed/311_cube.csv`: Aggregated multi-dimensional municipal cube.
- `data/processed/311_resolutions_cube.csv`: Resolution outcome cube.
- `data/processed/complaints_clean.csv`: Cleaned 25-column Brooklyn analytical dataset.
- `data/processed/flood_raw_aggregate_*.csv`: Historical raw query aggregates from Socrata.
- `data/processed/flood_raw_erm2_quarterly/`: Intermediate quarterly aggregate partitions.
- `data/processed/*_log.txt`: Extraction and transformation operational logs.

Large files: Drive link TODO

> **Prerequisite Notice**: Notebooks `01_311_window_and_groups.ipynb`, `02_brooklyn_cleaning_lab1.ipynb`, `03_features_and_projection.ipynb`, and `06_label_readiness_and_duplicates.ipynb` require the presence of these gitignored files (`complaints_window_all.csv.gz`, `brooklyn_stage1.csv.gz`, `brooklyn_stage2.csv.gz`, `311_cube.csv`, and `complaints_clean.csv`) in `data/processed/` to execute.

---

## 1. Preprocessing Pipeline Execution Order

All preprocessing scripts are located under `src/preprocessing/` and must be executed in the following sequential order from the repository root using the `citypulse` conda environment:

### Step 1.1: Local 311 Exploration
- **Command**: `python src/preprocessing/explore_311.py`
- **Inputs**: `data/raw/311_Service_Requests_from_2020_to_Present_20261107.csv`
- **Outputs**:
  - `data/processed/311_cube.csv`
  - `data/processed/311_resolutions_cube.csv`
  - `data/processed/311_boroughs.csv`
  - `data/processed/311_complaint_types.csv`
  - `data/processed/311_resolutions.csv`
  - `data/processed/311_group_candidates.csv`
  - `data/processed/311_borough_group_counts.csv`
  - `data/processed/explore_log.txt`
- **Runtime**: ~20-30 minutes (scans ~18M raw municipal records in chunks of 500,000)

### Step 1.2: Study Window Extraction & Group Assignment
- **Command**: `python src/preprocessing/extract_window.py`
- **Inputs**:
  - `data/raw/311_Service_Requests_from_2020_to_Present_20261107.csv`
  - `src/preprocessing/groups.py`
- **Outputs**:
  - `data/processed/group_definitions.csv`
  - `data/processed/complaints_window_all.csv.gz`
  - `data/processed/resolution_lookup.csv`
  - `data/processed/extract_log.txt`
- **Runtime**: ~15-20 minutes

### Step 1.3: Long-History Flood & Precipitation Extraction
- **Command**: `python src/preprocessing/flood_history.py`
- **Inputs**: Socrata Open Data APIs (`76ig-c548`, `erm2-nwe9`), Open-Meteo Historical Weather API
- **Outputs**:
  - `data/processed/flood_daily_all_years.csv`
  - `data/processed/rain_daily.csv`
  - `data/processed/rain_hourly_brooklyn.csv`
  - `data/processed/flood_raw_aggregate_2010_2019.csv`
  - `data/processed/flood_raw_aggregate_2020_onward.csv`
  - `data/processed/flood_descriptor_by_year.csv`
  - `data/processed/flood_type_discovery.csv`
  - `data/processed/flood_monthly_transition.csv`
  - `data/processed/flood_history_log.txt`
- **Runtime**: ~10-15 minutes (with rate-limit backoff and pacing)

### Step 2.1: Brooklyn Complaint Data Cleaning
- **Command**: `python -c "from src.preprocessing.complaints import load_extract, clean_complaints; df = load_extract('data/processed/complaints_window_all.csv.gz'); clean_complaints(df, 'data/processed/dropped_outside_area.csv')"`
- **Inputs**: `data/processed/complaints_window_all.csv.gz`
- **Outputs**:
  - `data/processed/brooklyn_stage1.csv.gz`
  - `data/processed/dropped_outside_area.csv`
  - `data/processed/report_tables/missing_before.csv`
  - `data/processed/report_tables/missing_after.csv`
  - `data/processed/report_tables/step_table.csv`
- **Runtime**: ~1-2 minutes

### Step 2.2: Feature Engineering & Planar Equirectangular Projection
- **Command**: `python -c "from src.preprocessing.features import add_time_features, add_projection, save_projection_constants; ..."`
- **Inputs**: `data/processed/brooklyn_stage1.csv.gz`
- **Outputs**:
  - `data/processed/brooklyn_stage2.csv.gz`
  - `data/processed/projection.json`
- **Runtime**: ~1-2 minutes

### Step 2.3: Final Clean Dataset Construction & Descriptor Grouping
- **Command**: `python src/preprocessing/build_clean.py`
- **Inputs**:
  - `data/processed/brooklyn_stage2.csv.gz`
  - `src/preprocessing/descriptor_groups.py`
- **Outputs**:
  - `data/processed/complaints_clean.csv`
  - `data/processed/complaints_clean_columns.csv`
  - `data/processed/descriptor_groups.csv`
- **Runtime**: ~1-2 minutes

### Step 2.4: Multi-Borough Comparative Evaluation
- **Command**: `python src/preprocessing/borough_check.py`
- **Inputs**:
  - `data/processed/complaints_window_all.csv.gz`
  - `data/processed/flood_vs_rain_all_boroughs.csv`
- **Outputs**: `data/processed/report_tables/t1_borough_summary.csv` through `t7_evidence_summary.csv`
- **Runtime**: ~30-45 seconds

### Step 2.5: Label Readiness & Spatial Duplicate Analysis
- **Command**: `python src/preprocessing/label_readiness.py`
- **Inputs**:
  - `data/processed/complaints_clean.csv`
  - `data/processed/rain_hourly_brooklyn.csv`
- **Outputs**: `data/processed/report_tables/t24_*.csv`
- **Runtime**: ~30-60 seconds

---

## 2. Jupyter Notebook Execution Order

All notebooks are executed in sequence using `jupyter nbconvert` with the `--execute` and `--inplace` flags:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/<notebook_name>.ipynb
```

| Order | Notebook | Primary Inputs | Key Outputs / Artifacts | Approx Runtime |
| :---: | :--- | :--- | :--- | :---: |
| 1 | `00_data_inventory.ipynb` | `data/processed/` directory | Directory file catalog and schema audit | ~5 s |
| 2 | `01_311_window_and_groups.ipynb` | `311_cube.csv`, `complaints_window_all.csv.gz`, `group_definitions.csv` | Monthly volume profiles, candidate group distributions | ~10 s |
| 3 | `02_brooklyn_cleaning_lab1.ipynb` | `complaints_window_all.csv.gz`, `brooklyn_stage1.csv.gz` | Cleaning step audit, missingness comparisons | ~15 s |
| 4 | `03_features_and_projection.ipynb` | `brooklyn_stage1.csv.gz`, `brooklyn_stage2.csv.gz`, `projection.json` | Spatial projection validation, temporal distributions | ~20 s |
| 5 | `04_borough_selection.ipynb` | `complaints_window_all.csv.gz`, `flood_vs_rain_all_boroughs.csv` | Multi-borough correlation profiles, bootstrap CIs | ~20 s |
| 6 | `05_descriptor_groups_and_clean_dataset.ipynb` | `complaints_clean.csv`, `descriptor_groups.csv` | Standardized descriptor categories, leakage audit | ~20 s |
| 7 | `06_label_readiness_and_duplicates.ipynb` | `complaints_clean.csv`, `rain_hourly_brooklyn.csv` | Sub-daily rain lags, duplicate clustering, label maturity | ~30 s |
| 8 | `07_flood_history_and_taxonomy_drift.ipynb` | `flood_daily_all_years.csv`, `rain_daily.csv`, aggregates, `311_cube.csv` | 16-year panel integrity, taxonomy drift, dry-spell tests | ~25 s |
| 9 | `08_hotspot_sanity.ipynb` | `complaints_clean.csv`, `projection.json` | Coordinate concentration, weak clusters, capping limits | ~20 s |

---

## 3. Automated Verification

After generating or updating notebooks, verify structural validity, cell executions, file sizes, and assertions:

```powershell
python tests/check_notebooks.py
```

