# CityPulse: A Multi-Sensor Urban Hazard Network

CityPulse is a multi-sensor machine learning pipeline and web application for detecting, prioritizing, and forecasting urban hazards—combining road-hazard imagery, urban-noise audio, municipal 311 service requests, and historical weather feeds. All models and analytics are trained and evaluated exclusively on public datasets (**UrbanSound8K**, **ESC-50**, **RDD2022**, public flood image sets, **NYC 311**, **Open-Meteo**, and **OpenStreetMap**).

---

## Overview

CityPulse unifies multimodal hazard sensing with municipal analytics across three core pillars:

- **Audio & Visual Hazard Sensing**: Audio classification for urban noise events and computer vision detection for road damage and surface flooding (`backend/`, `frontend/`).
- **Municipal Spatio-Temporal Analytics (`src/preprocessing/`, `src/clustering/`)**: Ingestion, cleaning, planar equirectangular projection, descriptor taxonomy mapping, spatio-temporal incident deduplication (`within 200 m and 4 h`), and balanced 8-zone hotspot profiling across Brooklyn 311 complaints (`data/processed/zones.json`).
- **Predictive Modeling & Triage (`src/models/`, `models/`)**:
  - **Daily Flood Risk (`predict_flood_risk`)**: Linear regression (`models/flood_model.joblib`) linking daily and lagged **Open-Meteo** precipitation to expected municipal flood complaints for risk-tier alerting.
  - **Hourly Noise Baseline (`predict_noise_v2`)**: Hour-by-weekday interaction regression (`models/noise_model_v2.joblib`) capturing late-night weekend surges and diurnal noise profiles.
  - **Complaint Trust Scoring (`score_trust`)**: Calibrated, prior-shifted logistic regression (`models/trust_model.joblib`) estimating empirical resolution validity from complaint metadata and causal 24-hour spatial density (`nearby_prior_24h_200m`).

For methodological caveats, proxy-label scope, and deployment guardrails, see [`report/ethics_and_limitations.md`](report/ethics_and_limitations.md).

---

## Repository Structure

```text
CityPulse/
├── src/
│   ├── preprocessing/   # 311 extraction, Open-Meteo ingestion, cleaning, feature engineering
│   ├── clustering/      # Spatio-temporal hierarchical deduplication & K-Means zone profiling
│   ├── models/          # Flood regression, noise interaction model, trust scorer & label rules
│   ├── inventory.py     # Processed dataset & schema inventory helpers
│   └── nbutils.py       # Shared notebook setup and programmatic Checker utilities
├── models/              # Serialized joblib model bundles (flood, noise, trust)
├── notebooks/           # Executed end-to-end analysis, audit, and modeling notebooks
├── tests/               # Unit tests (pytest) and notebook structural verifier (check_notebooks.py)
├── docs/                # Pipeline reproduction guide and integration handoff docs
├── report/              # Project proposal, execution guide, figures/, and ethics_and_limitations.md
├── backend/             # Backend API service workspace
├── frontend/            # Frontend web application workspace
├── environment.yml      # Conda environment specification (Python 3.11 + ffmpeg)
└── requirements.txt     # Python package dependencies
```

---

## Setup

### 1. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate citypulse
```

### 2. Register the Jupyter Kernel

Register the `citypulse` environment as a Jupyter kernel before running or executing notebooks:

```bash
python -m ipykernel install --user --name citypulse --display-name "Python (citypulse)"
```

### 3. Verify Installation

Confirm core audio, ML, and API dependencies import cleanly:

```bash
python -c "import librosa, sklearn, fastapi; print('ok')"
```

Run unit tests and the notebook structural/execution verification suite from the repository root:

```bash
python -m pytest tests
python tests/check_notebooks.py
```

---

## Running the Web Application (Full-Stack)

The FastAPI backend loads every team model and the processed 311 data, and serves the
dashboard frontend from the same origin — so the whole app runs from **one command**.

### Quick start

```bash
conda activate citypulse
```

Windows (PowerShell):

```powershell
.\run.ps1
```

macOS / Linux:

```bash
./run.sh
```

…or run it directly:

```bash
uvicorn backend.main:app --reload
```

Then open **http://localhost:8000/** for the dashboard (API docs at **/docs**).

> The first time you click **Plan route**, the backend downloads the Brooklyn drive
> network from OpenStreetMap (internet required) and caches it to
> `data/brooklyn_drive.graphml`; every call after that is offline and fast.
> On startup the backend builds `data/citypulse.db` from `incidents.csv` once.

### Endpoints

| Method & path | Purpose |
|---|---|
| `GET /` | Dashboard (served from `frontend/`) |
| `GET /incidents` | Incident hotspots, ranked by report count (`?limit=&group=&min_count=`) |
| `GET /zones` | 8 K-Means hotspot zones with per-group counts |
| `POST /predict` | `rain_mm` → flood risk, `hour`+`weekday` → noise, spread across zones |
| `POST /report` | Upload clip/photo → `predict_audio`/`predict_image` + trust score (webm→wav via ffmpeg) |
| `POST /route` | Calm & safe route: osmnx graph penalised near incidents, `networkx` shortest path |
| `GET /admin` | Incidents ranked by report count × severity |
| `GET /demo_list`, `GET /reports` | Demo samples list; persisted demo submissions |

### Dashboard features

- Dark Leaflet map with incident markers, livability-scored zones, and a legend.
- **Demo mode**: pick a bundled clip/photo, click the map to place it, get a live
  classification with confidence / relative-intensity / trust meters and a map pin.
- **Live mic** (optional): records 5 s and classifies it through the same path.
- **Forecast**: rainfall and time sliders recolour zones by flood / noise risk;
  a report-density mode flags low-data areas.
- **Calm & safe route**: draws the plain route and a hazard-avoiding route; dropping a
  report on a route makes the safe path detour around it.
- **Priority board** and a Chart.js livability chart.

### Screenshots

| Dashboard | Report result | Safe route |
|---|---|---|
| ![Dashboard](docs/screenshots/01_dashboard.jpg) | ![Report result](docs/screenshots/02_report_result.jpg) | ![Safe route](docs/screenshots/03_safe_route.jpg) |

See [`docs/demo_script.md`](docs/demo_script.md) for the rehearsed demo walkthrough.

---

## Docs (`docs/`)
* [`docs/image_rules.md`](docs/image_rules.md) — Part 2 image pipeline guidelines (`image_rules.md`: `GroupShuffleSplit` on `image_id`, relative manifest paths, bootstrap confidence intervals, and `scikit-learn 1.9.1` model packaging).

## Docs (`docs/`)
* [`docs/audio_rules.md`](docs/audio_rules.md) — Part 3 audio pipeline guidelines (source-grouped splits, 22,050 Hz / 4 s fixed clips, 64-band log-Mel features, train-only `relative_intensity` bounds, degraded-data training, and cross-dataset evaluation).

## Data

**Raw datasets are NOT committed to this repository.**

To keep the repository lightweight and respect upstream data terms, raw dumps under `data/raw/*` (such as full **NYC 311** CSV exports) and large intermediate/derived tables under `data/processed/` (including `*.csv.gz`, `311_cube.csv`, `311_resolutions_cube.csv`, `complaints_clean.csv`, `incidents.csv`, and `complaint_incident_map.csv`) are excluded via [`.gitignore`](.gitignore).

Primary public data sources:
- **NYC 311**: *311 Service Requests from 2020 to Present* via NYC Open Data / Socrata APIs.
- **Open-Meteo**: Historical daily and hourly precipitation feeds via the Open-Meteo Historical Weather API.
- **Audio, Road & Map Corpora**: UrbanSound8K, ESC-50, RDD2022, public flood image sets, and OpenStreetMap.

For full pipeline execution order, script commands, and intermediate artifact specifications, see [`docs/reproduce.md`](docs/reproduce.md). For downstream backend/frontend model signatures, CSV reading conventions, and schema contracts, see [`docs/handoff_zainab.md`](docs/handoff_zainab.md).

---

## Branch Workflow

Contributors develop features on dedicated domain branches (`audio`, `image`, `data-ml`, `fullstack`). Completed phases merge into `dev` via pull request, and finalized releases merge from `dev` into `main`. Never push directly to `main`.

---

## Dataset Licenses

Upstream public datasets carry their own terms and licenses—verify each source before redistribution or commercial reuse:
- **UrbanSound8K & ESC-50**: CC BY-NC (non-commercial)
- **RDD2022**: CC BY-SA 4.0
- **Open-Meteo**: CC BY 4.0
- **OpenStreetMap**: ODbL
- **NYC 311 (NYC Open Data)**: Public open data

---

## License

This project's source code and documentation are released under the **MIT License** — see [`LICENSE`](LICENSE) for the full license text.

```text
MIT License

Copyright (c) 2026 Abdul Hayy Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
