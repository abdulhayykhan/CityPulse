# CityPulse: A Multi-Sensor Urban Hazard Network

ML pipeline and web app for road-hazard photos and urban-noise audio. Trained and evaluated on public datasets only (UrbanSound8K, ESC-50, RDD2022, flood image sets, NYC 311, Open-Meteo).

## Setup
```bash
conda env create -f environment.yml
conda activate citypulse
python -c "import librosa, sklearn, fastapi; print('ok')"
```

## Branch workflow
Work on your own branch (`audio`, `image`, `data-ml`, `fullstack`). Phase complete -> merge into `dev` via PR. Finalized -> `dev` into `main`. Never push directly to `main`.

## Data
Raw datasets are not committed. Download instructions and licenses: see `data/README.md` (to be added).

## Dataset licenses
The MIT License below covers this repository's code only. UrbanSound8K and ESC-50 are CC BY-NC (non-commercial); RDD2022 is CC BY-SA 4.0; Open-Meteo is CC BY 4.0; OpenStreetMap is ODbL. Check each dataset's license before any reuse.

## 📄 License
This project is open-source and available for educational and commercial use under the MIT License.
---
**Made with ❤️ by [Abdul Hayy Khan](https://www.linkedin.com/in/abdulhayykhan/)**