# CityPulse Handoff Package — Zainab (Backend / Frontend Integration)

**Source Branch**: `data-ml`  
**Study Window (Seeded Data)**: `2025-11-01` through `2026-10-31` (ends `2026-10-31`)  
**Recommended Runtime Versions**: `Python 3.11.18`, `scikit-learn 1.9.1`, `joblib 1.6.0`, `pandas 3.1.1`, `numpy 2.4.6`, `scipy 1.17.1`

---

## 1. Handoff File Inventory

> Size in MB is shown as binary MB (`bytes / 1024^2`, matching Windows / repo utilities) alongside decimal MB (`bytes / 10^6`) and exact byte size for unambiguous checksum verification.

| Name | Rows / Structure | Size MB | SHA-256 | Columns / Keys in Exact Order |
| :--- | :--- | :--- | :--- | :--- |
| `complaints_clean.csv` | `236,507` rows (`25` cols) | `66.2623 MB` (`69.4811 MB` dec / `69,481,076 B`) | `fb0fa8f176b04c8258efa490d63855c572bc055f2272e9d138a3c0532b50a4d7` | `id`, `lat`, `lon`, `timestamp`, `group`, `descriptor`, `zone_id`, `descriptor_group`, `complaint_type`, `borough`, `status`, `closed_date`, `time_to_close_h`, `hour`, `weekday`, `month`, `date`, `is_midnight_timestamp`, `channel`, `location_type`, `incident_address`, `address_norm`, `resolution_id`, `x_m`, `y_m` |
| `incidents.csv` | `154,599` rows (`8` cols) | `13.7448 MB` (`14.4125 MB` dec / `14,412,491 B`) | `fe2eea51a8af06aa372930286e2aa64bfa015346d87837ed47543b3594e1583e` | `incident_id`, `lat`, `lon`, `count`, `group`, `first_timestamp`, `last_timestamp`, `zone_id` |
| `zones.json` | JSON array of `8` zone objects (`9` keys each) | `0.0035 MB` (`0.0037 MB` dec / `3,699 B`) | `4e36e3a9d0861257c50e700063bf49118da2787bf772d89f7cd751c39b728949` | `zone_id`, `centre_lat`, `centre_lon`, `count`, `dominant_group`, `counts_by_group`, `lift_by_group`, `dominant_lift`, `radius_m_p90` |
| `trust_model.joblib` | Joblib `dict` (`LogisticRegression(class_weight='balanced')` + `OneHotEncoder`, `83` encoded features, `Pooled_NoTop5Noise`) | `0.0084 MB` (`0.0088 MB` dec / `8,808 B`) | `1d51b001b243a9ce3b34b6b1bbb446eb9599f8d7ad4d614d5eca311e4bb00698` | Bundle keys: `model`, `encoder`, `feature_names`, `categorical_cols`, `numeric_cols`, `sklearn_version`, `created_date`, `variant`, `train_cutoff_ts`, `excluded_top5_noise_coords_train`<br>Raw inputs (`7`): `complaint_type`, `descriptor_group`, `channel`, `location_type`, `hour`, `weekday`, `nearby_prior_24h_200m` |
| `flood_model.joblib` | Joblib `dict` (`LinearRegression`, `4` features, `variant_a`) | `0.0008 MB` (`0.0008 MB` dec / `787 B`) | `7bd63d00b54a7578a1c1b41da28a64c98ddeac6831a4922ec209df78fa94b2ce` | Bundle keys: `model`, `feature_names`, `target`, `sklearn_version`, `created_date`, `variant`<br>`feature_names` (`4`): `rain_t`, `rain_t1`, `rain_t2`, `rain_sum_t1_t3` |
| `noise_model_v2.joblib` | Joblib `dict` (`LinearRegression`, `167` features, `interaction_hour_weekday_v2`) | `0.0097 MB` (`0.0102 MB` dec / `10,190 B`) | `852d251a4facf4b10188d7115db5e7043e3cee45bdef6fda67aac0758d9aae0b` | Bundle keys: `model`, `feature_names`, `target`, `sklearn_version`, `created_date`, `variant`, `interaction_type`<br>`feature_names` (`167`): `hour_1..hour_23`, `weekday_1..weekday_6`, `hour_{1..23}_x_weekday_{1..6}` |

Original repository locations:
- `data/processed/complaints_clean.csv`
- `data/processed/incidents.csv`
- `data/processed/zones.json`
- `models/trust_model.joblib`
- `models/flood_model.joblib`
- `models/noise_model_v2.joblib`

---

## 2. CSV Reading Notes

Always read `id` and `resolution_id` explicitly as strings (`str`) and set `keep_default_na=False` when loading `complaints_clean.csv`:

```python
import pandas as pd

complaints_df = pd.read_csv(
    "complaints_clean.csv",
    dtype={"id": str, "resolution_id": str},
    keep_default_na=False,
)

incidents_df = pd.read_csv(
    "incidents.csv",
    dtype={"incident_id": str},
    keep_default_na=False,
)
```

Why this matters:
- `id` is a numeric-looking unique identifier string.
- `resolution_id` contains `3,108` empty values stored as `""`; without `dtype={"resolution_id": str}` and `keep_default_na=False`, pandas converts blanks to `NaN` and parses integer IDs into floats (`"2.0"`), breaking lookups and trust labeling.
- `closed_date`, `incident_address`, `address_norm`, and `location_type` also use empty string `""` for unclosed/absent values, while literal strings such as `"UNKNOWN"` should not be silently converted to `NaN`.

---

## 3. `zones.json` Schema

`zones.json` is a JSON list of **8** Brooklyn spatial zone objects (`zone_id` `0` through `7`), sorted by `zone_id` (south-to-north / cluster order; `sum(count) == 236,507`).

Each object has **9** keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `zone_id` | `int` | Zone index (`0..7`). Matches `zone_id` in `complaints_clean.csv` and `incidents.csv`. |
| `centre_lat` | `float` | Zone cluster centroid latitude (WGS84). |
| `centre_lon` | `float` | Zone cluster centroid longitude (WGS84). |
| `count` | `int` | Total complaints assigned to this zone across all 4 groups. |
| `dominant_group` | `str` | Dominant problem concentration (`"mixed"`, `"road"`, `"sewer_backup"`, or `"flood"`). |
| `counts_by_group` | `dict[str, int]` | Per-group complaint counts (`flood`, `noise`, `road`, `sewer_backup`). |
| `lift_by_group` | `dict[str, float]` | Zone share divided by Brooklyn-wide share for each group (`flood`, `noise`, `road`, `sewer_backup`). |
| `dominant_lift` | `float` | Summary dominant lift metric for the zone. |
| `radius_m_p90` | `float` | 90th-percentile complaint distance from zone centroid in metres (~`2,143 m` to `3,088 m`). |

Example entry (`zone_id = 0`):

```json
{
  "zone_id": 0,
  "centre_lat": 40.713236,
  "centre_lon": -73.950159,
  "count": 29760,
  "dominant_group": "mixed",
  "counts_by_group": {
    "flood": 520,
    "noise": 27482,
    "road": 1549,
    "sewer_backup": 209
  },
  "lift_by_group": {
    "flood": 0.7486,
    "noise": 1.031,
    "road": 0.7628,
    "sewer_backup": 0.5525
  },
  "dominant_lift": 1.031,
  "radius_m_p90": 2143.15
}
```

---

## 4. Incidents Constraint (`GET /incidents`)

- `incidents.csv` contains **154,599** incidents mapping all `236,507` complaints (`max count = 85`), deduplicated at `time_scale = 50.0` m/h, `T* = 200.0` m.
- **75.24%** of incidents (`116,326 / 154,599`, exact `75.2437%`) are **singletons** (`count == 1`). Only **38,273** incidents (`24.76%`) have `count >= 2`.
- **`GET /incidents` MUST filter** on query parameters and/or sensible defaults (for example, `count >= 2` / `min_count=2` by default, spatial `bbox`, `group`, `zone_id`, date window, and pagination `limit`). Returning all `154,599` rows unfiltered on an API call or map view will degrade backend latency and freeze client map rendering.

---

## 5. Study Area & Projection

- **Borough**: Brooklyn (`borough == "BROOKLYN"` exclusively).
- **Study Bounding Box**:
  - Latitude: **`40.55` to `40.74`**
  - Longitude: **`-74.06` to `-73.83`**
  - Map centre: **`40.65, -73.95`**
- **Observed Data Bounds (`complaints_clean.csv`)**: lat `[40.571483, 40.739067]`, lon `[-74.040822, -73.855776]`.
- **Planar Equirectangular Projection** (`data/processed/projection.json`, `src/preprocessing/features.py`):
  - Fixed origin: `lat0 = 40.661641`, `lon0 = -73.952455`
  - `x_m = (lon - lon0) * 111320.0 * cos(radians(lat0))`
  - `y_m = (lat - lat0) * 110540.0`
  - Always project live `(lat, lon)` reports using these exact constants via `project(lat, lon, lat0=40.661641, lon0=-73.952455)`.

---

## 6. Live Demo Merging Note

- The seeded dataset (`complaints_clean.csv` and `incidents.csv`) ends on **`2026-10-31`** (`2025-11-01T00:00:35` to `2026-10-31T23:59:46`).
- Incoming **live demo reports merge only with each other** — treat `incidents.csv` as frozen historical seed data and maintain live demo incidents in a separate runtime collection/table rather than mutating historical seeded rows.
- Merge live demo submissions using `online_assign` from `src/clustering/hierarchical_dedup.py`:

```python
from src.preprocessing.features import project, load_projection_constants
from src.clustering.hierarchical_dedup import online_assign

lat0, lon0 = 40.661641, -73.952455  # or load_projection_constants()
x_m, y_m = project(lat, lon, lat0=lat0, lon0=lon0)

matched_incident_id = online_assign(
    x_m=x_m,
    y_m=y_m,
    t=timestamp_iso,
    group=group,               # one of {'noise', 'road', 'flood', 'sewer_backup'}
    incidents_xy=live_incidents_df,
    time_scale=50.0,
    T=200.0,
)
```

- **Merging window (`within 200m and 4h`)**: `online_assign` checks same-`group` live incidents using Euclidean spatio-temporal distance $\sqrt{(x_m - x_{\text{inc}})^2 + (y_m - y_{\text{inc}})^2 + (50.0 \cdot \Delta t_{\text{hours}})^2} \le 200.0$, where $\Delta t_{\text{hours}} = 0$ inside `[first_timestamp, last_timestamp]` and equals the hour gap to the nearest endpoint otherwise. At zero spatial offset, $50.0 \cdot \Delta t_{\text{hours}} \le 200.0 \iff \Delta t_{\text{hours}} \le 4.0\text{ h}$. If no match is within `T = 200.0`, `online_assign` returns `None` and the backend starts a new live incident.

---

## 7. Model Functions & API Contracts

All three production inference helpers are exported from `src/models/__init__.py`. If loading models from the standalone handoff folder outside the repo, pass `model_path=Path(...)` explicitly.

### A. Flood Risk — `predict_flood_risk(rain_mm)`
- **Import**: `from src.models.regression import predict_flood_risk` (backed by `models/flood_model.joblib`, `LinearRegression`, 4 features `['rain_t', 'rain_t1', 'rain_t2', 'rain_sum_t1_t3']`).
- **Signature**: `predict_flood_risk(rain_mm: float, model_path: Optional[Path] = None) -> float`
- **Behavior**: Returns expected Brooklyn-wide daily flood complaint count (`>= 0.0`) for today's rainfall `rain_mm` assuming dry antecedent days (`rain_t1 = rain_t2 = rain_sum_t1_t3 = 0.0`). Raises `ValueError` for negative, `NaN`, or `inf` inputs. Approximate linear response: `3.813 + 1.143 * rain_mm` (e.g., `0 mm -> 3.81`, `10 mm -> 15.24`, `25 mm -> 32.39`, `50 mm -> 60.96`).
- **Important Caveat (Risk Bands Only / Extreme Underprediction)**: **Use this output to map into categorical risk bands** (e.g., Low / Moderate / Elevated / High alert tiers), **not** as an exact count forecast during major storms. Because OLS lacks a nonlinear runoff/saturation threshold, it severely underpredicts extreme cloudburst events (`rain_t > 50 mm`: 2021–2025 rolling mean predicted `76.29` vs actual `188.78`; on `2023-09-29` with `62.1 mm` rain, rolling predicted `64.09` vs `970` actual complaints).

### B. Hourly Noise Profile — `predict_noise_v2(hour, weekday)`
- **Import**: `from src.models.regression import predict_noise_v2` (backed by `models/noise_model_v2.joblib`, `LinearRegression`, 167 `hour * weekday` interaction terms + intercept).
- **Signature**: `predict_noise_v2(hour: int, weekday: int, model_path: Optional[Path] = None) -> float`
- **Inputs & Encoding**:
  - `hour`: integer `0..23`
  - `weekday`: ISO integer `0..6` where **`0 = Monday` ... `6 = Sunday`** (`pd.Timestamp.dayofweek`)
  - Raises `ValueError` for booleans or out-of-range integers.
- **Behavior**: Returns expected city-wide Brooklyn hourly noise complaint volume (`>= 0.0`). Captures weekend late-night spillover and Sunday night drop-off (e.g., `Wed 12:00 (12, 2) -> 12.21`, `Fri 22:00 (22, 4) -> 70.23`, `Sat 23:00 (23, 5) -> 105.98`, `Sun 00:00 (0, 6) -> 101.31`).

### C. Complaint Trust Score — `score_trust(features)`
- **Import**: `from src.models.trust import score_trust` (backed by `models/trust_model.joblib`, variant `Pooled_NoTop5Noise`, `LogisticRegression(class_weight='balanced')` + `OneHotEncoder(handle_unknown='ignore')`).
- **Signature**: `score_trust(features, model_path: Optional[Path] = None, bundle=None) -> float | np.ndarray`
- **Inputs**:
  - Pass a `dict` (single complaint -> returns scalar `float` in `[0.0, 1.0]`) or a `pd.DataFrame` (batch -> returns 1D `np.ndarray` in `[0.0, 1.0]`).
  - Required raw keys/columns (`7`):
    - `complaint_type` (`str`, e.g. `"Noise - Residential"`, `"Street Condition"`, `"Sewer"`)
    - `descriptor_group` (`str`, 18 categories such as `"music_party"`, `"banging"`, `"pothole"`, `"catch_basin_clogged"`, `"backup"`)
    - `channel` (`str`: `"ONLINE"`, `"MOBILE"`, `"PHONE"`, `"UNKNOWN"`, `"OTHER"`)
    - `location_type` (`str`, e.g. `"Residential Building/House"`, `"Street/Sidewalk"`, `"Street"`, `""`)
    - `hour` (`int` `0..23`)
    - `weekday` (`int` `0..6`, `0 = Monday`)
    - `nearby_prior_24h_200m` (`int >= 0`: strictly prior complaints in `[t - 24h, t)` within `200 m` on planar `(x_m, y_m)`)
  - Unseen categorical values are safely ignored (`all-zero` OHE columns).
  - Example call:
    ```python
    from src.models.trust import score_trust

    p_valid = score_trust({
        "complaint_type": "Street Condition",
        "descriptor_group": "pothole",
        "channel": "ONLINE",
        "location_type": "Street/Sidewalk",
        "hour": 10,
        "weekday": 2,
        "nearby_prior_24h_200m": 1,
    })
    ```
- **Performance & Calibration Caveat**:
  - Strong overall ranking (`ROC AUC = 0.9635`, `PR AUC = 0.9489` on Sep–Oct 2026 holdout): `noise` (`ROC 0.9610`), `road` (`ROC 0.9340`), `flood` (`ROC 0.7364`), `sewer_backup` (`ROC 0.5415`, small positive sample).
  - Trained with `class_weight='balanced'`, so raw probabilities are shifted upward relative to empirical positive rates (mean predicted `0.3978` vs empirical `0.2788` overall). Treat the score as a **0–1 ranking / trust prioritization signal** rather than an uncalibrated probability percentage.

