"""
CityPulse - Part 3, Step 2, Chunk 2.2: Feature Engineering & Local Equirectangular Projection
=============================================================================================
Provides:
- add_time_features(df): Add hour, weekday, month, date, is_midnight_timestamp, time_to_close_h.
- project(lat, lon, lat0, lon0): Convert (lat, lon) to local planar coordinates (x_m, y_m).
- unproject(x_m, y_m, lat0, lon0): Inverse of project, converting (x_m, y_m) back to (lat, lon).
- add_projection(df, lat0, lon0): Add x_m and y_m coordinates to dataframe.
- save_projection_constants(lat0, lon0, path): Persist origin constants to projection.json.
- load_projection_constants(path): Load origin constants from projection.json.

NOTE: lat0 and lon0 are FIXED constants loaded from projection.json
(so new demo points and route queries use the exact same origin).
"""

import json
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROJECTION_JSON = PROCESSED_DIR / "projection.json"
STAGE1_GZ = PROCESSED_DIR / "brooklyn_stage1.csv.gz"
STAGE2_GZ = PROCESSED_DIR / "brooklyn_stage2.csv.gz"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from created_date and closed_date:
    - hour (0 to 23, int)
    - weekday (0=Monday to 6=Sunday, int)
    - month (1 to 12, int)
    - date (calendar day, datetime.date)
    - is_midnight_timestamp (created time exactly 00:00:00, bool)
    - time_to_close_h (hours between created_date and closed_date; NaN when closed_date is NaT)
    """
    df = df.copy()

    # Ensure datetime dtypes
    c_dt = pd.to_datetime(df["created_date"], errors="coerce")
    cl_dt = pd.to_datetime(df["closed_date"], errors="coerce")

    df["hour"] = c_dt.dt.hour.astype(int)
    df["weekday"] = c_dt.dt.day_of_week.astype(int)
    df["month"] = c_dt.dt.month.astype(int)
    df["date"] = c_dt.dt.date
    df["is_midnight_timestamp"] = (
        (c_dt.dt.hour == 0) & (c_dt.dt.minute == 0) & (c_dt.dt.second == 0)
    )

    diff_sec = (cl_dt - c_dt).dt.total_seconds()
    df["time_to_close_h"] = diff_sec / 3600.0

    return df


def save_projection_constants(
    lat0: float, lon0: float, path: Optional[Union[str, Path]] = None
) -> None:
    """
    Save fixed projection origin constants to projection.json.
    lat0 and lon0 are FIXED constants loaded from projection.json
    (so new demo points and route queries use the same origin).
    """
    if path is None:
        path = PROJECTION_JSON
    else:
        path = Path(path)

    payload = {
        "lat0": round(float(lat0), 6),
        "lon0": round(float(lon0), 6),
        "note": "x_m=(lon-lon0)*111320*cos(radians(lat0)); y_m=(lat-lat0)*110540",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_projection_constants(
    path: Optional[Union[str, Path]] = None
) -> Tuple[float, float]:
    """
    Load fixed projection origin constants (lat0, lon0) from projection.json.
    lat0 and lon0 are FIXED constants loaded from projection.json
    (so new demo points and route queries use the same origin).
    """
    if path is None:
        path = PROJECTION_JSON
    else:
        path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["lat0"]), float(data["lon0"])


def project(
    lat: Union[float, np.ndarray, pd.Series],
    lon: Union[float, np.ndarray, pd.Series],
    lat0: float,
    lon0: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Project (latitude, longitude) coordinates into local planar meter coordinates (x_m, y_m).
    Formula:
        x_m = (lon - lon0) * 111320 * cos(radians(lat0))
        y_m = (lat - lat0) * 110540

    Works on scalars, numpy arrays, and pandas Series.
    Uses np.radians(lat0); never passes degrees to np.cos.

    NOTE: lat0 and lon0 are FIXED constants loaded from projection.json
    (so new demo points and route queries use the same origin).
    """
    cos_lat0 = np.cos(np.radians(lat0))
    x_m = (lon - lon0) * 111320.0 * cos_lat0
    y_m = (lat - lat0) * 110540.0
    return x_m, y_m


def unproject(
    x_m: Union[float, np.ndarray, pd.Series],
    y_m: Union[float, np.ndarray, pd.Series],
    lat0: float,
    lon0: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Unproject local planar meter coordinates (x_m, y_m) back to (latitude, longitude).
    Exact inverse of project:
        lon = lon0 + x_m / (111320 * cos(radians(lat0)))
        lat = lat0 + y_m / 110540

    Works on scalars, numpy arrays, and pandas Series.

    NOTE: lat0 and lon0 are FIXED constants loaded from projection.json
    (so new demo points and route queries use the same origin).
    """
    cos_lat0 = np.cos(np.radians(lat0))
    lon = lon0 + x_m / (111320.0 * cos_lat0)
    lat = lat0 + y_m / 110540.0
    return lat, lon


def add_projection(df: pd.DataFrame, lat0: float, lon0: float) -> pd.DataFrame:
    """
    Add projected local meter coordinate columns (x_m, y_m) to the dataframe.

    NOTE: lat0 and lon0 are FIXED constants loaded from projection.json
    (so new demo points and route queries use the same origin).
    """
    df = df.copy()
    x_m, y_m = project(df["latitude"].values, df["longitude"].values, lat0, lon0)
    df["x_m"] = x_m
    df["y_m"] = y_m
    return df


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray, R: float = 6371008.8
) -> np.ndarray:
    """
    Vectorized haversine distance in meters using sphere radius R = 6371008.8 m.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    print("=" * 70)
    print("STEP 2, CHUNK 2.2: BROOKLYN FEATURES & PROJECTION")
    print("=" * 70)

    # 1. Read input dataset
    print(f"\nLoading stage 1 dataset from: {STAGE1_GZ}")
    str_cols = [
        "unique_key",
        "resolution_id",
        "status",
        "channel",
        "location_type",
        "complaint_type",
        "descriptor",
        "group",
        "borough",
        "incident_address",
        "address_norm",
    ]
    df = pd.read_csv(
        STAGE1_GZ,
        dtype={c: str for c in str_cols},
        keep_default_na=False,
    )
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows.")

    # 2. Compute projection constants
    lat0 = round(float(df["latitude"].mean()), 6)
    lon0 = round(float(df["longitude"].mean()), 6)
    print(f"\nComputed projection origin: lat0 = {lat0:.6f}, lon0 = {lon0:.6f}")
    save_projection_constants(lat0, lon0, PROJECTION_JSON)
    print(f"Saved projection constants to: {PROJECTION_JSON}")

    # 3. Add features
    print("\nAdding time features...")
    df = add_time_features(df)

    print("Adding projection coordinates (x_m, y_m)...")
    df = add_projection(df, lat0, lon0)

    # 4. Save stage 2 dataset
    print(f"\nSaving stage 2 dataset to: {STAGE2_GZ}...")
    save_df = df.copy()
    save_df["created_date"] = save_df["created_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    save_df["closed_date"] = save_df["closed_date"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    save_df["date"] = save_df["date"].astype(str)

    # Ensure empty string formatting for string columns
    for c in str_cols:
        save_df[c] = save_df[c].fillna("")

    save_df.to_csv(STAGE2_GZ, index=False, compression="gzip")
    print(f"Saved: {STAGE2_GZ} ({STAGE2_GZ.stat().st_size / (1024**2):.2f} MB, {len(save_df):,} rows)")

    # =======================================================================
    # TASK 3: VERIFICATIONS
    # =======================================================================
    print("\n" + "=" * 70)
    print("[TASK 3] VERIFICATIONS")
    print("=" * 70)

    # 3a. Projection accuracy
    print("\n--- 3a. Projection Accuracy Check ---")
    rng = np.random.default_rng(42)
    lats = df["latitude"].values
    lons = df["longitude"].values
    xs = df["x_m"].values
    ys = df["y_m"].values

    pairs_idx1 = []
    pairs_idx2 = []
    while len(pairs_idx1) < 2000:
        batch = 3000
        i1 = rng.integers(0, n_rows, size=batch)
        i2 = rng.integers(0, n_rows, size=batch)
        valid = i1 != i2
        i1, i2 = i1[valid], i2[valid]
        h_dists = haversine_distance(lats[i1], lons[i1], lats[i2], lons[i2])
        far = h_dists > 500.0
        pairs_idx1.extend(i1[far])
        pairs_idx2.extend(i2[far])

    idx1 = np.array(pairs_idx1[:2000])
    idx2 = np.array(pairs_idx2[:2000])

    h_true = haversine_distance(lats[idx1], lons[idx1], lats[idx2], lons[idx2])
    e_proj = np.sqrt((xs[idx1] - xs[idx2]) ** 2 + (ys[idx1] - ys[idx2]) ** 2)
    rel_err = np.abs(e_proj - h_true) / h_true

    # Wrong formula: cos applied directly to degrees
    x_wrong = (lons - lon0) * 111320.0 * np.cos(lat0)
    e_wrong = np.sqrt((x_wrong[idx1] - x_wrong[idx2]) ** 2 + (ys[idx1] - ys[idx2]) ** 2)
    rel_err_wrong = np.abs(e_wrong - h_true) / h_true

    mean_err = float(rel_err.mean())
    median_err = float(np.median(rel_err))
    p95_err = float(np.percentile(rel_err, 95))
    max_err = float(rel_err.max())
    mean_err_wrong = float(rel_err_wrong.mean())

    print(f"Sampled pairs count                      : {len(h_true)}")
    print(f"Haversine distance range (m)             : {h_true.min():.1f} to {h_true.max():.1f} (mean {h_true.mean():.1f})")
    print(f"Projection Relative Error (Mean)         : {mean_err:.6f} ({mean_err*100:.4f}%)")
    print(f"Projection Relative Error (Median)       : {median_err:.6f} ({median_err*100:.4f}%)")
    print(f"Projection Relative Error (95th %ile)    : {p95_err:.6f} ({p95_err*100:.4f}%)")
    print(f"Projection Relative Error (Max)          : {max_err:.6f} ({max_err*100:.4f}%)")
    print(f"WRONG Formula Relative Error (Mean)      : {mean_err_wrong:.6f} ({mean_err_wrong*100:.4f}%)")

    # 3b. Extent in km
    print("\n--- 3b. Extent in Kilometers ---")
    x_min_km, x_max_km = float(xs.min()) / 1000.0, float(xs.max()) / 1000.0
    y_min_km, y_max_km = float(ys.min()) / 1000.0, float(ys.max()) / 1000.0
    x_range_km = x_max_km - x_min_km
    y_range_km = y_max_km - y_min_km

    print(f"x_m (East-West width)  : min = {x_min_km:+.3f} km | max = {x_max_km:+.3f} km | range = {x_range_km:.3f} km")
    print(f"y_m (North-South height): min = {y_min_km:+.3f} km | max = {y_max_km:+.3f} km | range = {y_range_km:.3f} km")
    if 13.0 <= x_range_km <= 18.0 and 16.0 <= y_range_km <= 21.0:
        print("Extent assessment: PASS (Brooklyn is roughly 15.6 km wide and 18.5 km tall, as expected).")
    else:
        print("Extent assessment: WARNING (extent deviates noticeably from ~15 km wide by ~18 km tall).")

    # 3c. Round-trip accuracy
    print("\n--- 3c. Round-trip Accuracy ---")
    rec_lat, rec_lon = unproject(xs, ys, lat0, lon0)
    diff_lat = float(np.max(np.abs(rec_lat - lats)))
    diff_lon = float(np.max(np.abs(rec_lon - lons)))
    max_round_trip = max(diff_lat, diff_lon)
    print(f"Max absolute lat difference: {diff_lat:.2e} degrees")
    print(f"Max absolute lon difference: {diff_lon:.2e} degrees")
    print(f"Max absolute difference    : {max_round_trip:.2e} degrees (threshold < 1e-9: {'PASS' if max_round_trip < 1e-9 else 'FAIL'})")

    # 3d. Feature sanity
    print("\n--- 3d. Feature Sanity Checks ---")
    print(f"Hour min/max      : min={df['hour'].min()}, max={df['hour'].max()} (valid 0..23)")
    print(f"Weekday min/max   : min={df['weekday'].min()}, max={df['weekday'].max()} (valid 0..6)")

    print("\nMonth counts (expected exactly 12 months, 2025-11 to 2026-10):")
    month_counts = df["month"].value_counts().sort_index()
    for m_val, cnt in month_counts.items():
        print(f"  Month {m_val:02d}: {cnt:,} rows")

    print("\nWeekday counts (0=Mon ... 6=Sun):")
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = df["weekday"].value_counts().sort_index()
    for w_val, cnt in weekday_counts.items():
        print(f"  Weekday {w_val} ({weekday_names[w_val]}): {cnt:,} rows")

    n_midnight = int(df["is_midnight_timestamp"].sum())
    pct_midnight = (n_midnight / n_rows) * 100.0
    print(f"\nMidnight timestamps (00:00:00): {n_midnight:,} rows ({pct_midnight:.4f}% overall)")

    n_missing_close = int(df["time_to_close_h"].isna().sum())
    n_neg_close = int((df["time_to_close_h"] < 0).sum())
    print(f"Missing time_to_close_h      : {n_missing_close:,} rows (matches NaT closed_date)")
    print(f"Negative time_to_close_h     : {n_neg_close} rows (must be 0)")

    print("\ntime_to_close_h quantiles (hours):")
    quantiles_to_check = [0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]
    q_vals = df["time_to_close_h"].quantile(quantiles_to_check)
    for q_prob, q_val in zip(quantiles_to_check, q_vals):
        print(f"  {int(q_prob*100):3d}%: {q_val:10.4f} h")

    # 3e. Read-back test
    print("\n--- 3e. Read-back Test ---")
    df_readback = pd.read_csv(
        STAGE2_GZ,
        dtype={c: str for c in str_cols},
        keep_default_na=False,
    )
    df_readback["x_m"] = df_readback["x_m"].astype(float)
    df_readback["y_m"] = df_readback["y_m"].astype(float)

    rb_n_rows = len(df_readback)
    rb_unique = df_readback["unique_key"].is_unique
    rb_empty_res = int((df_readback["resolution_id"] == "").sum())
    rb_xm_nonan = df_readback["x_m"].notna().all()
    rb_ym_nonan = df_readback["y_m"].notna().all()

    print(f"Read-back row count == 236,507      : {'PASS' if rb_n_rows == 236507 else 'FAIL'} ({rb_n_rows:,})")
    print(f"Read-back unique_key is unique      : {'PASS' if rb_unique else 'FAIL'}")
    print(f"Read-back empty resolution_id == 3108: {'PASS' if rb_empty_res == 3108 else 'FAIL'} ({rb_empty_res:,})")
    print(f"Read-back x_m and y_m have no NaN   : {'PASS' if (rb_xm_nonan and rb_ym_nonan) else 'FAIL'}")

    print("\n" + "=" * 70)
    print("STEP 2, CHUNK 2.2 EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

