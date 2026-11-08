"""
CityPulse - Regression Models for Flood and Noise Forecasting (Step 5 / Lab 2)
==============================================================================
Provides feature engineering, temporal validation splitting, training, evaluation,
and production inference routines for:
1. Daily Brooklyn Flood Complaint Regression (Precipitation & Antecedent Lags)
   - Models city-wide Brooklyn flood counts as a function of same-day and antecedent rain.
   - Model describes a single storm day after dry antecedent conditions at the city-wide level,
     not spatial zones.
2. Hourly Brooklyn Noise Complaint Regression (Temporal One-Hot Profile)
   - Models hourly Brooklyn noise complaints as a function of hour-of-day and day-of-week.
   - Deviation from project guide: Borough is NOT included as a feature because the cleaned
     dataset contains Brooklyn exclusively.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cache for lazy loading models
_FLOOD_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_NOISE_MODEL_BUNDLE: Optional[Dict[str, Any]] = None


def build_flood_features(panel_brooklyn: pd.DataFrame) -> pd.DataFrame:
    """
    Construct same-day and antecedent lag features from daily precipitation.
    
    Columns produced:
    - rain_t: precip_mm on day t (same day)
    - rain_t1: precip_mm on day t-1
    - rain_t2: precip_mm on day t-2
    - rain_sum_t1_t3: sum of rain on days t-1, t-2, and t-3 (excludes today).
    
    Note on rain_sum_t1_t3:
    The 3-day antecedent sum strictly excludes today's rain (days t-1..t-3). Including
    today's rain would introduce severe collinearity with rain_t, rain_t1, and rain_t2,
    destabilizing OLS parameter estimates.
    
    Drops the first 3 days because they have NaN lags and prints the dropped count.
    """
    df = panel_brooklyn.copy()
    if "day" in df.columns:
        df = df.sort_values(by="day").reset_index(drop=True)
    elif "date" in df.columns:
        df = df.sort_values(by="date").reset_index(drop=True)

    precip_col = "precip_mm"
    df["rain_t"] = df[precip_col]
    df["rain_t1"] = df[precip_col].shift(1)
    df["rain_t2"] = df[precip_col].shift(2)
    # Exclude today: shift(1) rolling sum of 3 days
    df["rain_sum_t1_t3"] = df[precip_col].shift(1).rolling(3, min_periods=3).sum()

    nan_mask = df[["rain_t", "rain_t1", "rain_t2", "rain_sum_t1_t3"]].isna().any(axis=1)
    n_dropped = int(nan_mask.sum())
    print(f"build_flood_features: Dropped {n_dropped} initial days with incomplete lag history.")

    df_clean = df[~nan_mask].copy().reset_index(drop=True)
    return df_clean


def rolling_origin_splits(
    days: Union[pd.Series, np.ndarray, List[str]],
    test_years: List[int],
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate rolling-origin (expanding window) temporal splits for specified test years.
    
    For each test year Y:
    - train: all days strictly before Y-01-01
    - test: days falling in year Y (up to Y-12-31 or end of series)
    Never shuffles.
    
    Returns list of (train_indices, test_indices, year).
    """
    days_arr = pd.Series(days).astype(str).values
    splits = []
    for y in test_years:
        train_idx = np.where(days_arr < f"{y}-01-01")[0]
        test_idx = np.where((days_arr >= f"{y}-01-01") & (days_arr <= f"{y}-12-31"))[0]
        splits.append((train_idx, test_idx, y))
    return splits


def chrono_split(
    days: Union[pd.Series, np.ndarray, List[str]],
    frac: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronological train/test split.
    
    When frac=0.8 on the 6148-day panel (or aligned to test start 2023-06-20),
    evaluates the final 1230 days as test.
    Never shuffles.
    """
    days_arr = pd.Series(days).astype(str).values
    # If aligned to notebook 07 fixed test cutoff 2023-06-20:
    if "2023-06-20" in days_arr:
        train_idx = np.where(days_arr < "2023-06-20")[0]
        test_idx = np.where(days_arr >= "2023-06-20")[0]
    else:
        n_train = int(len(days_arr) * frac)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, len(days_arr))
    return train_idx, test_idx


def fit_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target: str = "count",
) -> Dict[str, Any]:
    """
    Fit LinearRegression on X_train, y_train and evaluate on X_test, y_test.
    
    target in {"count", "log1p"}:
    - For target="log1p", fits on log1p(y_train), back-transforms test predictions with expm1,
      and clips at 0. No smearing correction is applied, as specified in project contract.
      
    Returns dictionary with:
    - mse, rmse, mae, r2 (count scale)
    - r2_log (log1p scale)
    - top2_share: share (%) of test set's total squared deviation from mean contributed by 2 largest days
    - r2_without_top2: r2 after removing the 2 largest-|y - mean(y_test)| test days
    - coefficients, intercept
    - n_train, n_test
    """
    if target not in {"count", "log1p"}:
        raise ValueError(f"target must be 'count' or 'log1p', got '{target}'")

    lr = LinearRegression(fit_intercept=True)
    if target == "count":
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, None)
    else:
        lr.fit(X_train, np.log1p(y_train))
        y_pred_log = lr.predict(X_test)
        y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)

    # Count scale metrics
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Log1p scale metric
    r2_log = float(r2_score(np.log1p(y_test), np.log1p(y_pred)))

    # top2_share: share of total squared deviation from mean contributed by 2 largest days
    y_test_arr = np.asarray(y_test, dtype=float)
    y_test_mean = float(np.mean(y_test_arr))
    sq_dev = (y_test_arr - y_test_mean) ** 2
    tot_sq_dev = float(np.sum(sq_dev))
    if tot_sq_dev > 0:
        top2_sq_dev = float(np.sort(sq_dev)[-2:].sum())
        top2_share = (top2_sq_dev / tot_sq_dev) * 100.0
    else:
        top2_share = 0.0

    # r2_without_top2: remove the 2 largest-|y - mean(y_test)| test days
    abs_dev = np.abs(y_test_arr - y_test_mean)
    if len(y_test_arr) > 2:
        top2_indices = np.argsort(abs_dev)[-2:]
        mask_keep = np.ones(len(y_test_arr), dtype=bool)
        mask_keep[top2_indices] = False
        r2_without_top2 = float(r2_score(y_test_arr[mask_keep], y_pred[mask_keep]))
    else:
        r2_without_top2 = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_log": r2_log,
        "top2_share": top2_share,
        "r2_without_top2": r2_without_top2,
        "coefficients": lr.coef_,
        "intercept": float(lr.intercept_),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model": lr,
    }


def baseline_mean(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Baseline predictor returning the training mean for all test days.
    Evaluates identical metrics to fit_eval.
    """
    train_mean = float(np.mean(y_train))
    y_pred = np.full_like(y_test, fill_value=train_mean, dtype=float)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    r2_log = float(r2_score(np.log1p(y_test), np.log1p(y_pred)))

    y_test_arr = np.asarray(y_test, dtype=float)
    y_test_mean = float(np.mean(y_test_arr))
    sq_dev = (y_test_arr - y_test_mean) ** 2
    tot_sq_dev = float(np.sum(sq_dev))
    top2_share = (float(np.sort(sq_dev)[-2:].sum()) / tot_sq_dev * 100.0) if tot_sq_dev > 0 else 0.0

    abs_dev = np.abs(y_test_arr - y_test_mean)
    if len(y_test_arr) > 2:
        top2_indices = np.argsort(abs_dev)[-2:]
        mask_keep = np.ones(len(y_test_arr), dtype=bool)
        mask_keep[top2_indices] = False
        r2_without_top2 = float(r2_score(y_test_arr[mask_keep], y_pred[mask_keep]))
    else:
        r2_without_top2 = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_log": r2_log,
        "top2_share": top2_share,
        "r2_without_top2": r2_without_top2,
        "coefficients": np.array([]),
        "intercept": train_mean,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def build_noise_hourly(
    df: pd.DataFrame,
    start_date: str = "2025-11-01",
    end_date: str = "2026-10-31",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct hourly noise complaint count grid and one-hot features.
    
    Filters complaints_clean.csv to group == 'noise'.
    Excludes rows with is_midnight_timestamp == True (placeholder hour).
    Reindexes to the complete 365 x 24 grid (8760 hours) with zero fill.
    
    Features created:
    - hour one-hot (23 dummy columns, hour_1..hour_23, drop_first=True, ref=hour_0)
    - weekday one-hot (6 dummy columns, weekday_1..weekday_6, drop_first=True, ref=weekday_0)
    
    Weekday encoding:
    Uses ISO convention matching complaints_clean.csv:
    0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday.
    
    Deviation note:
    Borough is NOT used as a feature because complaints_clean.csv is single-borough (Brooklyn).
    """
    noise_df = df[df["group"] == "noise"].copy()
    n_total_noise = len(noise_df)

    # Exclude midnight timestamp placeholders
    is_midnight = noise_df["is_midnight_timestamp"].astype(bool)
    n_excluded = int(is_midnight.sum())
    share_excluded = float(n_excluded / n_total_noise) if n_total_noise > 0 else 0.0

    print(f"build_noise_hourly: Excluded {n_excluded:,} midnight placeholder rows ({share_excluded:.4%})")
    print("Weekday encoding in complaints_clean.csv: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday.")

    valid_noise = noise_df[~is_midnight].copy()

    # Aggregate counts per (date, hour)
    counts = valid_noise.groupby(["date", "hour"]).size().reset_index(name="count")

    # Complete 365 x 24 date-hour grid
    all_dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    all_hours = list(range(24))
    grid = pd.MultiIndex.from_product([all_dates, all_hours], names=["date", "hour"]).to_frame().reset_index(drop=True)

    merged = grid.merge(counts, on=["date", "hour"], how="left")
    merged["count"] = merged["count"].fillna(0).astype(int)

    # Add weekday: pd.to_datetime dt.day_of_week (0=Monday, ..., 6=Sunday)
    merged["dt"] = pd.to_datetime(merged["date"])
    merged["weekday"] = merged["dt"].dt.day_of_week

    zero_count_hours = int((merged["count"] == 0).sum())
    print(f"build_noise_hourly: Reindexed to {len(merged):,} hours. Zero-count hours: {zero_count_hours:,}")

    # One-hot features with drop_first=True
    # Hours: hour_1 .. hour_23
    for h in range(1, 24):
        merged[f"hour_{h}"] = (merged["hour"] == h).astype(float)

    # Weekdays: weekday_1 .. weekday_6
    for w in range(1, 7):
        merged[f"weekday_{w}"] = (merged["weekday"] == w).astype(float)

    meta = {
        "n_total_noise": n_total_noise,
        "n_excluded_midnight": n_excluded,
        "share_excluded": share_excluded,
        "zero_count_hours": zero_count_hours,
        "total_hours": len(merged),
        "hour_features": [f"hour_{h}" for h in range(1, 24)],
        "weekday_features": [f"weekday_{w}" for w in range(1, 7)],
    }
    return merged, meta


def load_flood_model(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Lazy load the frozen flood regression model bundle."""
    global _FLOOD_MODEL_BUNDLE
    if _FLOOD_MODEL_BUNDLE is None:
        if model_path is None:
            # Look up relative to current workspace or repo root
            root = Path(__file__).resolve().parents[2]
            model_path = root / "models/flood_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Flood model not found at {model_path}. Train and save it first.")
        _FLOOD_MODEL_BUNDLE = joblib.load(model_path)
    return _FLOOD_MODEL_BUNDLE


def load_noise_model(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Lazy load the frozen noise regression model bundle."""
    global _NOISE_MODEL_BUNDLE
    if _NOISE_MODEL_BUNDLE is None:
        if model_path is None:
            root = Path(__file__).resolve().parents[2]
            model_path = root / "models/noise_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Noise model not found at {model_path}. Train and save it first.")
        _NOISE_MODEL_BUNDLE = joblib.load(model_path)
    return _NOISE_MODEL_BUNDLE


def predict_flood_risk(rain_mm: float, model_path: Optional[Path] = None) -> float:
    """
    Predict expected Brooklyn city-wide flood complaints for a single storm day
    with dry antecedent conditions (rain_t = rain_mm, lags = 0).
    
    Parameters
    ----------
    rain_mm : float
        Total rainfall in millimetres for today.
    model_path : Path, optional
        Path to flood_model.joblib. If None, loads from default models/ dir.
        
    Returns
    -------
    float
        Expected complaint count >= 0.
        
    Raises
    ------
    ValueError
        If rain_mm is negative, NaN, or infinite.
    """
    if np.isnan(rain_mm) or np.isinf(rain_mm) or rain_mm < 0:
        raise ValueError(f"rain_mm must be a non-negative finite float, got {rain_mm}")

    bundle = load_flood_model(model_path)
    model = bundle["model"]
    target = bundle.get("target", "count")

    if hasattr(model, "feature_names_in_"):
        feature_names = bundle.get("feature_names", list(model.feature_names_in_))
        X = pd.DataFrame([[float(rain_mm), 0.0, 0.0, 0.0]], columns=feature_names)
    else:
        X = np.array([[float(rain_mm), 0.0, 0.0, 0.0]], dtype=float)

    if target == "count":
        pred = model.predict(X)[0]
    elif target == "log1p":
        pred_log = model.predict(X)[0]
        pred = np.expm1(pred_log)
    else:
        pred = model.predict(X)[0]

    return float(max(0.0, pred))


def predict_noise(hour: int, weekday: int, model_path: Optional[Path] = None) -> float:
    """
    Predict expected Brooklyn noise complaints for a given hour and weekday.
    
    Parameters
    ----------
    hour : int
        Hour of day (0..23).
    weekday : int
        Weekday index (0..6 where 0=Monday, 6=Sunday).
    model_path : Path, optional
        Path to noise_model.joblib.
        
    Returns
    -------
    float
        Expected complaint count >= 0.
        
    Raises
    ------
    ValueError
        If hour or weekday is outside valid ranges.
    """
    if not (isinstance(hour, (int, np.integer)) and 0 <= hour <= 23):
        raise ValueError(f"hour must be an integer between 0 and 23, got {hour}")
    if not (isinstance(weekday, (int, np.integer)) and 0 <= weekday <= 6):
        raise ValueError(f"weekday must be an integer between 0 and 6 (0=Monday), got {weekday}")

    bundle = load_noise_model(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Construct one-hot vector matching feature_names
    feat_dict = {name: 0.0 for name in feature_names}
    if hour > 0 and f"hour_{hour}" in feat_dict:
        feat_dict[f"hour_{hour}"] = 1.0
    if weekday > 0 and f"weekday_{weekday}" in feat_dict:
        feat_dict[f"weekday_{weekday}"] = 1.0

    if hasattr(model, "feature_names_in_"):
        X = pd.DataFrame([[feat_dict[col] for col in feature_names]], columns=feature_names)
    else:
        X = np.array([[feat_dict[col] for col in feature_names]], dtype=float)
    pred = model.predict(X)[0]
    return float(max(0.0, pred))
