"""
CityPulse - Citizen Complaint Trust Scoring Model (Step 6.2 / Lab 3)
=====================================================================
Provides standardized trust-label assignment, leak-free causal feature
engineering, balanced logistic regression training, evaluation helpers,
and production inference routines (`score_trust`).

Features constructed by `build_trust_features(df)`:
- One-hot encoded submission-time categorical fields:
  - `complaint_type`
  - `descriptor_group`
  - `channel`
  - `location_type`
  - `hour` (0..23)
  - `weekday` (0=Monday..6=Sunday)
- Strictly causal spatio-temporal activity count:
  - `nearby_prior_24h_200m`: count of prior complaints in [t - 24h, t)
    within 200 m Euclidean distance on local planar `(x_m, y_m)` coordinates.
    Strictly excludes self and simultaneous complaints (`t_prior < t_current`).

Labeling rules (Step 6.1c standardized order):
  0. Gate: `unclosed == 1` (`closed_date == ""` or `status != "Closed"`) or empty `resolution_id` -> `'excluded'`
  1. `EXCLUDED` substrings (27) -> `'excluded'`
  2. `VALID` substrings (19) -> `'valid'` (target = 1)
  3. `INVALID` substrings (15) -> `'invalid'` (target = 0)
  4. Everything else -> `'unmatched'` (dropped with `'excluded'` during supervised training)
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

TRUST_CATEGORICAL_COLS: List[str] = [
    "complaint_type",
    "descriptor_group",
    "channel",
    "location_type",
    "hour",
    "weekday",
]
TRUST_NUMERIC_COLS: List[str] = ["nearby_prior_24h_200m"]

# Step 6.1c standardized keyword rules (applied in strict priority order: excluded > valid > invalid)
EXCLUDED_KEYWORDS: List[str] = [
    "duplicate",
    "previously reported",
    "re-inspected",
    "could be closed",
    "referred",
    "additional information",
    "requires contact",
    "insufficient",
    "undeliverable",
    "incorrect",
    "jurisdiction",
    "will inspect",
    "capital project",
    "not accessible",
    "didn't include",
    "complainant's request",
    "observation",
    "regular schedule",
    "further information",
    "assigned to a unit",
    "home or business owner",
    "re-assignment",
    "not available online",
    "sufficient information",
    "report was prepared",
    "requires an appointment",
    "specific tow is required",
]

VALID_KEYWORDS: List[str] = [
    "corrected",
    "repaired",
    "took action",
    "summons",
    "arrest",
    "obstructing",
    "cleaned",
    "warning letter",
    "surcharge conditions",
    "resolved",
    "flushing",
    "made a repair",
    "hydrant",
    "defect",
    "problem was fixed",
    "repairs were made",
    "repair work",
    "correct the condition",
    "issued a notice",
]

INVALID_KEYWORDS: List[str] = [
    "no evidence",
    "not necessary",
    "unable to gain",
    "didn't observe",
    "no criminal violation",
    "meets its standards",
    "no sewer",
    "could not find",
    "no indication",
    "were gone",
    "not found",
    "overtaxed",
    "no encampment",
    "did not find",
    "functioning properly",
]

# Cache loaded trust model bundles keyed by resolved file path
_TRUST_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def classify_resolution_text(text: Any) -> str:
    """
    Classify a 311 resolution description using Step 6.1c case-insensitive
    substring rules in strict priority order:
      1. EXCLUDED -> 'excluded'
      2. VALID    -> 'valid'
      3. INVALID  -> 'invalid'
      4. else     -> 'unmatched'
    """
    if not isinstance(text, str) or not text.strip():
        return "unmatched"
    t_lower = text.lower()
    if any(kw in t_lower for kw in EXCLUDED_KEYWORDS):
        return "excluded"
    if any(kw in t_lower for kw in VALID_KEYWORDS):
        return "valid"
    if any(kw in t_lower for kw in INVALID_KEYWORDS):
        return "invalid"
    return "unmatched"


def assign_trust_labels(
    df: pd.DataFrame,
    lookup_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Assign trust class labels ('valid', 'invalid', 'excluded', 'unmatched')
    following Step 6.1c exact gate and keyword priority rules.

    Gate 0: rows with unclosed == 1 (missing closed_date or status != 'Closed')
    or missing resolution_id are always labeled 'excluded'.
    """
    if lookup_df is None:
        root = Path(__file__).resolve().parents[2]
        lookup_df = pd.read_csv(
            root / "data/processed/resolution_lookup.csv",
            dtype={"resolution_id": str},
            keep_default_na=False,
        )
    else:
        lookup_df = lookup_df.copy()
        lookup_df["resolution_id"] = lookup_df["resolution_id"].fillna("").astype(str)

    res_class_map = {
        rid: classify_resolution_text(txt)
        for rid, txt in zip(lookup_df["resolution_id"], lookup_df["text"])
    }

    res_ids = (
        df["resolution_id"].fillna("").astype(str)
        if "resolution_id" in df.columns
        else pd.Series("", index=df.index)
    )
    labels = res_ids.map(res_class_map).fillna("unmatched")

    closed_vals = (
        df["closed_date"].fillna("").astype(str)
        if "closed_date" in df.columns
        else pd.Series("", index=df.index)
    )
    status_vals = (
        df["status"].fillna("").astype(str)
        if "status" in df.columns
        else pd.Series("Closed", index=df.index)
    )
    unclosed_mask = (closed_vals == "") | (status_vals != "Closed")
    empty_res_mask = res_ids == ""

    labels = labels.where(~(unclosed_mask | empty_res_mask), "excluded")
    return pd.Series(labels.values, index=df.index, name="trust_label")


def _extract_time_series(df: pd.DataFrame) -> pd.Series:
    """Extract parsed datetime series from `timestamp` or `created_date`."""
    for col in ("timestamp", "created_date"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.isna().any():
                raise ValueError(f"Column '{col}' contains unparseable or missing timestamps.")
            return ts
    raise ValueError("DataFrame must include a 'timestamp' or 'created_date' column.")


def _extract_xy_meters(df: pd.DataFrame) -> np.ndarray:
    """Extract planar (x_m, y_m) array in meters, projecting from lat/lon if needed."""
    if "x_m" in df.columns and "y_m" in df.columns:
        xy = df[["x_m", "y_m"]].astype(float).values
        if np.isnan(xy).any():
            raise ValueError("Columns 'x_m' / 'y_m' contain NaN values.")
        return xy

    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    if lat_col is not None and lon_col is not None:
        from src.preprocessing.features import load_projection_constants, project

        lat0, lon0 = load_projection_constants()
        x_m, y_m = project(df[lat_col].astype(float).values, df[lon_col].astype(float).values, lat0, lon0)
        return np.column_stack([x_m, y_m])

    raise ValueError("DataFrame must provide planar ('x_m', 'y_m') or geographic ('lat', 'lon') coordinates.")


def compute_nearby_prior_24h_200m(
    df: pd.DataFrame,
    radius_m: float = 200.0,
    window_hours: float = 24.0,
) -> np.ndarray:
    """
    Compute strictly causal spatio-temporal complaint activity counts:
    for each row with timestamp `t_current` and location `(x_m, y_m)`, counts rows
    within Euclidean distance <= `radius_m` (200 m) whose timestamp `t_prior` satisfies:
        t_current - window_hours <= t_prior < t_current

    Strictly earlier inequality (`<`) excludes self and any simultaneous rows.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=int)

    ts = _extract_time_series(df)
    xy = _extract_xy_meters(df)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]")
    window_ns = np.timedelta64(int(round(window_hours * 3600 * 1_000_000_000)), "ns")

    if n == 1:
        return np.zeros(1, dtype=int)

    day_dates = ts.dt.date.to_numpy()
    orig_indices = np.arange(n)

    work = pd.DataFrame({
        "_day": day_dates,
        "_pos": orig_indices,
    })
    sorted_days = sorted(work["_day"].unique())
    day_groups = {d: grp["_pos"].to_numpy() for d, grp in work.groupby("_day", sort=False)}
    lookback_days = max(1, int(np.ceil(window_hours / 24.0)))

    counts = np.zeros(n, dtype=int)
    for idx_d, d in enumerate(sorted_days):
        cur_pos = day_groups[d]
        pool_parts = [cur_pos]
        # Scan backwards through prior days within lookback window
        j = idx_d - 1
        while j >= 0 and (d - sorted_days[j]).days <= lookback_days:
            pool_parts.append(day_groups[sorted_days[j]])
            j -= 1
        pool_pos = np.concatenate(pool_parts) if len(pool_parts) > 1 else cur_pos

        tree = cKDTree(xy[pool_pos])
        pool_times = ts_ns[pool_pos]
        cur_times = ts_ns[cur_pos]

        neighbors_list = tree.query_ball_point(xy[cur_pos], r=float(radius_m))
        for i, nbr_idxs in enumerate(neighbors_list):
            t_cur = cur_times[i]
            t_low = t_cur - window_ns
            n_times = pool_times[nbr_idxs]
            counts[cur_pos[i]] = int(np.sum((n_times >= t_low) & (n_times < t_cur)))

    return counts


def _normalize_categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize required submission-time categorical columns to canonical string representations."""
    out = pd.DataFrame(index=df.index)

    # Hour & weekday can come directly from columns or be derived from timestamp/created_date
    derived_ts: Optional[pd.Series] = None
    if "hour" not in df.columns or "weekday" not in df.columns:
        if "timestamp" in df.columns or "created_date" in df.columns:
            derived_ts = _extract_time_series(df)

    for col in ("complaint_type", "descriptor_group", "channel", "location_type"):
        if col not in df.columns:
            raise ValueError(f"Missing required categorical column '{col}' for trust feature engineering.")
        out[col] = df[col].fillna("").astype(str)

    if "hour" in df.columns:
        hr_vals = df["hour"]
    elif derived_ts is not None:
        hr_vals = derived_ts.dt.hour
    else:
        raise ValueError("Missing required column 'hour' (and no timestamp/created_date column to derive it).")

    if "weekday" in df.columns:
        wd_vals = df["weekday"]
    elif derived_ts is not None:
        wd_vals = derived_ts.dt.dayofweek
    else:
        raise ValueError("Missing required column 'weekday' (and no timestamp/created_date column to derive it).")

    # Normalize integer-like hours/weekdays ("0", "23", etc.) consistently regardless of int/float/str dtype
    out["hour"] = pd.to_numeric(hr_vals, errors="raise").astype(int).astype(str)
    out["weekday"] = pd.to_numeric(wd_vals, errors="raise").astype(int).astype(str)
    return out[TRUST_CATEGORICAL_COLS]


def build_trust_features(
    df: Union[pd.DataFrame, Dict[str, Any]],
    encoder: Optional[OneHotEncoder] = None,
    recompute_nearby: bool = False,
    return_encoder: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, OneHotEncoder]]:
    """
    Build submission-time feature matrix for the complaint trust scorer:
    - One-hot encoded `complaint_type`, `descriptor_group`, `channel`,
      `location_type`, `hour`, `weekday` (using `OneHotEncoder(handle_unknown='ignore')`).
    - Numeric `nearby_prior_24h_200m` computed strictly backwards (`t_prior < t_current`)
      unless already precomputed in `df` (pass `recompute_nearby=True` to force recomputation).

    To enforce zero train/test leakage (Rule 10):
      X_train, enc = build_trust_features(train_df, return_encoder=True)
      X_test       = build_trust_features(test_df, encoder=enc)
    """
    if isinstance(df, dict):
        work = pd.DataFrame([df])
    elif isinstance(df, pd.DataFrame):
        work = df
    else:
        raise TypeError(f"Expected pandas DataFrame or dict, got {type(df).__name__}")

    cat_df = _normalize_categorical_frame(work)

    if not recompute_nearby and "nearby_prior_24h_200m" in work.columns:
        nearby_arr = pd.to_numeric(work["nearby_prior_24h_200m"], errors="raise").astype(float).to_numpy()
    else:
        nearby_arr = compute_nearby_prior_24h_200m(work).astype(float)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = encoder.fit_transform(cat_df)
    else:
        X_cat = encoder.transform(cat_df)

    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()

    cat_feature_names = list(encoder.get_feature_names_out(TRUST_CATEGORICAL_COLS))
    all_feature_names = cat_feature_names + TRUST_NUMERIC_COLS

    X_mat = np.column_stack([np.asarray(X_cat, dtype=float), nearby_arr.reshape(-1, 1)])
    X_df = pd.DataFrame(X_mat, index=work.index, columns=all_feature_names)
    X_df.attrs["encoder"] = encoder
    X_df.attrs["feature_names"] = all_feature_names

    if return_encoder:
        return X_df, encoder
    return X_df


def _filter_valid_invalid_targets(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, Sequence[Any]],
) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """
    Retain ONLY valid (1 / 'valid') and invalid (0 / 'invalid') rows.
    Drops 'excluded', 'unmatched', -1, NaN, and any non-binary labels.
    """
    y_seq = list(y) if not isinstance(y, (np.ndarray, pd.Series)) else y
    n_y = len(y_seq)
    if n_y != X.shape[0]:
        raise ValueError(f"Length mismatch between X ({X.shape[0]}) and y ({n_y}).")

    keep_mask = np.zeros(n_y, dtype=bool)
    y_bin = np.zeros(n_y, dtype=int)

    for idx, val in enumerate(y_seq):
        if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            continue
        if isinstance(val, (bool, np.bool_)):
            keep_mask[idx] = True
            y_bin[idx] = int(val)
        elif isinstance(val, (int, np.integer, float, np.floating)):
            if val == 1 or val == 1.0:
                keep_mask[idx] = True
                y_bin[idx] = 1
            elif val == 0 or val == 0.0:
                keep_mask[idx] = True
                y_bin[idx] = 0
        elif isinstance(val, str):
            s = val.strip().lower()
            if s in {"valid", "1", "true"}:
                keep_mask[idx] = True
                y_bin[idx] = 1
            elif s in {"invalid", "0", "false"}:
                keep_mask[idx] = True
                y_bin[idx] = 0

    n_kept = int(keep_mask.sum())
    if n_kept < 2:
        raise ValueError("Need at least 2 valid/invalid rows after dropping excluded labels.")

    y_clean = y_bin[keep_mask]
    if len(np.unique(y_clean)) < 2:
        raise ValueError("Both valid (1) and invalid (0) classes must be present after dropping excluded labels.")

    if isinstance(X, pd.DataFrame):
        X_clean = X.iloc[keep_mask]
    else:
        X_clean = X[keep_mask]

    return X_clean, y_clean


def train_trust_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, Sequence[Any]],
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train `LogisticRegression(class_weight='balanced')` strictly on binary
    `valid` (1) vs `invalid` (0) observations, automatically dropping any
    `excluded` / `unmatched` / `-1` / NaN labels.
    """
    X_clean, y_clean = _filter_valid_invalid_targets(X, y)
    clf = LogisticRegression(
        class_weight="balanced",
        C=float(C),
        max_iter=int(max_iter),
        random_state=random_state,
    )
    clf.fit(X_clean, y_clean)
    return clf


def save_trust_model(
    model: LogisticRegression,
    encoder: OneHotEncoder,
    feature_names: List[str],
    model_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist trained trust model bundle (classifier + OHE encoder + feature schema) via joblib."""
    if model_path is None:
        root = Path(__file__).resolve().parents[2]
        model_path = root / "models/trust_model.joblib"
    else:
        model_path = Path(model_path)

    bundle: Dict[str, Any] = {
        "model": model,
        "encoder": encoder,
        "feature_names": list(feature_names),
        "categorical_cols": list(TRUST_CATEGORICAL_COLS),
        "numeric_cols": list(TRUST_NUMERIC_COLS),
        "sklearn_version": sklearn.__version__,
        "created_date": datetime.now().strftime("%Y-%m-%d"),
    }
    if metadata:
        bundle.update(metadata)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    _TRUST_MODEL_CACHE.pop(str(model_path.resolve()), None)
    return model_path


def load_trust_model(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Lazy load the frozen trust scoring model bundle keyed by resolved path."""
    if model_path is None:
        root = Path(__file__).resolve().parents[2]
        model_path = root / "models/trust_model.joblib"
    else:
        model_path = Path(model_path)

    cache_key = str(model_path.resolve())
    if cache_key not in _TRUST_MODEL_CACHE:
        if not model_path.exists():
            raise FileNotFoundError(f"Trust model not found at {model_path}. Train and save it first.")
        _TRUST_MODEL_CACHE[cache_key] = joblib.load(model_path)
    return _TRUST_MODEL_CACHE[cache_key]


def score_trust(
    features: Union[pd.DataFrame, Dict[str, Any], pd.Series, np.ndarray, Sequence[float]],
    model_path: Optional[Union[str, Path]] = None,
    bundle: Optional[Dict[str, Any]] = None,
) -> Union[float, np.ndarray]:
    """
    Score probability P(valid | features) in [0.0, 1.0].

    Accepts:
    - Raw complaint `dict` or `pd.Series` (returns scalar `float` in [0, 1])
    - Raw complaint `pd.DataFrame` containing submission-time columns + `nearby_prior_24h_200m`
      (or `timestamp` + coordinates), automatically encoded using the bundle's fitted encoder
    - Pre-encoded feature matrix (`pd.DataFrame`, 2D `np.ndarray`, or sparse matrix)
      or 1D numeric vector matching `bundle['feature_names']` (returns `float` for 1D, `np.ndarray` for 2D).
    """
    if bundle is None:
        bundle = load_trust_model(model_path=model_path)

    model: LogisticRegression = bundle["model"]
    encoder: Optional[OneHotEncoder] = bundle.get("encoder")
    feature_names: List[str] = bundle.get("feature_names") or list(getattr(model, "feature_names_in_", []))

    scalar_output = False

    if isinstance(features, dict):
        scalar_output = True
        if feature_names and set(feature_names).issubset(features.keys()):
            X_in = pd.DataFrame([[float(features[c]) for c in feature_names]], columns=feature_names)
        else:
            row_df = pd.DataFrame([features])
            X_in = build_trust_features(row_df, encoder=encoder)
    elif isinstance(features, pd.Series):
        return score_trust(features.to_dict(), model_path=model_path, bundle=bundle)
    elif isinstance(features, pd.DataFrame):
        if len(features) == 0:
            return np.zeros(0, dtype=float)
        if feature_names and set(feature_names).issubset(features.columns):
            X_in = features[feature_names].astype(float)
        else:
            X_in = build_trust_features(features, encoder=encoder)
            if feature_names:
                X_in = X_in[feature_names]
    else:
        arr = np.asarray(features, dtype=float) if not hasattr(features, "tocsr") else features
        if hasattr(arr, "ndim") and arr.ndim == 1:
            scalar_output = True
            arr2d = arr.reshape(1, -1)
        elif hasattr(arr, "ndim") and arr.ndim == 2:
            arr2d = arr
        else:
            raise ValueError(f"Unsupported feature input shape for score_trust.")

        if feature_names and arr2d.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {arr2d.shape[1]}.")

        if hasattr(model, "feature_names_in_") and not hasattr(arr2d, "tocsr"):
            X_in = pd.DataFrame(arr2d, columns=list(model.feature_names_in_))
        else:
            X_in = arr2d

    probs = np.clip(model.predict_proba(X_in)[:, 1].astype(float), 0.0, 1.0)
    if scalar_output:
        return float(probs[0])
    return probs


def evaluate_trust_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute classification, ranking, and probability calibration metrics
    for binary trust predictions (1 = valid, 0 = invalid).
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    preds = (p_arr >= float(threshold)).astype(int)

    n_total = int(len(y_arr))
    n_pos = int(y_arr.sum())
    n_neg = n_total - n_pos
    has_both = (n_pos > 0) and (n_neg > 0)

    roc = float(roc_auc_score(y_arr, p_arr)) if has_both else float("nan")
    pr = float(average_precision_score(y_arr, p_arr)) if n_pos > 0 else float("nan")
    prec = float(precision_score(y_arr, preds, zero_division=0))
    rec = float(recall_score(y_arr, preds, zero_division=0))
    f1 = float(f1_score(y_arr, preds, zero_division=0))
    acc = float(accuracy_score(y_arr, preds))
    brier = float(brier_score_loss(y_arr, p_arr)) if n_total > 0 else float("nan")

    # Quantile-based calibration bins and Expected Calibration Error (ECE)
    bin_rows: List[Dict[str, Any]] = []
    ece = 0.0
    if n_total > 0:
        edges = np.quantile(p_arr, np.linspace(0.0, 1.0, n_bins + 1))
        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (p_arr >= edges[b]) & (p_arr < edges[b + 1])
            else:
                mask = (p_arr >= edges[b]) & (p_arr <= edges[b + 1])
            cnt = int(mask.sum())
            if cnt > 0:
                mean_p = float(p_arr[mask].mean())
                frac_pos = float(y_arr[mask].mean())
                ece += (cnt / n_total) * abs(frac_pos - mean_p)
                bin_rows.append({
                    "bin": b + 1,
                    "n": cnt,
                    "mean_pred": mean_p,
                    "frac_pos": frac_pos,
                    "abs_gap": abs(frac_pos - mean_p),
                })

    return {
        "n": n_total,
        "n_valid": n_pos,
        "n_invalid": n_neg,
        "valid_rate": float(n_pos / n_total) if n_total > 0 else float("nan"),
        "mean_pred": float(p_arr.mean()) if n_total > 0 else float("nan"),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
        "ece": float(ece),
        "calibration_bins": bin_rows,
    }


def evaluate_by_group(
    df_eval: pd.DataFrame,
    y_prob: Union[pd.Series, np.ndarray],
    group_col: str = "group",
    y_col: str = "y",
    groups: Optional[List[str]] = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Evaluate trust predictions overall and disaggregated per complaint group."""
    if groups is None:
        groups = sorted(df_eval[group_col].unique().tolist())

    y_all = df_eval[y_col].to_numpy(dtype=int)
    p_all = np.asarray(y_prob, dtype=float)

    rows = []
    for label, mask in [("ALL", np.ones(len(df_eval), dtype=bool))] + [
        (g, (df_eval[group_col].to_numpy() == g)) for g in groups
    ]:
        if not mask.any():
            continue
        m = evaluate_trust_predictions(y_all[mask], p_all[mask], threshold=threshold, n_bins=n_bins)
        m.pop("calibration_bins", None)
        m["group"] = label
        rows.append(m)

    cols = [
        "group", "n", "n_valid", "n_invalid", "valid_rate", "mean_pred",
        "precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc", "brier", "ece",
    ]
    return pd.DataFrame(rows)[cols]

