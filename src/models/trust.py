"""
CityPulse - Citizen Complaint Trust Scoring Model (Step 6.1d & Step 6.2 / Lab 3)
=================================================================================
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

Labeling rules (Step 6.1d standardized CSV rules):
  0. Rule source: `src/models/trust_label_rules.csv` ONLY (lowercase, collapsed whitespace,
     case-insensitive substring match).
     - Matching rules from BOTH classes (`valid` & `invalid`) or >1 distinct rule raises `ValueError`.
     - No rule match defaults to `'excluded'`.
  1. Gate (`apply_gate=True`): resolved rows with empty `closed_date`, or rows with empty
     `resolution_id`, are labeled `'excluded'`.
  2. Sensitivity overrides (`S1`, `S2`, `S3`, `S4`):
     - `S1`: text containing `"no criminal violation existed"` -> `'invalid'`
     - `S2`: text containing `"no criminal violation existed"` -> `'valid'`
     - `S3`: text containing `"unable to gain entry"` -> `'invalid'`
     - `S4`: text containing `"sent the dog owner a warning letter"` -> `'excluded'`
"""

import hashlib
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

TRUST_RULES_PATH: Path = Path(__file__).resolve().parent / "trust_label_rules.csv"

NEARBY_PRIOR_DEFINITION: str = (
    "nearby_prior_24h_200m: count of prior complaints across ANY group "
    "(flood, noise, road, sewer_backup pooled in the input stream) within Euclidean distance "
    "<= 200.0 m on local planar (x_m, y_m) coordinates over the strictly backward-looking "
    "24-hour window [t_current - 24h, t_current). Self and exact timestamp ties "
    "(t_prior == t_current) are strictly excluded via `t_prior < t_current`."
)

# Cache parsed rules keyed by (resolved_path, mtime_ns)
_RULES_CACHE: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}

# Cache loaded trust model bundles keyed by resolved file path
_TRUST_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def compute_file_sha256(path: Union[str, Path], uppercase: bool = False) -> str:
    """Compute SHA-256 hex digest for a local file."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest.upper() if uppercase else digest.lower()


def _normalize_rule_text(text: Any) -> str:
    """Lowercase text and collapse all whitespace runs to single spaces."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def load_trust_label_rules(
    rules_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str]]:
    """
    Load `(class, substring)` trust classification rules strictly from CSV
    (`src/models/trust_label_rules.csv` by default), normalizing whitespace and case.
    """
    path = TRUST_RULES_PATH if rules_path is None else Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Trust label rules file not found: {path}")

    stat = path.stat()
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns))
    if cache_key in _RULES_CACHE:
        return list(_RULES_CACHE[cache_key])

    rules_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"class", "substring"}.issubset(rules_df.columns):
        raise ValueError(f"Rules CSV must contain 'class' and 'substring' columns: {path}")

    seen: set = set()
    rules: List[Tuple[str, str]] = []
    for _, row in rules_df.iterrows():
        cls = row["class"].strip().lower()
        sub = _normalize_rule_text(row["substring"])
        if cls not in {"valid", "invalid"}:
            raise ValueError(f"Invalid rule class {cls!r} in {path}; expected 'valid' or 'invalid'.")
        if not sub:
            raise ValueError(f"Empty rule substring for class {cls!r} in {path}.")
        if (cls, sub) not in seen:
            seen.add((cls, sub))
            rules.append((cls, sub))

    _RULES_CACHE[cache_key] = rules
    return list(rules)


def classify_resolution_text(
    text: Any,
    rules_path: Optional[Union[str, Path]] = None,
    sensitivity: Optional[str] = None,
    s1: bool = False,
    s2: bool = False,
    s3: bool = False,
    s4: bool = False,
    return_rule: bool = False,
) -> Union[str, Tuple[str, str]]:
    """
    Classify a 311 resolution description using ONLY `src/models/trust_label_rules.csv`.

    Matching rules:
    - Both rule substrings and input text are lowercased with whitespace collapsed.
    - Case-insensitive substring matching across all rules in the CSV.
    - Raises `ValueError` (listing matched rules and offending text) if a text matches
      rules of BOTH classes ('valid' and 'invalid') OR matches more than one distinct rule.
    - Unmatched or empty text returns `'excluded'`.

    Optional sensitivity parameters:
    - `S1` / `s1=True`: text containing `"no criminal violation existed"` -> `'invalid'`
    - `S2` / `s2=True`: text containing `"no criminal violation existed"` -> `'valid'`
    - `S3` / `s3=True`: text containing `"unable to gain entry"` -> `'invalid'`
    - `S4` / `s4=True`: text containing `"sent the dog owner a warning letter"` -> `'excluded'`
    """
    if sensitivity is not None:
        sens_key = str(sensitivity).strip().upper()
        if sens_key in {"", "NONE", "BASE"}:
            pass
        elif sens_key == "S1":
            s1 = True
        elif sens_key == "S2":
            s2 = True
        elif sens_key == "S3":
            s3 = True
        elif sens_key == "S4":
            s4 = True
        else:
            raise ValueError(
                f"Unknown sensitivity {sensitivity!r}; expected None, 'S1', 'S2', 'S3', or 'S4'."
            )

    if s1 and s2:
        raise ValueError(
            "Sensitivity options S1 and S2 are mutually exclusive "
            "(both match 'no criminal violation existed' with opposite classes)."
        )

    norm_text = _normalize_rule_text(text)
    if not norm_text:
        return ("excluded", "") if return_rule else "excluded"

    if s4 and "sent the dog owner a warning letter" in norm_text:
        return ("excluded", "S4: sent the dog owner a warning letter") if return_rule else "excluded"

    rules = load_trust_label_rules(rules_path=rules_path)
    matched: List[Tuple[str, str]] = [
        (cls, sub) for cls, sub in rules if sub in norm_text
    ]

    if s1 and "no criminal violation existed" in norm_text:
        matched.append(("invalid", "S1: no criminal violation existed"))
    if s2 and "no criminal violation existed" in norm_text:
        matched.append(("valid", "S2: no criminal violation existed"))
    if s3 and "unable to gain entry" in norm_text:
        matched.append(("invalid", "S3: unable to gain entry"))

    # Deduplicate exact (class, rule) entries while preserving conflict detection across distinct rules
    unique_matched: List[Tuple[str, str]] = []
    seen_rules: set = set()
    for cls, sub in matched:
        if (cls, sub) not in seen_rules:
            seen_rules.add((cls, sub))
            unique_matched.append((cls, sub))

    distinct_classes = sorted({cls for cls, _ in unique_matched})
    formatted_matches = [f"{cls}:{sub}" for cls, sub in unique_matched]

    if len(distinct_classes) > 1:
        raise ValueError(
            f"Resolution text matches rules of BOTH classes ({distinct_classes}): "
            f"matched_rules={formatted_matches}; text={norm_text!r}"
        )
    if len(unique_matched) > 1:
        raise ValueError(
            f"Resolution text matches more than one distinct rule ({len(unique_matched)}): "
            f"matched_rules={formatted_matches}; text={norm_text!r}"
        )

    if not unique_matched:
        return ("excluded", "") if return_rule else "excluded"

    chosen_cls, chosen_sub = unique_matched[0]
    if return_rule:
        return chosen_cls, chosen_sub
    return chosen_cls


def assign_trust_labels(
    df: pd.DataFrame,
    lookup_df: Optional[pd.DataFrame] = None,
    rules_path: Optional[Union[str, Path]] = None,
    apply_gate: bool = True,
    sensitivity: Optional[str] = None,
    s1: bool = False,
    s2: bool = False,
    s3: bool = False,
    s4: bool = False,
) -> pd.Series:
    """
    Assign trust class labels ('valid', 'invalid', 'excluded') from
    `src/models/trust_label_rules.csv` and the Step 6.1d gate.

    Gate (`apply_gate=True`):
    - Rows with no `resolution_id` (empty string / NaN) -> `'excluded'`.
    - Resolved rows with empty `closed_date` -> `'excluded'`.
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

    res_class_map: Dict[str, str] = {}
    for rid, txt in zip(lookup_df["resolution_id"], lookup_df["text"]):
        try:
            res_class_map[rid] = str(
                classify_resolution_text(
                    txt,
                    rules_path=rules_path,
                    sensitivity=sensitivity,
                    s1=s1,
                    s2=s2,
                    s3=s3,
                    s4=s4,
                )
            )
        except ValueError as exc:
            raise ValueError(f"Rule conflict on resolution_id={rid!r}: {exc}") from exc

    res_ids = (
        df["resolution_id"].fillna("").astype(str).str.strip()
        if "resolution_id" in df.columns
        else pd.Series("", index=df.index)
    )
    labels = res_ids.map(res_class_map).fillna("excluded")
    empty_res_mask = res_ids == ""
    labels = labels.where(~empty_res_mask, "excluded")

    if apply_gate:
        closed_vals = (
            df["closed_date"].fillna("").astype(str).str.strip()
            if "closed_date" in df.columns
            else pd.Series("", index=df.index)
        )
        gate_mask = (closed_vals == "") | empty_res_mask
        labels = labels.where(~gate_mask, "excluded")

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


def build_trust_feature_subset(
    df: Union[pd.DataFrame, Dict[str, Any]],
    categorical_cols: Sequence[str] = TRUST_CATEGORICAL_COLS,
    include_nearby: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    recompute_nearby: bool = False,
    return_encoder: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[OneHotEncoder]]]:
    """
    Generalized submission-time feature builder supporting categorical column subsets
    and optional exclusion of `nearby_prior_24h_200m` for ablation experiments.
    When `categorical_cols` is empty, only `nearby_prior_24h_200m` is included.
    """
    if isinstance(df, dict):
        work = pd.DataFrame([df])
    elif isinstance(df, pd.DataFrame):
        work = df
    else:
        raise TypeError(f"Expected pandas DataFrame or dict, got {type(df).__name__}")

    cat_list = list(categorical_cols)
    if not cat_list and not include_nearby:
        raise ValueError("At least one categorical column or include_nearby=True is required.")

    parts: List[np.ndarray] = []
    feature_names: List[str] = []

    if cat_list:
        full_cat_df = _normalize_categorical_frame(work)
        sub_cat_df = full_cat_df[cat_list]
        if encoder is None:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat = encoder.fit_transform(sub_cat_df)
        else:
            X_cat = encoder.transform(sub_cat_df)
        if hasattr(X_cat, "toarray"):
            X_cat = X_cat.toarray()
        parts.append(np.asarray(X_cat, dtype=float))
        feature_names.extend(list(encoder.get_feature_names_out(cat_list)))

    if include_nearby:
        if not recompute_nearby and "nearby_prior_24h_200m" in work.columns:
            nearby_arr = pd.to_numeric(work["nearby_prior_24h_200m"], errors="raise").astype(float).to_numpy()
        else:
            nearby_arr = compute_nearby_prior_24h_200m(work).astype(float)
        parts.append(nearby_arr.reshape(-1, 1))
        feature_names.extend(TRUST_NUMERIC_COLS)

    X_mat = np.column_stack(parts) if len(parts) > 1 else parts[0]
    X_df = pd.DataFrame(X_mat, index=work.index, columns=feature_names)
    X_df.attrs["encoder"] = encoder
    X_df.attrs["feature_names"] = feature_names

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


def prior_shift(
    p: Union[float, np.ndarray, pd.Series, Sequence[float]],
    logit_shift: float = 0.0,
    eps: float = 1e-15,
) -> Union[float, np.ndarray]:
    """
    Apply prior-odds logit shift:
        prior_shift(p, logit_shift) = sigmoid(logit(p) + logit_shift)
    Used to recalibrate `class_weight='balanced'` predictions to the training prior
    via `logit_shift = log(n_valid_train / n_invalid_train)`.
    Returns an exact clipped identity when `logit_shift == 0.0`.
    """
    is_scalar = np.isscalar(p)
    p_arr = np.asarray(p, dtype=float)
    shift = float(logit_shift)
    if shift == 0.0:
        shifted = np.clip(p_arr, 0.0, 1.0)
    else:
        p_clip = np.clip(p_arr, float(eps), 1.0 - float(eps))
        logits = np.log(p_clip / (1.0 - p_clip)) + shift
        logits_safe = np.clip(logits, -500.0, 500.0)
        shifted = np.where(
            logits_safe >= 0.0,
            1.0 / (1.0 + np.exp(-logits_safe)),
            np.exp(logits_safe) / (1.0 + np.exp(logits_safe)),
        )
        shifted = np.clip(shifted, 0.0, 1.0)

    if is_scalar:
        return float(shifted)
    return np.asarray(shifted, dtype=float)


def select_top_noise_coords(
    train_df: pd.DataFrame,
    top_k: int = 5,
    label_col: Optional[str] = "trust_label",
) -> Tuple[List[Dict[str, Any]], set]:
    """
    Identify the top-k highest-volume noise coordinates strictly from TRAIN noise rows.
    Returns `(records_list, set_of_(x_m, y_m))`.
    """
    noise_tr = train_df[train_df["group"] == "noise"].copy()
    if len(noise_tr) == 0:
        return [], set()

    has_addr = "address_norm" in noise_tr.columns
    has_latlon = ("lat" in noise_tr.columns) and ("lon" in noise_tr.columns)
    has_lbl = label_col is not None and label_col in noise_tr.columns

    grouped = (
        noise_tr.groupby(["x_m", "y_m"], sort=True)
        .agg(train_total=("id", "count") if "id" in noise_tr.columns else ("x_m", "count"))
        .sort_values("train_total", ascending=False)
        .head(int(top_k))
    )

    records: List[Dict[str, Any]] = []
    coord_set: set = set()
    for (xm, ym), row in grouped.iterrows():
        sub = noise_tr[(noise_tr["x_m"] == xm) & (noise_tr["y_m"] == ym)]
        rec: Dict[str, Any] = {
            "x_m": float(xm),
            "y_m": float(ym),
            "train_total": int(row["train_total"]),
        }
        if has_addr:
            rec["address"] = str(sub["address_norm"].value_counts().index[0])
        if has_latlon:
            rec["lat"] = float(sub["lat"].iloc[0])
            rec["lon"] = float(sub["lon"].iloc[0])
        if has_lbl:
            vc = sub[label_col].value_counts()
            n_val = int(vc.get("valid", 0))
            n_inv = int(vc.get("invalid", 0))
            rec["train_valid"] = n_val
            rec["train_invalid"] = n_inv
            rec["train_labelled"] = n_val + n_inv
        records.append(rec)
        coord_set.add((float(xm), float(ym)))

    return records, coord_set


def save_trust_model(
    model: LogisticRegression,
    encoder: OneHotEncoder,
    feature_names: List[str],
    model_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Persist trained trust model bundle via joblib with full Step 6.2b schema:
    model, encoder, feature_names, categorical_cols, numeric_cols, sklearn_version,
    created_date, variant, train_cutoff_ts, excluded_top5_noise_coords_train,
    label_rules_sha256, usable_groups, logit_shift, calibrated, group_train_valid_rate.
    """
    if model_path is None:
        root = Path(__file__).resolve().parents[2]
        model_path = root / "models/trust_model.joblib"
    else:
        model_path = Path(model_path)

    rules_sha = compute_file_sha256(TRUST_RULES_PATH) if TRUST_RULES_PATH.exists() else ""

    bundle: Dict[str, Any] = {
        "model": model,
        "encoder": encoder,
        "feature_names": list(feature_names),
        "categorical_cols": list(TRUST_CATEGORICAL_COLS),
        "numeric_cols": list(TRUST_NUMERIC_COLS),
        "sklearn_version": sklearn.__version__,
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "variant": "",
        "train_cutoff_ts": "",
        "excluded_top5_noise_coords_train": [],
        "label_rules_sha256": rules_sha,
        "usable_groups": [],
        "logit_shift": 0.0,
        "calibrated": False,
        "group_train_valid_rate": {},
    }
    if metadata:
        bundle.update(metadata)

    bundle["calibrated"] = bool(bundle.get("calibrated", False))
    bundle["logit_shift"] = float(bundle.get("logit_shift", 0.0))
    bundle["usable_groups"] = list(bundle.get("usable_groups", []))
    bundle["group_train_valid_rate"] = dict(bundle.get("group_train_valid_rate", {}))

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
    if bool(bundle.get("calibrated", False)):
        probs = prior_shift(probs, float(bundle.get("logit_shift", 0.0)))
    probs = np.clip(probs, 0.0, 1.0)
    if scalar_output:
        return float(probs[0])
    return probs


def compute_calibration_table(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute reliability table, ECE, and Brier score across `n_bins`.
    - `strategy="uniform"` uses equal-width probability bins on [0.0, 1.0].
    - `strategy="quantile"` uses empirical quantile edges.
    Returns `(reliability_df, ece, brier)`.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n_total = int(len(y_arr))
    brier = float(brier_score_loss(y_arr, p_arr)) if n_total > 0 else float("nan")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    elif strategy == "quantile":
        edges = np.quantile(p_arr, np.linspace(0.0, 1.0, int(n_bins) + 1)) if n_total > 0 else np.linspace(0.0, 1.0, int(n_bins) + 1)
    else:
        raise ValueError(f"Unknown binning strategy {strategy!r}; expected 'uniform' or 'quantile'.")

    rows: List[Dict[str, Any]] = []
    ece = 0.0
    for b in range(int(n_bins)):
        lo = float(edges[b])
        hi = float(edges[b + 1])
        if n_total == 0:
            mask = np.zeros(0, dtype=bool)
        elif b < int(n_bins) - 1:
            mask = (p_arr >= lo) & (p_arr < hi)
        else:
            mask = (p_arr >= lo) & (p_arr <= hi)

        cnt = int(mask.sum())
        if cnt > 0:
            mean_p = float(p_arr[mask].mean())
            obs_r = float(y_arr[mask].mean())
            gap = abs(obs_r - mean_p)
            ece += (cnt / n_total) * gap
        else:
            mean_p = float("nan")
            obs_r = float("nan")
            gap = float("nan")

        rows.append({
            "bin": b + 1,
            "bin_range": f"[{lo:.1f}, {hi:.1f}" + ("]" if b == int(n_bins) - 1 else ")"),
            "n": cnt,
            "mean_pred": mean_p,
            "obs_rate": obs_r,
            "abs_gap": gap,
        })

    return pd.DataFrame(rows), float(ece), brier


def bootstrap_auc_ci(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute ROC AUC, PR AUC, and percentile bootstrap 95% CIs (`seed=42`, `n_boot=1000`).
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_arr)
    has_both = (n > 0) and (y_arr.sum() > 0) and (y_arr.sum() < n)

    roc_pt = float(roc_auc_score(y_arr, p_arr)) if has_both else float("nan")
    pr_pt = float(average_precision_score(y_arr, p_arr)) if (n > 0 and y_arr.sum() > 0) else float("nan")

    if not has_both or int(n_boot) <= 0:
        return {
            "roc_auc": roc_pt,
            "roc_ci_low": float("nan"),
            "roc_ci_high": float("nan"),
            "pr_auc": pr_pt,
            "pr_ci_low": float("nan"),
            "pr_ci_high": float("nan"),
        }

    rng = np.random.default_rng(int(seed))
    roc_boots: List[float] = []
    pr_boots: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        pb = p_arr[idx]
        n_pos_b = int(yb.sum())
        if n_pos_b == 0 or n_pos_b == n:
            continue
        roc_boots.append(float(roc_auc_score(yb, pb)))
        pr_boots.append(float(average_precision_score(yb, pb)))

    q_lo = 100.0 * (float(alpha) / 2.0)
    q_hi = 100.0 * (1.0 - float(alpha) / 2.0)
    return {
        "roc_auc": roc_pt,
        "roc_ci_low": float(np.percentile(roc_boots, q_lo)) if roc_boots else float("nan"),
        "roc_ci_high": float(np.percentile(roc_boots, q_hi)) if roc_boots else float("nan"),
        "pr_auc": pr_pt,
        "pr_ci_low": float(np.percentile(pr_boots, q_lo)) if pr_boots else float("nan"),
        "pr_ci_high": float(np.percentile(pr_boots, q_hi)) if pr_boots else float("nan"),
    }


def evaluate_with_ci_and_bss(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    train_base_rate: float,
    n_boot: int = 1000,
    seed: int = 42,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute full Step 6.2b evaluation metrics:
    n_test, n_valid, n_invalid, test_base_rate, train_base_rate,
    ROC/PR AUC + 95% percentile bootstrap CIs, Brier, constant train-rate Brier,
    Brier skill score (BSS = 1 - brier / brier_const), equal-width 10-bin ECE, and mean_pred.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n_test = int(len(y_arr))
    n_valid = int(y_arr.sum())
    n_invalid = n_test - n_valid

    auc_stats = bootstrap_auc_ci(y_arr, p_arr, n_boot=n_boot, seed=seed)
    _, ece, brier = compute_calibration_table(y_arr, p_arr, n_bins=n_bins, strategy="uniform")
    brier_const = float(np.mean((float(train_base_rate) - y_arr.astype(float)) ** 2)) if n_test > 0 else float("nan")
    bss = float(1.0 - (brier / brier_const)) if (n_test > 0 and brier_const > 0.0) else float("nan")

    return {
        "n_test": n_test,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "test_base_rate": float(n_valid / n_test) if n_test > 0 else float("nan"),
        "train_base_rate": float(train_base_rate),
        "roc_auc": auc_stats["roc_auc"],
        "roc_ci_low": auc_stats["roc_ci_low"],
        "roc_ci_high": auc_stats["roc_ci_high"],
        "pr_auc": auc_stats["pr_auc"],
        "pr_ci_low": auc_stats["pr_ci_low"],
        "pr_ci_high": auc_stats["pr_ci_high"],
        "brier": brier,
        "brier_const": brier_const,
        "bss": bss,
        "ece": ece,
        "mean_pred": float(p_arr.mean()) if n_test > 0 else float("nan"),
    }


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


def evaluate_ablation_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "trust_label",
    group_col: str = "group",
    groups: Sequence[str] = ("noise", "road", "flood", "sewer_backup"),
) -> pd.DataFrame:
    """
    Evaluate Step 6.2b feature ablations (A..E) trained ONLY on labelled rows of
    `train_df` and evaluated ONLY on labelled rows of `test_df`:
    - A: full feature set
    - B: complaint_type + descriptor_group only
    - C: full minus nearby_prior_24h_200m
    - D: without complaint_type and descriptor_group
    - E: no model, empirical train valid rate per complaint_type lookup
    """
    tr_lab = train_df[train_df[label_col].isin(["valid", "invalid"])].copy()
    te_lab = test_df[test_df[label_col].isin(["valid", "invalid"])].copy()
    y_tr = (tr_lab[label_col] == "valid").astype(int).to_numpy()
    y_te = (te_lab[label_col] == "valid").astype(int).to_numpy()

    ablation_specs = [
        ("A: full", list(TRUST_CATEGORICAL_COLS), True),
        ("B: complaint_type + descriptor_group only", ["complaint_type", "descriptor_group"], False),
        ("C: full minus nearby_prior_24h_200m", list(TRUST_CATEGORICAL_COLS), False),
        ("D: without complaint_type and descriptor_group", ["channel", "location_type", "hour", "weekday"], True),
    ]

    rows: List[Dict[str, Any]] = []
    for name, cat_cols, inc_nearby in ablation_specs:
        X_tr, enc = build_trust_feature_subset(
            tr_lab, categorical_cols=cat_cols, include_nearby=inc_nearby, return_encoder=True
        )
        clf = train_trust_model(X_tr, y_tr)
        X_te = build_trust_feature_subset(
            te_lab, categorical_cols=cat_cols, include_nearby=inc_nearby, encoder=enc
        )
        probs = np.clip(clf.predict_proba(X_te)[:, 1].astype(float), 0.0, 1.0)
        rec: Dict[str, Any] = {"ablation": name, "n_features": int(X_tr.shape[1])}
        for g in list(groups) + ["POOLED"]:
            mask = np.ones(len(te_lab), dtype=bool) if g == "POOLED" else (te_lab[group_col].to_numpy() == g)
            yt_g = y_te[mask]
            rec[g] = float(roc_auc_score(yt_g, probs[mask])) if (yt_g.sum() > 0 and yt_g.sum() < len(yt_g)) else float("nan")
        rows.append(rec)

    # E: empirical train valid rate per complaint_type lookup (fallback to pooled train valid rate)
    tr_tmp = tr_lab[["complaint_type"]].copy()
    tr_tmp["_y"] = y_tr
    ct_rates = tr_tmp.groupby("complaint_type")["_y"].mean().to_dict()
    global_rate = float(y_tr.mean())
    probs_e = te_lab["complaint_type"].map(ct_rates).fillna(global_rate).to_numpy(dtype=float)
    rec_e: Dict[str, Any] = {"ablation": "E: complaint_type train valid-rate lookup (no model)", "n_features": 1}
    for g in list(groups) + ["POOLED"]:
        mask = np.ones(len(te_lab), dtype=bool) if g == "POOLED" else (te_lab[group_col].to_numpy() == g)
        yt_g = y_te[mask]
        rec_e[g] = float(roc_auc_score(yt_g, probs_e[mask])) if (yt_g.sum() > 0 and yt_g.sum() < len(yt_g)) else float("nan")
    rows.append(rec_e)

    return pd.DataFrame(rows)


def classify_noise_agency(text: Any) -> str:
    """Classify resolving agency from resolution text substring."""
    norm = _normalize_rule_text(text)
    if "police department" in norm:
        return "NYPD"
    if "environmental protection" in norm:
        return "DEP"
    return "OTHER"


def classify_noise_valid_text_group(text: Any) -> str:
    """Group valid noise resolutions into interpretable agency/action buckets."""
    norm = _normalize_rule_text(text)
    if "police department" in norm:
        if "summons" in norm:
            return "NYPD summons"
        if "arrest" in norm:
            return "NYPD arrest"
        if "took action" in norm:
            return "NYPD action"
        return "NYPD other"
    if "environmental protection" in norm:
        if "sent the dog owner a warning letter" in norm:
            return "dog letters (DEP)"
        if "summons" in norm:
            return "DEP summons"
        if "obstructing the catch basin" in norm:
            return "DEP catch basin debris"
        return "DEP other"
    return "OTHER"


def evaluate_noise_agency_slices(
    noise_test_df: pd.DataFrame,
    y_prob: Union[pd.Series, np.ndarray],
    lookup_df: pd.DataFrame,
    y_col: str = "y",
) -> pd.DataFrame:
    """Evaluate M2 test noise performance stratified by resolving agency (NYPD vs DEP)."""
    lkp_map = dict(zip(lookup_df["resolution_id"].astype(str), lookup_df["text"]))
    agencies = noise_test_df["resolution_id"].astype(str).map(lkp_map).map(classify_noise_agency).to_numpy()
    y_arr = noise_test_df[y_col].to_numpy(dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    rows: List[Dict[str, Any]] = []
    for ag in ("NYPD", "DEP", "OTHER"):
        mask = agencies == ag
        if not mask.any():
            continue
        yt = y_arr[mask]
        pt = p_arr[mask]
        n_val = int(yt.sum())
        n_inv = int(len(yt) - n_val)
        roc = float(roc_auc_score(yt, pt)) if (n_val > 0 and n_inv > 0) else float("nan")
        pr = float(average_precision_score(yt, pt)) if n_val > 0 else float("nan")
        rows.append({
            "agency": ag,
            "n_test": int(len(yt)),
            "n_valid": n_val,
            "n_invalid": n_inv,
            "test_base_rate": float(n_val / len(yt)),
            "roc_auc": roc,
            "pr_auc": pr,
        })
    return pd.DataFrame(rows)


def evaluate_sensitivity_suite(
    df: pd.DataFrame,
    train_cutoff_ts: Any,
    top5_noise_coords: set,
    lookup_df: pd.DataFrame,
    sensitivities: Sequence[Optional[str]] = (None, "S1", "S2", "S3", "S4"),
    groups: Sequence[str] = ("noise", "road", "flood", "sewer_backup"),
) -> pd.DataFrame:
    """
    Re-assign trust labels under Base and S1..S4 sensitivities, fit M2 strictly on
    labelled train rows (excluding top-5 noise train coordinates), and evaluate
    per-group and pooled ROC AUC, PR AUC, and test base rate on labelled test rows.
    """
    ts_col = pd.to_datetime(df["ts"]) if "ts" in df.columns else _extract_time_series(df)
    cutoff = pd.Timestamp(train_cutoff_ts)
    coord_in_top5 = pd.Series(list(zip(df["x_m"], df["y_m"])), index=df.index).isin(top5_noise_coords)
    is_noise_top5 = (df["group"] == "noise") & coord_in_top5

    rows: List[Dict[str, Any]] = []
    for sens in sensitivities:
        sens_name = "Base" if sens is None else str(sens)
        lbl = assign_trust_labels(df, lookup_df=lookup_df, apply_gate=True, sensitivity=sens)
        is_lab = lbl.isin(["valid", "invalid"])
        tr_mask = (ts_col < cutoff) & is_lab & (~is_noise_top5)
        te_mask = (ts_col >= cutoff) & is_lab

        tr_df = df.loc[tr_mask].copy()
        te_df = df.loc[te_mask].copy()
        y_tr = (lbl.loc[tr_mask] == "valid").astype(int).to_numpy()
        y_te = (lbl.loc[te_mask] == "valid").astype(int).to_numpy()

        X_tr, enc = build_trust_features(tr_df, return_encoder=True)
        clf = train_trust_model(X_tr, y_tr)
        X_te = build_trust_features(te_df, encoder=enc)
        probs = np.clip(clf.predict_proba(X_te)[:, 1].astype(float), 0.0, 1.0)

        for g in list(groups) + ["POOLED"]:
            g_mask = np.ones(len(te_df), dtype=bool) if g == "POOLED" else (te_df["group"].to_numpy() == g)
            yt = y_te[g_mask]
            pt = probs[g_mask]
            n_val = int(yt.sum())
            n_inv = int(len(yt) - n_val)
            roc = float(roc_auc_score(yt, pt)) if (n_val > 0 and n_inv > 0) else float("nan")
            pr = float(average_precision_score(yt, pt)) if n_val > 0 else float("nan")
            rows.append({
                "sensitivity": sens_name,
                "group": g,
                "n_test": int(len(yt)),
                "n_valid": n_val,
                "n_invalid": n_inv,
                "base_rate": float(n_val / len(yt)) if len(yt) > 0 else float("nan"),
                "roc_auc": roc,
                "pr_auc": pr,
            })

    return pd.DataFrame(rows)


