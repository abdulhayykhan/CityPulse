"""
CityPulse - Optional Baseline Classifiers (Labs 4, 8, 9)
========================================================
Clean, leak-free multi-class baselines predicting complaint `group`
(`noise`, `road`, `flood`, `sewer_backup`) from non-leaky submission metadata
(`hour`, `weekday`, `channel`, `location_type`) using:
- Lab 9: Multinomial Naive Bayes (`MultinomialNB`)
- Lab 4: Decision Tree Classifier (`DecisionTreeClassifier`) & depth study
- Lab 8: k-Nearest Neighbors (`KNeighborsClassifier`) & K-sensitivity curve

Explicitly excludes `complaint_type`, `descriptor`, and `descriptor_group`
to prevent 100% deterministic target leakage.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASELINE_FEATURE_COLS: List[str] = ["hour", "weekday", "channel", "location_type"]
BASELINE_TARGET_COL: str = "group"
BASELINE_FORBIDDEN_COLS: Tuple[str, ...] = ("complaint_type", "descriptor", "descriptor_group")


def _non_missing_mask(series: pd.Series) -> pd.Series:
    """Return boolean mask identifying valid non-missing entries (not NaN/None or blank string)."""
    mask = ~series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        stripped = series.where(mask, "").astype(str).str.strip()
        mask = mask & (stripped != "") & (~stripped.str.lower().isin({"nan", "none"}))
    return mask


def prepare_baseline_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract leak-free submission baseline features (`hour`, `weekday`, `channel`, `location_type`)
    and target (`group`), dropping rows with missing values in any of these columns.

    Strictly excludes `complaint_type` and `descriptor` (and any derived descriptor fields)
    to prevent deterministic target leakage. Treats both `NaN`/`None` and empty/whitespace
    strings (such as absent `location_type` loaded with `keep_default_na=False`) as missing,
    while preserving explicit categories such as `'UNKNOWN'`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    required = BASELINE_FEATURE_COLS + [BASELINE_TARGET_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required baseline columns: {missing_cols}")

    work = df[required].copy()

    # Coerce integer temporal fields first so non-numeric/empty strings become NaN and drop cleanly
    hour_num = pd.to_numeric(work["hour"], errors="coerce")
    weekday_num = pd.to_numeric(work["weekday"], errors="coerce")

    valid_mask = (
        hour_num.notna()
        & np.isfinite(hour_num.fillna(0.0))
        & weekday_num.notna()
        & np.isfinite(weekday_num.fillna(0.0))
        & _non_missing_mask(work["channel"])
        & _non_missing_mask(work["location_type"])
        & _non_missing_mask(work["group"])
    )

    clean_idx = work.index[valid_mask]
    X = pd.DataFrame(
        {
            "hour": hour_num.loc[clean_idx].astype(int),
            "weekday": weekday_num.loc[clean_idx].astype(int),
            "channel": work.loc[clean_idx, "channel"].astype(str),
            "location_type": work.loc[clean_idx, "location_type"].astype(str),
        },
        index=clean_idx,
        columns=BASELINE_FEATURE_COLS,
    )
    y = work.loc[clean_idx, BASELINE_TARGET_COL].astype(str)
    y.name = BASELINE_TARGET_COL

    # Defensive guarantee against target leakage columns
    for forbidden in BASELINE_FORBIDDEN_COLS:
        assert forbidden not in X.columns, f"Leakage column {forbidden!r} must not appear in baseline features."

    return X, y


def build_baseline_pipeline(model: Any, sparse_output: bool = False) -> Pipeline:
    """
    Build a scikit-learn `Pipeline` with `OneHotEncoder(handle_unknown='ignore')`
    followed by the provided classifier `model`.

    Uses `sparse_output=False` by default for compact 53-column one-hot inputs so distance-based
    models like `KNeighborsClassifier` run several times faster without changing predictions.
    """
    if model is None:
        raise ValueError("model must be a valid scikit-learn estimator instance, got None.")
    return Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
            ("model", model),
        ]
    )


def evaluate_baselines(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray, Sequence[Any]],
) -> pd.DataFrame:
    """
    Evaluate a dictionary of fitted baseline models/pipelines on `(X_test, y_test)` and
    return a comparison `pd.DataFrame` with `model`, `accuracy`, and `weighted_f1`.
    """
    if not models_dict:
        raise ValueError("models_dict must contain at least one fitted model.")

    y_arr = np.asarray(y_test)
    rows: List[Dict[str, Any]] = []
    for name, estimator in models_dict.items():
        y_pred = np.asarray(estimator.predict(X_test))
        acc = float(accuracy_score(y_arr, y_pred))
        wf1 = float(f1_score(y_arr, y_pred, average="weighted", zero_division=0))
        rows.append(
            {
                "model": str(name),
                "accuracy": acc,
                "weighted_f1": wf1,
            }
        )
    return pd.DataFrame(rows, columns=["model", "accuracy", "weighted_f1"])


def chronological_train_test_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Prepare leak-free baseline `(X, y)` and split strictly chronologically 80/20
    by `time_col` after stable timestamp sorting.
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing timestamp column {time_col!r} for chronological split.")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")

    X_raw, y_raw = prepare_baseline_data(df)
    ts = pd.to_datetime(df.loc[X_raw.index, time_col], errors="coerce")
    valid_ts = ts.notna()
    X_valid = X_raw.loc[valid_ts].copy()
    y_valid = y_raw.loc[valid_ts].copy()
    ts_valid = ts.loc[valid_ts]

    order = ts_valid.sort_values(kind="mergesort").index
    X_sorted = X_valid.loc[order]
    y_sorted = y_valid.loc[order]
    ts_sorted = ts_valid.loc[order]

    n_total = len(X_sorted)
    if n_total < 2:
        raise ValueError(f"Need at least 2 clean rows to split chronologically, got {n_total}.")
    n_train = max(1, min(n_total - 1, int(n_total * float(train_frac))))

    X_train = X_sorted.iloc[:n_train].copy()
    X_test = X_sorted.iloc[n_train:].copy()
    y_train = y_sorted.iloc[:n_train].copy()
    y_test = y_sorted.iloc[n_train:].copy()

    split_meta = {
        "n_input_rows": int(len(df)),
        "n_clean_rows": int(n_total),
        "n_dropped_missing": int(len(df) - n_total),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_start": str(ts_sorted.iloc[0]),
        "train_end": str(ts_sorted.iloc[n_train - 1]),
        "test_start": str(ts_sorted.iloc[n_train]),
        "test_end": str(ts_sorted.iloc[-1]),
    }
    return X_train, X_test, y_train, y_test, split_meta


def downsample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_rows: int = 50000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Deterministically downsample `(X_train, y_train)` to at most `max_rows` rows."""
    n_avail = len(X_train)
    if n_avail <= int(max_rows):
        return X_train.copy(), y_train.copy()
    X_sub = X_train.sample(n=int(max_rows), random_state=int(random_state))
    y_sub = y_train.loc[X_sub.index]
    return X_sub, y_sub


def compute_binary_calibration(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    pos_class: str = "noise",
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute 1-vs-rest reliability curve points, Brier score, and ECE for `pos_class`
    from a fitted probabilistic classifier (`predict_proba`).
    """
    classes = list(estimator.classes_)
    if pos_class not in classes:
        raise ValueError(f"Class {pos_class!r} not found in estimator.classes_={classes}")
    pos_idx = classes.index(pos_class)

    y_bin = (np.asarray(y_test).astype(str) == str(pos_class)).astype(int)
    y_prob = np.clip(np.asarray(estimator.predict_proba(X_test)[:, pos_idx], dtype=float), 0.0, 1.0)

    prob_true, prob_pred = calibration_curve(y_bin, y_prob, n_bins=int(n_bins), strategy=strategy)
    brier = float(brier_score_loss(y_bin, y_prob))

    # Bin counts and equal-width Expected Calibration Error (ECE)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_rows: List[Dict[str, Any]] = []
    ece = 0.0
    n_total = len(y_bin)
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = ((y_prob >= lo) & (y_prob < hi)) if b < int(n_bins) - 1 else ((y_prob >= lo) & (y_prob <= hi))
        cnt = int(mask.sum())
        if cnt > 0:
            mp = float(y_prob[mask].mean())
            or_val = float(y_bin[mask].mean())
            ece += (cnt / n_total) * abs(or_val - mp)
        else:
            mp = float("nan")
            or_val = float("nan")
        bin_rows.append(
            {
                "bin": b + 1,
                "range": f"[{lo:.1f}, {hi:.1f}" + ("]" if b == int(n_bins) - 1 else ")"),
                "n": cnt,
                "mean_pred": mp,
                "obs_rate": or_val,
            }
        )

    curve_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    stats = {
        "pos_class": pos_class,
        "n_test": int(n_total),
        "pos_rate": float(y_bin.mean()) if n_total > 0 else float("nan"),
        "mean_pred": float(y_prob.mean()) if n_total > 0 else float("nan"),
        "brier": brier,
        "ece": float(ece),
        "bin_df": pd.DataFrame(bin_rows),
        "probs": y_prob,
    }
    return curve_df, stats


def run_dt_depth_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    depths: Sequence[Optional[int]] = (2, 5, 10, 20, None),
    random_state: int = 42,
    prefitted_models: Optional[Dict[Optional[int], Pipeline]] = None,
) -> pd.DataFrame:
    """
    Evaluate `DecisionTreeClassifier` train vs test accuracy and weighted F1 across `max_depth` values.
    """
    prefitted = prefitted_models or {}
    y_tr_arr = np.asarray(y_train).astype(str)
    y_te_arr = np.asarray(y_test).astype(str)

    rows: List[Dict[str, Any]] = []
    for d in depths:
        if d in prefitted:
            pipe = prefitted[d]
        else:
            pipe = build_baseline_pipeline(DecisionTreeClassifier(max_depth=d, random_state=int(random_state)))
            pipe.fit(X_train, y_tr_arr)

        pred_tr = np.asarray(pipe.predict(X_train)).astype(str)
        pred_te = np.asarray(pipe.predict(X_test)).astype(str)
        tree_step = pipe.named_steps["model"]
        rows.append(
            {
                "max_depth": d,
                "depth_label": "None" if d is None else str(d),
                "train_accuracy": float(accuracy_score(y_tr_arr, pred_tr)),
                "test_accuracy": float(accuracy_score(y_te_arr, pred_te)),
                "train_weighted_f1": float(f1_score(y_tr_arr, pred_tr, average="weighted", zero_division=0)),
                "test_weighted_f1": float(f1_score(y_te_arr, pred_te, average="weighted", zero_division=0)),
                "actual_depth": int(tree_step.get_depth()),
                "n_leaves": int(tree_step.get_n_leaves()),
            }
        )
    return pd.DataFrame(rows)


def run_knn_k_curve(
    X_train_sub: pd.DataFrame,
    y_train_sub: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
    n_jobs: int = -1,
    prefitted_models: Optional[Dict[int, Pipeline]] = None,
) -> pd.DataFrame:
    """
    Evaluate `KNeighborsClassifier` test Accuracy and weighted F1 across neighborhood sizes `k_values`.
    """
    prefitted = prefitted_models or {}
    y_tr_arr = np.asarray(y_train_sub).astype(str)
    y_te_arr = np.asarray(y_test).astype(str)

    rows: List[Dict[str, Any]] = []
    for k in k_values:
        k_int = int(k)
        if k_int in prefitted:
            pipe = prefitted[k_int]
        else:
            pipe = build_baseline_pipeline(KNeighborsClassifier(n_neighbors=k_int, n_jobs=n_jobs))
            pipe.fit(X_train_sub, y_tr_arr)

        pred_te = np.asarray(pipe.predict(X_test)).astype(str)
        rows.append(
            {
                "k": k_int,
                "accuracy": float(accuracy_score(y_te_arr, pred_te)),
                "weighted_f1": float(f1_score(y_te_arr, pred_te, average="weighted", zero_division=0)),
            }
        )
    return pd.DataFrame(rows)
