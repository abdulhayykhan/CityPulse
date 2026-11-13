"""
Unit tests for src/models/baselines.py (Optional Baselines: Labs 4, 8, 9)
========================================================================
Tests:
- `prepare_baseline_data` strictly extracts (hour, weekday, channel, location_type) -> `group`,
  never exposes `complaint_type` / `descriptor` (preventing 100% target leakage),
  drops NaN and empty/whitespace missing rows while keeping valid strings like `'UNKNOWN'`.
- `build_baseline_pipeline` wraps `OneHotEncoder(handle_unknown='ignore')` + estimator,
  and safely handles unseen categories at inference time.
- `evaluate_baselines` returns comparison DataFrame with `model`, `accuracy`, `weighted_f1` and no NaNs.
- `chronological_train_test_split` enforces temporal ordering without shuffling or leakage.
- Public exports in `src/models/__init__.py`.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import src.models as pkg_models
from src.models.baselines import (
    BASELINE_FEATURE_COLS,
    build_baseline_pipeline,
    chronological_train_test_split,
    downsample_training_data,
    evaluate_baselines,
    prepare_baseline_data,
)


def test_prepare_baseline_data_excludes_leakage_and_drops_missing():
    """Verify exact feature columns, zero leakage columns, and NaN / blank dropping."""
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6"],
        "hour": [22, 10, np.nan, 14, 18, 9],
        "weekday": [5, 2, 1, 3, 4, 0],
        "channel": ["ONLINE", "UNKNOWN", "PHONE", "", "MOBILE", "ONLINE"],
        "location_type": [
            "Residential Building/House",
            "Street",
            "Street/Sidewalk",
            "Sewer",
            "",  # absent location_type when loaded with keep_default_na=False
            "Sewer",
        ],
        "complaint_type": [
            "Noise - Residential",
            "Street Condition",
            "Noise - Street/Sidewalk",
            "Sewer",
            "Noise - Commercial",
            "Sewer",
        ],
        "descriptor": ["Loud Music", "Pothole", "Banging", "Catch Basin", "Loud Talking", np.nan],
        "group": ["noise", "road", "noise", "flood", "noise", "sewer_backup"],
    })

    X, y = prepare_baseline_data(df)

    # Row index 0, 1, 5 survive (even though row 5 has missing descriptor, descriptor is excluded!)
    assert list(X.columns) == BASELINE_FEATURE_COLS
    assert "complaint_type" not in X.columns
    assert "descriptor" not in X.columns
    assert list(X.index) == [0, 1, 5]
    assert X.index.equals(y.index)
    assert y.tolist() == ["noise", "road", "sewer_backup"]
    assert pd.api.types.is_integer_dtype(X["hour"])
    assert pd.api.types.is_integer_dtype(X["weekday"])
    assert "UNKNOWN" in X["channel"].values


def test_prepare_baseline_data_missing_required_column_raises():
    df_bad = pd.DataFrame({"hour": [1], "weekday": [2], "channel": ["ONLINE"]})
    with pytest.raises(ValueError, match="Missing required baseline columns"):
        prepare_baseline_data(df_bad)


def test_build_baseline_pipeline_and_unseen_categories():
    """Verify Pipeline structure and safe handling of unseen test categories."""
    train_df = pd.DataFrame({
        "hour": [23, 22, 9, 10, 14, 15],
        "weekday": [5, 6, 1, 2, 3, 4],
        "channel": ["ONLINE", "MOBILE", "UNKNOWN", "PHONE", "ONLINE", "PHONE"],
        "location_type": [
            "Residential Building/House",
            "Street/Sidewalk",
            "Street",
            "Street",
            "Sewer",
            "Sewer",
        ],
        "group": ["noise", "noise", "road", "road", "flood", "flood"],
    })
    test_df = pd.DataFrame({
        "hour": [12, 8],
        "weekday": [0, 2],
        "channel": ["BRAND_NEW_CHANNEL", "UNKNOWN"],
        "location_type": ["Unseen Location Type", "Street"],
        "group": ["noise", "road"],
    })

    X_tr, y_tr = prepare_baseline_data(train_df)
    X_te, y_te = prepare_baseline_data(test_df)

    for model_name, clf in [
        ("MultinomialNB", MultinomialNB()),
        ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=3)),
    ]:
        pipe = build_baseline_pipeline(clf)
        assert isinstance(pipe, Pipeline)
        assert isinstance(pipe.steps[0][1], OneHotEncoder)
        assert pipe.steps[0][1].handle_unknown == "ignore"
        assert pipe.steps[1][1] is clf

        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert len(preds) == len(X_te)

    fitted = {
        "nb": build_baseline_pipeline(MultinomialNB()).fit(X_tr, y_tr),
        "dt": build_baseline_pipeline(DecisionTreeClassifier(max_depth=5, random_state=42)).fit(X_tr, y_tr),
        "knn": build_baseline_pipeline(KNeighborsClassifier(n_neighbors=3)).fit(X_tr, y_tr),
    }
    comp = evaluate_baselines(fitted, X_te, y_te)
    assert list(comp.columns) == ["model", "accuracy", "weighted_f1"]
    assert len(comp) == 3
    assert not comp.isna().any().any()
    assert ((comp["accuracy"] >= 0.0) & (comp["accuracy"] <= 1.0)).all()
    assert ((comp["weighted_f1"] >= 0.0) & (comp["weighted_f1"] <= 1.0)).all()


def test_chronological_split_and_downsample_and_exports():
    """Verify unsorted timestamps are sorted before 80/20 split and package exports exist."""
    df = pd.DataFrame({
        "timestamp": [
            "2026-10-01T12:00:00",
            "2026-01-01T08:00:00",
            "2026-05-01T10:00:00",
            "2026-03-01T09:00:00",
            "2026-07-01T15:00:00",
        ],
        "hour": [12, 8, 10, 9, 15],
        "weekday": [3, 3, 4, 6, 2],
        "channel": ["ONLINE"] * 5,
        "location_type": ["Street", "Residential Building/House", "Sewer", "Street", "Sewer"],
        "group": ["road", "noise", "flood", "road", "flood"],
    })
    X_tr, X_te, y_tr, y_te, meta = chronological_train_test_split(df, train_frac=0.8)
    assert len(X_tr) == 4
    assert len(X_te) == 1
    assert meta["train_end"] == "2026-07-01 15:00:00"
    assert meta["test_start"] == "2026-10-01 12:00:00"
    assert y_te.iloc[0] == "road"

    X_sub, y_sub = downsample_training_data(X_tr, y_tr, max_rows=2, random_state=42)
    assert len(X_sub) == 2 and X_sub.index.equals(y_sub.index)

    for sym in ("prepare_baseline_data", "build_baseline_pipeline", "evaluate_baselines"):
        assert hasattr(pkg_models, sym)

