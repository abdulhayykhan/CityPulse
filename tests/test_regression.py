"""
Unit tests for src/models/regression.py (Step 5 / Lab 2)
=========================================================
Tests:
- Lag feature construction on synthetic frame (no leakage; verified by hand)
- Rolling-origin splits (train strictly before test, zero overlap)
- top2_share on a hand-built case
- predict_flood_risk returns float >= 0 and raises ValueError on negative/NaN/inf
- predict_noise range errors on invalid hour/weekday
- noise grid has exactly 24 hours per date
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

from src.models.regression import (
    build_flood_features,
    rolling_origin_splits,
    chrono_split,
    fit_eval,
    build_noise_hourly,
    add_noise_interaction_features,
    predict_flood_risk,
    predict_noise,
)


def test_flood_lag_construction_no_leakage():
    """
    Verify lag construction on a tiny synthetic frame:
    - Exactly 3 initial days dropped due to NaN lags.
    - Day t uses precip_mm[t] as rain_t.
    - rain_t1 and rain_t2 match t-1 and t-2.
    - rain_sum_t1_t3 strictly equals precip_mm[t-1] + precip_mm[t-2] + precip_mm[t-3]
      (excludes today's rain to prevent collinearity).
    """
    days = pd.date_range("2026-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    precip = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    df = pd.DataFrame({"day": days, "precip_mm": precip, "n": [0, 1, 2, 3, 4, 5]})

    feat_df = build_flood_features(df)

    # First 3 days dropped (lags for t=0,1,2 contain NaNs)
    assert len(feat_df) == 3
    assert feat_df["day"].iloc[0] == "2026-01-04"

    # Verify first remaining row (day index 3: 2026-01-04)
    # rain_t: today's rain (40.0)
    assert feat_df["rain_t"].iloc[0] == 40.0
    # rain_t1: yesterday's rain (30.0)
    assert feat_df["rain_t1"].iloc[0] == 30.0
    # rain_t2: day t-2 rain (20.0)
    assert feat_df["rain_t2"].iloc[0] == 20.0
    # rain_sum_t1_t3: 30.0 + 20.0 + 10.0 = 60.0 (strictly excludes today's 40.0)
    assert feat_df["rain_sum_t1_t3"].iloc[0] == 60.0

    # Verify second remaining row (day index 4: 2026-01-05)
    assert feat_df["rain_t"].iloc[1] == 50.0
    assert feat_df["rain_t1"].iloc[1] == 40.0
    assert feat_df["rain_t2"].iloc[1] == 30.0
    assert feat_df["rain_sum_t1_t3"].iloc[1] == 40.0 + 30.0 + 20.0  # 90.0


def test_rolling_origin_splits():
    """Verify rolling-origin splits: train is strictly before test year with zero temporal overlap."""
    days = pd.date_range("2018-01-01", "2022-12-31", freq="D").strftime("%Y-%m-%d")
    test_years = [2020, 2021, 2022]

    splits = rolling_origin_splits(days, test_years)
    assert len(splits) == 3

    for train_idx, test_idx, y in splits:
        train_days = days[train_idx]
        test_days = days[test_idx]

        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Zero overlap
        assert set(train_idx).isdisjoint(set(test_idx))
        # Strict temporal ordering: all train days strictly before year Y
        assert (train_days < f"{y}-01-01").all()
        # All test days in year Y
        assert (test_days >= f"{y}-01-01").all()
        assert (test_days <= f"{y}-12-31").all()


def test_top2_share_hand_built():
    """Verify top2_share calculation on hand-built case with known variance contribution."""
    # Test values: 4 points at 0, 1 point at 3, 1 point at 9
    # Sum = 12, N = 6, Mean = 2.0
    # Deviations from mean: [-2, -2, -2, -2, 1, 7]
    # Squared deviations: [4, 4, 4, 4, 1, 49]
    # Total sum of squared deviations = 4*4 + 1 + 49 = 16 + 1 + 49 = 66
    # Top 2 squared deviations: 49 and 4 -> sum = 53
    # Expected top2_share = (53 / 66) * 100 = 80.303030... %
    y_test = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 9.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    X_dummy = np.zeros((6, 1))

    res = fit_eval(X_dummy, y_test, X_dummy, y_test, target="count")
    expected_top2_share = (53.0 / 66.0) * 100.0
    assert pytest.approx(res["top2_share"], abs=1e-4) == expected_top2_share


def test_predict_flood_risk():
    """Verify predict_flood_risk returns float >= 0 and raises on negative, NaN, or inf."""
    # Create and save a minimal linear model bundle
    lr = LinearRegression()
    # Fit simple model: y = 2.0 * rain_t + 1.0 * rain_t1 + 5.0
    X_train = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [20, 0, 0, 0]], dtype=float)
    y_train = np.array([5.0, 25.0, 45.0], dtype=float)
    lr.fit(X_train, y_train)

    bundle = {
        "model": lr,
        "feature_names": ["rain_t", "rain_t1", "rain_t2", "rain_sum_t1_t3"],
        "target": "count",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "flood_test.joblib"
        joblib.dump(bundle, model_path)

        # Valid prediction (rain = 10mm -> 25.0)
        pred = predict_flood_risk(10.0, model_path=model_path)
        assert isinstance(pred, float)
        assert pred >= 0.0
        assert pytest.approx(pred, abs=1e-4) == 25.0

        # Zero rain -> 5.0
        pred_zero = predict_flood_risk(0.0, model_path=model_path)
        assert pytest.approx(pred_zero, abs=1e-4) == 5.0

        # Input error handling
        with pytest.raises(ValueError):
            predict_flood_risk(-5.0, model_path=model_path)
        with pytest.raises(ValueError):
            predict_flood_risk(float("nan"), model_path=model_path)
        with pytest.raises(ValueError):
            predict_flood_risk(float("inf"), model_path=model_path)


def test_predict_noise():
    """Verify predict_noise range errors on invalid hour or weekday."""
    # Create minimal noise model bundle
    feature_names = [f"hour_{h}" for h in range(1, 24)] + [f"weekday_{w}" for w in range(1, 7)]
    lr = LinearRegression()
    X_dummy = np.zeros((5, len(feature_names)))
    y_dummy = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    lr.fit(X_dummy, y_dummy)

    bundle = {
        "model": lr,
        "feature_names": feature_names,
        "target": "count",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "noise_test.joblib"
        joblib.dump(bundle, model_path)

        # Valid calls
        pred = predict_noise(hour=14, weekday=2, model_path=model_path)
        assert isinstance(pred, float)
        assert pred >= 0.0

        # Invalid hour (< 0 or > 23)
        with pytest.raises(ValueError):
            predict_noise(hour=-1, weekday=2, model_path=model_path)
        with pytest.raises(ValueError):
            predict_noise(hour=24, weekday=2, model_path=model_path)

        # Invalid weekday (< 0 or > 6)
        with pytest.raises(ValueError):
            predict_noise(hour=14, weekday=-1, model_path=model_path)
        with pytest.raises(ValueError):
            predict_noise(hour=14, weekday=7, model_path=model_path)


def test_noise_grid_has_24_hours_per_date():
    """Verify build_noise_hourly constructs exactly 24 hours per date with full zero-fill."""
    # Synthetic frame across 3 dates
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    records = []
    # Only 2 complaints on Jan 1, hour 10 and hour 14
    records.append({"group": "noise", "date": "2026-01-01", "hour": 10, "is_midnight_timestamp": False})
    records.append({"group": "noise", "date": "2026-01-01", "hour": 14, "is_midnight_timestamp": False})
    # 1 midnight complaint (should be excluded)
    records.append({"group": "noise", "date": "2026-01-02", "hour": 0, "is_midnight_timestamp": True})
    df_sample = pd.DataFrame(records)

    grid_df, meta = build_noise_hourly(df_sample, start_date="2026-01-01", end_date="2026-01-03")

    # 3 days * 24 hours = 72 rows
    assert len(grid_df) == 72
    # Exactly 24 hours per date
    counts_per_date = grid_df.groupby("date")["hour"].count()
    assert (counts_per_date == 24).all()
    # 1 midnight row excluded
    assert meta["n_excluded_midnight"] == 1
    # Check that hour 10 on Jan 1 has count 1, hour 11 has count 0
    sub_10 = grid_df[(grid_df["date"] == "2026-01-01") & (grid_df["hour"] == 10)]
    assert sub_10["count"].iloc[0] == 1
    sub_11 = grid_df[(grid_df["date"] == "2026-01-01") & (grid_df["hour"] == 11)]
    assert sub_11["count"].iloc[0] == 0


def test_noise_interaction_features_and_predict():
    """Verify hour * weekday interaction columns (23 * 6 = 138) and predict_noise interaction inference."""
    df_sample = pd.DataFrame([
        {"group": "noise", "date": "2026-01-01", "hour": 23, "is_midnight_timestamp": False},  # Thu (weekday=3)
        {"group": "noise", "date": "2026-01-03", "hour": 23, "is_midnight_timestamp": False},  # Sat (weekday=5)
    ])
    grid_df, meta = build_noise_hourly(
        df_sample,
        start_date="2026-01-01",
        end_date="2026-01-03",
        include_interactions=True,
    )
    assert len(meta["hour_features"]) == 23
    assert len(meta["weekday_features"]) == 6
    assert len(meta["interaction_features"]) == 23 * 6
    assert len(meta["all_features"]) == 167

    # Check exact cell indicator for Sat (weekday=5), hour=23
    sat_23 = grid_df[(grid_df["weekday"] == 5) & (grid_df["hour"] == 23)].iloc[0]
    assert sat_23["hour_23"] == 1.0
    assert sat_23["weekday_5"] == 1.0
    assert sat_23["hour_23_x_weekday_5"] == 1.0
    assert sat_23["hour_23_x_weekday_3"] == 0.0

    # Fit synthetic interaction model where hour=23 & weekday=5 has a +50 surge
    feature_names = meta["all_features"]
    lr = LinearRegression()
    X_syn = np.eye(len(feature_names) + 1)[:, :len(feature_names)]
    y_syn = np.full(len(feature_names) + 1, 10.0)
    # Last row is all zeros (hour=0, weekday=0 -> intercept 10.0)
    idx_h23 = feature_names.index("hour_23")
    idx_w5 = feature_names.index("weekday_5")
    idx_inter = feature_names.index("hour_23_x_weekday_5")
    y_syn[idx_h23] = 20.0   # +10 for hour 23
    y_syn[idx_w5] = 15.0    # +5 for weekday 5
    y_syn[idx_inter] = 60.0 # +50 extra for interaction (23, 5)
    lr.fit(X_syn, y_syn)

    bundle = {
        "model": lr,
        "feature_names": feature_names,
        "target": "count",
        "variant": "interaction_hour_weekday_v2",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        v2_path = Path(tmpdir) / "noise_model_v2.joblib"
        joblib.dump(bundle, v2_path)
        # Expected prediction for (23, 5) = 10 + 10 + 5 + 50 = 75.0
        pred_sat_23 = predict_noise(hour=23, weekday=5, model_path=v2_path)
        assert pytest.approx(pred_sat_23, abs=1e-4) == 75.0
        # Expected prediction for (23, 4) = 10 + 10 + 0 + 0 = 20.0 (no Sat interaction)
        pred_fri_23 = predict_noise(hour=23, weekday=4, model_path=v2_path)
        assert pytest.approx(pred_fri_23, abs=1e-4) == 20.0


