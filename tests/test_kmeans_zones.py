"""
Tests for K-Means Spatial Zones (Chunk 3.1)
===========================================
Unit tests on synthetic data for:
- distinct_locations aggregation
- build_weights under raw, capped, and capped_balanced variants
- fit_kmeans wrapper
- weighted_silhouette_sample calculation
- zone_table construction, lift computation, and dominant_group identification
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering.kmeans_zones import (
    distinct_locations,
    build_weights,
    fit_kmeans,
    weighted_silhouette_sample,
    zone_table,
)
from src.preprocessing.features import project, unproject


@pytest.fixture
def synthetic_complaints():
    """Tiny synthetic dataframe with known coordinates and groups."""
    data = [
        # Group A at coord 1 (2 rows)
        {"group": "group_a", "x_m": 100.0, "y_m": 200.0},
        {"group": "group_a", "x_m": 100.0, "y_m": 200.0},
        # Group A at coord 2 (1 row)
        {"group": "group_a", "x_m": 300.0, "y_m": 400.0},
        # Group B at coord 1 (3 rows)
        {"group": "group_b", "x_m": 100.0, "y_m": 200.0},
        {"group": "group_b", "x_m": 100.0, "y_m": 200.0},
        {"group": "group_b", "x_m": 100.0, "y_m": 200.0},
        # Group B at coord 3 (1 row)
        {"group": "group_b", "x_m": 500.0, "y_m": 600.0},
    ]
    return pd.DataFrame(data)


def test_distinct_locations(synthetic_complaints):
    locs = distinct_locations(synthetic_complaints)
    assert len(locs) == 4, f"Expected 4 distinct locations, got {len(locs)}"
    assert set(locs.columns) == {"group", "x_m", "y_m", "count"}

    # Verify counts
    row_a1 = locs[(locs["group"] == "group_a") & (locs["x_m"] == 100.0)]
    assert row_a1["count"].iloc[0] == 2

    row_b1 = locs[(locs["group"] == "group_b") & (locs["x_m"] == 100.0)]
    assert row_b1["count"].iloc[0] == 3

    assert locs["count"].sum() == len(synthetic_complaints)


def test_build_weights_raw(synthetic_complaints):
    locs = distinct_locations(synthetic_complaints)
    w_raw = build_weights(locs, "raw")

    assert len(w_raw) == len(locs)
    assert np.allclose(w_raw, locs["count"].values)
    assert np.isclose(w_raw.sum(), len(synthetic_complaints))


def test_build_weights_capped():
    # Frame with extreme hotspot in group A
    locs = pd.DataFrame([
        {"group": "A", "x_m": 0.0, "y_m": 0.0, "count": 1},
        {"group": "A", "x_m": 1.0, "y_m": 1.0, "count": 1},
        {"group": "A", "x_m": 2.0, "y_m": 2.0, "count": 1},
        {"group": "A", "x_m": 3.0, "y_m": 3.0, "count": 100},  # Outlier
        {"group": "B", "x_m": 10.0, "y_m": 10.0, "count": 5},
    ])
    w_capped = build_weights(locs, "capped")
    p99_A = float(locs[locs["group"] == "A"]["count"].quantile(0.99))

    assert w_capped[3] == pytest.approx(p99_A)
    assert w_capped[3] < 100.0
    assert w_capped[4] == 5.0


def test_build_weights_capped_balanced():
    locs = pd.DataFrame([
        {"group": "A", "x_m": 0.0, "y_m": 0.0, "count": 10},
        {"group": "A", "x_m": 1.0, "y_m": 1.0, "count": 20},
        {"group": "B", "x_m": 10.0, "y_m": 10.0, "count": 100},
        {"group": "B", "x_m": 11.0, "y_m": 11.0, "count": 200},
    ])
    w_bal = build_weights(locs, "capped_balanced")

    sum_A = w_bal[locs["group"] == "A"].sum()
    sum_B = w_bal[locs["group"] == "B"].sum()

    assert np.isclose(sum_A, sum_B, atol=1e-6)
    w_capped = build_weights(locs, "capped")
    assert np.isclose(w_bal.sum(), w_capped.sum(), atol=1e-6)


def test_build_weights_invalid_variant():
    locs = pd.DataFrame([{"group": "A", "x_m": 0.0, "y_m": 0.0, "count": 1}])
    with pytest.raises(ValueError, match="Unknown variant"):
        build_weights(locs, "unsupported_variant")


def test_fit_kmeans():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    km = fit_kmeans(X, w, k=2, seed=42)

    assert km.n_clusters == 2
    assert len(km.labels_) == 4
    assert km.labels_[0] == km.labels_[1]
    assert km.labels_[2] == km.labels_[3]
    assert km.labels_[0] != km.labels_[2]


def test_weighted_silhouette_sample():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.0, 10.1]])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    score = weighted_silhouette_sample(X, w, labels, n=20, seed=42)
    assert isinstance(score, float)
    assert 0.8 < score <= 1.0


def test_zone_table_and_lift():
    lat0, lon0 = 40.66, -73.95
    # Two well-separated clusters
    # Cluster 0: mainly Group A
    # Cluster 1: mainly Group B
    locs = pd.DataFrame([
        {"group": "A", "x_m": 0.0, "y_m": 0.0, "count": 80},
        {"group": "B", "x_m": 0.0, "y_m": 0.0, "count": 20},
        {"group": "A", "x_m": 5000.0, "y_m": 5000.0, "count": 20},
        {"group": "B", "x_m": 5000.0, "y_m": 5000.0, "count": 80},
    ])
    labels = np.array([0, 0, 1, 1])
    centers = np.array([[0.0, 0.0], [5000.0, 5000.0]])

    zt = zone_table(locs, labels, centers, lat0=lat0, lon0=lon0)

    assert len(zt) == 2
    assert "zone" in zt.columns
    assert "centre_lat" in zt.columns
    assert "centre_lon" in zt.columns
    assert "dominant_group" in zt.columns

    # City-wide totals: A = 100, B = 100, total = 200.
    # City-wide share: A = 0.5, B = 0.5.
    # Zone 0: total = 100. count_A = 80 (share=0.8), count_B = 20 (share=0.2).
    # Lift in Zone 0: A = 0.8 / 0.5 = 1.6, B = 0.2 / 0.5 = 0.4.
    assert zt.loc[0, "count_A"] == 80
    assert zt.loc[0, "share_A"] == pytest.approx(0.8)
    assert zt.loc[0, "lift_A"] == pytest.approx(1.6)
    assert zt.loc[0, "dominant_group"] == "A"

    # Zone 1: total = 100. count_A = 20 (share=0.2), count_B = 80 (share=0.8).
    # Lift in Zone 1: A = 0.2 / 0.5 = 0.4, B = 0.8 / 0.5 = 1.6.
    assert zt.loc[1, "count_B"] == 80
    assert zt.loc[1, "share_B"] == pytest.approx(0.8)
    assert zt.loc[1, "lift_B"] == pytest.approx(1.6)
    assert zt.loc[1, "dominant_group"] == "B"

