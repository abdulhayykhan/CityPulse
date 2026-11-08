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
    davies_bouldin_sample,
    relabel_by_latitude,
    assign_zones,
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
    assert "radius_m_p90" in zt.columns
    assert "dominant_lift" in zt.columns


def test_davies_bouldin_sample():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.0, 10.1]])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    score = davies_bouldin_sample(X, w, labels, n=20, seed=42)
    assert isinstance(score, float)
    assert score >= 0.0


def test_relabel_by_latitude():
    # 4 centers with varying lat and lon
    centers_latlon = np.array([
        [40.5, -73.9],  # Old 0 -> lowest lat (South) -> becomes zone 3
        [40.8, -73.9],  # Old 1 -> highest lat (North) -> becomes zone 0
        [40.7, -73.8],  # Old 2 -> lat 40.7, lon -73.8 -> becomes zone 2
        [40.7, -74.0],  # Old 3 -> lat 40.7, lon -74.0 -> becomes zone 1 (ties broken by lon asc)
    ])
    labels = np.array([0, 1, 2, 3, 1, 0])

    new_labels, order = relabel_by_latitude(centers_latlon, labels, return_order=True)

    # Order should be: Old 1, Old 3, Old 2, Old 0
    assert list(order) == [1, 3, 2, 0]
    # Mapping: 1 -> 0, 3 -> 1, 2 -> 2, 0 -> 3
    # Original labels [0, 1, 2, 3, 1, 0] -> [3, 0, 2, 1, 0, 3]
    assert list(new_labels) == [3, 0, 2, 1, 0, 3]

    # Verify monotonic non-increasing latitude order
    sorted_centers = centers_latlon[order]
    for i in range(len(sorted_centers) - 1):
        assert sorted_centers[i, 0] >= sorted_centers[i + 1, 0]


def test_assign_zones():
    centers_xy = np.array([
        [0.0, 0.0],      # Center 0
        [1000.0, 0.0],   # Center 1
        [0.0, 1000.0],   # Center 2
    ])
    x = np.array([10.0, 990.0, 5.0, 1050.0])
    y = np.array([10.0, 10.0, 995.0, -10.0])

    assigned = assign_zones(x, y, centers_xy)
    assert list(assigned) == [0, 1, 2, 1]


def test_zone_table_mixed_rule():
    lat0, lon0 = 40.66, -73.95
    # Two clusters:
    # Zone 0: noise 91, flood 9
    # Zone 1: noise 89, flood 11
    # Citywide: noise 180 (90%), flood 20 (10%)
    # Zone 0: noise lift = 0.91 / 0.90 = 1.011, flood lift = 0.09 / 0.10 = 0.90
    # Both lifts < 1.15 -> dominant_group MUST be "mixed"
    #
    # Zone 2: flood 40, noise 10 -> flood share = 40/50 = 0.80
    # Citywide total with Zone 2:
    # noise: 180 + 10 = 190 / 250 = 0.76
    # flood: 20 + 40 = 60 / 250 = 0.24
    # In Zone 2: flood lift = 0.80 / 0.24 ≈ 3.33 >= 1.15 -> dominant_group "flood"
    locs = pd.DataFrame([
        # Zone 0
        {"group": "noise", "x_m": 0.0, "y_m": 0.0, "count": 91},
        {"group": "flood", "x_m": 0.0, "y_m": 0.0, "count": 9},
        # Zone 1 (flood specialized)
        {"group": "noise", "x_m": 5000.0, "y_m": 5000.0, "count": 10},
        {"group": "flood", "x_m": 5000.0, "y_m": 5000.0, "count": 90},
    ])
    labels = np.array([0, 0, 1, 1])
    centers = np.array([[0.0, 0.0], [5000.0, 5000.0]])

    zt = zone_table(locs, labels, centers, lat0=lat0, lon0=lon0)

    # City totals: noise 101/200 = 0.505, flood 99/200 = 0.495
    # Zone 0: noise count = 91 (share=0.91), lift = 0.91 / 0.505 ≈ 1.802 > 1.15 -> "noise"
    # Zone 1: flood count = 90 (share=0.90), lift = 0.90 / 0.495 ≈ 1.818 > 1.15 -> "flood"
    assert zt.loc[0, "dominant_group"] == "noise"
    assert zt.loc[1, "dominant_group"] == "flood"

    # Now test case where max lift is < 1.15
    locs_balanced = pd.DataFrame([
        {"group": "noise", "x_m": 0.0, "y_m": 0.0, "count": 52},
        {"group": "flood", "x_m": 0.0, "y_m": 0.0, "count": 48},
        {"group": "noise", "x_m": 5000.0, "y_m": 5000.0, "count": 48},
        {"group": "flood", "x_m": 5000.0, "y_m": 5000.0, "count": 52},
    ])
    # Citywide shares: noise = 100/200 = 0.5, flood = 100/200 = 0.5
    # Zone 0: noise share = 52/100 = 0.52 -> lift = 0.52 / 0.50 = 1.04 (< 1.15)
    # Zone 1: flood share = 52/100 = 0.52 -> lift = 0.52 / 0.50 = 1.04 (< 1.15)
    zt_bal = zone_table(locs_balanced, labels, centers, lat0=lat0, lon0=lon0)
    assert zt_bal.loc[0, "dominant_lift"] == pytest.approx(1.04)
    assert zt_bal.loc[0, "dominant_group"] == "mixed"
    assert zt_bal.loc[1, "dominant_lift"] == pytest.approx(1.04)
    assert zt_bal.loc[1, "dominant_group"] == "mixed"


