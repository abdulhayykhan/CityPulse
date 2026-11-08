"""
Unit tests for src.clustering.hierarchical_dedup
Tests anchor vs chain mode, blocking behavior, T thresholding, and empty address handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering.hierarchical_dedup import (
    weak_labels,
    make_blocks,
    cluster_block,
    cluster_all,
    make_time_chunks,
    linkage_per_chunk,
    cut_incidents,
    pair_prf,
    build_incidents,
    build_complaint_incident_map,
    online_assign,
)


@pytest.fixture
def synthetic_complaints():
    """Tiny synthetic complaints dataframe for unit tests."""
    return pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6"],
        "group": ["noise", "noise", "noise", "noise", "road", "noise"],
        "complaint_type": [
            "Noise - Residential",
            "Noise - Residential",
            "Noise - Residential",
            "Noise - Residential",
            "Pothole",
            "Noise - Street",
        ],
        "address_norm": [
            "100 MAIN ST",
            "100 MAIN ST",
            "100 MAIN ST",
            "",             # empty address
            "100 MAIN ST",
            "200 PARK AVE",
        ],
        "timestamp": [
            "2026-01-01T00:00:00",
            "2026-01-01T18:00:00",  # +18h (gap=18h, from first=18h)
            "2026-01-02T12:00:00",  # +36h (gap=18h, from first=36h)
            "2026-01-01T05:00:00",
            "2026-01-01T08:00:00",
            "2026-01-01T10:00:00",
        ],
        "date": [
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-01",
            "2026-01-01",
            "2026-01-01",
        ],
        "x_m": [100.0, 120.0, 110.0, 50.0, 100.0, 1500.0],
        "y_m": [200.0, 210.0, 205.0, 80.0, 200.0, 2500.0],
    })


def test_weak_labels_anchor_vs_chain(synthetic_complaints):
    """
    Test the fundamental difference between anchor and chain weak clustering.
    Row 0: t = 0h
    Row 1: t = 18h (gap 18h <= 24h, anchor diff 18h <= 24h) -> same cluster for both.
    Row 2: t = 36h (gap 18h <= 24h, anchor diff 36h > 24h)  -> new cluster in anchor, same in chain.
    """
    sub_df = synthetic_complaints.iloc[:3].copy()

    anchor_lbls = weak_labels(sub_df, window_hours=24, mode="anchor")
    chain_lbls = weak_labels(sub_df, window_hours=24, mode="chain")

    # In anchor mode, row 0 and 1 are cluster 0, row 2 is cluster 1
    assert anchor_lbls[0] == anchor_lbls[1]
    assert anchor_lbls[0] != anchor_lbls[2]
    assert len(np.unique(anchor_lbls)) == 2

    # In chain mode, transitive chaining groups all 3 rows into cluster 0
    assert chain_lbls[0] == chain_lbls[1]
    assert chain_lbls[1] == chain_lbls[2]
    assert len(np.unique(chain_lbls)) == 1


def test_weak_labels_empty_address(synthetic_complaints):
    """Assert rows with empty address_norm receive label -1."""
    anchor_lbls = weak_labels(synthetic_complaints, window_hours=24, mode="anchor")
    chain_lbls = weak_labels(synthetic_complaints, window_hours=24, mode="chain")

    # Row 3 has empty address_norm ("")
    assert anchor_lbls[3] == -1
    assert chain_lbls[3] == -1

    # Non-empty rows receive non-negative labels
    for i in [0, 1, 2, 4, 5]:
        assert anchor_lbls[i] >= 0
        assert chain_lbls[i] >= 0


def test_make_blocks(synthetic_complaints):
    """Assert blocking partitions rows by group, day, and 500m grid cells."""
    blocks = make_blocks(synthetic_complaints, cell_m=500.0)

    assert len(blocks) == len(synthetic_complaints)
    assert isinstance(blocks.iloc[0], tuple)
    assert len(blocks.iloc[0]) == 4  # (group, day, gx, gy)

    # Row 0 and Row 4 have same day and location, but different groups ('noise' vs 'road')
    assert blocks.iloc[0] != blocks.iloc[4]
    assert blocks.iloc[0][0] == "noise"
    assert blocks.iloc[4][0] == "road"

    # Row 0 and Row 2 have same group ('noise'), but different days ('2026-01-01' vs '2026-01-02')
    assert blocks.iloc[0] != blocks.iloc[2]
    assert blocks.iloc[0][1] == "2026-01-01"
    assert blocks.iloc[2][1] == "2026-01-02"

    # Row 0 (x=100, y=200) and Row 5 (x=1500, y=2500) are in different grid cells
    assert blocks.iloc[0][2] != blocks.iloc[5][2]
    assert blocks.iloc[0][3] != blocks.iloc[5][3]


def test_cluster_block_size_one():
    """Assert blocks of size 1 skip the model and return [0]."""
    df_one = pd.DataFrame({
        "id": ["1"],
        "x_m": [10.0],
        "y_m": [20.0],
        "timestamp": ["2026-01-01T12:00:00"],
    })
    labels = cluster_block(df_one, T=100.0)
    assert len(labels) == 1
    assert labels[0] == 0


def test_cluster_block_t_behaviour():
    """
    Test agglomerative distance threshold T behaviour on points within a single block.
    p0: (0, 0, t=0h)
    p1: (60, 0, t=0h)  -> spatial dist = 60m
    p2: (200, 0, t=0h) -> spatial dist = 200m from p0, 140m from p1
    """
    block_df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "x_m": [0.0, 60.0, 200.0],
        "y_m": [0.0, 0.0, 0.0],
        "timestamp": [
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ],
    })

    # T = 50: distance 60 > 50, all 3 points in separate clusters
    lbls_50 = cluster_block(block_df, T=50.0)
    assert len(np.unique(lbls_50)) == 3

    # T = 100: distance 60 < 100, p0 and p1 merge into 1 cluster, p2 separate
    lbls_100 = cluster_block(block_df, T=100.0)
    assert len(np.unique(lbls_100)) == 2
    assert lbls_100[0] == lbls_100[1]
    assert lbls_100[0] != lbls_100[2]

    # T = 250: average linkage merges p2 as well (avg dist between {p0, p1} and p2 is (200+140)/2 = 170 < 250)
    lbls_250 = cluster_block(block_df, T=250.0)
    assert len(np.unique(lbls_250)) == 1
    assert lbls_250[0] == lbls_250[1] == lbls_250[2]


def test_cluster_block_deterministic_order():
    """Assert cluster_block produces identical labels regardless of initial row order."""
    block_df = pd.DataFrame({
        "id": ["a", "b", "c"],
        "x_m": [10.0, 20.0, 30.0],
        "y_m": [10.0, 20.0, 30.0],
        "timestamp": [
            "2026-01-01T01:00:00",
            "2026-01-01T02:00:00",
            "2026-01-01T03:00:00",
        ],
    })
    block_permuted = block_df.iloc[[2, 0, 1]].copy()

    lbls1 = cluster_block(block_df, T=100.0)
    lbls2 = cluster_block(block_permuted, T=100.0)

    # Check that pairwise equality is preserved
    assert (lbls1[0] == lbls1[1]) == (lbls2[1] == lbls2[2])


def test_cluster_all_disjoint_and_block_sizes(synthetic_complaints):
    """Assert cluster_all produces disjoint global IDs across blocks and returns block sizes."""
    global_labels, block_sizes = cluster_all(synthetic_complaints, T=100.0, cell_m=500.0)

    assert len(global_labels) == len(synthetic_complaints)
    assert block_sizes.sum() == len(synthetic_complaints)
    assert len(np.unique(global_labels)) == len(set(global_labels))


def test_make_time_chunks_gaps_and_forced_split():
    """Test time-gap splitting and recursive max_chunk splitting in make_time_chunks."""
    # 6 complaints in same group and 500m cell
    # Timestamps spaced by: 1h, 1h, 5h, 1h, 1h
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6"],
        "group": ["noise"] * 6,
        "x_m": [10.0] * 6,
        "y_m": [10.0] * 6,
        "timestamp": [
            "2026-01-01T00:00:00",
            "2026-01-01T01:00:00",
            "2026-01-01T02:00:00",
            "2026-01-01T07:00:00",  # gap 5h
            "2026-01-01T08:00:00",
            "2026-01-01T09:00:00",
        ],
    })

    # Gap threshold 3h: splits row 0,1,2 and 3,4,5 into 2 chunks
    chunk_ids_3h, rep_3h = make_time_chunks(df, cell_m=500.0, t_gap_hours=3.0, max_chunk=10)
    assert rep_3h["n_chunks"] == 2
    assert rep_3h["n_forced_splits"] == 0
    assert rep_3h["rows_in_forced_splits"] == 0
    assert chunk_ids_3h[0] == chunk_ids_3h[1] == chunk_ids_3h[2]
    assert chunk_ids_3h[3] == chunk_ids_3h[4] == chunk_ids_3h[5]
    assert chunk_ids_3h[0] != chunk_ids_3h[3]

    # Gap threshold 10h: without max_chunk, all 6 would be in 1 chunk.
    # With max_chunk=3: must split recursively at largest internal gap (5h).
    chunk_ids_forced, rep_forced = make_time_chunks(df, cell_m=500.0, t_gap_hours=10.0, max_chunk=3)
    assert rep_forced["n_chunks"] == 2
    assert rep_forced["n_forced_splits"] == 1
    assert rep_forced["rows_in_forced_splits"] == 6
    assert chunk_ids_forced[0] == chunk_ids_forced[1] == chunk_ids_forced[2]
    assert chunk_ids_forced[3] == chunk_ids_forced[4] == chunk_ids_forced[5]
    assert chunk_ids_forced[0] != chunk_ids_forced[3]


def test_equivalence_scipy_sklearn():
    """
    Equivalence test: On 30 random chunks of size 3..200 and T in {50, 150, 450},
    scipy fcluster partition vs sklearn AgglomerativeClustering partition must match (ARI == 1.0).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score

    np.random.seed(42)
    mismatches = 0
    for i in range(30):
        size = np.random.randint(3, 201)
        pts = np.random.uniform(0, 1000, size=(size, 3))
        Z = linkage(pts, method="average")
        for T in [50.0, 150.0, 450.0]:
            scipy_lbls = fcluster(Z, t=T, criterion="distance")
            sk_model = AgglomerativeClustering(n_clusters=None, distance_threshold=T, linkage="average")
            sk_lbls = sk_model.fit_predict(pts)
            ari = adjusted_rand_score(scipy_lbls, sk_lbls)
            if ari < 1.0:
                mismatches += 1
    assert mismatches == 0


def test_pair_prf_hand_counted():
    """Test pair_prf on tiny frames with hand-counted true and predicted pairs."""
    # Case 1: 5 rows
    # true: 0, 0, 0, 1, 1 -> true pairs: (0,1), (0,2), (1,2) in cluster 0 (3 pairs); (3,4) in cluster 1 (1 pair). Total true = 4.
    # pred: 0, 0, 1, 1, 1 -> pred pairs: (0,1) in pred 0 (1 pair); (2,3), (2,4), (3,4) in pred 1 (3 pairs). Total pred = 4.
    # TP: (0,1) in true 0 and pred 0; (3,4) in true 1 and pred 1. TP = 2.
    # FP = 4 - 2 = 2.
    # FN = 4 - 2 = 2.
    # Precision = 2/4 = 0.5. Recall = 2/4 = 0.5. F1 = 0.5.
    true1 = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    pred1 = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    res1 = pair_prf(pred1, true1)
    assert res1["tp"] == 2
    assert res1["fp"] == 2
    assert res1["fn"] == 2
    assert pytest.approx(res1["precision"]) == 0.5
    assert pytest.approx(res1["recall"]) == 0.5
    assert pytest.approx(res1["f1"]) == 0.5

    # Case 2: true has label -1 (excluded)
    # true: [0, 0, -1, 1, 1]
    # pred: [0, 0, 0, 1, 1]
    # Valid indices: 0, 1, 3, 4.
    # true pairs: (0,1) and (3,4) -> 2 pairs.
    # pred pairs: (0,1) and (3,4) -> 2 pairs.
    # TP = 2, FP = 0, FN = 0. Precision = 1.0, Recall = 1.0, F1 = 1.0.
    true2 = np.array([0, 0, -1, 1, 1], dtype=np.int64)
    pred2 = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    res2 = pair_prf(pred2, true2)
    assert res2["tp"] == 2
    assert res2["fp"] == 0
    assert res2["fn"] == 0
    assert pytest.approx(res2["precision"]) == 1.0
    assert pytest.approx(res2["recall"]) == 1.0
    assert pytest.approx(res2["f1"]) == 1.0


def test_build_incidents_formatting_and_ties():
    """Test build_incidents deterministic INC-%06d IDs, tie-breaking for zone_id, and unprojection."""
    df = pd.DataFrame({
        "id": ["10", "20", "30", "40"],
        "timestamp": [
            "2026-01-01T01:00:00",
            "2026-01-01T02:00:00",
            "2026-01-01T00:30:00",
            "2026-01-01T00:30:00",
        ],
        "group": ["noise", "noise", "road", "road"],
        "x_m": [100.0, 200.0, 300.0, 400.0],
        "y_m": [500.0, 600.0, 700.0, 800.0],
        "zone_id": [3, 1, 2, 2],
    })
    # Cluster 0: rows 0, 1 (noise, zone_ids 3 and 1 -> tie broken to smallest -> 1)
    # Cluster 1: rows 2, 3 (road, zone_ids 2 and 2 -> 2)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    incidents, mapping = build_incidents(df, labels, lat0=40.661641, lon0=-73.952455, return_map=True)

    assert len(incidents) == 2
    # Check column names
    assert list(incidents.columns) == [
        "incident_id",
        "lat",
        "lon",
        "count",
        "group",
        "first_timestamp",
        "last_timestamp",
        "zone_id",
    ]
    # Check deterministic numbering: Cluster 1 has first_timestamp 00:30:00 < Cluster 0 (01:00:00)
    # So Cluster 1 gets INC-000000, Cluster 0 gets INC-000001
    assert incidents["incident_id"].tolist() == ["INC-000000", "INC-000001"]
    assert incidents["group"].tolist() == ["road", "noise"]
    # Check zone_id for noise incident (tie between 3 and 1 broken to smallest -> 1)
    noise_inc = incidents[incidents["group"] == "noise"].iloc[0]
    assert noise_inc["zone_id"] == 1

    # Check mapping
    assert len(mapping) == 4
    assert list(mapping.columns) == ["id", "incident_id"]
    assert mapping["id"].tolist() == ["10", "20", "30", "40"]
    assert mapping.loc[mapping["id"] == "10", "incident_id"].iloc[0] == "INC-000001"
    assert mapping.loc[mapping["id"] == "30", "incident_id"].iloc[0] == "INC-000000"


def test_online_assign():
    """Test online_assign real-time assignment logic."""
    incidents = pd.DataFrame({
        "incident_id": ["INC-000000", "INC-000001"],
        "group": ["road", "noise"],
        "x_m": [100.0, 500.0],
        "y_m": [100.0, 500.0],
        "first_timestamp": ["2026-01-01T10:00:00", "2026-01-01T10:00:00"],
        "last_timestamp": ["2026-01-01T14:00:00", "2026-01-01T14:00:00"],
    })

    # Exact location, inside interval (dt=0) -> distance = 0 <= 100 -> assigns to INC-000000
    assigned = online_assign(
        x_m=100.0,
        y_m=100.0,
        t="2026-01-01T12:00:00",
        group="road",
        incidents_xy=incidents,
        time_scale=50.0,
        T=100.0,
    )
    assert assigned == "INC-000000"

    # Outside time interval by 1h: dt=1h -> time penalty = 50 * 1 = 50m.
    # dx = 30m, dy = 0m -> total dist = sqrt(30^2 + 50^2) = sqrt(3400) = 58.3m <= 100m -> assigned
    assigned_outside = online_assign(
        x_m=130.0,
        y_m=100.0,
        t="2026-01-01T15:00:00",  # 1h after last_timestamp
        group="road",
        incidents_xy=incidents,
        time_scale=50.0,
        T=100.0,
    )
    assert assigned_outside == "INC-000000"

    # Far away (distance > T) -> None
    assigned_far = online_assign(
        x_m=1000.0,
        y_m=1000.0,
        t="2026-01-01T12:00:00",
        group="road",
        incidents_xy=incidents,
        time_scale=50.0,
        T=100.0,
    )
    assert assigned_far is None

    # Different group ('flood' not in incidents) -> None
    assigned_diff_grp = online_assign(
        x_m=100.0,
        y_m=100.0,
        t="2026-01-01T12:00:00",
        group="flood",
        incidents_xy=incidents,
        time_scale=50.0,
        T=100.0,
    )
    assert assigned_diff_grp is None


