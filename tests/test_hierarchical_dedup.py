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

