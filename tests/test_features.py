"""
Tests for Feature Engineering and Local Projection (Chunk 2.2)
==============================================================
Asserts:
- Property a: Local projection mean relative error vs. haversine is below 1%.
- Property c: Round-trip unproject(project(lat, lon)) has max error below 1e-9 degrees.
- Property e: Read-back of brooklyn_stage2.csv.gz preserves schema, counts, and non-null values.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features import (
    load_projection_constants,
    project,
    unproject,
    haversine_distance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROJECTION_JSON = PROCESSED_DIR / "projection.json"
STAGE1_GZ = PROCESSED_DIR / "brooklyn_stage1.csv.gz"
STAGE2_GZ = PROCESSED_DIR / "brooklyn_stage2.csv.gz"


@pytest.fixture(scope="module")
def projection_origin():
    assert PROJECTION_JSON.exists(), f"projection.json not found at {PROJECTION_JSON}"
    lat0, lon0 = load_projection_constants(PROJECTION_JSON)
    return lat0, lon0


@pytest.fixture(scope="module")
def stage1_df():
    assert STAGE1_GZ.exists(), f"Stage 1 dataset not found at {STAGE1_GZ}"
    df = pd.read_csv(
        STAGE1_GZ,
        usecols=["unique_key", "latitude", "longitude"],
        dtype={"unique_key": str, "latitude": float, "longitude": float},
    )
    return df


@pytest.fixture(scope="module")
def stage2_df():
    assert STAGE2_GZ.exists(), f"Stage 2 dataset not found at {STAGE2_GZ}"
    str_cols = [
        "unique_key",
        "resolution_id",
        "status",
        "channel",
        "location_type",
        "complaint_type",
        "descriptor",
        "group",
        "borough",
        "incident_address",
        "address_norm",
    ]
    df = pd.read_csv(
        STAGE2_GZ,
        dtype={c: str for c in str_cols},
        keep_default_na=False,
    )
    df["x_m"] = df["x_m"].astype(float)
    df["y_m"] = df["y_m"].astype(float)
    return df


def test_property_a_projection_accuracy(stage1_df, projection_origin):
    """
    Test 3a: Sample 2000 pairs with haversine distance > 500 m (seed 42).
    Verify that mean relative error of planar Euclidean vs. spherical haversine is < 1%.
    """
    lat0, lon0 = projection_origin
    lats = stage1_df["latitude"].values
    lons = stage1_df["longitude"].values
    n = len(stage1_df)

    xs, ys = project(lats, lons, lat0, lon0)

    rng = np.random.default_rng(42)
    pairs_idx1 = []
    pairs_idx2 = []
    while len(pairs_idx1) < 2000:
        batch = 3000
        i1 = rng.integers(0, n, size=batch)
        i2 = rng.integers(0, n, size=batch)
        valid = i1 != i2
        i1, i2 = i1[valid], i2[valid]
        h_dists = haversine_distance(lats[i1], lons[i1], lats[i2], lons[i2])
        far = h_dists > 500.0
        pairs_idx1.extend(i1[far])
        pairs_idx2.extend(i2[far])

    idx1 = np.array(pairs_idx1[:2000])
    idx2 = np.array(pairs_idx2[:2000])

    h_true = haversine_distance(lats[idx1], lons[idx1], lats[idx2], lons[idx2])
    e_proj = np.sqrt((xs[idx1] - xs[idx2]) ** 2 + (ys[idx1] - ys[idx2]) ** 2)
    rel_err = np.abs(e_proj - h_true) / h_true

    mean_rel_err = float(rel_err.mean())
    assert mean_rel_err < 0.01, f"Mean relative error {mean_rel_err:.6f} exceeded 1% (0.01)"


def test_property_c_round_trip(stage1_df, projection_origin):
    """
    Test 3c: unproject(project(lat, lon)) vs original for all 236,507 rows.
    Verify that max absolute difference in degrees is below 1e-9.
    """
    lat0, lon0 = projection_origin
    lats = stage1_df["latitude"].values
    lons = stage1_df["longitude"].values

    xs, ys = project(lats, lons, lat0, lon0)
    rec_lat, rec_lon = unproject(xs, ys, lat0, lon0)

    max_diff_lat = float(np.max(np.abs(rec_lat - lats)))
    max_diff_lon = float(np.max(np.abs(rec_lon - lons)))
    max_err = max(max_diff_lat, max_diff_lon)

    assert max_err < 1e-9, f"Max round trip error {max_err:.2e} exceeded 1e-9 degrees"


def test_property_e_readback_checks(stage2_df):
    """
    Test 3e: Read-back checks on brooklyn_stage2.csv.gz:
    - Row count equals 236,507
    - unique_key is unique
    - Count of empty resolution_id equals 3,108
    - x_m and y_m have no NaN
    """
    assert len(stage2_df) == 236507, f"Expected 236,507 rows, found {len(stage2_df)}"
    assert stage2_df["unique_key"].is_unique, "unique_key column contains duplicate values"

    empty_res_count = int((stage2_df["resolution_id"] == "").sum())
    assert empty_res_count == 3108, f"Expected 3,108 empty resolution_id, found {empty_res_count}"

    assert stage2_df["x_m"].notna().all(), "x_m column contains NaN values"
    assert stage2_df["y_m"].notna().all(), "y_m column contains NaN values"

