"""
Tests for complaints_clean.csv dataset integrity and schema compliance.
Part 3, Step 2, Chunk 2.3.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = REPO_ROOT / "data" / "processed" / "complaints_clean.csv"
COLUMNS_CSV = REPO_ROOT / "data" / "processed" / "complaints_clean_columns.csv"

EXPECTED_COLUMNS = [
    "id",
    "lat",
    "lon",
    "timestamp",
    "group",
    "descriptor",
    "zone_id",
    "descriptor_group",
    "complaint_type",
    "borough",
    "status",
    "closed_date",
    "time_to_close_h",
    "hour",
    "weekday",
    "month",
    "date",
    "is_midnight_timestamp",
    "channel",
    "location_type",
    "incident_address",
    "address_norm",
    "resolution_id",
    "x_m",
    "y_m",
]

EXPECTED_FIRST_7 = [
    "id",
    "lat",
    "lon",
    "timestamp",
    "group",
    "descriptor",
    "zone_id",
]

ALLOWED_GROUPS = {"noise", "road", "flood", "sewer_backup"}


@pytest.fixture(scope="module")
def clean_df():
    """Load complaints_clean.csv with dtype=str for id and resolution_id."""
    assert CLEAN_CSV.exists(), f"File not found: {CLEAN_CSV}"
    df = pd.read_csv(
        CLEAN_CSV,
        dtype={"id": str, "resolution_id": str},
        keep_default_na=False,
    )
    return df


def test_column_list_and_order(clean_df):
    """Assert column list and order exactly as specified (25 columns)."""
    assert list(clean_df.columns) == EXPECTED_COLUMNS, (
        f"Columns do not match expected list or order.\n"
        f"Actual: {list(clean_df.columns)}\n"
        f"Expected: {EXPECTED_COLUMNS}"
    )


def test_first_seven_columns(clean_df):
    """Assert the first 7 columns are exactly id, lat, lon, timestamp, group, descriptor, zone_id."""
    assert list(clean_df.columns[:7]) == EXPECTED_FIRST_7, (
        f"First 7 columns do not match.\n"
        f"Actual: {list(clean_df.columns[:7])}\n"
        f"Expected: {EXPECTED_FIRST_7}"
    )


def test_row_count(clean_df):
    """Assert row count is exactly 236,507 rows."""
    assert len(clean_df) == 236507, f"Expected 236,507 rows, got {len(clean_df)}"


def test_id_uniqueness(clean_df):
    """Assert id is unique across all rows."""
    assert clean_df["id"].nunique() == 236507, (
        f"Expected 236,507 unique IDs, got {clean_df['id'].nunique()}"
    )


def test_no_nans_in_required_fields(clean_df):
    """Assert no NaN in id, lat, lon, timestamp, group, descriptor, descriptor_group, x_m, y_m."""
    critical_cols = [
        "id",
        "lat",
        "lon",
        "timestamp",
        "group",
        "descriptor",
        "descriptor_group",
        "x_m",
        "y_m",
    ]
    for col in critical_cols:
        # Check both pandas nulls and empty strings for these mandatory fields
        if col in ["lat", "lon", "x_m", "y_m"]:
            # Ensure float parsing works and no NaNs
            val_float = pd.to_numeric(clean_df[col], errors="coerce")
            nan_cnt = val_float.isna().sum()
            assert nan_cnt == 0, f"Column {col} contains {nan_cnt} NaN values"
        else:
            nan_cnt = (clean_df[col].isna() | (clean_df[col] == "")).sum()
            assert nan_cnt == 0, f"Column {col} contains {nan_cnt} missing/empty values"


def test_group_values_subset(clean_df):
    """Assert group values are a subset of {noise, road, flood, sewer_backup}."""
    unique_groups = set(clean_df["group"].unique())
    assert unique_groups.issubset(ALLOWED_GROUPS), (
        f"Disallowed group values found: {unique_groups - ALLOWED_GROUPS}"
    )


def test_zone_id_valid(clean_df):
    """Assert zone_id has int dtype, no NaN, and set of values is {0..7}."""
    assert pd.api.types.is_integer_dtype(clean_df["zone_id"]), (
        f"Expected int dtype for zone_id, got {clean_df['zone_id'].dtype}"
    )
    assert not clean_df["zone_id"].isna().any(), "zone_id contains NaN values"
    assert set(clean_df["zone_id"].unique()) == set(range(8)), (
        f"Expected zone_id values to be {{0..7}}, got {set(clean_df['zone_id'].unique())}"
    )


def test_resolution_id_empty_count(clean_df):
    """Assert empty resolution_id count equals 3,108."""
    empty_cnt = (clean_df["resolution_id"] == "").sum()
    assert empty_cnt == 3108, (
        f"Expected exactly 3,108 empty resolution_id values, got {empty_cnt}"
    )


def test_timestamp_iso_regex(clean_df):
    r"""Assert timestamp matches ^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$ for every row."""
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
    matches = clean_df["timestamp"].str.match(pattern)
    non_matches = (~matches).sum()
    assert non_matches == 0, f"Found {non_matches} timestamps not matching ISO regex"


def test_columns_metadata_file():
    """Assert complaints_clean_columns.csv exists and has correct schema and annotations."""
    assert COLUMNS_CSV.exists(), f"File not found: {COLUMNS_CSV}"
    df_cols = pd.read_csv(COLUMNS_CSV)
    assert list(df_cols.columns) == [
        "column",
        "dtype",
        "description",
        "safe_as_trust_feature",
        "notes",
    ]
    assert len(df_cols) == 25
    assert list(df_cols["column"]) == EXPECTED_COLUMNS

    # Check leakages
    leak_cols = ["closed_date", "time_to_close_h", "status", "resolution_id"]
    for col in leak_cols:
        row = df_cols[df_cols["column"] == col].iloc[0]
        assert row["safe_as_trust_feature"] == "no"
        assert "label leakage" in row["notes"]

    # Check zone_id note
    zone_row = df_cols[df_cols["column"] == "zone_id"].iloc[0]
    assert "empty until K-Means step" in zone_row["notes"]
