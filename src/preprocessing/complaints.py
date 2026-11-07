"""
CityPulse - Part 3, Step 2, Chunk 2.1: Complaint Data Cleaning for Brooklyn
==========================================================================
Provides:
- load_extract(path): Load and parse the window extract CSV.
- compute_missing_table(df): Compute missing value statistics per column.
- clean_complaints(df, dropped_path): Clean complaints according to the 7-step protocol.
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORT_TABLES_DIR = PROCESSED_DIR / "report_tables"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_GZ = PROCESSED_DIR / "complaints_window_all.csv.gz"
OUT_STAGE1_GZ = PROCESSED_DIR / "brooklyn_stage1.csv.gz"
DROPPED_CSV = PROCESSED_DIR / "dropped_outside_area.csv"
STEP_TABLE_CSV = REPORT_TABLES_DIR / "step_table.csv"
MISSING_BEFORE_CSV = REPORT_TABLES_DIR / "missing_before.csv"
MISSING_AFTER_CSV = REPORT_TABLES_DIR / "missing_after.csv"


def load_extract(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read the extracted gzip CSV.
    - unique_key as string.
    - Parse created_date and closed_date with pd.to_datetime(errors='coerce').
    - Convert latitude and longitude to float.
    - Print the count of unparseable created_date and closed_date values
      (separating empty from unparseable).
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        dtype={
            "unique_key": str,
            "complaint_type": str,
            "descriptor": str,
            "group": str,
            "status": str,
            "resolution_id": str,
            "borough": str,
            "incident_address": str,
            "location_type": str,
            "channel": str,
            "created_date": str,
            "closed_date": str,
            "latitude": str,
            "longitude": str,
        },
    )

    # Check created_date: empty vs unparseable
    raw_created = df["created_date"].fillna("").astype(str).str.strip()
    empty_created = int((raw_created == "").sum())
    parsed_created = pd.to_datetime(raw_created, errors="coerce")
    unparseable_created = int(((raw_created != "") & parsed_created.isna()).sum())

    # Check closed_date: empty vs unparseable
    raw_closed = df["closed_date"].fillna("").astype(str).str.strip()
    empty_closed = int((raw_closed == "").sum())
    parsed_closed = pd.to_datetime(raw_closed, errors="coerce")
    unparseable_closed = int(((raw_closed != "") & parsed_closed.isna()).sum())

    print(f"created_date parsing: {empty_created:,} empty, {unparseable_created:,} unparseable")
    print(f"closed_date parsing : {empty_closed:,} empty, {unparseable_closed:,} unparseable")

    df["created_date"] = parsed_created
    df["closed_date"] = parsed_closed
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype(float)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype(float)

    return df


def compute_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count and percent missing per column.
    Treat empty strings, whitespace-only strings, NaNs, and NaTs as missing.
    """
    n_rows = len(df)
    records = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s):
            n_missing = int(s.isna().sum())
        else:
            n_missing = int(
                (s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.strip() == "nan") | (s.astype(str).str.strip() == "None")).sum()
            )
        pct_missing = round((n_missing / n_rows * 100), 4) if n_rows > 0 else 0.0
        records.append({
            "column": col,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
        })
    return pd.DataFrame(records)


def clean_complaints(
    df: pd.DataFrame,
    dropped_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean complaints according to the 7-step sequence:
    1. all rows (baseline)
    2. keep borough == 'BROOKLYN'
    3. drop duplicate unique_key
    4. drop rows with missing or zero latitude or longitude
    5. drop rows outside the box lat 40.55 to 40.74, lon -74.06 to -73.83;
       write dropped rows to dropped_outside_area.csv
    6. where closed_date < created_date set closed_date to NaT; count them; do NOT drop rows
    7. create address_norm = upper-case, whitespace-stripped incident_address, empty string if missing

    Returns:
        (clean_df, step_table)
    """
    if dropped_path is None:
        dropped_path = PROCESSED_DIR / "dropped_outside_area.csv"
    else:
        dropped_path = Path(dropped_path)

    steps = []

    # Step 1: Baseline
    n1 = len(df)
    steps.append({
        "step": 1,
        "description": "all rows (baseline)",
        "rows_before": n1,
        "rows_removed": 0,
        "rows_after": n1,
    })

    # Step 2: Keep borough == 'BROOKLYN'
    df_step2 = df[df["borough"].fillna("").astype(str).str.strip().str.upper() == "BROOKLYN"].copy()
    n2 = len(df_step2)
    steps.append({
        "step": 2,
        "description": "keep borough == 'BROOKLYN'",
        "rows_before": n1,
        "rows_removed": n1 - n2,
        "rows_after": n2,
    })

    # Step 3: Drop duplicate unique_key
    df_step3 = df_step2.drop_duplicates(subset=["unique_key"], keep="first").copy()
    n3 = len(df_step3)
    steps.append({
        "step": 3,
        "description": "drop duplicate unique_key",
        "rows_before": n2,
        "rows_removed": n2 - n3,
        "rows_after": n3,
    })

    # Step 4: Drop rows with missing or zero latitude or longitude
    valid_coords = (
        df_step3["latitude"].notna()
        & df_step3["longitude"].notna()
        & (df_step3["latitude"] != 0.0)
        & (df_step3["longitude"] != 0.0)
    )
    df_step4 = df_step3[valid_coords].copy()
    n4 = len(df_step4)
    steps.append({
        "step": 4,
        "description": "drop rows with missing or zero latitude or longitude",
        "rows_before": n3,
        "rows_removed": n3 - n4,
        "rows_after": n4,
    })

    # Step 5: Coordinate percentile check & box filtering
    lat = df_step4["latitude"]
    lon = df_step4["longitude"]

    lat_min = float(lat.min())
    lat_p01 = float(lat.quantile(0.001))
    lat_p999 = float(lat.quantile(0.999))
    lat_max = float(lat.max())

    lon_min = float(lon.min())
    lon_p01 = float(lon.quantile(0.001))
    lon_p999 = float(lon.quantile(0.999))
    lon_max = float(lon.max())

    print("\n--- Coordinate Percentiles for Brooklyn Rows (before Step 5) ---")
    print(f"Latitude  : min={lat_min:.6f}, 0.1%={lat_p01:.6f}, 99.9%={lat_p999:.6f}, max={lat_max:.6f}")
    print(f"Longitude : min={lon_min:.6f}, 0.1%={lon_p01:.6f}, 99.9%={lon_p999:.6f}, max={lon_max:.6f}")

    outside_mask = (
        (lat < 40.55) | (lat > 40.74) | (lon < -74.06) | (lon > -73.83)
    )
    n_outside = int(outside_mask.sum())
    pct_outside = (n_outside / n4 * 100) if n4 > 0 else 0.0
    print(f"Rows outside box [lat 40.55 to 40.74, lon -74.06 to -73.83]: {n_outside:,} ({pct_outside:.4f}%)")

    if n4 > 0 and (n_outside / n4) > 0.01:
        msg = f"Step 5 would remove {n_outside:,} rows ({pct_outside:.2f}%), exceeding 1% limit. Stopping."
        print(f"STOPPING: {msg}")
        raise RuntimeError(msg)

    # Save dropped rows (even if 0, write CSV to document)
    dropped_df = df_step4[outside_mask].copy()
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    # When saving dropped rows, convert datetimes to ISO strings
    dropped_save = dropped_df.copy()
    dropped_save["created_date"] = dropped_save["created_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    dropped_save["closed_date"] = dropped_save["closed_date"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    dropped_save.to_csv(dropped_path, index=False)
    print(f"Saved {len(dropped_df):,} dropped rows outside area to: {dropped_path}")

    df_step5 = df_step4[~outside_mask].copy()
    n5 = len(df_step5)
    steps.append({
        "step": 5,
        "description": "drop rows outside the box lat 40.55 to 40.74, lon -74.06 to -73.83",
        "rows_before": n4,
        "rows_removed": n4 - n5,
        "rows_after": n5,
    })

    # Step 6: Where closed_date < created_date set closed_date to NaT; count them; do NOT drop rows
    df_step6 = df_step5.copy()
    bad_closed = (df_step6["closed_date"].notna()) & (df_step6["closed_date"] < df_step6["created_date"])
    n_bad_closed = int(bad_closed.sum())
    df_step6.loc[bad_closed, "closed_date"] = pd.NaT
    print(f"\nStep 6: Found {n_bad_closed:,} rows where closed_date < created_date; set closed_date to NaT (0 rows dropped).")
    n6 = len(df_step6)
    steps.append({
        "step": 6,
        "description": f"where closed_date < created_date set closed_date to NaT (updated {n_bad_closed:,} rows)",
        "rows_before": n5,
        "rows_removed": 0,
        "rows_after": n6,
    })

    # Step 7: Create address_norm
    df_step7 = df_step6.copy()
    df_step7["address_norm"] = df_step7["incident_address"].fillna("").astype(str).str.strip().str.upper()
    n7 = len(df_step7)
    steps.append({
        "step": 7,
        "description": "create address_norm = upper-case, whitespace-stripped incident_address, empty string if missing",
        "rows_before": n6,
        "rows_removed": 0,
        "rows_after": n7,
    })

    step_table = pd.DataFrame(steps)
    return df_step7, step_table


def main():
    print("=" * 70)
    print("STEP 2, CHUNK 2.1: BROOKLYN COMPLAINTS CLEANING")
    print("=" * 70)
    print(f"Input file: {INPUT_GZ}")

    # TASK 1: Load and parse
    print("\n[TASK 1] Loading and parsing extract...")
    df = load_extract(INPUT_GZ)
    print(f"Total rows loaded: {len(df):,}")

    # Initial borough counts
    borough_counts = df["borough"].value_counts()
    print("\nBorough value counts before filtering:")
    print(borough_counts.to_string())

    # Brooklyn rows before step 3 for missing_before.csv
    df_bk_raw = df[df["borough"].fillna("").astype(str).str.strip().str.upper() == "BROOKLYN"].copy()
    missing_before = compute_missing_table(df_bk_raw)
    missing_before.to_csv(MISSING_BEFORE_CSV, index=False)
    print(f"\nSaved missing_before.csv to: {MISSING_BEFORE_CSV} ({len(missing_before)} columns)")

    # TASK 2: Clean complaints
    print("\n[TASK 2] Applying 7-step cleaning sequence...")
    clean_df, step_table = clean_complaints(df, dropped_path=DROPPED_CSV)

    # Save step table
    step_table.to_csv(STEP_TABLE_CSV, index=False)
    print(f"Saved step_table.csv to: {STEP_TABLE_CSV}")

    # Compute and save missing_after
    missing_after = compute_missing_table(clean_df)
    missing_after.to_csv(MISSING_AFTER_CSV, index=False)
    print(f"Saved missing_after.csv to: {MISSING_AFTER_CSV} ({len(missing_after)} columns)")

    # Save stage 1 cleaned complaints
    print(f"\nSaving cleaned dataset to {OUT_STAGE1_GZ}...")
    save_df = clean_df.copy()
    save_df["created_date"] = save_df["created_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    save_df["closed_date"] = save_df["closed_date"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    save_df.to_csv(OUT_STAGE1_GZ, index=False, compression="gzip")
    print(f"Saved: {OUT_STAGE1_GZ} ({OUT_STAGE1_GZ.stat().st_size / (1024**2):.2f} MB, {len(save_df):,} rows)")

    # TASK 3: Checks and Assertions
    print("\n" + "=" * 70)
    print("[TASK 3] CHECKS & VERIFICATIONS")
    print("=" * 70)

    # 1. Borough counts before filtering
    print("\n--- 1. Borough value_counts before filtering ---")
    print(borough_counts.to_string())

    # 2. Group value_counts after cleaning & group by status
    print("\n--- 2. Group value_counts after cleaning ---")
    group_counts = clean_df["group"].value_counts()
    print(group_counts.to_string())

    print("\n--- Group by Status counts after cleaning ---")
    group_status = pd.crosstab(clean_df["group"], clean_df["status"], margins=True)
    print(group_status.to_string())

    # 3. Empty address_norm per group
    print("\n--- 3. Empty address_norm per group ---")
    empty_addr_mask = clean_df["address_norm"] == ""
    empty_addr_per_group = clean_df[empty_addr_mask].groupby("group").size()
    # Reindex over all groups present in cleaned frame
    all_groups = clean_df["group"].unique()
    empty_addr_table = pd.DataFrame({
        "group": all_groups,
        "empty_address_count": [empty_addr_per_group.get(g, 0) for g in all_groups],
        "total_group_rows": [len(clean_df[clean_df["group"] == g]) for g in all_groups],
    })
    empty_addr_table["pct_empty"] = (empty_addr_table["empty_address_count"] / empty_addr_table["total_group_rows"] * 100).round(2)
    print(empty_addr_table.to_string(index=False))

    # 4. Dtypes and head(3)
    print("\n--- 4. Dtypes of the cleaned frame ---")
    print(clean_df.dtypes.to_string())

    print("\n--- Head(3) of the cleaned frame ---")
    print(clean_df.head(3).to_string())

    # 5. Assertions
    print("\n--- 5. Assertions ---")
    a1 = clean_df["unique_key"].is_unique
    a2_lat = clean_df["latitude"].notna().all()
    a2_lon = clean_df["longitude"].notna().all()
    a2_cdate = clean_df["created_date"].notna().all()
    a2_grp = clean_df["group"].notna().all()
    a2 = a2_lat and a2_lon and a2_cdate and a2_grp

    in_box = (
        (clean_df["latitude"] >= 40.55)
        & (clean_df["latitude"] <= 40.74)
        & (clean_df["longitude"] >= -74.06)
        & (clean_df["longitude"] <= -73.83)
    )
    a3 = in_box.all()
    a4 = (clean_df["borough"] == "BROOKLYN").all()
    
    # closed_date is never earlier than created_date
    has_closed = clean_df["closed_date"].notna()
    closed_earlier = has_closed & (clean_df["closed_date"] < clean_df["created_date"])
    a5 = (~closed_earlier).all()

    print(f"Assertion 1 (unique_key is unique)                                    : {'PASS' if a1 else 'FAIL'}")
    print(f"Assertion 2 (no NaN in latitude, longitude, created_date, group)       : {'PASS' if a2 else 'FAIL'}")
    print(f"Assertion 3 (all rows inside box lat [40.55, 40.74], lon [-74.06,-73.83]): {'PASS' if a3 else 'FAIL'}")
    print(f"Assertion 4 (all borough values equal 'BROOKLYN')                      : {'PASS' if a4 else 'FAIL'}")
    print(f"Assertion 5 (closed_date is never earlier than created_date)           : {'PASS' if a5 else 'FAIL'}")

    # SAFETY: git status and file sizes
    print("\n" + "=" * 70)
    print("SAFETY & FILE SIZES")
    print("=" * 70)
    git_st = subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True)
    print("git status --short:")
    print(git_st if git_st.strip() else "(clean)")

    print("\nCreated files size:")
    target_files = [
        OUT_STAGE1_GZ,
        DROPPED_CSV,
        STEP_TABLE_CSV,
        MISSING_BEFORE_CSV,
        MISSING_AFTER_CSV,
    ]
    for p in target_files:
        if p.exists():
            mb = p.stat().st_size / (1024**2)
            print(f"  {p.relative_to(REPO_ROOT)}: {mb:.3f} MB ({p.stat().st_size:,} bytes)")
        else:
            print(f"  {p.relative_to(REPO_ROOT)}: NOT FOUND")

    print("\n" + "=" * 70)
    print("CHUNK 2.1 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

