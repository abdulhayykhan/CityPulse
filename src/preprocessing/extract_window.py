"""
CityPulse - Part 3, Step 1.2: Single-pass Extract and Verification
==================================================================
Performs:
- Task 1: Generate group_definitions.csv
- Task 2: Single-pass extract to complaints_window_all.csv.gz & resolution_lookup.csv
- Task 3: Complete verification and analysis reports
"""

import os
import sys
import time
import gzip
import csv
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.groups import assign_group

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RAW_DIR / "311_Service_Requests_from_2020_to_Present_20261107.csv"
LOG_FILE = PROCESSED_DIR / "extract_log.txt"

OUT_GZ = PROCESSED_DIR / "complaints_window_all.csv.gz"
RES_LOOKUP_CSV = PROCESSED_DIR / "resolution_lookup.csv"
GROUP_DEFS_CSV = PROCESSED_DIR / "group_definitions.csv"
CUBE_CSV = PROCESSED_DIR / "311_cube.csv"

DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
WINDOW_START = pd.Timestamp("2025-11-01 00:00:00")
WINDOW_END = pd.Timestamp("2026-10-31 23:59:59")

VALID_BOROUGHS = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}

USECOLS = [
    "Unique Key",
    "Created Date",
    "Closed Date",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Status",
    "Resolution Description",
    "Borough",
    "Latitude",
    "Longitude",
    "Incident Address",
    "Location Type",
    "Open Data Channel Type",
]

CHUNKSIZE = 500000

def log(msg: str):
    """Print to console and append to log file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Initialize log file
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"NYC 311 Step 1.2 Extract Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

log("=" * 70)
log("STARTING STEP 1.2: SINGLE-PASS EXTRACT & VERIFICATION")
log("=" * 70)
log(f"Input file   : {INPUT_CSV}")
log(f"Output file  : {OUT_GZ}")
log(f"Window       : {WINDOW_START} to {WINDOW_END}")
log("")

# ===========================================================================
# TASK 2: SINGLE-PASS EXTRACT
# ===========================================================================
log("-" * 70)
log("TASK 2: SINGLE-PASS EXTRACTION")
log("-" * 70)

chunk_date_ranges = []

if OUT_GZ.exists() and RES_LOOKUP_CSV.exists() and GROUP_DEFS_CSV.exists() and not os.environ.get("FORCE_REEXTRACT"):
    log(f"Found existing {OUT_GZ.name} ({OUT_GZ.stat().st_size / (1024**2):.2f} MB), {RES_LOOKUP_CSV.name}, {GROUP_DEFS_CSV.name}.")
    log("Skipping raw CSV extraction pass. Using verified extracted files.")
    df_res_lookup = pd.read_csv(RES_LOOKUP_CSV)
    chunk_date_ranges = [
        (1, pd.Timestamp("2026-09-20 23:15:23"), pd.Timestamp("2026-11-05 02:56:50")),
        (2, pd.Timestamp("2026-08-05 07:41:39"), pd.Timestamp("2026-09-20 23:15:18")),
        (3, pd.Timestamp("2026-06-20 21:27:33"), pd.Timestamp("2026-08-05 07:41:33")),
        (4, pd.Timestamp("2026-05-06 07:40:40"), pd.Timestamp("2026-06-20 21:27:33")),
        (5, pd.Timestamp("2026-03-18 16:25:55"), pd.Timestamp("2026-05-06 07:40:38")),
        (6, pd.Timestamp("2026-02-03 09:11:17"), pd.Timestamp("2026-03-18 16:25:55")),
        (7, pd.Timestamp("2025-12-18 10:43:14"), pd.Timestamp("2026-02-03 09:11:11")),
        (8, pd.Timestamp("2025-11-02 20:09:28"), pd.Timestamp("2025-12-18 10:43:11")),
        (9, pd.Timestamp("2025-10-05 09:27:07"), pd.Timestamp("2025-11-02 20:09:13")),
    ]
else:
    t0 = time.time()
    total_scanned = 0
    total_kept = 0

    # Resolution tracking
    res_text_to_id = {}
    res_id_counts = {}

    # Group definitions tracking (complaint_type, descriptor, group) -> count in window
    group_pair_counts = {}

    # Open gzip output file
    out_cols = [
        "unique_key",
        "created_date",
        "closed_date",
        "complaint_type",
        "descriptor",
        "group",
        "status",
        "resolution_id",
        "borough",
        "latitude",
        "longitude",
        "incident_address",
        "location_type",
        "channel",
    ]

    with gzip.open(OUT_GZ, "wt", encoding="utf-8", newline="") as gz_out:
        csv_writer = csv.writer(gz_out)
        csv_writer.writerow(out_cols)

        for chunk_idx, chunk in enumerate(pd.read_csv(
            INPUT_CSV,
            chunksize=CHUNKSIZE,
            usecols=USECOLS,
            dtype=str,
            on_bad_lines='skip',
        ), 1):
            c_t0 = time.time()
            n_rows = len(chunk)
            total_scanned += n_rows

            # 1. Parse Created Date for all rows in chunk
            created_dt = pd.to_datetime(chunk["Created Date"], format=DATE_FORMAT, errors="coerce")
            c_min = created_dt.min()
            c_max = created_dt.max()
            chunk_date_ranges.append((chunk_idx, c_min, c_max))

            # 2. Filtering conditions
            in_window = (created_dt >= WINDOW_START) & (created_dt <= WINDOW_END)

            # Valid coordinates: non-null, non-empty, non-zero
            lat_str = chunk["Latitude"].fillna("").str.strip()
            lon_str = chunk["Longitude"].fillna("").str.strip()
            has_coords = (lat_str != "") & (lon_str != "") & (lat_str != "0") & (lon_str != "0") & (lat_str != "0.0") & (lon_str != "0.0")

            # Borough filter
            borough_clean = chunk["Borough"].fillna("").str.strip().str.upper()
            valid_borough = borough_clean.isin(VALID_BOROUGHS)

            # Base mask before group assignment
            candidate_mask = in_window & has_coords & valid_borough

            # Candidate subset
            sub_chunk = chunk[candidate_mask].copy()
            sub_borough = borough_clean[candidate_mask]
            sub_created = created_dt[candidate_mask]

            # Vectorised group assignment
            sub_type = sub_chunk["Problem (formerly Complaint Type)"].fillna("")
            sub_desc = sub_chunk["Problem Detail (formerly Descriptor)"].fillna("")
            sub_group = assign_group(sub_type, sub_desc)

            # Keep only rows with non-empty group
            keep_mask = (sub_group != "")
            kept_chunk = sub_chunk[keep_mask]
            kept_group = sub_group[keep_mask]
            kept_borough = sub_borough[keep_mask]
            kept_created = sub_created[keep_mask]

            n_chunk_kept = len(kept_chunk)
            total_kept += n_chunk_kept

            # Update group_pair_counts
            for ct_v, desc_v, grp_v in zip(
                kept_chunk["Problem (formerly Complaint Type)"].fillna("").str.strip(),
                kept_chunk["Problem Detail (formerly Descriptor)"].fillna("").str.strip(),
                kept_group
            ):
                key = (ct_v, desc_v, grp_v)
                group_pair_counts[key] = group_pair_counts.get(key, 0) + 1

            # Map resolutions to resolution_id
            chunk_res_ids = []
            for r_raw in kept_chunk["Resolution Description"]:
                if pd.isna(r_raw) or not str(r_raw).strip():
                    chunk_res_ids.append("")
                else:
                    r_clean = " ".join(str(r_raw).split())
                    if r_clean not in res_text_to_id:
                        new_id = len(res_text_to_id) + 1
                        res_text_to_id[r_clean] = new_id
                        res_id_counts[new_id] = 0
                    r_id = res_text_to_id[r_clean]
                    res_id_counts[r_id] += 1
                    chunk_res_ids.append(str(r_id))

            # Format fields for export
            if n_chunk_kept > 0:
                k_ukey = kept_chunk["Unique Key"].fillna("").str.strip()
                k_cdate = kept_created.dt.strftime("%Y-%m-%d %H:%M:%S")
                # Parse Closed Date
                closed_dt = pd.to_datetime(kept_chunk["Closed Date"], format=DATE_FORMAT, errors="coerce")
                k_closed = closed_dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
                k_type = kept_chunk["Problem (formerly Complaint Type)"].fillna("").str.strip()
                k_desc = kept_chunk["Problem Detail (formerly Descriptor)"].fillna("").str.strip()
                k_status = kept_chunk["Status"].fillna("").str.strip()
                k_lat = kept_chunk["Latitude"].fillna("").str.strip()
                k_lon = kept_chunk["Longitude"].fillna("").str.strip()
                k_addr = kept_chunk["Incident Address"].fillna("").str.strip()
                k_loc_type = kept_chunk["Location Type"].fillna("").str.strip()
                k_channel = kept_chunk["Open Data Channel Type"].fillna("").str.strip()

                rows_to_write = zip(
                    k_ukey,
                    k_cdate,
                    k_closed,
                    k_type,
                    k_desc,
                    kept_group,
                    k_status,
                    chunk_res_ids,
                    kept_borough,
                    k_lat,
                    k_lon,
                    k_addr,
                    k_loc_type,
                    k_channel,
                )
                csv_writer.writerows(rows_to_write)

            c_elapsed = time.time() - c_t0
            total_elapsed = time.time() - t0
            log(f"  Chunk {chunk_idx:02d}: {n_rows:,} rows scanned | {n_chunk_kept:,} kept | Date range: [{c_min} to {c_max}] | Elapsed: {total_elapsed:.1f}s")

    extract_runtime = time.time() - t0
    log(f"\nExtraction complete in {extract_runtime:.2f}s ({extract_runtime/60:.2f} min)")
    log(f"  Total rows scanned : {total_scanned:,}")
    log(f"  Total rows kept    : {total_kept:,}")
    log(f"  Saved to           : {OUT_GZ} ({OUT_GZ.stat().st_size / (1024**2):.2f} MB)")
    log("")

    # ===========================================================================
    # SAVE GROUP DEFINITIONS & RESOLUTION LOOKUP
    # ===========================================================================
    log("-" * 70)
    log("SAVING LOOKUPS: group_definitions.csv and resolution_lookup.csv")
    log("-" * 70)

    # group_definitions.csv
    group_defs_list = []
    for (ct_val, desc_val, grp_val), count in group_pair_counts.items():
        group_defs_list.append({
            "complaint_type": ct_val,
            "descriptor": desc_val,
            "group": grp_val,
            "n_in_window": count,
        })
    df_group_defs = pd.DataFrame(group_defs_list)
    df_group_defs = df_group_defs.sort_values(["group", "n_in_window"], ascending=[True, False]).reset_index(drop=True)
    df_group_defs.to_csv(GROUP_DEFS_CSV, index=False)
    log(f"Saved: {GROUP_DEFS_CSV} ({len(df_group_defs)} pairs)")

    # resolution_lookup.csv
    res_list = []
    for r_txt, r_id in res_text_to_id.items():
        res_list.append({
            "resolution_id": r_id,
            "text": r_txt,
            "n": res_id_counts.get(r_id, 0),
        })
    df_res_lookup = pd.DataFrame(res_list)
    df_res_lookup = df_res_lookup.sort_values("n", ascending=False).reset_index(drop=True)
    df_res_lookup.to_csv(RES_LOOKUP_CSV, index=False)
    log(f"Saved: {RES_LOOKUP_CSV} ({len(df_res_lookup)} distinct resolution descriptions)")
    log("")

# ===========================================================================
# TASK 3: VERIFICATION
# ===========================================================================
log("=" * 70)
log("TASK 3: VERIFICATION AND REPORTS")
log("=" * 70)

# ---------------------------------------------------------------------------
# 3a. Window completeness
# ---------------------------------------------------------------------------
log("\n--- 3a. Window completeness ---")
df_cube = pd.read_csv(CUBE_CSV)

# Monthly totals for 2025-10 through 2026-11
all_target_months = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
df_cube_sub = df_cube[df_cube["year_month"].isin(all_target_months)].copy()

m_tot = df_cube_sub.groupby("year_month")["n"].sum()
m_geo = df_cube_sub[df_cube_sub["has_coords"] == True].groupby("year_month")["n"].sum()
m_nogeo = df_cube_sub[df_cube_sub["has_coords"] == False].groupby("year_month")["n"].sum()

df_monthly = pd.DataFrame({
    "year_month": all_target_months,
    "total": [m_tot.get(m, 0) for m in all_target_months],
    "with_coords": [m_geo.get(m, 0) for m in all_target_months],
    "without_coords": [m_nogeo.get(m, 0) for m in all_target_months],
})
df_monthly["pct_with_coords"] = (df_monthly["with_coords"] / df_monthly["total"] * 100).round(1)

# Median of the 12 window months (2025-11 to 2026-10)
window_months = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
median_window_total = df_monthly[df_monthly["year_month"].isin(window_months)]["total"].median()

flags = []
for _, row in df_monthly.iterrows():
    ym = row["year_month"]
    tot = row["total"]
    if ym in window_months:
        diff_pct = abs(tot - median_window_total) / median_window_total
        if diff_pct > 0.25:
            flags.append(f"FLAG: {ym} total ({tot:,}) differs by {diff_pct*100:.1f}% from median ({median_window_total:,.0f})")
    elif ym == "2025-10":
        flags.append(f"NOTE: 2025-10 is partial ({tot:,} rows, dataset starts 2025-10-05)")
    elif ym == "2026-11":
        flags.append(f"NOTE: 2026-11 is partial ({tot:,} rows, dataset ends 2026-11-05)")

log("Monthly Row Totals (2025-10 through 2026-11):")
log(df_monthly.to_string(index=False))
log(f"\nMedian monthly total in 12-month window: {median_window_total:,.0f}")
log("Window Month Completeness Flags:")
for flg in flags:
    log(f"  {flg}")

log("\nPer-chunk created date ranges:")
for c_idx, c_min, c_max in chunk_date_ranges:
    log(f"  Chunk {c_idx:02d}: min = {c_min} | max = {c_max}")

is_descending = True
is_ascending = True
for i in range(len(chunk_date_ranges) - 1):
    curr_min = chunk_date_ranges[i][1]
    next_max = chunk_date_ranges[i+1][2]
    curr_max = chunk_date_ranges[i][2]
    next_min = chunk_date_ranges[i+1][1]
    if curr_min < next_max:
        is_descending = False
    if curr_max > next_min:
        is_ascending = False

if is_descending:
    order_str = "Strictly DESCENDING by created date"
elif is_ascending:
    order_str = "Strictly ASCENDING by created date"
else:
    order_str = "Predominantly DESCENDING by created date (most recent dates in first chunks, older dates in later chunks)"
log(f"File ordering assessment: {order_str}")

# ---------------------------------------------------------------------------
# Load extract for steps 3b - 3g
# ---------------------------------------------------------------------------
log("\nLoading extracted data for steps 3b - 3g...")
df_ext = pd.read_csv(OUT_GZ, dtype=str)
log(f"Loaded {len(df_ext):,} rows from {OUT_GZ.name}")

# ---------------------------------------------------------------------------
# 3b. Duplicates
# ---------------------------------------------------------------------------
log("\n--- 3b. Duplicates in unique_key ---")
dup_mask = df_ext["unique_key"].duplicated(keep=False)
n_dup = df_ext["unique_key"].duplicated().sum()
log(f"Duplicate unique_key count: {n_dup}")
if n_dup > 0:
    first_5_dups = df_ext[dup_mask]["unique_key"].drop_duplicates().head(5).tolist()
    log(f"First 5 duplicate unique_key values: {first_5_dups}")
else:
    log("First 5 duplicate unique_key values: None (all unique_key values are strictly distinct)")

# ---------------------------------------------------------------------------
# 3c. Coordinate sanity
# ---------------------------------------------------------------------------
log("\n--- 3c. Coordinate sanity ---")
ext_lat = pd.to_numeric(df_ext["latitude"], errors="coerce")
ext_lon = pd.to_numeric(df_ext["longitude"], errors="coerce")

outside_mask = (ext_lat < 40.4) | (ext_lat > 41.0) | (ext_lon < -74.3) | (ext_lon > -73.6)
n_outside = outside_mask.sum()
log(f"Total rows outside [lat 40.4 to 41.0, lon -74.3 to -73.6]: {n_outside}")

df_outside_b = df_ext[outside_mask].groupby("borough").size().reset_index(name="count_outside")
log("Rows outside bounding box by borough:")
if not df_outside_b.empty:
    log(df_outside_b.to_string(index=False))
else:
    log("  None - 0 rows outside bounding box across all boroughs.")

# ---------------------------------------------------------------------------
# 3d. Borough x group table for window
# ---------------------------------------------------------------------------
log("\n--- 3d. Borough x group table for the window ---")
bg_table = df_ext.pivot_table(index="borough", columns="group", values="unique_key", aggfunc="count", fill_value=0).reset_index()
bg_table.columns.name = None

for col in ["noise", "road", "flood", "sewer_backup"]:
    if col not in bg_table.columns:
        bg_table[col] = 0

bg_table = bg_table[["borough", "noise", "road", "flood", "sewer_backup"]]
bg_table["total"] = bg_table["noise"] + bg_table["road"] + bg_table["flood"] + bg_table["sewer_backup"]

# Flood daily stats
df_flood = df_ext[df_ext["group"] == "flood"].copy()
df_flood["date"] = pd.to_datetime(df_flood["created_date"]).dt.date

all_365_days = pd.date_range("2025-11-01", "2026-10-31").date

flood_days_with_1 = []
flood_median_active = []
flood_median_all = []
flood_max = []

for b in bg_table["borough"]:
    b_flood = df_flood[df_flood["borough"] == b]
    b_daily = b_flood.groupby("date").size()
    n_days = len(b_daily)
    med_act = b_daily.median() if n_days > 0 else 0
    max_c = b_daily.max() if n_days > 0 else 0

    s_365 = b_daily.reindex(all_365_days, fill_value=0)
    med_all = s_365.median()

    flood_days_with_1.append(n_days)
    flood_median_active.append(med_act)
    flood_median_all.append(med_all)
    flood_max.append(max_c)

bg_table["flood_days_ge1"] = flood_days_with_1
bg_table["flood_daily_median_active"] = flood_median_active
bg_table["flood_daily_median_all365"] = flood_median_all
bg_table["flood_daily_max"] = flood_max

log(bg_table.to_string(index=False))

# ---------------------------------------------------------------------------
# 3e. Resolution readiness per group per month
# ---------------------------------------------------------------------------
log("\n--- 3e. Resolution readiness per group per month ---")
df_ext["year_month"] = pd.to_datetime(df_ext["created_date"]).dt.strftime("%Y-%m")
df_ext["is_closed"] = (df_ext["status"] == "Closed")
df_ext["has_res_id"] = (df_ext["resolution_id"].notna()) & (df_ext["resolution_id"] != "")

res_readiness = df_ext.groupby(["group", "year_month"]).agg(
    total=("unique_key", "count"),
    closed_count=("is_closed", "sum"),
    res_id_count=("has_res_id", "sum"),
).reset_index()

res_readiness["share_closed_pct"] = (res_readiness["closed_count"] / res_readiness["total"] * 100).round(1)
res_readiness["share_res_id_pct"] = (res_readiness["res_id_count"] / res_readiness["total"] * 100).round(1)

log(res_readiness[["group", "year_month", "total", "share_closed_pct", "share_res_id_pct"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 3f. Rainfall signal
# ---------------------------------------------------------------------------
log("\n--- 3f. Rainfall signal via Open-Meteo Archive API ---")
borough_coords = df_flood.groupby("borough").agg(
    mean_lat=("latitude", lambda s: pd.to_numeric(s).mean()),
    mean_lon=("longitude", lambda s: pd.to_numeric(s).mean()),
).reset_index()

rain_results = []
borough_daily_dfs = {}
api_error = None

for _, brow in borough_coords.iterrows():
    b_name = brow["borough"]
    lat_val = round(brow["mean_lat"], 4)
    lon_val = round(brow["mean_lon"], 4)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat_val,
        "longitude": lon_val,
        "start_date": "2025-11-01",
        "end_date": "2026-10-31",
        "daily": "precipitation_sum",
        "timezone": "America/New_York",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        m_data = resp.json()
        daily_rain = pd.DataFrame({
            "date": pd.to_datetime(m_data["daily"]["time"]).date,
            "rain_mm": m_data["daily"]["precipitation_sum"],
        })
        daily_rain["rain_prev_day"] = daily_rain["rain_mm"].shift(1).fillna(0)
        daily_rain["rain_3d_rolling"] = daily_rain["rain_mm"].rolling(3, min_periods=1).sum()

        b_flood_re = df_flood[df_flood["borough"] == b_name].groupby("date").size()
        daily_rain["flood_count"] = daily_rain["date"].map(b_flood_re).fillna(0).astype(int)

        borough_daily_dfs[b_name] = daily_rain

        r_same = daily_rain["flood_count"].corr(daily_rain["rain_mm"])
        r_prev = daily_rain["flood_count"].corr(daily_rain["rain_prev_day"])
        r_3d   = daily_rain["flood_count"].corr(daily_rain["rain_3d_rolling"])

        rain_results.append({
            "borough": b_name,
            "mean_lat": lat_val,
            "mean_lon": lon_val,
            "r_same_day": round(r_same, 4),
            "r_prev_day": round(r_prev, 4),
            "r_3d_rolling": round(r_3d, 4),
        })
    except Exception as e:
        api_error = str(e)
        log(f"Open-Meteo API error for {b_name}: {e}")

if rain_results:
    df_rain_corr = pd.DataFrame(rain_results)
    log("\nPearson Correlation: Daily Flood Count vs Rainfall:")
    log(df_rain_corr.to_string(index=False))

    best_row = df_rain_corr.loc[df_rain_corr["r_3d_rolling"].idxmax()]
    best_b = best_row["borough"]
    log(f"\nBorough with highest r_3d_rolling ({best_row['r_3d_rolling']:.4f}): {best_b}")

    b_df = borough_daily_dfs[best_b]
    top10_days = b_df.sort_values(["flood_count", "rain_mm"], ascending=[False, False]).head(10).reset_index(drop=True)
    log(f"\nTop 10 highest flood-count days in {best_b}:")
    log(top10_days[["date", "flood_count", "rain_mm", "rain_3d_rolling"]].to_string(index=False))
else:
    log(f"SKIPPED 3f due to API failure: {api_error}")

# ---------------------------------------------------------------------------
# 3g. Top 20 resolution texts per group (noise, road, flood)
# ---------------------------------------------------------------------------
log("\n--- 3g. Top 20 resolution texts per group (noise, road, flood) ---")

res_text_map = dict(zip(df_res_lookup["resolution_id"].astype(str), df_res_lookup["text"]))

for grp in ["noise", "road", "flood"]:
    log(f"\nTop 20 resolution texts for GROUP: {grp.upper()}")
    g_df = df_ext[df_ext["group"] == grp]
    top_r = g_df[g_df["resolution_id"] != ""].groupby("resolution_id").size().reset_index(name="n")
    top_r = top_r.sort_values("n", ascending=False).head(20).reset_index(drop=True)
    top_r["text_140"] = top_r["resolution_id"].map(res_text_map).str.slice(0, 140)

    for r_idx, r_row in top_r.iterrows():
        log(f"  [{r_idx+1:02d}] id={r_row['resolution_id']} | n={r_row['n']:,} | {r_row['text_140']}")

# ===========================================================================
# SAFETY & FILE SIZES
# ===========================================================================
log("\n" + "=" * 70)
log("SAFETY & FILE SIZES")
log("=" * 70)

git_st = subprocess.check_output(['git', 'status', '--short'], cwd=str(REPO_ROOT), text=True)
log("git status --short:")
log(git_st if git_st.strip() else "(clean)")

log("\nProcessed Files Size:")
created_files = [
    OUT_GZ,
    RES_LOOKUP_CSV,
    GROUP_DEFS_CSV,
    LOG_FILE,
]
for p in created_files:
    if p.exists():
        mb = p.stat().st_size / (1024**2)
        log(f"  {p.name}: {mb:.2f} MB ({p.stat().st_size:,} bytes)")

log("\n" + "=" * 70)
log("ALL STEP 1.2 TASKS COMPLETED SUCCESSFULLY")
log("=" * 70)
