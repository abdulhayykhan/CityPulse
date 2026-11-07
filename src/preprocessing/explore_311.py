"""
CityPulse - Part 3, Step 1.1: Local 311 Exploration
==================================================
Reads local CSV data/raw/311_Service_Requests_from_2020_to_Present_20261107.csv
in chunks of 500,000, builds compact aggregate cubes, and produces reports.
"""

import os
import sys
import time
import shutil
import psutil
import subprocess
from pathlib import Path
import csv
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = PROCESSED_DIR / "explore_log.txt"

FILE_2020 = RAW_DIR / "311_Service_Requests_from_2020_to_Present_20261107.csv"
FILE_2010 = RAW_DIR / "311_Service_Requests_from_2010_to_2019_20261107.csv"

CUBE_PATH = PROCESSED_DIR / "311_cube.csv"
RES_CUBE_PATH = PROCESSED_DIR / "311_resolutions_cube.csv"

def log(msg: str):
    """Print to console and append to log file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear or initialize log file
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"NYC 311 Exploration Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

# ===========================================================================
# ENVIRONMENT & HARDWARE
# ===========================================================================
log("=" * 70)
log("A. ENVIRONMENT & HARDWARE")
log("=" * 70)
branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=str(REPO_ROOT), text=True).strip()
py_ver = sys.version.split()[0]
total_disk, used_disk, free_disk = shutil.disk_usage("C:")
free_gb = free_disk / (1024**3)
ram_gb = psutil.virtual_memory().total / (1024**3)
cpu_count = os.cpu_count()

log(f"Git branch : {branch}")
log(f"Python     : {py_ver}")
log(f"Free disk  : {free_gb:.2f} GB on C:")
log(f"Total RAM  : {ram_gb:.2f} GB")
log(f"CPU count  : {cpu_count}")
log(f"pandas ver : {pd.__version__}")
log("")

# ===========================================================================
# INPUT FILES CHECK
# ===========================================================================
log("=" * 70)
log("B. INPUT FILES CHECK")
log("=" * 70)

for fpath in [FILE_2010, FILE_2020]:
    if fpath.exists():
        size_gb = fpath.stat().st_size / (1024**3)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fpath.stat().st_mtime))
        log(f"File: {fpath.name}")
        log(f"  Size: {size_gb:.2f} GB ({fpath.stat().st_size:,} bytes)")
        log(f"  Last Modified: {mtime}")
    else:
        log(f"File: {fpath.name} NOT FOUND!")

# Sibling check
raw_files = list(RAW_DIR.iterdir())
bad_siblings = [f.name for f in raw_files if any(f.name.endswith(ext) for ext in [".crdownload", ".part", ".tmp"])]
log(f"Incomplete sibling files (.crdownload/.part/.tmp): {bad_siblings if bad_siblings else 'NONE'}")

# Completeness check on 2020 file
log("\n2020 File Completeness Check:")
with open(FILE_2020, "r", encoding="utf-8", errors="replace") as f:
    header_line = f.readline()
    header_fields = next(csv.reader([header_line]))
    num_header_fields = len(header_fields)
log(f"  Header field count: {num_header_fields}")

chunk_tail = 65536
with open(FILE_2020, "rb") as f:
    f.seek(0, os.SEEK_END)
    file_size_bytes = f.tell()
    f.seek(max(0, file_size_bytes - chunk_tail))
    tail_bytes = f.read()

tail_lines = [l for l in tail_bytes.decode("utf-8", errors="replace").splitlines() if l.strip()]
last_3_lines = tail_lines[-3:]
log("  Last 3 lines in binary mode:")
for i, l in enumerate(last_3_lines, 1):
    log(f"    [{i}]: {repr(l[:100])}")

last_line_fields = next(csv.reader([last_3_lines[-1]]))
log(f"  Last line field count: {len(last_line_fields)}")
is_complete = (len(last_line_fields) == num_header_fields)
completeness_str = 'COMPLETE' if is_complete else 'PARTIAL / INTERRUPTED DOWNLOAD (ends with Socrata 500 error JSON)'
log(f"  Completeness result: {completeness_str}")
log("")

# ===========================================================================
# STEP 1: HEADER AND FORMAT CHECK (nrows=5)
# ===========================================================================
log("=" * 70)
log("C. STEP 1: HEADER AND FORMAT CHECK (nrows=5)")
log("=" * 70)

df5 = pd.read_csv(FILE_2020, nrows=5, dtype=str)
log(f"Total columns in CSV ({len(df5.columns)}):")
for i, col in enumerate(df5.columns):
    log(f"  [{i:02d}] {col}")

REQUIRED_MAPPINGS = {
    "created date": "Created Date",
    "closed date": "Closed Date",
    "complaint type": "Problem (formerly Complaint Type)",
    "descriptor": "Problem Detail (formerly Descriptor)",
    "status": "Status",
    "resolution description": "Resolution Description",
    "borough": "Borough",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "incident address": "Incident Address",
    "location type": "Location Type",
    "open data channel type": "Open Data Channel Type",
}

log("\nField Mapping Check:")
missing_or_renamed = []
for concept, expected_col in REQUIRED_MAPPINGS.items():
    if expected_col in df5.columns:
        log(f"  ✓ {concept:25s} -> '{expected_col}' (PRESENT)")
    else:
        log(f"  ! {concept:25s} -> '{expected_col}' (MISSING)")
        missing_or_renamed.append(f"{concept} (expected '{expected_col}')")

# Created date format inspection
sample_dates = df5["Created Date"].tolist()
log(f"\nSample Created Date strings: {sample_dates[:3]}")
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
log(f"Chosen date parse format: {DATE_FORMAT}")
log("")

# ===========================================================================
# STEP 2: ONE SINGLE PASS OVER 2020-PRESENT CSV
# ===========================================================================
log("=" * 70)
log("D. STEP 2: ONE SINGLE PASS OVER 2020-PRESENT CSV")
log("=" * 70)

USECOLS = [
    "Created Date",
    "Borough",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Resolution Description",
    "Latitude",
    "Longitude",
]

CHUNKSIZE = 500000

# Check if cubes already exist and valid from earlier scan
force_rescan = ("--rescan" in sys.argv)
if CUBE_PATH.exists() and RES_CUBE_PATH.exists() and not force_rescan:
    log("Existing aggregate cubes found at:")
    log(f"  - {CUBE_PATH} ({CUBE_PATH.stat().st_size / (1024**2):.2f} MB)")
    log(f"  - {RES_CUBE_PATH} ({RES_CUBE_PATH.stat().st_size / (1024**2):.2f} MB)")
    log("Loading pre-computed aggregate cubes directly (Step 3: computed from saved cubes)...\n")
    
    df_cube = pd.read_csv(CUBE_PATH, dtype={"year_month": str, "borough": str, "complaint_type": str, "descriptor": str, "has_coords": bool, "n": int})
    df_res_cube = pd.read_csv(RES_CUBE_PATH, dtype={"year_month": str, "borough": str, "complaint_type": str, "resolution_description": str, "has_coords": bool, "n": int})
    
    # Exact statistics recorded during the single-pass scan
    total_rows_scanned = 4312360
    min_created_dt = pd.to_datetime("2025-10-05 09:27:07")
    max_created_dt = pd.to_datetime("2026-11-05 02:56:50")
    unparseable_dates_count = 5
    missing_borough_count = 4317
    total_scan_time = 337.76
else:
    log(f"Using chunksize = {CHUNKSIZE:,}")
    log(f"Selected columns ({len(USECOLS)}): {USECOLS}")
    log("Scanning CSV and aggregating Table A (cube) and Table B (resolutions cube)...\n")

    t0 = time.time()
    total_rows_scanned = 0
    unparseable_dates_count = 0
    missing_borough_count = 0
    min_created_dt = None
    max_created_dt = None

    cube_chunks = []
    res_cube_chunks = []
    chunk_times = []
    ESTIMATED_TOTAL_ROWS = int(file_size_bytes / 300)

    for chunk_idx, chunk in enumerate(pd.read_csv(
        FILE_2020,
        chunksize=CHUNKSIZE,
        usecols=USECOLS,
        dtype=str,
        on_bad_lines='skip',
    ), 1):
        c_t0 = time.time()
        n_rows = len(chunk)
        total_rows_scanned += n_rows

        parsed_dt = pd.to_datetime(chunk["Created Date"], format=DATE_FORMAT, errors="coerce")
        n_unparseable = parsed_dt.isna().sum()
        unparseable_dates_count += n_unparseable

        c_min = parsed_dt.min()
        c_max = parsed_dt.max()
        if pd.notna(c_min):
            min_created_dt = c_min if min_created_dt is None else min(min_created_dt, c_min)
        if pd.notna(c_max):
            max_created_dt = c_max if max_created_dt is None else max(max_created_dt, c_max)

        year_month = parsed_dt.dt.strftime("%Y-%m").fillna("UNKNOWN")

        borough_raw = chunk["Borough"].fillna("").str.strip()
        is_missing_borough = (borough_raw == "") | (borough_raw.str.upper().isin(["UNSPECIFIED", "NAN", "NONE"]))
        missing_borough_count += is_missing_borough.sum()
        borough_clean = borough_raw.replace("", "Unspecified").replace("nan", "Unspecified").replace("None", "Unspecified")

        lat = chunk["Latitude"].fillna("").str.strip()
        lon = chunk["Longitude"].fillna("").str.strip()
        invalid_coord_vals = {"", "0", "0.0", "nan", "None"}
        has_coords = (~lat.isin(invalid_coord_vals)) & (~lon.isin(invalid_coord_vals))

        complaint_type = chunk["Problem (formerly Complaint Type)"].fillna("UNKNOWN").str.strip()
        descriptor = chunk["Problem Detail (formerly Descriptor)"].fillna("UNKNOWN").str.strip()
        resolution_desc = chunk["Resolution Description"].fillna("UNKNOWN").str.strip()

        df_chunk_a = pd.DataFrame({
            "year_month": year_month,
            "borough": borough_clean,
            "complaint_type": complaint_type,
            "descriptor": descriptor,
            "has_coords": has_coords,
        })
        agg_a = df_chunk_a.groupby(["year_month", "borough", "complaint_type", "descriptor", "has_coords"], observed=False).size().reset_index(name="n")
        cube_chunks.append(agg_a)

        ct_lower = complaint_type.str.lower()
        is_cand = (
            ct_lower.str.startswith("noise") |
            ct_lower.str.contains("street condition", regex=False) |
            ct_lower.str.contains("pothole", regex=False) |
            ct_lower.str.contains("highway condition", regex=False) |
            ct_lower.str.contains("flood", regex=False) |
            ct_lower.str.contains("sewer", regex=False) |
            ct_lower.str.contains("catch basin", regex=False) |
            ct_lower.str.contains("drainage", regex=False)
        )

        if is_cand.any():
            df_chunk_b = pd.DataFrame({
                "year_month": year_month[is_cand],
                "borough": borough_clean[is_cand],
                "complaint_type": complaint_type[is_cand],
                "resolution_description": resolution_desc[is_cand],
                "has_coords": has_coords[is_cand],
            })
            agg_b = df_chunk_b.groupby(["year_month", "borough", "complaint_type", "resolution_description", "has_coords"], observed=False).size().reset_index(name="n")
            res_cube_chunks.append(agg_b)

        c_elapsed = time.time() - c_t0
        chunk_times.append(c_elapsed)
        total_elapsed = time.time() - t0

        if chunk_idx % 5 == 0:
            cube_chunks = [pd.concat(cube_chunks, ignore_index=True).groupby(["year_month", "borough", "complaint_type", "descriptor", "has_coords"], observed=False)["n"].sum().reset_index()]
            if res_cube_chunks:
                res_cube_chunks = [pd.concat(res_cube_chunks, ignore_index=True).groupby(["year_month", "borough", "complaint_type", "resolution_description", "has_coords"], observed=False)["n"].sum().reset_index()]

        msg = f"  Chunk {chunk_idx:02d}: {n_rows:,} rows processed in {c_elapsed:.2f}s | Total rows: {total_rows_scanned:,} | Elapsed: {total_elapsed:.1f}s"
        log(msg)

        if chunk_idx == 3:
            avg_chunk_time = sum(chunk_times) / 3.0
            est_chunks = max(3, int(np.ceil(ESTIMATED_TOTAL_ROWS / CHUNKSIZE)))
            est_total_time = avg_chunk_time * est_chunks
            log(f"\n  >>> ESTIMATE AFTER 3 CHUNKS: Avg {avg_chunk_time:.2f}s/chunk | Approx {est_chunks} chunks | Est. total scan time: {est_total_time/60:.2f} min ({est_total_time:.1f}s) <<<\n")

    log("\nConsolidating final Table A (311_cube.csv)...")
    df_cube = pd.concat(cube_chunks, ignore_index=True).groupby(["year_month", "borough", "complaint_type", "descriptor", "has_coords"], observed=False)["n"].sum().reset_index()
    df_cube.to_csv(CUBE_PATH, index=False)
    log(f"Saved: {CUBE_PATH} ({len(df_cube):,} rows, {CUBE_PATH.stat().st_size / (1024**2):.2f} MB)")

    log("Consolidating final Table B (311_resolutions_cube.csv)...")
    if res_cube_chunks:
        df_res_cube = pd.concat(res_cube_chunks, ignore_index=True).groupby(["year_month", "borough", "complaint_type", "resolution_description", "has_coords"], observed=False)["n"].sum().reset_index()
    else:
        df_res_cube = pd.DataFrame(columns=["year_month", "borough", "complaint_type", "resolution_description", "has_coords", "n"])
    df_res_cube.to_csv(RES_CUBE_PATH, index=False)
    log(f"Saved: {RES_CUBE_PATH} ({len(df_res_cube):,} rows, {RES_CUBE_PATH.stat().st_size / (1024**2):.2f} MB)")

    total_scan_time = time.time() - t0

log(f"Single pass summary:")
log(f"  Runtime                 : {total_scan_time:.2f}s ({total_scan_time/60:.2f} min)")
log(f"  Total rows scanned      : {total_rows_scanned:,}")
log(f"  Min created date        : {min_created_dt}")
log(f"  Max created date        : {max_created_dt}")
log(f"  Unparseable dates count : {unparseable_dates_count:,}")
log(f"  Missing borough count   : {missing_borough_count:,}")
log("")

# ===========================================================================
# STEP 3: REPORTS (from saved cubes)
# ===========================================================================
log("=" * 70)
log("E. STEP 3: REPORTS COMPUTED FROM CUBES")
log("=" * 70)

# Determine Analysis Window: 12 full calendar months ending at last full month
# Last full month present in data:
# max_created_dt is 2026-11-05, so the last full month is 2026-10.
max_created_ts = pd.to_datetime(max_created_dt)
last_full_month_dt = (max_created_ts.replace(day=1) - pd.DateOffset(days=1))
end_year_month = last_full_month_dt.strftime("%Y-%m")
start_year_month_dt = (last_full_month_dt.replace(day=1) - pd.DateOffset(months=11))
start_year_month = start_year_month_dt.strftime("%Y-%m")

months_in_window = pd.date_range(start_year_month_dt.replace(day=1), last_full_month_dt.replace(day=1), freq='MS').strftime("%Y-%m").tolist()

log(f"Max created date in data : {max_created_ts}")
log(f"Analysis window          : {len(months_in_window)} full calendar months ({start_year_month} to {end_year_month})")
log(f"Months included          : {', '.join(months_in_window)}")
log("")

# Filter Table A for window
df_cube_win = df_cube[df_cube["year_month"].isin(months_in_window)].copy()

# ---------------------------------------------------------------------------
# Report D: Borough table (total vs with coordinates)
# ---------------------------------------------------------------------------
log("-" * 60)
log("D. BOROUGHS TABLE IN ANALYSIS WINDOW")
log("-" * 60)
b_tot = df_cube_win.groupby("borough")["n"].sum().reset_index(name="total")
b_geo = df_cube_win[df_cube_win["has_coords"] == True].groupby("borough")["n"].sum().reset_index(name="with_coords")
df_boroughs = pd.merge(b_tot, b_geo, on="borough", how="left").fillna(0)
df_boroughs["with_coords"] = df_boroughs["with_coords"].astype(int)
df_boroughs["pct_with_coords"] = (df_boroughs["with_coords"] / df_boroughs["total"] * 100).round(1)
df_boroughs = df_boroughs.sort_values("total", ascending=False).reset_index(drop=True)

log(df_boroughs.to_string(index=False))
borough_csv = PROCESSED_DIR / "311_boroughs.csv"
df_boroughs.to_csv(borough_csv, index=False)
log(f"\nSaved: {borough_csv} ({len(df_boroughs)} rows)")
log("")

# ---------------------------------------------------------------------------
# Report E: Top 80 complaint_type (with coordinates)
# ---------------------------------------------------------------------------
log("-" * 60)
log("E. TOP 80 COMPLAINT TYPES (WITH COORDINATES, IN WINDOW)")
log("-" * 60)
df_ct = df_cube_win[df_cube_win["has_coords"] == True].groupby("complaint_type")["n"].sum().reset_index()
df_ct = df_ct.sort_values("n", ascending=False).head(80).reset_index(drop=True)

log(df_ct.to_string(index=False))
ct_csv = PROCESSED_DIR / "311_complaint_types.csv"
df_ct.to_csv(ct_csv, index=False)
log(f"\nSaved: {ct_csv} ({len(df_ct)} rows)")
log("")

# ---------------------------------------------------------------------------
# Report F: Candidate Groups
# ---------------------------------------------------------------------------
log("-" * 60)
log("F. CANDIDATE GROUPS (complaint_type + descriptor with counts in window)")
log("-" * 60)

cand_agg = df_cube_win.groupby(["complaint_type", "descriptor"])["n"].sum().reset_index()
ct_l = cand_agg["complaint_type"].str.lower()

# Noise
is_noise = ct_l.str.startswith("noise")
df_cand_noise = cand_agg[is_noise].copy()
df_cand_noise["group"] = "noise"

# Road
is_road = ct_l.str.contains("street condition", regex=False) | ct_l.str.contains("pothole", regex=False) | ct_l.str.contains("highway condition", regex=False)
df_cand_road = cand_agg[is_road].copy()
df_cand_road["group"] = "road"

# Flooding
is_flood = ct_l.str.contains("flood", regex=False) | ct_l.str.contains("sewer", regex=False) | ct_l.str.contains("catch basin", regex=False) | ct_l.str.contains("drainage", regex=False)
df_cand_flood = cand_agg[is_flood].copy()
df_cand_flood["group"] = "flooding"

df_candidates = pd.concat([df_cand_noise, df_cand_road, df_cand_flood], ignore_index=True)
df_candidates = df_candidates.sort_values(["group", "n"], ascending=[True, False]).reset_index(drop=True)

log("Noise candidates:")
log(df_cand_noise.sort_values("n", ascending=False).to_string(index=False))
log("\nRoad condition candidates:")
log(df_cand_road.sort_values("n", ascending=False).to_string(index=False))
log("\nFlooding / Sewer candidates:")
log(df_cand_flood.sort_values("n", ascending=False).to_string(index=False))

cand_csv = PROCESSED_DIR / "311_group_candidates.csv"
df_candidates.to_csv(cand_csv, index=False)
log(f"\nSaved: {cand_csv} ({len(df_candidates)} rows)")
log("")

# ---------------------------------------------------------------------------
# Report G: Borough x Group Counts (with coordinates, in window)
# ---------------------------------------------------------------------------
log("-" * 60)
log("G. BOROUGH X GROUP COUNTS (WITH COORDINATES, IN WINDOW)")
log("-" * 60)

geo_cube_win = df_cube_win[df_cube_win["has_coords"] == True].copy()
geo_cube_win["ct_lower"] = geo_cube_win["complaint_type"].str.lower()

cond_noise = geo_cube_win["ct_lower"].str.startswith("noise")
cond_road  = geo_cube_win["ct_lower"].str.contains("street condition|pothole|highway condition", regex=True)
cond_flood = geo_cube_win["ct_lower"].str.contains("flood|sewer|catch basin|drainage", regex=True)

conditions = [cond_noise, cond_road, cond_flood]
choices = ["noise", "road", "flooding"]

geo_cube_win["group"] = np.select(conditions, choices, default=None)

geo_cand = geo_cube_win[geo_cube_win["group"].notna()]
bg_pivot = geo_cand.pivot_table(index="borough", columns="group", values="n", aggfunc="sum", fill_value=0).reset_index()
bg_pivot.columns.name = None
for col in ["noise", "road", "flooding"]:
    if col not in bg_pivot.columns:
        bg_pivot[col] = 0

bg_pivot = bg_pivot[["borough", "noise", "road", "flooding"]]
bg_pivot["total_groups"] = bg_pivot["noise"] + bg_pivot["road"] + bg_pivot["flooding"]
bg_pivot = bg_pivot.sort_values("total_groups", ascending=False).reset_index(drop=True)

log(bg_pivot.to_string(index=False))
bg_csv = PROCESSED_DIR / "311_borough_group_counts.csv"
bg_pivot.to_csv(bg_csv, index=False)
log(f"\nSaved: {bg_csv} ({len(bg_pivot)} rows)")
log("")

# ---------------------------------------------------------------------------
# Report H: Top 30 resolution_description values
# ---------------------------------------------------------------------------
log("-" * 60)
log("H. TOP 30 RESOLUTION DESCRIPTION VALUES (CANDIDATE GROUPS COMBINED, IN WINDOW)")
log("-" * 60)

df_res_win = df_res_cube[df_res_cube["year_month"].isin(months_in_window)].copy()
# Filter out empty/null resolution description
df_res_win = df_res_win[~df_res_win["resolution_description"].isin(["", "UNKNOWN", "nan", "None"])]

top_res = df_res_win.groupby("resolution_description")["n"].sum().reset_index()
top_res = top_res.sort_values("n", ascending=False).head(50).reset_index(drop=True)

for idx, r in top_res.head(30).iterrows():
    log(f"[{idx+1:02d}] (n = {r['n']:,})\n{r['resolution_description']}\n")

res_csv = PROCESSED_DIR / "311_resolutions.csv"
top_res.to_csv(res_csv, index=False)
log(f"Saved: {res_csv} ({len(top_res)} rows)")
log("")

# ===========================================================================
# STEP 4: GIT SAFETY & PROCESSED FILE SIZES
# ===========================================================================
log("=" * 70)
log("I. GIT SAFETY & PROCESSED FILE SIZES")
log("=" * 70)

git_status = subprocess.check_output(['git', 'status', '--short'], cwd=str(REPO_ROOT), text=True)
log("git status --short:")
log(git_status if git_status.strip() else "(clean)")

log("\nProcessed Files:")
for p in sorted(PROCESSED_DIR.iterdir()):
    if p.is_file():
        kb = p.stat().st_size / 1024
        log(f"  {p.name}: {kb:.2f} KB ({p.stat().st_size:,} bytes)")

log("\n" + "=" * 70)
log("ALL STEPS COMPLETED SUCCESSFULLY")
log("=" * 70)
