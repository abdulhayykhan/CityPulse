"""
CityPulse - Long-History Daily Flood & Rain Pipeline (Step 1.3)
==============================================================
Pulls long-history daily flood complaint counts per borough (2010-01-01 to 2026-10-31)
from NYC 311 Socrata APIs (76ig-c548 and erm2-nwe9) and daily rainfall from Open-Meteo.
"""

import os
import sys
import time
import datetime
import hashlib
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.groups import assign_group

LOG_PATH = "data/processed/flood_history_log.txt"
SOCRATA_76IG = "76ig-c548"
SOCRATA_ERM2 = "erm2-nwe9"

BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]

BOROUGH_COORDS = {
    "BROOKLYN": (40.65, -73.95),
    "QUEENS": (40.73, -73.80),
    "BRONX": (40.85, -73.87),
    "MANHATTAN": (40.78, -73.97),
    "STATEN ISLAND": (40.58, -74.15),
}


def log_request(resource_id: str, params: Dict[str, Any], status_code: int, rows_returned: int, elapsed_sec: float, error_msg: str = ""):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    ts = datetime.datetime.now().isoformat()
    log_line = f"[{ts}] ID: {resource_id} | Status: {status_code} | Rows: {rows_returned} | Duration: {elapsed_sec:.2f}s | Params: {params}"
    if error_msg:
        log_line += f" | Error: {error_msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def socrata_get(resource_id: str, params: Dict[str, Any], max_retries: int = 3, sleep_after: float = 2.5) -> List[Dict[str, Any]]:
    url = f"https://data.cityofnewyork.us/resource/{resource_id}.json"
    token = os.environ.get("SOCRATA_APP_TOKEN")
    headers = {"X-App-Token": token} if token else {}

    last_error = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=90)
            elapsed = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                log_request(resource_id, params, 200, len(data), elapsed)
                time.sleep(sleep_after)
                return data
            elif resp.status_code in [429, 500, 502, 503, 504]:
                log_request(resource_id, params, resp.status_code, 0, elapsed, resp.text[:200])
                backoff = attempt * 15.0
                print(f"[RETRY {attempt}/{max_retries}] {resource_id} status {resp.status_code}. Backing off {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                log_request(resource_id, params, resp.status_code, 0, elapsed, resp.text[:200])
                raise RuntimeError(f"Socrata error {resp.status_code} on {resource_id}: {resp.text[:300]}")
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            elapsed = time.time() - t0
            log_request(resource_id, params, 0, 0, elapsed, str(e))
            last_error = e
            backoff = attempt * 15.0
            print(f"[RETRY {attempt}/{max_retries}] {resource_id} exception: {e}. Backing off {backoff:.1f}s...")
            time.sleep(backoff)

    error_msg = f"Failed to fetch {resource_id} after {max_retries} retries. Last error: {last_error}"
    print(f"FATAL: {error_msg}")
    raise RuntimeError(error_msg)


def socrata_paginate(resource_id: str, base_params: Dict[str, Any], limit: int = 50000) -> List[Dict[str, Any]]:
    all_rows = []
    offset = 0
    while True:
        p = dict(base_params)
        p["$limit"] = limit
        p["$offset"] = offset
        page = socrata_get(resource_id, p)
        all_rows.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return all_rows


def generate_quarterly_ranges(start_year: int, end_year: int, end_date_str: str = "2026-10-31T23:59:59"):
    quarters = []
    q_defs = [
        (1, "-01-01T00:00:00", "-03-31T23:59:59"),
        (2, "-04-01T00:00:00", "-06-30T23:59:59"),
        (3, "-07-01T00:00:00", "-09-30T23:59:59"),
        (4, "-10-01T00:00:00", "-12-31T23:59:59"),
    ]
    for y in range(start_year, end_year + 1):
        for q_num, q_start, q_end in q_defs:
            s = f"{y}{q_start}"
            e = f"{y}{q_end}"
            if s > end_date_str:
                break
            if e > end_date_str:
                e = end_date_str
            quarters.append((f"{y}Q{q_num}", s, e))
            if e == end_date_str:
                break
    return quarters


def run_task1():
    print("=================================================================")
    print("TASK 1: SCHEMA AND DISCOVERY")
    print("=================================================================")

    # 1a: Fetch 3 sample rows from each dataset
    print("\n--- 1a: 3 Sample Rows & Exact Field Names ---")
    s76 = socrata_get(SOCRATA_76IG, {"$limit": 3, "$where": "created_date >= '2019-12-31T00:00:00'"})
    s_erm = socrata_get(SOCRATA_ERM2, {"$limit": 3, "$where": "created_date >= '2026-10-01T00:00:00'"})

    cols_76 = list(s76[0].keys()) if s76 else []
    cols_erm = list(s_erm[0].keys()) if s_erm else []

    print(f"76ig-c548 columns ({len(cols_76)}): {cols_76}")
    print(f"erm2-nwe9 columns ({len(cols_erm)}): {cols_erm}")

    for req_field in ["created_date", "complaint_type", "descriptor", "borough"]:
        in_76 = req_field in cols_76
        in_erm = req_field in cols_erm
        print(f"  Field '{req_field}': in 76ig-c548 -> {in_76}, in erm2-nwe9 -> {in_erm}")
        if not (in_76 and in_erm):
            raise RuntimeError(f"Missing required field {req_field}")

    # 1b: Per dataset, per YEAR: counts of complaint_type containing SEWER, FLOOD, CATCH BASIN, DRAIN
    print("\n--- 1b: Complaint Type Discovery Per Year ---")
    disc_path = "data/processed/flood_type_discovery.csv"
    if os.path.exists(disc_path) and os.path.getsize(disc_path) > 100:
        print(f"Loading existing {disc_path}")
        df_discovery = pd.read_csv(disc_path)
    else:
        discovery_rows = []

        # 76ig: 2010 to 2019
        types_76 = ["Root/Sewer/Sidewalk Condition", "Sewer"]
        for y in range(2010, 2020):
            params = {
                "$select": "complaint_type, count(*) as n",
                "$where": f"complaint_type in ('Root/Sewer/Sidewalk Condition', 'Sewer') and created_date >= '{y}-01-01T00:00:00' and created_date <= '{y}-12-31T23:59:59'",
                "$group": "complaint_type",
                "$order": "complaint_type",
            }
            res = socrata_get(SOCRATA_76IG, params)
            for r in res:
                discovery_rows.append({
                    "dataset": "76ig-c548",
                    "year": y,
                    "complaint_type": r["complaint_type"],
                    "count": int(r["n"]),
                })

        # erm2: 2020 to 2026
        types_erm = ["Root/Sewer/Sidewalk Condition", "Sewer", "Sewer Maintenance", "Water Drainage"]
        for y in range(2020, 2027):
            end_str = "2026-10-31T23:59:59" if y == 2026 else f"{y}-12-31T23:59:59"
            params = {
                "$select": "complaint_type, count(*) as n",
                "$where": f"complaint_type in ('Root/Sewer/Sidewalk Condition', 'Sewer', 'Sewer Maintenance', 'Water Drainage') and created_date >= '{y}-01-01T00:00:00' and created_date <= '{end_str}'",
                "$group": "complaint_type",
                "$order": "complaint_type",
            }
            res = socrata_get(SOCRATA_ERM2, params)
            for r in res:
                discovery_rows.append({
                    "dataset": "erm2-nwe9",
                    "year": y,
                    "complaint_type": r["complaint_type"],
                    "count": int(r["n"]),
                })

        df_discovery = pd.DataFrame(discovery_rows)
        df_discovery.to_csv(disc_path, index=False)
        print(f"Saved {disc_path}")

    print(df_discovery.to_string(index=False))

    # 1c: Per dataset, per YEAR, per complaint_type in ('Sewer','Sewer Maintenance'), per descriptor: counts
    print("\n--- 1c: Descriptors by Year ---")
    desc_path = "data/processed/flood_descriptor_by_year.csv"
    if os.path.exists(desc_path) and os.path.getsize(desc_path) > 100:
        print(f"Loading existing {desc_path}")
        df_desc = pd.read_csv(desc_path)
    else:
        descriptor_rows = []

        # 76ig: 2010 to 2019
        for y in range(2010, 2020):
            params = {
                "$select": "complaint_type, descriptor, count(*) as n",
                "$where": f"complaint_type in ('Sewer', 'Sewer Maintenance') and created_date >= '{y}-01-01T00:00:00' and created_date <= '{y}-12-31T23:59:59'",
                "$group": "complaint_type, descriptor",
                "$order": "complaint_type, descriptor",
                "$limit": 50000,
            }
            res = socrata_paginate(SOCRATA_76IG, params)
            for r in res:
                descriptor_rows.append({
                    "dataset": "76ig-c548",
                    "year": y,
                    "complaint_type": r.get("complaint_type", ""),
                    "descriptor": r.get("descriptor", ""),
                    "count": int(r["n"]),
                })

        # erm2: 2020 to 2026
        for y in range(2020, 2027):
            end_str = "2026-10-31T23:59:59" if y == 2026 else f"{y}-12-31T23:59:59"
            params = {
                "$select": "complaint_type, descriptor, count(*) as n",
                "$where": f"complaint_type in ('Sewer', 'Sewer Maintenance') and created_date >= '{y}-01-01T00:00:00' and created_date <= '{end_str}'",
                "$group": "complaint_type, descriptor",
                "$order": "complaint_type, descriptor",
                "$limit": 50000,
            }
            res = socrata_paginate(SOCRATA_ERM2, params)
            for r in res:
                descriptor_rows.append({
                    "dataset": "erm2-nwe9",
                    "year": y,
                    "complaint_type": r.get("complaint_type", ""),
                    "descriptor": r.get("descriptor", ""),
                    "count": int(r["n"]),
                })

        df_desc = pd.DataFrame(descriptor_rows)
        df_desc.to_csv(desc_path, index=False)
        print(f"Saved {desc_path}")

    print(df_desc.to_string(index=False))

    # 1d: erm2 monthly counts from 2025-06-01 to 2026-10-31 for ('Sewer', 'Sewer Maintenance')
    print("\n--- 1d: erm2-nwe9 Monthly Counts (2025-06 to 2026-10) ---")
    path_1d = "data/processed/flood_monthly_transition.csv"
    if os.path.exists(path_1d) and os.path.getsize(path_1d) > 100:
        print(f"Loading existing {path_1d}")
        df_1d = pd.read_csv(path_1d)
    else:
        params_1d = {
            "$select": "date_trunc_ym(created_date) as month, complaint_type, count(*) as n",
            "$where": "complaint_type in ('Sewer', 'Sewer Maintenance') and created_date >= '2025-06-01T00:00:00' and created_date <= '2026-10-31T23:59:59'",
            "$group": "month, complaint_type",
            "$order": "month, complaint_type",
            "$limit": 50000,
        }
        res_1d = socrata_get(SOCRATA_ERM2, params_1d)
        rows_1d = []
        for r in res_1d:
            m_str = r.get("month", "")[:7]
            rows_1d.append({
                "month": m_str,
                "complaint_type": r.get("complaint_type", ""),
                "count": int(r["n"]),
            })
        df_1d = pd.DataFrame(rows_1d)
        df_1d.to_csv(path_1d, index=False)
        print(f"Saved {path_1d}")

    print(df_1d.to_string(index=False))

    sm_df = df_1d[df_1d["complaint_type"] == "Sewer Maintenance"]
    first_sm_month = sm_df["month"].min() if not sm_df.empty else "None"
    print(f"\nFirst month in which 'Sewer Maintenance' appears: {first_sm_month}")

    return df_discovery, df_desc, df_1d, first_sm_month


def run_task2():
    print("=================================================================")
    print("TASK 2: DAILY AGGREGATE PULL")
    print("=================================================================")

    # Pull 76ig-c548 (2010 to 2019)
    print("\n--- Pulling 76ig-c548 (2010-01-01 to 2019-12-31) in quarterly slices ---")
    raw_76_path = "data/processed/flood_raw_aggregate_2010_2019.csv"
    if os.path.exists(raw_76_path) and os.path.getsize(raw_76_path) > 1000:
        print(f"Loading existing {raw_76_path}")
        df_raw_76 = pd.read_csv(raw_76_path)
        req_count_76 = 40
        print(f"Loaded {raw_76_path}: {len(df_raw_76)} rows from {req_count_76} quarterly queries.")
    else:
        q_76 = generate_quarterly_ranges(2010, 2019, "2019-12-31T23:59:59")
        rows_76 = []
        req_count_76 = 0
        for q_name, s, e in q_76:
            params = {
                "$select": "date_trunc_ymd(created_date) AS day, borough, complaint_type, descriptor, count(*) AS n",
                "$where": f"complaint_type in ('Sewer', 'Sewer Maintenance') and created_date >= '{s}' and created_date <= '{e}'",
                "$group": "day, borough, complaint_type, descriptor",
                "$order": "day, borough, complaint_type, descriptor",
                "$limit": 50000,
            }
            res = socrata_paginate(SOCRATA_76IG, params)
            req_count_76 += 1
            print(f"  {q_name} ({s[:10]} to {e[:10]}): {len(res)} rows")
            for r in res:
                d_str = r.get("day", "")[:10]
                rows_76.append({
                    "day": d_str,
                    "borough": r.get("borough", ""),
                    "complaint_type": r.get("complaint_type", ""),
                    "descriptor": r.get("descriptor", ""),
                    "n": int(r.get("n", 0)),
                })

        df_raw_76 = pd.DataFrame(rows_76)
        df_raw_76.to_csv(raw_76_path, index=False)
        print(f"Saved {raw_76_path}: {len(df_raw_76)} rows from {req_count_76} quarterly queries.")

    # Pull erm2-nwe9 (2020 to 2026-10-31)
    print("\n--- Pulling erm2-nwe9 (2020-01-01 to 2026-10-31) in quarterly slices ---")
    raw_erm_path = "data/processed/flood_raw_aggregate_2020_onward.csv"
    q_erm_dir = "data/processed/flood_raw_erm2_quarterly"
    os.makedirs(q_erm_dir, exist_ok=True)
    q_erm = generate_quarterly_ranges(2020, 2026, "2026-10-31T23:59:59")
    rows_erm = []
    req_count_erm = 0
    for q_name, s, e in q_erm:
        q_file = os.path.join(q_erm_dir, f"{q_name}.csv")
        if os.path.exists(q_file) and os.path.getsize(q_file) > 10:
            df_q = pd.read_csv(q_file)
            print(f"  {q_name} ({s[:10]} to {e[:10]}): {len(df_q)} rows (cached)")
            rows_erm.extend(df_q.to_dict("records"))
            req_count_erm += 1
        else:
            params = {
                "$select": "date_trunc_ymd(created_date) AS day, borough, complaint_type, descriptor, count(*) AS n",
                "$where": f"complaint_type in ('Sewer', 'Sewer Maintenance') and created_date >= '{s}' and created_date <= '{e}'",
                "$group": "day, borough, complaint_type, descriptor",
                "$order": "day, borough, complaint_type, descriptor",
                "$limit": 50000,
            }
            res = socrata_paginate(SOCRATA_ERM2, params)
            req_count_erm += 1
            print(f"  {q_name} ({s[:10]} to {e[:10]}): {len(res)} rows")
            q_rows = []
            for r in res:
                d_str = r.get("day", "")[:10]
                q_rows.append({
                    "day": d_str,
                    "borough": r.get("borough", ""),
                    "complaint_type": r.get("complaint_type", ""),
                    "descriptor": r.get("descriptor", ""),
                    "n": int(r.get("n", 0)),
                })
            df_q = pd.DataFrame(q_rows)
            df_q.to_csv(q_file, index=False)
            rows_erm.extend(q_rows)

    df_raw_erm = pd.DataFrame(rows_erm)
    df_raw_erm.to_csv(raw_erm_path, index=False)
    print(f"Saved {raw_erm_path}: {len(df_raw_erm)} rows from {req_count_erm} quarterly queries.")

    return df_raw_76, df_raw_erm, req_count_76, req_count_erm


def run_task3():
    print("=================================================================")
    print("TASK 3: BUILD THE DAILY TABLE")
    print("=================================================================")

    df_76 = pd.read_csv("data/processed/flood_raw_aggregate_2010_2019.csv")
    df_erm = pd.read_csv("data/processed/flood_raw_aggregate_2020_onward.csv")

    df_76["dataset"] = "76ig-c548"
    df_erm["dataset"] = "erm2-nwe9"

    # Confirm no day appears in both raw extracts
    days_76 = set(df_76["day"].dropna().unique())
    days_erm = set(df_erm["day"].dropna().unique())
    overlap = days_76.intersection(days_erm)
    print(f"Max day in 76ig-c548: {max(days_76)}")
    print(f"Min day in erm2-nwe9: {min(days_erm)}")
    print(f"Overlapping days between 2010-2019 and 2020-onward: {len(overlap)} (Expected: 0)")
    assert len(overlap) == 0, f"Overlap detected: {overlap}"

    df_comb = pd.concat([df_76, df_erm], ignore_index=True)
    df_comb["borough"] = df_comb["borough"].fillna("").astype(str).str.strip().str.upper()
    df_comb["descriptor"] = df_comb["descriptor"].fillna("").astype(str)
    df_comb["complaint_type"] = df_comb["complaint_type"].fillna("").astype(str)
    df_comb["year"] = pd.to_datetime(df_comb["day"]).dt.year

    # 3a: Apply assign_group
    print("\n--- 3a: Applying assign_group from groups.py ---")
    df_comb["group"] = assign_group(df_comb["complaint_type"], df_comb["descriptor"])
    print(f"Total aggregate rows: {len(df_comb)}")
    print("Group distribution on raw aggregates:")
    print(df_comb["group"].value_counts(dropna=False))

    # Keep rows with group == 'flood' and boroughs in 5 boroughs
    is_valid_boro = df_comb["borough"].isin(BOROUGHS)
    is_flood = df_comb["group"] == "flood"
    df_flood = df_comb[is_flood & is_valid_boro].copy()
    print(f"Filtered flood rows in 5 boroughs: {len(df_flood)} (Total complaints n = {df_flood['n'].sum()})")

    # 3b: Report distinct (complaint_type, descriptor) matching (?i)flood|overflow|clogged not labeled flood
    print("\n--- 3b: Potential Flood Descriptors NOT Labeled 'flood' ---")
    flood_kw = df_comb["descriptor"].str.contains(r"(?i)flood|overflow|clogged", regex=True)
    not_flood = df_comb["group"] != "flood"
    df_mismatched = df_comb[flood_kw & not_flood]

    if not df_mismatched.empty:
        summary_mismatch = (
            df_mismatched.groupby(["complaint_type", "descriptor", "group"])
            .agg(total_n=("n", "sum"), min_year=("year", "min"), max_year=("year", "max"))
            .reset_index()
            .sort_values(by="total_n", ascending=False)
        )
        print(summary_mismatch.to_string(index=False))
    else:
        print("None. All matching descriptors were assigned to flood.")

    # 3c: Save flood_daily_all_years.csv with columns day, borough, n zero-filled for EVERY calendar day
    print("\n--- 3c: Creating Complete Zero-Filled Daily Panel ---")
    all_dates = pd.date_range("2010-01-01", "2026-10-31", freq="D").strftime("%Y-%m-%d")
    idx = pd.MultiIndex.from_product([all_dates, BOROUGHS], names=["day", "borough"])
    panel = pd.DataFrame(index=idx).reset_index()
    print(f"Expected daily panel rows: {len(all_dates)} days * 5 boroughs = {len(panel)} rows.")

    daily_agg = df_flood.groupby(["day", "borough"])["n"].sum().reset_index()
    panel = panel.merge(daily_agg, on=["day", "borough"], how="left")
    panel["n"] = panel["n"].fillna(0).astype(int)

    panel.to_csv("data/processed/flood_daily_all_years.csv", index=False)
    print(f"Saved data/processed/flood_daily_all_years.csv: {len(panel)} rows.")
    print(f"Total flood complaints: {panel['n'].sum()}")

    return panel, summary_mismatch if not df_mismatched.empty else pd.DataFrame()


def run_task4():
    print("=================================================================")
    print("TASK 4: RAIN (DAILY ONLY FROM OPEN-METEO ARCHIVE)")
    print("=================================================================")

    url = "https://archive-api.open-meteo.com/v1/archive"
    rain_dfs = []

    for boro, (lat, lon) in BOROUGH_COORDS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2010-01-01",
            "end_date": "2026-10-31",
            "daily": ["precipitation_sum", "precipitation_hours"],
            "timezone": "America/New_York",
        }
        print(f"Fetching rain for {boro} ({lat}, {lon})...")
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 200:
                    d = resp.json()["daily"]
                    b_df = pd.DataFrame({
                        "date": d["time"],
                        "borough": boro,
                        "precip_mm": d["precipitation_sum"],
                        "precip_hours": d["precipitation_hours"],
                    })
                    rain_dfs.append(b_df)
                    print(f"  {boro}: {len(b_df)} rows, time {time.time()-t0:.2f}s")
                    break
                else:
                    print(f"  Attempt {attempt}: status {resp.status_code}, error: {resp.text[:100]}")
                    time.sleep(2.0 * attempt)
            except Exception as e:
                print(f"  Attempt {attempt} exception: {e}")
                time.sleep(2.0 * attempt)
        time.sleep(1.0)

    df_rain = pd.concat(rain_dfs, ignore_index=True)
    df_rain.to_csv("data/processed/rain_daily.csv", index=False)
    print(f"\nSaved data/processed/rain_daily.csv: {len(df_rain)} rows.")

    null_counts = df_rain.isnull().sum().to_dict()
    print("Null counts:", null_counts)

    # Check for missing dates
    all_dates = set(pd.date_range("2010-01-01", "2026-10-31", freq="D").strftime("%Y-%m-%d"))
    for boro in BOROUGHS:
        b_dates = set(df_rain[df_rain["borough"] == boro]["date"])
        missing = all_dates - b_dates
        print(f"  {boro} missing dates count: {len(missing)}")

    return df_rain


def run_task5():
    print("=================================================================")
    print("TASK 5: VERIFICATION FROM SAVED FILES ONLY")
    print("=================================================================")

    df_flood = pd.read_csv("data/processed/flood_daily_all_years.csv")
    df_rain = pd.read_csv("data/processed/rain_daily.csv")
    df_desc = pd.read_csv("data/processed/flood_descriptor_by_year.csv")

    df_flood["year"] = pd.to_datetime(df_flood["day"]).dt.year
    df_flood["date"] = df_flood["day"]

    # 5a: Flood complaints per borough per year (flag >3x or <1/3x previous year)
    print("\n--- 5a: Flood Complaints Per Borough Per Year ---")
    piv_5a = df_flood.pivot_table(index="year", columns="borough", values="n", aggfunc="sum")
    print("Yearly totals:")
    print(piv_5a.to_string())

    flags_5a = []
    years = sorted(piv_5a.index)
    for i in range(1, len(years)):
        prev_y = years[i - 1]
        curr_y = years[i]
        for boro in BOROUGHS:
            prev_val = piv_5a.loc[prev_y, boro]
            curr_val = piv_5a.loc[curr_y, boro]
            ratio = (curr_val / prev_val) if prev_val > 0 else np.nan
            flag = ""
            if prev_val > 0:
                if curr_val > 3.0 * prev_val:
                    flag = ">3x"
                elif curr_val < (1.0 / 3.0) * prev_val:
                    flag = "<1/3x"
            if flag:
                flags_5a.append({
                    "borough": boro,
                    "prev_year": prev_y,
                    "curr_year": curr_y,
                    "prev_val": prev_val,
                    "curr_val": curr_val,
                    "ratio": f"{ratio:.2f}x",
                    "flag": flag,
                })
    df_flags_5a = pd.DataFrame(flags_5a)
    print("\nYearly fluctuation flags (>3x or <1/3x previous year):")
    if not df_flags_5a.empty:
        print(df_flags_5a.to_string(index=False))
    else:
        print("None.")

    # 5b: Descriptor stability: first and last year each flood descriptor appears
    print("\n--- 5b: Descriptor Stability ---")
    # Determine flood descriptors by applying assign_group
    df_desc["group"] = assign_group(df_desc["complaint_type"], df_desc["descriptor"])
    flood_desc_df = df_desc[df_desc["group"] == "flood"].copy()
    desc_stability = (
        flood_desc_df.groupby(["complaint_type", "descriptor"])
        .agg(first_year=("year", "min"), last_year=("year", "max"), total_count=("count", "sum"))
        .reset_index()
    )
    desc_stability["vanished_mid_span"] = desc_stability["last_year"] < 2026
    desc_stability["appeared_mid_span"] = desc_stability["first_year"] > 2010
    desc_stability["flag"] = desc_stability.apply(
        lambda r: ("Vanished in " + str(r["last_year"]) if r["vanished_mid_span"] else "")
        + ("; " if r["vanished_mid_span"] and r["appeared_mid_span"] else "")
        + ("Appeared in " + str(r["first_year"]) if r["appeared_mid_span"] else ""),
        axis=1
    )
    print(desc_stability.to_string(index=False))

    # 5c: Boundary check: 2019-12-25 to 2020-01-05
    print("\n--- 5c: Boundary Check (2019-12-25 to 2020-01-05) ---")
    df_boundary = df_flood[(df_flood["day"] >= "2019-12-25") & (df_flood["day"] <= "2020-01-05")]
    piv_boundary = df_boundary.pivot_table(index="day", columns="borough", values="n", aggfunc="sum")
    print(piv_boundary.to_string())

    # 5d: Cross-check: Brooklyn daily flood counts for 2025-11-01 to 2026-10-31 vs complaints_clean.csv
    print("\n--- 5d: Brooklyn Cross-Check vs complaints_clean.csv ---")
    clean_df = pd.read_csv("data/processed/complaints_clean.csv", usecols=["id", "date", "group"], dtype={"id": str})
    clean_flood = clean_df[clean_df["group"] == "flood"].copy()
    clean_daily = clean_flood.groupby("date").size().rename("clean_n")

    bk_flood = df_flood[
        (df_flood["borough"] == "BROOKLYN") &
        (df_flood["day"] >= "2025-11-01") &
        (df_flood["day"] <= "2026-10-31")
    ].copy().set_index("day")["n"].rename("socrata_n")

    cmp_df = pd.DataFrame({"socrata_n": bk_flood, "clean_n": clean_daily}).fillna(0)
    cmp_df["diff"] = cmp_df["socrata_n"] - cmp_df["clean_n"]
    cmp_df["abs_diff"] = cmp_df["diff"].abs()

    total_diff = cmp_df["diff"].sum()
    days_differ = (cmp_df["diff"] != 0).sum()
    max_abs_diff = cmp_df["abs_diff"].max()
    pearson_r = cmp_df["socrata_n"].corr(cmp_df["clean_n"])

    print(f"Total Socrata flood sum: {cmp_df['socrata_n'].sum():.0f}")
    print(f"Total Clean flood sum:   {cmp_df['clean_n'].sum():.0f}")
    print(f"Total difference (Socrata - Clean): {total_diff:.0f}")
    print(f"Number of days differing: {days_differ} / {len(cmp_df)}")
    print(f"Largest absolute daily difference: {max_abs_diff:.0f}")
    print(f"Pearson r between the two series: {pearson_r:.6f}")

    # 5e: Days with n >= 1 per borough, median and max daily count
    print("\n--- 5e: Days with n >= 1, Median, and Max Daily Count ---")
    stats_5e = []
    for boro in BOROUGHS:
        b_series = df_flood[df_flood["borough"] == boro]["n"]
        total_days = len(b_series)
        days_ge_1 = (b_series >= 1).sum()
        pct_ge_1 = (days_ge_1 / total_days) * 100.0
        med_all = b_series.median()
        med_active = b_series[b_series >= 1].median() if days_ge_1 > 0 else np.nan
        max_val = b_series.max()
        stats_5e.append({
            "borough": boro,
            "total_days": total_days,
            "days_n>=1": days_ge_1,
            "pct_days_n>=1": f"{pct_ge_1:.2f}%",
            "median_all_days": med_all,
            "median_active_days": med_active,
            "max_daily_count": max_val,
        })
    df_stats_5e = pd.DataFrame(stats_5e)
    print(df_stats_5e.to_string(index=False))

    # 5f: Pearson and Spearman r vs rain same day, prev day, 3-day sum
    print("\n--- 5f: Pearson and Spearman Correlations vs Rain ---")
    # Merge flood and rain
    merged = df_flood[["date", "borough", "n"]].merge(df_rain[["date", "borough", "precip_mm"]], on=["date", "borough"])
    merged = merged.sort_values(by=["borough", "date"]).reset_index(drop=True)

    # Compute lag-1 and 3-day rolling sum within each borough
    merged["precip_lag1"] = merged.groupby("borough")["precip_mm"].shift(1)
    merged["precip_3d"] = merged.groupby("borough")["precip_mm"].transform(lambda s: s.rolling(3, min_periods=3).sum())

    spans = [
        ("Full: 2010 to 2026-10-31", "2010-01-01", "2026-10-31"),
        ("2020 Onward", "2020-01-01", "2026-10-31"),
        ("Last 12 Months: 2025-11-01 to 2026-10-31", "2025-11-01", "2026-10-31"),
    ]

    corr_rows = []
    for span_name, s_date, e_date in spans:
        span_sub = merged[(merged["date"] >= s_date) & (merged["date"] <= e_date)]
        for boro in BOROUGHS:
            b_sub = span_sub[span_sub["borough"] == boro]
            # Same day
            p_same = b_sub["n"].corr(b_sub["precip_mm"], method="pearson")
            s_same = b_sub["n"].corr(b_sub["precip_mm"], method="spearman")
            # Lag 1
            valid_lag = b_sub.dropna(subset=["precip_lag1"])
            p_lag1 = valid_lag["n"].corr(valid_lag["precip_lag1"], method="pearson")
            s_lag1 = valid_lag["n"].corr(valid_lag["precip_lag1"], method="spearman")
            # 3-day sum
            valid_3d = b_sub.dropna(subset=["precip_3d"])
            p_3d = valid_3d["n"].corr(valid_3d["precip_3d"], method="pearson")
            s_3d = valid_3d["n"].corr(valid_3d["precip_3d"], method="spearman")

            corr_rows.append({
                "span": span_name,
                "borough": boro,
                "pearson_sameday": round(p_same, 4),
                "spearman_sameday": round(s_same, 4),
                "pearson_lag1": round(p_lag1, 4),
                "spearman_lag1": round(s_lag1, 4),
                "pearson_3d": round(p_3d, 4),
                "spearman_3d": round(s_3d, 4),
            })
    df_corr = pd.DataFrame(corr_rows)
    print(df_corr.to_string(index=False))

    # 5g: Storm days (precip_mm >= 20) per borough
    print("\n--- 5g: Storm Days (precip_mm >= 20) ---")
    storm_rows = []
    # Total days in full span = 6148. Latest 20% = round(6148 * 0.20) = 1230 days.
    n_days_total = 6148
    n_latest_20 = int(np.round(n_days_total * 0.20))
    latest_20_cutoff = all_dates_sorted = sorted(df_rain["date"].unique())[-n_latest_20]
    print(f"Latest 20% cutoff date ({n_latest_20} days): {latest_20_cutoff} to {all_dates_sorted[-1]}")

    for boro in BOROUGHS:
        b_rain = df_rain[df_rain["borough"] == boro]
        storm_full = (b_rain["precip_mm"] >= 20.0).sum()
        storm_l12 = (
            (b_rain["date"] >= "2025-11-01") &
            (b_rain["date"] <= "2026-10-31") &
            (b_rain["precip_mm"] >= 20.0)
        ).sum()
        storm_latest_20 = (
            (b_rain["date"] >= latest_20_cutoff) &
            (b_rain["precip_mm"] >= 20.0)
        ).sum()
        pct_in_latest_20 = (storm_latest_20 / storm_full * 100.0) if storm_full > 0 else 0.0

        storm_rows.append({
            "borough": boro,
            "storm_days_full_span": storm_full,
            "storm_days_last_12m": storm_l12,
            "storm_days_latest_20pct": storm_latest_20,
            "pct_storm_in_latest_20pct": f"{pct_in_latest_20:.1f}%",
        })
    df_storm = pd.DataFrame(storm_rows)
    print(df_storm.to_string(index=False))

    # 5h: Top 10 daily flood-count days per borough over full span
    print("\n--- 5h: Top 10 Daily Flood-Count Days Per Borough ---")
    top10_rows = []
    for boro in BOROUGHS:
        b_m = merged[merged["borough"] == boro].sort_values(by="n", ascending=False).head(10).copy()
        b_m["rank"] = range(1, len(b_m) + 1)
        for _, r in b_m.iterrows():
            top10_rows.append({
                "borough": boro,
                "rank": r["rank"],
                "date": r["date"],
                "flood_count_n": int(r["n"]),
                "precip_mm_same_day": round(r["precip_mm"], 2),
                "precip_mm_3d_sum": round(r["precip_3d"], 2) if pd.notnull(r["precip_3d"]) else np.nan,
            })
    df_top10 = pd.DataFrame(top10_rows)
    print(df_top10.to_string(index=False))

    return piv_5a, df_flags_5a, desc_stability, piv_boundary, cmp_df, df_stats_5e, df_corr, df_storm, df_top10


if __name__ == "__main__":
    t_start = time.time()
    print("Starting Step 1.3 Pipeline...")
    run_task1()
    run_task2()
    run_task3()
    run_task4()
    run_task5()
    print(f"\nPipeline completed in {time.time()-t_start:.1f}s.")

