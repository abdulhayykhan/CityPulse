"""
CityPulse - Part 3, Step 2, Chunk 2.5: Multi-Borough Comparative Analysis
=========================================================================
Performs comparative evaluation across all five NYC boroughs:
- Daily aggregation across flood, road, noise, and sewer_backup for 365 calendar days
- Open-Meteo precipitation archive fetch for all 5 borough centers
- Generation of Tables T1 through T7:
  * T1: Borough summary metrics (totals, active days, medians, 95th percentiles)
  * T2: Full-window precipitation correlations (Pearson & Spearman)
  * T3: Top-10 flood spike days with dry-day flags (rain_mm < 10)
  * T4: Robust correlations excluding spike_no_rain days and log1p correlations
  * T5: Storm-day impact analysis (rain_mm >= 20)
  * T6: Deep dive for Brooklyn high-volume flood dates (2026-05-20, 2026-05-21, 2026-07-06)
  * T7: Comparative evidence table sorted by weakest group size
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List
import requests
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORT_TABLES_DIR = PROCESSED_DIR / "report_tables"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COMPLAINTS_GZ = PROCESSED_DIR / "complaints_window_all.csv.gz"
FLOOD_RAIN_CSV = PROCESSED_DIR / "flood_vs_rain_all_boroughs.csv"

BOROUGH_CENTRES = {
    "BROOKLYN": (40.65, -73.95),
    "QUEENS": (40.73, -73.80),
    "BRONX": (40.85, -73.87),
    "MANHATTAN": (40.78, -73.97),
    "STATEN ISLAND": (40.58, -74.15),
}

START_DATE = "2025-11-01"
END_DATE = "2026-10-31"


def fetch_open_meteo_rain() -> Dict[str, pd.DataFrame]:
    """
    Fetch daily precipitation sum from Open-Meteo archive API for all 5 borough centres.
    Returns dictionary mapping borough name to DataFrame with columns: date, rain_mm.
    """
    rain_dfs = {}
    print("\nFetching precipitation from Open-Meteo Archive API (2025-11-01 to 2026-10-31)...")
    for b_name, (lat, lon) in BOROUGH_CENTRES.items():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": "precipitation_sum",
            "timezone": "America/New_York",
        }
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        df_b = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]).strftime("%Y-%m-%d"),
            "rain_mm": data["daily"]["precipitation_sum"],
        })
        # Check nulls
        n_null = df_b["rain_mm"].isna().sum()
        null_dates = df_b[df_b["rain_mm"].isna()]["date"].tolist()
        print(f"  {b_name:13s} (lat {lat}, lon {lon}): {len(df_b)} days fetched | {n_null} null values | null dates: {null_dates}")

        rain_dfs[b_name] = df_b

    return rain_dfs


def build_daily_dataset(complaints_path: Path, rain_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build daily counts per borough across all 365 calendar days for flood, road, noise, sewer_backup.
    Merge with rain features.
    """
    print(f"\nLoading complaints extract from: {complaints_path}")
    df_raw = pd.read_csv(
        complaints_path,
        dtype={"unique_key": str, "resolution_id": str, "incident_address": str},
        keep_default_na=False,
    )
    df_raw["created_dt"] = pd.to_datetime(df_raw["created_date"], errors="coerce")
    df_raw["date"] = df_raw["created_dt"].dt.strftime("%Y-%m-%d")

    all_dates = pd.date_range(START_DATE, END_DATE).strftime("%Y-%m-%d").tolist()
    assert len(all_dates) == 365, f"Expected 365 days, got {len(all_dates)}"

    daily_records = []

    for b_name in BOROUGH_CENTRES.keys():
        b_raw = df_raw[df_raw["borough"] == b_name]
        b_rain = rain_dfs[b_name].set_index("date")["rain_mm"]

        # Daily counts per group
        flood_counts = b_raw[b_raw["group"] == "flood"].groupby("date").size()
        road_counts = b_raw[b_raw["group"] == "road"].groupby("date").size()
        noise_counts = b_raw[b_raw["group"] == "noise"].groupby("date").size()
        sewer_counts = b_raw[b_raw["group"] == "sewer_backup"].groupby("date").size()

        for d in all_dates:
            r_mm = b_rain.get(d, np.nan)
            daily_records.append({
                "date": d,
                "borough": b_name,
                "flood_count": int(flood_counts.get(d, 0)),
                "road_count": int(road_counts.get(d, 0)),
                "noise_count": int(noise_counts.get(d, 0)),
                "sewer_backup_count": int(sewer_counts.get(d, 0)),
                "rain_mm": r_mm,
            })

    daily_df = pd.DataFrame(daily_records)

    # Compute previous day rain and rolling 3-day sum per borough
    processed_dfs = []
    for b_name, group in daily_df.groupby("borough", sort=False):
        g = group.sort_values("date").copy()
        g["rain_prev_day"] = g["rain_mm"].shift(1).fillna(0.0)
        g["rain_3d_sum"] = g["rain_mm"].rolling(3, min_periods=1).sum()
        processed_dfs.append(g)

    daily_df = pd.concat(processed_dfs, ignore_index=True)

    # Save data/processed/flood_vs_rain_all_boroughs.csv
    export_df = daily_df[["date", "borough", "flood_count", "road_count", "noise_count", "rain_mm"]].copy()
    export_df.to_csv(FLOOD_RAIN_CSV, index=False)
    print(f"Saved: {FLOOD_RAIN_CSV} ({len(export_df):,} rows)")

    return daily_df, df_raw


def compute_t1(daily_df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    T1. Per borough: total flood, road, noise, sewer_backup rows;
    days with flood_count >= 1; median, max, and 95th percentile daily flood count.
    """
    rows = []
    for b_name in BOROUGH_CENTRES.keys():
        b_raw = df_raw[df_raw["borough"] == b_name]
        b_daily = daily_df[daily_df["borough"] == b_name]
        f_counts = b_daily["flood_count"]

        flood_tot = int((b_raw["group"] == "flood").sum())
        road_tot = int((b_raw["group"] == "road").sum())
        noise_tot = int((b_raw["group"] == "noise").sum())
        sewer_tot = int((b_raw["group"] == "sewer_backup").sum())

        days_ge_1 = int((f_counts >= 1).sum())
        med_daily = float(f_counts.median())
        max_daily = int(f_counts.max())
        p95_daily = float(f_counts.quantile(0.95))

        rows.append({
            "borough": b_name,
            "flood_total": flood_tot,
            "road_total": road_tot,
            "noise_total": noise_tot,
            "sewer_backup_total": sewer_tot,
            "days_flood_ge_1": days_ge_1,
            "median_daily_flood": med_daily,
            "max_daily_flood": max_daily,
            "p95_daily_flood": p95_daily,
        })

    t1_df = pd.DataFrame(rows)
    t1_df.to_csv(REPORT_TABLES_DIR / "t1_borough_summary.csv", index=False)
    return t1_df


def compute_t2(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    T2. Correlation on all days, per borough:
    Pearson and Spearman of flood_count vs (rain same day, rain previous day, rolling 3-day rain sum).
    """
    rows = []
    for b_name in BOROUGH_CENTRES.keys():
        b_df = daily_df[daily_df["borough"] == b_name].copy()
        # Drop any null rain rows if present
        valid_df = b_df.dropna(subset=["rain_mm", "rain_prev_day", "rain_3d_sum"])
        fc = valid_df["flood_count"]

        rows.append({
            "borough": b_name,
            "pearson_same_day": round(fc.corr(valid_df["rain_mm"], method="pearson"), 4),
            "spearman_same_day": round(fc.corr(valid_df["rain_mm"], method="spearman"), 4),
            "pearson_prev_day": round(fc.corr(valid_df["rain_prev_day"], method="pearson"), 4),
            "spearman_prev_day": round(fc.corr(valid_df["rain_prev_day"], method="spearman"), 4),
            "pearson_3d_sum": round(fc.corr(valid_df["rain_3d_sum"], method="pearson"), 4),
            "spearman_3d_sum": round(fc.corr(valid_df["rain_3d_sum"], method="spearman"), 4),
        })

    t2_df = pd.DataFrame(rows)
    t2_df.to_csv(REPORT_TABLES_DIR / "t2_correlations_all_days.csv", index=False)
    return t2_df


def compute_t3(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    T3. Spike days: for each borough, 10 highest flood-count days with:
    date, flood_count, rain_mm, rain_3d_sum, and spike_no_rain = True when rain_mm < 10.
    """
    rows = []
    for b_name in BOROUGH_CENTRES.keys():
        b_df = daily_df[daily_df["borough"] == b_name].copy()
        top10 = b_df.sort_values(by=["flood_count", "rain_mm"], ascending=[False, False]).head(10).reset_index(drop=True)

        for rank, r in enumerate(top10.itertuples(), start=1):
            is_dry_spike = bool(r.rain_mm < 10.0)
            rows.append({
                "borough": b_name,
                "rank": rank,
                "date": r.date,
                "flood_count": r.flood_count,
                "rain_mm": round(r.rain_mm, 2),
                "rain_3d_sum": round(r.rain_3d_sum, 2),
                "spike_no_rain": is_dry_spike,
            })

    t3_df = pd.DataFrame(rows)
    t3_df.to_csv(REPORT_TABLES_DIR / "t3_spike_days.csv", index=False)
    return t3_df


def compute_t4(daily_df: pd.DataFrame, t3_df: pd.DataFrame) -> pd.DataFrame:
    """
    T4. Robust correlation per borough:
    Pearson and Spearman (same-day, previous-day, 3-day sum) after dropping spike_no_rain days;
    also Pearson of log1p(flood_count) vs the same rain features on all days.
    """
    # Extract spike_no_rain dates per borough
    spike_dry = t3_df[t3_df["spike_no_rain"] == True]

    rows = []
    for b_name in BOROUGH_CENTRES.keys():
        b_df = daily_df[daily_df["borough"] == b_name].copy()
        drop_dates = spike_dry[spike_dry["borough"] == b_name]["date"].tolist()
        n_dropped = len(drop_dates)

        b_robust = b_df[~b_df["date"].isin(drop_dates)].copy()

        # Robust correlations
        fc_rob = b_robust["flood_count"]
        r_rob_same = fc_rob.corr(b_robust["rain_mm"], method="pearson")
        s_rob_same = fc_rob.corr(b_robust["rain_mm"], method="spearman")
        r_rob_prev = fc_rob.corr(b_robust["rain_prev_day"], method="pearson")
        s_rob_prev = fc_rob.corr(b_robust["rain_prev_day"], method="spearman")
        r_rob_3d = fc_rob.corr(b_robust["rain_3d_sum"], method="pearson")
        s_rob_3d = fc_rob.corr(b_robust["rain_3d_sum"], method="spearman")

        # log1p correlations on ALL days
        fc_all_log = np.log1p(b_df["flood_count"])
        r_log_same = fc_all_log.corr(b_df["rain_mm"], method="pearson")
        r_log_prev = fc_all_log.corr(b_df["rain_prev_day"], method="pearson")
        r_log_3d = fc_all_log.corr(b_df["rain_3d_sum"], method="pearson")

        rows.append({
            "borough": b_name,
            "days_dropped": n_dropped,
            "robust_pearson_same_day": round(r_rob_same, 4),
            "robust_spearman_same_day": round(s_rob_same, 4),
            "robust_pearson_prev_day": round(r_rob_prev, 4),
            "robust_spearman_prev_day": round(s_rob_prev, 4),
            "robust_pearson_3d_sum": round(r_rob_3d, 4),
            "robust_spearman_3d_sum": round(s_rob_3d, 4),
            "log1p_pearson_same_day": round(r_log_same, 4),
            "log1p_pearson_prev_day": round(r_log_prev, 4),
            "log1p_pearson_3d_sum": round(r_log_3d, 4),
        })

    t4_df = pd.DataFrame(rows)
    t4_df.to_csv(REPORT_TABLES_DIR / "t4_robust_correlations.csv", index=False)
    return t4_df


def compute_t5(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    T5. Storm days (rain_mm >= 20):
    per borough number of storm days, mean and median flood_count on storm vs non-storm days,
    and ratio of the means.
    """
    rows = []
    for b_name in BOROUGH_CENTRES.keys():
        b_df = daily_df[daily_df["borough"] == b_name].copy()
        is_storm = b_df["rain_mm"] >= 20.0

        storm_df = b_df[is_storm]
        non_storm_df = b_df[~is_storm]

        n_storm = len(storm_df)
        n_non_storm = len(non_storm_df)

        mean_storm = float(storm_df["flood_count"].mean()) if n_storm > 0 else 0.0
        med_storm = float(storm_df["flood_count"].median()) if n_storm > 0 else 0.0
        mean_non_storm = float(non_storm_df["flood_count"].mean()) if n_non_storm > 0 else 0.0
        med_non_storm = float(non_storm_df["flood_count"].median()) if n_non_storm > 0 else 0.0

        ratio = (mean_storm / mean_non_storm) if mean_non_storm > 0 else np.nan

        rows.append({
            "borough": b_name,
            "storm_days_count": n_storm,
            "non_storm_days_count": n_non_storm,
            "mean_flood_storm": round(mean_storm, 2),
            "median_flood_storm": round(med_storm, 2),
            "mean_flood_non_storm": round(mean_non_storm, 2),
            "median_flood_non_storm": round(med_non_storm, 2),
            "mean_ratio": round(ratio, 2),
        })

    t5_df = pd.DataFrame(rows)
    t5_df.to_csv(REPORT_TABLES_DIR / "t5_storm_days.csv", index=False)
    return t5_df


def compute_t6(df_raw: pd.DataFrame, daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    T6. Brooklyn deep dive for 2026-05-20, 2026-05-21 and 2026-07-06 (flood group only).
    """
    target_dates = ["2026-05-20", "2026-05-21", "2026-07-06"]
    bk_flood = df_raw[(df_raw["borough"] == "BROOKLYN") & (df_raw["group"] == "flood")].copy()
    bk_rain = daily_df[daily_df["borough"] == "BROOKLYN"].set_index("date")["rain_mm"]

    summary_rows = []
    deep_dive_details = {}

    for d in target_dates:
        sub = bk_flood[bk_flood["date"] == d].copy()
        n_flood = len(sub)

        # Rain on day t and 3 preceding days
        d_ts = pd.Timestamp(d)
        r_t = bk_rain.get(d, 0.0)
        r_t1 = bk_rain.get((d_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), 0.0)
        r_t2 = bk_rain.get((d_ts - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), 0.0)
        r_t3 = bk_rain.get((d_ts - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), 0.0)

        # Types and descriptors
        type_desc = sub.groupby(["complaint_type", "descriptor"]).size().reset_index(name="count")
        type_desc = type_desc.sort_values("count", ascending=False)

        # Hours
        hours_series = sub["created_dt"].dt.hour.value_counts().sort_index()

        # Addresses
        addr_counts = sub["incident_address"].value_counts()
        distinct_addr = sub["incident_address"].nunique()
        top5_addr = addr_counts.head(5)

        top1_addr = addr_counts.index[0] if len(addr_counts) > 0 else ""
        top1_cnt = addr_counts.iloc[0] if len(addr_counts) > 0 else 0
        top1_share = (top1_cnt / n_flood * 100.0) if n_flood > 0 else 0.0

        # Named address (excluding empty string)
        named_addr_counts = sub[sub["incident_address"].str.strip() != ""]["incident_address"].value_counts()
        top_named_addr = named_addr_counts.index[0] if len(named_addr_counts) > 0 else "None"
        top_named_cnt = named_addr_counts.iloc[0] if len(named_addr_counts) > 0 else 0
        top_named_share = (top_named_cnt / n_flood * 100.0) if n_flood > 0 else 0.0

        summary_rows.append({
            "date": d,
            "flood_count": n_flood,
            "rain_day_t_mm": round(r_t, 2),
            "rain_t_minus_1_mm": round(r_t1, 2),
            "rain_t_minus_2_mm": round(r_t2, 2),
            "rain_t_minus_3_mm": round(r_t3, 2),
            "rain_3d_sum_mm": round(r_t + r_t1 + r_t2, 2),
            "distinct_addresses": distinct_addr,
            "top1_address": repr(top1_addr),
            "top1_address_count": top1_cnt,
            "top1_address_share_pct": round(top1_share, 2),
            "top_named_address": top_named_addr,
            "top_named_address_count": top_named_cnt,
            "top_named_address_share_pct": round(top_named_share, 2),
        })

        deep_dive_details[d] = {
            "type_desc": type_desc,
            "hours": hours_series,
            "top5_addresses": top5_addr,
        }

    t6_df = pd.DataFrame(summary_rows)
    t6_df.to_csv(REPORT_TABLES_DIR / "t6_brooklyn_deep_dive.csv", index=False)
    return t6_df, deep_dive_details


def compute_t7(t1_df: pd.DataFrame, t4_df: pd.DataFrame, t5_df: pd.DataFrame) -> pd.DataFrame:
    """
    T7. Evidence table, one row per borough:
    flood_total, road_total, noise_total, robust Spearman (3-day sum, from T4),
    storm-day count, and minimum of (flood_total, road_total) as 'weakest_group_size'.
    Sorted by weakest_group_size descending. (NO recommendation column).
    """
    m = t1_df.merge(t4_df[["borough", "robust_spearman_3d_sum"]], on="borough")
    m = m.merge(t5_df[["borough", "storm_days_count"]], on="borough")

    m["weakest_group_size"] = m[["flood_total", "road_total"]].min(axis=1)

    t7_df = m[[
        "borough",
        "flood_total",
        "road_total",
        "noise_total",
        "robust_spearman_3d_sum",
        "storm_days_count",
        "weakest_group_size",
    ]].sort_values("weakest_group_size", ascending=False).reset_index(drop=True)

    t7_df.to_csv(REPORT_TABLES_DIR / "t7_evidence_summary.csv", index=False)
    return t7_df


def main():
    print("=" * 70)
    print("STEP 2, CHUNK 2.5: MULTI-BOROUGH COMPARATIVE ANALYSIS")
    print("=" * 70)

    # 1. Fetch rain data
    rain_dfs = fetch_open_meteo_rain()

    # 2. Build daily dataset
    daily_df, df_raw = build_daily_dataset(INPUT_COMPLAINTS_GZ, rain_dfs)

    # 3. Generate tables
    print("\nComputing Table T1 (Borough summary metrics)...")
    t1_df = compute_t1(daily_df, df_raw)

    print("Computing Table T2 (Full-window correlations)...")
    t2_df = compute_t2(daily_df)

    print("Computing Table T3 (Spike days)...")
    t3_df = compute_t3(daily_df)

    print("Computing Table T4 (Robust correlations)...")
    t4_df = compute_t4(daily_df, t3_df)

    print("Computing Table T5 (Storm days analysis)...")
    t5_df = compute_t5(daily_df)

    print("Computing Table T6 (Brooklyn deep dive)...")
    t6_df, t6_details = compute_t6(df_raw, daily_df)

    print("Computing Table T7 (Evidence summary)...")
    t7_df = compute_t7(t1_df, t4_df, t5_df)

    # Print tables
    print("\n" + "=" * 70)
    print("TABLE T1: BOROUGH SUMMARY")
    print("=" * 70)
    print(t1_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLE T2: CORRELATIONS (ALL DAYS)")
    print("=" * 70)
    print(t2_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLE T3: TOP 10 FLOOD SPIKE DAYS PER BOROUGH")
    print("=" * 70)
    print(t3_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLE T4: ROBUST CORRELATIONS (EXCLUDING SPIKE_NO_RAIN DAYS)")
    print("=" * 70)
    print(t4_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLE T5: STORM DAYS (RAIN >= 20 MM)")
    print("=" * 70)
    print(t5_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLE T6: BROOKLYN DEEP DIVE (SUMMARY)")
    print("=" * 70)
    print(t6_df.to_string(index=False))

    for d in ["2026-05-20", "2026-05-21", "2026-07-06"]:
        print(f"\n--- Brooklyn Deep Dive Details: {d} ---")
        det = t6_details[d]
        print("Complaint Types & Descriptors:")
        print(det["type_desc"].to_string(index=False))
        print("\nHourly Distribution (hour 0..23):")
        print(det["hours"].to_string())
        print("\nTop 5 Incident Addresses:")
        print(det["top5_addresses"].to_string())

    print("\n" + "=" * 70)
    print("TABLE T7: EVIDENCE SUMMARY (SORTED BY WEAKEST GROUP SIZE)")
    print("=" * 70)
    print(t7_df.to_string(index=False))


if __name__ == "__main__":
    main()
