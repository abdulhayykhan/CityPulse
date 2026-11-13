"""
CityPulse - Part 3, Step 2, Chunk 2.4: Label Readiness & Spatial Duplicate Analysis
===================================================================================
Diagnostic utilities for evaluating label readiness, duplicate complaints,
and hourly meteorological correlations in Brooklyn:
- Hourly precipitation-flood lag and rolling correlation analysis
- Chained vs. non-chaining spatial-temporal incident clustering
- Coordinate-level duplicate detection and spatial spread evaluation
- Administrative reporting channel and location taxonomy distributions
- Label maturity, lifecycle resolution rate, and closure duration tracking
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

GROUPS = ["flood", "noise", "road", "sewer_backup"]


def build_hourly_flood_rain(
    clean_df: pd.DataFrame,
    rain_hourly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct continuous 8760-hour Brooklyn dataset merging hourly precipitation
    with zero-filled flood complaint counts, including lags and rolling sums.
    """
    rain = rain_hourly_df.copy()
    rain["time_dt"] = pd.to_datetime(rain["time"])
    rain = rain.sort_values("time_dt").reset_index(drop=True)

    flood_df = clean_df[clean_df["group"] == "flood"].copy()
    flood_df["time_dt"] = pd.to_datetime(flood_df["timestamp"]).dt.floor("h")

    hourly_counts = flood_df.groupby("time_dt").size().rename("flood_count")
    merged = rain.merge(hourly_counts, on="time_dt", how="left")
    merged["flood_count"] = merged["flood_count"].fillna(0).astype(int)

    # Precipitation features
    merged["rain_lag_0h"] = merged["precip_mm"]
    merged["rain_lag_1h"] = merged["precip_mm"].shift(1)
    merged["rain_lag_2h"] = merged["precip_mm"].shift(2)
    merged["rain_lag_3h"] = merged["precip_mm"].shift(3)
    merged["rain_lag_6h"] = merged["precip_mm"].shift(6)

    merged["rain_rolling_3h"] = merged["precip_mm"].rolling(3, min_periods=1).sum()
    merged["rain_rolling_6h"] = merged["precip_mm"].rolling(6, min_periods=1).sum()
    merged["rain_rolling_24h"] = merged["precip_mm"].rolling(24, min_periods=1).sum()

    merged["date"] = merged["time_dt"].dt.strftime("%Y-%m-%d")
    merged["hour"] = merged["time_dt"].dt.hour
    return merged


def compute_hourly_correlations(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations of hourly flood count vs precipitation lags and rolling windows.
    """
    features = [
        "rain_lag_0h", "rain_lag_1h", "rain_lag_2h", "rain_lag_3h", "rain_lag_6h",
        "rain_rolling_3h", "rain_rolling_6h", "rain_rolling_24h"
    ]
    records = []
    fc = hourly_df["flood_count"]

    for feat in features:
        sub = hourly_df.dropna(subset=[feat])
        r_p, p_p = pearsonr(sub["flood_count"], sub[feat])
        r_s, p_s = spearmanr(sub["flood_count"], sub[feat])
        records.append({
            "feature": feat,
            "n_hours": len(sub),
            "pearson_r": round(float(r_p), 4),
            "pearson_p": f"{p_p:.2e}",
            "spearman_r": round(float(r_s), 4),
            "spearman_p": f"{p_s:.2e}",
        })
    return pd.DataFrame(records)


def compute_daily_correlations(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to 365 calendar days and compute correlations vs flood count:
    - rain_sum
    - rain_max_hourly
    - rain_hours_ge_5mm
    """
    daily = hourly_df.groupby("date").agg(
        flood_count=("flood_count", "sum"),
        rain_sum=("precip_mm", "sum"),
        rain_max_hourly=("precip_mm", "max"),
        rain_hours_ge_5mm=("precip_mm", lambda s: int((s >= 5.0).sum())),
    ).reset_index()

    fc = daily["flood_count"]
    metrics = ["rain_sum", "rain_max_hourly", "rain_hours_ge_5mm"]
    records = []

    for m in metrics:
        r_p, p_p = pearsonr(fc, daily[m])
        r_s, p_s = spearmanr(fc, daily[m])
        records.append({
            "metric": m,
            "pearson_r": round(float(r_p), 4),
            "pearson_p": f"{p_p:.2e}",
            "spearman_r": round(float(r_s), 4),
            "spearman_p": f"{p_s:.2e}",
        })
    return pd.DataFrame(records)


def compute_storm_days_summary(
    hourly_df: pd.DataFrame,
    target_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarize meteorological and incident metrics for peak precipitation event days.
    """
    if target_dates is None:
        target_dates = ["2026-05-20", "2026-05-21", "2026-07-06"]

    records = []
    for d in target_dates:
        sub = hourly_df[hourly_df["date"] == d].sort_values("hour")
        d_rain_sum = sub["precip_mm"].sum()
        max_rain = sub["precip_mm"].max()
        hr_max_rain = int(sub.loc[sub["precip_mm"].idxmax(), "hour"])
        max_flood = sub["flood_count"].max()
        hr_max_flood = int(sub.loc[sub["flood_count"].idxmax(), "hour"])
        d_flood_sum = sub["flood_count"].sum()

        records.append({
            "date": d,
            "daily_rain_sum_mm": round(float(d_rain_sum), 1),
            "max_hourly_rain_mm": round(float(max_rain), 1),
            "hour_of_max_rain": hr_max_rain,
            "max_hourly_flood_count": int(max_flood),
            "hour_of_max_flood": hr_max_flood,
            "daily_flood_count": int(d_flood_sum),
        })
    return pd.DataFrame(records)


def compute_weak_clusters_chained(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized chained clustering for records with non-empty normalized address:
    Key = (group, address_norm, complaint_type).
    A record joins the previous record's cluster when gap <= 24 hours.
    """
    df = clean_df[clean_df["address_norm"].str.strip() != ""].copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")

    sort_cols = ["group", "address_norm", "complaint_type", "timestamp_dt"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    key_cols = ["group", "address_norm", "complaint_type"]
    same_key = (df[key_cols] == df[key_cols].shift(1)).all(axis=1)
    time_diff_h = (df["timestamp_dt"] - df["timestamp_dt"].shift(1)).dt.total_seconds() / 3600.0

    new_cluster = (~same_key) | (time_diff_h > 24.0) | time_diff_h.isna()
    df["cluster_id"] = new_cluster.cumsum()

    summary_records = []
    for grp in GROUPS:
        sub = df[df["group"] == grp]
        rows_with_addr = len(sub)
        c_sizes = sub.groupby("cluster_id").size()
        n_clusters = len(c_sizes)
        c_ge2 = c_sizes[c_sizes >= 2]
        n_ge2 = len(c_ge2)
        rows_ge2 = int(c_ge2.sum())
        share_ge2 = round(rows_ge2 / rows_with_addr * 100.0, 2) if rows_with_addr > 0 else 0.0

        s2 = int((c_sizes == 2).sum())
        s3 = int((c_sizes == 3).sum())
        s4 = int((c_sizes == 4).sum())
        s5p = int((c_sizes >= 5).sum())
        max_sz = int(c_sizes.max()) if len(c_sizes) > 0 else 0

        summary_records.append({
            "group": grp,
            "rows_with_addr": rows_with_addr,
            "n_clusters": n_clusters,
            "clusters_ge_2": n_ge2,
            "rows_in_clusters_ge_2": rows_ge2,
            "share_rows_ge_2_pct": share_ge2,
            "size_2": s2,
            "size_3": s3,
            "size_4": s4,
            "size_5plus": s5p,
            "max_cluster_size": max_sz,
        })

    return pd.DataFrame(summary_records), df


def compute_cluster_alternatives(clean_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare chained clustering with non-chaining alternatives:
    - Alternative A: Calendar-day clusters (key + created date)
    - Alternative B: Anchor windows (starts at first row, includes rows within 24h of anchor)
    """
    df = clean_df[clean_df["address_norm"].str.strip() != ""].copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

    groups = ["noise", "road", "flood", "sewer_backup"]
    records = []

    # Alternative A overall
    cal_clusters_all = df.groupby(["group", "address_norm", "complaint_type", "date"]).size()
    max_a = int(cal_clusters_all.max())

    max_b_overall = 1
    for grp in groups:
        gdf = df[df["group"] == grp].copy()
        gdf_sorted = gdf.sort_values(["address_norm", "complaint_type", "timestamp_dt"]).reset_index(drop=True)

        same_key = (
            (gdf_sorted["address_norm"] == gdf_sorted["address_norm"].shift(1)) &
            (gdf_sorted["complaint_type"] == gdf_sorted["complaint_type"].shift(1))
        )
        time_diff = gdf_sorted["timestamp_dt"].diff().dt.total_seconds()
        new_clust = (~same_key) | (time_diff > 24 * 3600)
        chained_id = new_clust.cumsum()
        ch_sizes = chained_id.value_counts()
        ch_max = int(ch_sizes.max()) if len(ch_sizes) > 0 else 0
        ch_n = int(chained_id.nunique())

        cal_clusters = gdf.groupby(["address_norm", "complaint_type", "date"]).size()
        cal_max = int(cal_clusters.max()) if len(cal_clusters) > 0 else 0
        cal_n = int(len(cal_clusters))

        key_ids = (~same_key).cumsum().values
        ts_int = gdf_sorted["timestamp_dt"].values.astype("datetime64[ns]").astype(np.int64)
        boundaries = np.where(np.diff(key_ids) != 0)[0] + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(key_ids)]
        delta_24h = int(24 * 3600 * 1e9)

        anc_max = 1
        anc_n = 0
        for s, e in zip(starts, ends):
            i = s
            while i < e:
                anc_n += 1
                anchor_t = ts_int[i]
                limit_t = anchor_t + delta_24h
                j = i + 1
                while j < e and ts_int[j] <= limit_t:
                    j += 1
                sz = j - i
                if sz > anc_max:
                    anc_max = sz
                i = j

        if anc_max > max_b_overall:
            max_b_overall = anc_max

        records.append({
            "group": grp,
            "chained_max_size": ch_max,
            "chained_n_clusters": ch_n,
            "calendar_day_max_size": cal_max,
            "calendar_day_n_clusters": cal_n,
            "anchor_24h_max_size": anc_max,
            "anchor_24h_n_clusters": anc_n,
        })

    table_df = pd.DataFrame(records)

    return {
        "max_size_calendar_day": max_a,
        "max_size_anchor_window": max_b_overall,
        "comparison_table": table_df,
        "ran_anchor_window": True,
    }


def compute_spatial_spread(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Compute maximum pairwise spatial distance inside clusters of size >= 2 using (x_m, y_m).
    """
    c_sizes = df_clustered.groupby("cluster_id").size()
    ge2_ids = c_sizes[c_sizes >= 2].index
    sub = df_clustered[df_clustered["cluster_id"].isin(ge2_ids)].copy()

    # Identify clusters with identical coords
    x_min = sub.groupby("cluster_id")["x_m"].min()
    x_max = sub.groupby("cluster_id")["x_m"].max()
    y_min = sub.groupby("cluster_id")["y_m"].min()
    y_max = sub.groupby("cluster_id")["y_m"].max()
    same_coords = (x_min == x_max) & (y_min == y_max)

    max_dists = {cid: 0.0 for cid in ge2_ids[same_coords]}
    diff_ids = ge2_ids[~same_coords]
    diff_df = sub[sub["cluster_id"].isin(diff_ids)]

    for cid, grp in diff_df.groupby("cluster_id"):
        pts = grp[["x_m", "y_m"]].values
        dists = pdist(pts)
        max_dists[cid] = float(dists.max())

    cluster_grp = sub.groupby("cluster_id")["group"].first()
    spread_df = pd.DataFrame({"group": cluster_grp, "max_dist": pd.Series(max_dists)})

    records = []
    # Overall ALL
    all_d = spread_df["max_dist"]
    all_le_05 = int((all_d <= 0.5).sum())
    records.append({
        "group": "ALL",
        "n_clusters_ge_2": len(all_d),
        "p50_dist_m": round(float(all_d.median()), 2),
        "p90_dist_m": round(float(all_d.quantile(0.9)), 2),
        "p99_dist_m": round(float(all_d.quantile(0.99)), 2),
        "max_dist_m": round(float(all_d.max()), 2),
        "clusters_le_0_5m": all_le_05,
        "share_le_0_5m_pct": round(all_le_05 / len(all_d) * 100.0, 2),
    })

    for grp in GROUPS:
        g_d = spread_df[spread_df["group"] == grp]["max_dist"]
        n_c = len(g_d)
        le_05 = int((g_d <= 0.5).sum())
        records.append({
            "group": grp,
            "n_clusters_ge_2": n_c,
            "p50_dist_m": round(float(g_d.median()), 2) if n_c > 0 else 0.0,
            "p90_dist_m": round(float(g_d.quantile(0.9)), 2) if n_c > 0 else 0.0,
            "p99_dist_m": round(float(g_d.quantile(0.99)), 2) if n_c > 0 else 0.0,
            "max_dist_m": round(float(g_d.max()), 2) if n_c > 0 else 0.0,
            "clusters_le_0_5m": le_05,
            "share_le_0_5m_pct": round(le_05 / n_c * 100.0, 2) if n_c > 0 else 0.0,
        })

    return pd.DataFrame(records)


def compute_coord_duplicates(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate complaints at exact identical coordinates and complaint_type within 24h.
    """
    df = clean_df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df_sorted = df.sort_values(["lat", "lon", "complaint_type", "timestamp_dt"]).reset_index(drop=True)

    match_prev = (
        (df_sorted["lat"] == df_sorted["lat"].shift(1)) &
        (df_sorted["lon"] == df_sorted["lon"].shift(1)) &
        (df_sorted["complaint_type"] == df_sorted["complaint_type"].shift(1))
    )
    match_next = (
        (df_sorted["lat"] == df_sorted["lat"].shift(-1)) &
        (df_sorted["lon"] == df_sorted["lon"].shift(-1)) &
        (df_sorted["complaint_type"] == df_sorted["complaint_type"].shift(-1))
    )

    gap_prev_h = (df_sorted["timestamp_dt"] - df_sorted["timestamp_dt"].shift(1)).dt.total_seconds() / 3600.0
    gap_next_h = (df_sorted["timestamp_dt"].shift(-1) - df_sorted["timestamp_dt"]).dt.total_seconds() / 3600.0

    is_dup = (match_prev & (gap_prev_h <= 24.0)) | (match_next & (gap_next_h <= 24.0))
    df_sorted["is_coord_dup_24h"] = is_dup

    records = []
    for grp in GROUPS:
        sub = df_sorted[df_sorted["group"] == grp]
        tot = len(sub)
        n_dup = int(sub["is_coord_dup_24h"].sum())
        share_dup = round(n_dup / tot * 100.0, 2)
        dup_sub = sub[sub["is_coord_dup_24h"]]
        n_with_addr = int((dup_sub["address_norm"].str.strip() != "").sum())
        share_with_addr = round(n_with_addr / n_dup * 100.0, 2) if n_dup > 0 else 0.0

        records.append({
            "group": grp,
            "total_rows": tot,
            "rows_with_coord_dup_24h": n_dup,
            "share_coord_dup_pct": share_dup,
            "dup_rows_with_address": n_with_addr,
            "share_dup_with_address_pct": share_with_addr,
        })
    return pd.DataFrame(records)


def compute_distinct_coordinates(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total rows, distinct (lat, lon) coordinates, and ratio per group.
    """
    records = []
    for grp in GROUPS:
        sub = clean_df[clean_df["group"] == grp]
        tot = len(sub)
        distinct_coords = sub[["lat", "lon"]].drop_duplicates().shape[0]
        ratio = round(tot / distinct_coords, 3)
        records.append({
            "group": grp,
            "total_rows": tot,
            "distinct_lat_lon_pairs": distinct_coords,
            "ratio_rows_per_coord": ratio,
        })
    return pd.DataFrame(records)


def compute_channel_and_location_stats(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute top 10 channels and location types per group and missing location_type percentages.
    """
    ch_records = []
    loc_records = []
    missing_loc_records = []

    for grp in GROUPS:
        sub = clean_df[clean_df["group"] == grp]
        tot = len(sub)

        # Channel
        ch_counts = sub["channel"].value_counts().head(10)
        for rank, (ch, cnt) in enumerate(ch_counts.items(), 1):
            ch_records.append({
                "group": grp,
                "rank": rank,
                "channel": ch,
                "count": int(cnt),
                "share_pct": round(cnt / tot * 100.0, 2),
            })

        # Location Type
        sub_loc = sub[sub["location_type"].str.strip() != ""]
        loc_counts = sub_loc["location_type"].value_counts().head(10)
        for rank, (lt, cnt) in enumerate(loc_counts.items(), 1):
            loc_records.append({
                "group": grp,
                "rank": rank,
                "location_type": lt,
                "count": int(cnt),
                "share_pct": round(cnt / tot * 100.0, 2),
            })

        # Missing location_type
        n_missing_loc = int((sub["location_type"].str.strip() == "").sum())
        missing_loc_records.append({
            "group": grp,
            "total_rows": tot,
            "missing_location_type_count": n_missing_loc,
            "missing_location_type_pct": round(n_missing_loc / tot * 100.0, 2),
        })

    return pd.DataFrame(ch_records), pd.DataFrame(loc_records), pd.DataFrame(missing_loc_records)


def compute_label_readiness_tables(clean_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute monthly and weekly label readiness, time-to-close distribution, and date extrema.
    """
    df = clean_df.copy()
    df["created_dt"] = pd.to_datetime(df["timestamp"])
    df["date_dt"] = pd.to_datetime(df["date"])
    df["time_to_close_h"] = pd.to_numeric(df["time_to_close_h"], errors="coerce")

    # Monthly readiness
    monthly_records = []
    months = sorted(df["month"].astype(int).unique())
    for m in months:
        m_df = df[df["month"].astype(int) == m]
        for grp in GROUPS:
            sub = m_df[m_df["group"] == grp]
            tot = len(sub)
            if tot == 0:
                continue
            pct_closed = round((sub["status"] == "Closed").mean() * 100.0, 2)
            pct_res = round((sub["resolution_id"].str.strip() != "").mean() * 100.0, 2)
            pct_cdate = round((sub["closed_date"].str.strip() != "").mean() * 100.0, 2)
            monthly_records.append({
                "month": m,
                "group": grp,
                "total_rows": tot,
                "pct_closed": pct_closed,
                "pct_with_resolution_id": pct_res,
                "pct_with_closed_date": pct_cdate,
            })
    monthly_df = pd.DataFrame(monthly_records)

    # Weekly readiness (Monday start)
    df["week_start"] = (df["date_dt"] - pd.to_timedelta(df["date_dt"].dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    weekly_records = []
    unique_weeks = sorted(df["week_start"].unique())
    target_weeks = [w for w in unique_weeks if w >= "2026-08-17" and w <= "2026-10-26"]

    for w in target_weeks:
        w_df = df[df["week_start"] == w]
        for grp in GROUPS:
            sub = w_df[w_df["group"] == grp]
            tot = len(sub)
            pct_res = round((sub["resolution_id"].str.strip() != "").mean() * 100.0, 2) if tot > 0 else 0.0
            pct_cdate = round((sub["closed_date"].str.strip() != "").mean() * 100.0, 2) if tot > 0 else 0.0
            weekly_records.append({
                "week_start": w,
                "group": grp,
                "total_rows": tot,
                "pct_with_resolution_id": pct_res,
                "pct_with_closed_date": pct_cdate,
            })
    weekly_df = pd.DataFrame(weekly_records)

    # Time to close among rows with non-empty closed_date
    ttc_records = []
    for grp in GROUPS:
        sub = df[(df["group"] == grp) & (df["closed_date"].str.strip() != "")]
        n_closed = len(sub)
        med = round(float(sub["time_to_close_h"].median()), 2)
        p90 = round(float(sub["time_to_close_h"].quantile(0.9)), 2)
        p99 = round(float(sub["time_to_close_h"].quantile(0.99)), 2)
        gt14 = round(float((sub["time_to_close_h"] > 14 * 24).mean() * 100.0), 2)
        ttc_records.append({
            "group": grp,
            "rows_with_closed_date": n_closed,
            "median_ttc_h": med,
            "p90_ttc_h": p90,
            "p99_ttc_h": p99,
            "share_gt_14d_pct": gt14,
        })
    ttc_df = pd.DataFrame(ttc_records)

    # Date extrema
    extrema_records = []
    for grp in GROUPS:
        sub = df[df["group"] == grp]
        first_t = sub["timestamp"].min()
        last_t = sub["timestamp"].max()
        closed_sub = sub[sub["closed_date"].str.strip() != ""]
        latest_c = closed_sub["closed_date"].max() if len(closed_sub) > 0 else ""
        extrema_records.append({
            "group": grp,
            "first_timestamp": first_t,
            "last_timestamp": last_t,
            "latest_closed_date": latest_c,
        })
    extrema_df = pd.DataFrame(extrema_records)

    return {
        "monthly": monthly_df,
        "weekly": weekly_df,
        "time_to_close": ttc_df,
        "extrema": extrema_df,
    }


def plot_storm_days_hourly(
    hourly_df: pd.DataFrame,
    output_path: Path,
    target_dates: Optional[List[str]] = None,
) -> None:
    """
    Generate three-panel plot showing precipitation bars and complaint line per hour for storm days.
    """
    if target_dates is None:
        target_dates = ["2026-05-20", "2026-05-21", "2026-07-06"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    for ax, d in zip(axes, target_dates):
        sub = hourly_df[hourly_df["date"] == d].sort_values("hour")
        ax2 = ax.twinx()

        # Precipitation bars
        ax.bar(sub["hour"], sub["precip_mm"], width=0.6, color="#3498db", alpha=0.6, label="Precipitation (mm)")
        # Flood complaints line
        ax2.plot(sub["hour"], sub["flood_count"], color="#e74c3c", marker="o", linewidth=2.0, label="Flood Complaints")

        ax.set_ylabel("Rain (mm)", color="#2980b9", fontsize=9)
        ax2.set_ylabel("Flood Complaints", color="#c0392b", fontsize=9)
        ax.set_title(f"Hourly Event Dynamics: {d}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(0, 24))
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("Hour of Day (0..23)", fontsize=10)
    plt.suptitle("Precipitation Intensity and Incident Volume on High-Impact Days", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_weekly_resolution_rate(
    weekly_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot weekly percentage of tickets with resolution_id per group.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"noise": "#3498db", "road": "#e67e22", "flood": "#2ecc71", "sewer_backup": "#9b59b6"}

    for grp in GROUPS:
        sub = weekly_df[weekly_df["group"] == grp]
        ax.plot(
            sub["week_start"],
            sub["pct_with_resolution_id"],
            marker="o",
            linewidth=2.0,
            label=grp.capitalize(),
            color=colors.get(grp, "#333333"),
        )

    ax.set_title("Weekly Label Readiness: % Complaints with Resolution ID (Aug - Oct 2026)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Week Start Date", fontsize=10)
    ax.set_ylabel("% with Resolution ID", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

