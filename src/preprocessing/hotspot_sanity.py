"""
CityPulse - Hotspot Sanity & Spatial Concentration Diagnostics (Chunk 2.6)
==========================================================================
Analysis and diagnostic utilities for:
- Spatial concentration and high-volume (lat, lon) coordinates
- Address normalization and distinct coordinate mapping
- Extreme coordinate filtering thresholds (>50, >200, >1000 rows)
- Chained weak cluster identification (24-hour gap rule)
- Weekly temporal evolution of the largest persistent cluster
- Projection origin proximity sanity check
- Effective row retention under coordinate count capping.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUPS = ["flood", "noise", "road", "sewer_backup"]


def load_clean_and_projection(clean_path: Path, proj_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load complaints_clean.csv and projection.json with strict dtypes and parsing."""
    df = pd.read_csv(
        clean_path,
        dtype={
            "id": str,
            "resolution_id": str,
            "group": str,
            "descriptor": str,
            "descriptor_group": str,
            "complaint_type": str,
            "borough": str,
            "status": str,
            "channel": str,
            "location_type": str,
            "incident_address": str,
            "address_norm": str,
            "date": str,
        },
        keep_default_na=False,
        parse_dates=["timestamp"],
    )
    with open(proj_path, "r", encoding="utf-8") as f:
        proj_info = json.load(f)
    return df, proj_info


def compute_top_coordinates(df: pd.DataFrame, n: int = 10) -> Dict[str, pd.DataFrame]:
    """Compute top N (lat, lon) pairs overall and per group."""
    def _top_coords_for_sub(sub: pd.DataFrame, group_label: str, total_count: int) -> pd.DataFrame:
        counts = sub.groupby(["lat", "lon"]).size().sort_values(ascending=False).head(n)
        results = []
        for rank, ((lat, lon), rows) in enumerate(counts.items(), start=1):
            c_rows = sub[(sub["lat"] == lat) & (sub["lon"] == lon)]
            distinct_addr = c_rows["address_norm"].nunique()
            addr_counts = c_rows["address_norm"].value_counts()
            top_addr = addr_counts.index[0] if len(addr_counts) > 0 else ""
            if top_addr == "":
                top_addr = "[EMPTY]"
            share = float(rows / total_count * 100.0) if total_count > 0 else 0.0

            results.append({
                "group": group_label,
                "rank": rank,
                "lat": float(lat),
                "lon": float(lon),
                "rows": int(rows),
                "share_pct": share,
                "distinct_address_norm": distinct_addr,
                "top_address": top_addr,
            })
        return pd.DataFrame(results)

    tables = {"overall": _top_coords_for_sub(df, "overall", len(df))}
    for g in GROUPS:
        g_df = df[df["group"] == g]
        tables[g] = _top_coords_for_sub(g_df, g, len(g_df))
    return tables


def compute_top_addresses(df: pd.DataFrame, n: int = 10) -> Dict[str, pd.DataFrame]:
    """Compute top N address_norm values overall and per group."""
    def _top_addrs_for_sub(sub: pd.DataFrame, group_label: str, total_count: int) -> pd.DataFrame:
        counts = sub.groupby("address_norm").size().sort_values(ascending=False).head(n)
        results = []
        for rank, (addr, rows) in enumerate(counts.items(), start=1):
            a_rows = sub[sub["address_norm"] == addr]
            distinct_coords = a_rows.groupby(["lat", "lon"]).ngroups
            share = float(rows / total_count * 100.0) if total_count > 0 else 0.0
            display_addr = addr if addr != "" else "[EMPTY]"

            results.append({
                "group": group_label,
                "rank": rank,
                "address_norm": display_addr,
                "rows": int(rows),
                "share_pct": share,
                "distinct_coords": distinct_coords,
            })
        return pd.DataFrame(results)

    tables = {"overall": _top_addrs_for_sub(df, "overall", len(df))}
    for g in GROUPS:
        g_df = df[df["group"] == g]
        tables[g] = _top_addrs_for_sub(g_df, g, len(g_df))
    return tables


def compute_coordinate_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute concentration of rows at coordinates exceeding 50, 200, and 1000 rows per group."""
    df_group_coords = df.groupby(["group", "lat", "lon"]).size().reset_index(name="group_coord_count")
    overall_coord_counts = df.groupby(["lat", "lon"]).size().rename("overall_coord_count")
    df_with_overall = df.merge(overall_coord_counts, on=["lat", "lon"], how="left")

    rows = []
    for g in GROUPS:
        g_coords = df_group_coords[df_group_coords["group"] == g]
        total_g_rows = g_coords["group_coord_count"].sum()
        g_all_rows = df_with_overall[df_with_overall["group"] == g]

        for th in [50, 200, 1000]:
            m_within = g_coords[g_coords["group_coord_count"] > th]
            n_within = len(m_within)
            rows_within = int(m_within["group_coord_count"].sum())
            share_within = float(rows_within / total_g_rows * 100.0) if total_g_rows > 0 else 0.0

            m_overall = g_all_rows[g_all_rows["overall_coord_count"] > th]
            n_overall = m_overall.groupby(["lat", "lon"]).ngroups
            rows_overall = len(m_overall)
            share_overall = float(rows_overall / total_g_rows * 100.0) if total_g_rows > 0 else 0.0

            rows.append({
                "group": g,
                "threshold": f">{th}",
                "coords_within_group": n_within,
                "rows_within_group": rows_within,
                "share_within_group_pct": share_within,
                "coords_overall_dataset": n_overall,
                "rows_overall_dataset": rows_overall,
                "share_overall_dataset_pct": share_overall,
            })

    return pd.DataFrame(rows)


def compute_percentiles_per_group(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rows per distinct coordinate percentiles (p50, p90, p99, max) per group and overall."""
    rows = []
    for g in GROUPS + ["overall"]:
        if g == "overall":
            counts = df.groupby(["lat", "lon"]).size()
        else:
            counts = df[df["group"] == g].groupby(["lat", "lon"]).size()

        rows.append({
            "group": g,
            "distinct_coords": len(counts),
            "total_rows": int(counts.sum()),
            "p50": float(np.percentile(counts, 50)),
            "p90": float(np.percentile(counts, 90)),
            "p99": float(np.percentile(counts, 99)),
            "max": int(counts.max()),
        })
    return pd.DataFrame(rows)


def compute_weak_clusters(df: pd.DataFrame, gap_hours: float = 24.0) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute chained weak clusters using key=(group, address_norm, complaint_type),
    excluding empty addresses, with gap <= gap_hours.
    Returns:
    1. Top 5 clusters per group
    2. Largest cluster series overall
    3. Rows of the largest cluster
    """
    df_nonempty = df[df["address_norm"].str.strip() != ""].copy()
    keys = ["group", "address_norm", "complaint_type"]
    df_nonempty = df_nonempty.sort_values(by=keys + ["timestamp"]).reset_index(drop=True)

    df_nonempty["prev_time"] = df_nonempty.groupby(keys)["timestamp"].shift(1)
    df_nonempty["gap_h"] = (df_nonempty["timestamp"] - df_nonempty["prev_time"]).dt.total_seconds() / 3600.0
    df_nonempty["new_cluster"] = df_nonempty["gap_h"].isna() | (df_nonempty["gap_h"] > gap_hours)
    df_nonempty["cluster_id"] = df_nonempty.groupby(keys)["new_cluster"].cumsum()
    df_nonempty["coord_str"] = df_nonempty["lat"].astype(str) + "_" + df_nonempty["lon"].astype(str)

    cluster_aggs = df_nonempty.groupby(keys + ["cluster_id"]).agg(
        size=("id", "count"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        distinct_coords=("coord_str", "nunique"),
    ).reset_index()

    cluster_aggs["span_days"] = (cluster_aggs["last_timestamp"] - cluster_aggs["first_timestamp"]).dt.total_seconds() / 86400.0

    top5_list = []
    for g in GROUPS:
        g_clusters = cluster_aggs[cluster_aggs["group"] == g].sort_values(
            by=["size", "first_timestamp"], ascending=[False, True]
        ).head(5).copy()
        g_clusters["group_rank"] = range(1, len(g_clusters) + 1)
        top5_list.append(g_clusters)

    df_top5 = pd.concat(top5_list, ignore_index=True)

    largest_cluster = cluster_aggs.sort_values(by="size", ascending=False).iloc[0]
    large_rows = df_nonempty[
        (df_nonempty["group"] == largest_cluster["group"]) &
        (df_nonempty["address_norm"] == largest_cluster["address_norm"]) &
        (df_nonempty["complaint_type"] == largest_cluster["complaint_type"]) &
        (df_nonempty["cluster_id"] == largest_cluster["cluster_id"])
    ].copy()

    return df_top5, largest_cluster, large_rows


def compute_largest_cluster_weekly(large_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly complaint counts for the largest weak cluster."""
    lr = large_rows.copy()
    lr["week_start"] = lr["timestamp"].dt.to_period("W-MON").dt.start_time
    df_weekly = lr.groupby("week_start").size().reset_index(name="count")
    df_weekly["week_start_str"] = df_weekly["week_start"].dt.strftime("%Y-%m-%d")
    return df_weekly


def compute_origin_proximity(df: pd.DataFrame, proj_info: Dict[str, Any]) -> Tuple[pd.DataFrame, float]:
    """Compute count of rows within 20 m of projection origin per group and minimum distance."""
    df_dist = df.copy()
    df_dist["dist_origin"] = np.sqrt(df_dist["x_m"]**2 + df_dist["y_m"]**2)
    min_dist_overall = float(df_dist["dist_origin"].min())

    rows = []
    for g in GROUPS + ["overall"]:
        sub = df_dist if g == "overall" else df_dist[df_dist["group"] == g]
        n_within = int((sub["dist_origin"] <= 20.0).sum())
        min_d = float(sub["dist_origin"].min())
        rows.append({
            "group": g,
            "rows_within_20m": n_within,
            "min_dist_m": min_d,
        })
    return pd.DataFrame(rows), min_dist_overall


def compute_effective_rows_capping(df: pd.DataFrame, caps: List[int] = [10, 25, 50, 100]) -> pd.DataFrame:
    """Compute effective row count retained per group when capping rows per distinct coordinate."""
    rows = []
    for g in GROUPS:
        g_df = df[df["group"] == g]
        coord_sizes = g_df.groupby(["lat", "lon"]).size()
        r_dict = {"group": g, "total_rows": len(g_df)}
        for c in caps:
            r_dict[f"cap_{c}"] = int(coord_sizes.clip(upper=c).sum())
        rows.append(r_dict)
    return pd.DataFrame(rows)


def plot_weekly_largest_cluster(weekly_df: pd.DataFrame, out_path: Path) -> None:
    """Plot weekly complaint counts of the largest persistent weak cluster."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(weekly_df["week_start"], weekly_df["count"], marker="o", color="#d62728", linewidth=1.8, markersize=4)
    ax.set_title("Weekly Complaint Trajectory for Single Largest Weak Cluster (1869 83 STREET)", fontsize=11, pad=10)
    ax.set_xlabel("Week Start Date", fontsize=10)
    ax.set_ylabel("Weekly Complaint Volume", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_coordinate_histograms(df: pd.DataFrame, out_path: Path) -> None:
    """Plot log-log histograms of rows per coordinate per group."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axes = axes.flatten()

    for idx, g in enumerate(GROUPS):
        ax = axes[idx]
        counts = df[df["group"] == g].groupby(["lat", "lon"]).size()
        ax.hist(counts, bins=np.logspace(0, np.log10(counts.max() + 1), 30), color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Group: {g} (Max = {counts.max():,})", fontsize=10)
        ax.set_xlabel("Rows per Coordinate (log scale)", fontsize=9)
        ax.set_ylabel("Frequency (log scale)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Log-Log Distribution of Complaint Volume per Distinct Coordinate", fontsize=12, y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_noise_top300_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Plot spatial scatter of top 300 noise coordinates sized by complaint volume."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    noise_df = df[df["group"] == "noise"]
    top300 = noise_df.groupby(["x_m", "y_m"]).size().sort_values(ascending=False).head(300).reset_index(name="count")

    scatter = ax.scatter(
        top300["x_m"],
        top300["y_m"],
        s=np.sqrt(top300["count"]) * 4.0,
        c=top300["count"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Complaint Count", fontsize=10)

    ax.set_title("Spatial Distribution of Top 300 Noise Incident Hotspots in Brooklyn", fontsize=11, pad=10)
    ax.set_xlabel("Local X Coordinate (meters from origin)", fontsize=10)
    ax.set_ylabel("Local Y Coordinate (meters from origin)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

