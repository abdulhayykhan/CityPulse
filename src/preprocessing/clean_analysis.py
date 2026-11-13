"""
CityPulse - Part 3, Step 2: Clean Dataset Analysis & Diagnostics Helpers
========================================================================
Diagnostic utilities for validating complaints_clean.csv:
- Descriptor group recomputation and comparison against reference
- Target leakage feature audit
- Temporal group distributions and time-based train/test splitting
- Diagnostic visual distributions
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUPS = ["flood", "noise", "road", "sewer_backup"]


def recompute_descriptor_group_summary(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute (group, complaint_type, descriptor, descriptor_group) counts from clean dataset.
    """
    summary = (
        df_clean.groupby(["group", "complaint_type", "descriptor", "descriptor_group"])
        .size()
        .reset_index(name="n")
        .sort_values(by=["group", "n"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def compute_group_month_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute group x month contingency table.
    """
    pivot = df_clean.pivot_table(
        index="month",
        columns="group",
        values="id",
        aggfunc="count",
        fill_value=0,
    )
    pivot["ALL"] = pivot.sum(axis=1)
    return pivot.reset_index()


def compute_descriptor_group_shares(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptor_group row counts and percentage shares within each group.
    """
    records = []
    for grp in GROUPS:
        sub = df_clean[df_clean["group"] == grp]
        tot = len(sub)
        counts = sub["descriptor_group"].value_counts()
        for dg, cnt in counts.items():
            records.append({
                "group": grp,
                "descriptor_group": dg,
                "count": int(cnt),
                "total_in_group": tot,
                "share_pct": round(cnt / tot * 100.0, 2),
            })
    return pd.DataFrame(records)


def compute_time_split(
    df_clean: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Sort 365 calendar days chronologically and compute time-based split:
    Train = earliest 80% (292 days), Test = latest 20% (73 days).
    """
    unique_dates = sorted(df_clean["date"].unique())
    n_days = len(unique_dates)
    n_train_days = int(round(n_days * train_frac))

    train_dates = set(unique_dates[:n_train_days])
    test_dates = set(unique_dates[n_train_days:])

    split_info = {
        "total_days": n_days,
        "train_days": n_train_days,
        "test_days": n_days - n_train_days,
        "train_start": unique_dates[0],
        "train_end": unique_dates[n_train_days - 1],
        "test_start": unique_dates[n_train_days],
        "test_end": unique_dates[-1],
    }

    records = []
    for grp in GROUPS:
        sub = df_clean[df_clean["group"] == grp]
        tr_cnt = int(sub["date"].isin(train_dates).sum())
        te_cnt = int(sub["date"].isin(test_dates).sum())
        tot = tr_cnt + te_cnt
        records.append({
            "group": grp,
            "train_count": tr_cnt,
            "test_count": te_cnt,
            "total_count": tot,
            "train_share_pct": round(tr_cnt / tot * 100.0, 2),
            "test_share_pct": round(te_cnt / tot * 100.0, 2),
        })

    # Summary row for ALL
    all_tr = int(df_clean["date"].isin(train_dates).sum())
    all_te = int(df_clean["date"].isin(test_dates).sum())
    all_tot = all_tr + all_te
    records.append({
        "group": "ALL",
        "train_count": all_tr,
        "test_count": all_te,
        "total_count": all_tot,
        "train_share_pct": round(all_tr / all_tot * 100.0, 2),
        "test_share_pct": round(all_te / all_tot * 100.0, 2),
    })

    return pd.DataFrame(records), split_info


def plot_descriptor_group_shares(
    share_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot horizontal bars of descriptor_group percentage share for each group.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, grp in zip(axes, GROUPS):
        sub = share_df[share_df["group"] == grp].sort_values("share_pct", ascending=True)
        bars = ax.barh(sub["descriptor_group"], sub["share_pct"], color="#2c3e50", edgecolor="black", alpha=0.85)
        ax.set_title(f"Group: {grp.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Share within Group (%)", fontsize=9)
        ax.set_xlim(0, max(sub["share_pct"].max() * 1.15, 10))
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8)

    plt.suptitle("Descriptor Group Distribution Across Target Groups", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_weekly_trend_split(
    df_clean: pd.DataFrame,
    split_date_str: str,
    output_path: Path,
) -> None:
    """
    Plot weekly complaint counts per group with train/test cutoff line.
    """
    df = df_clean.copy()
    df["created_dt"] = pd.to_datetime(df["timestamp"])
    df["week_start"] = (df["created_dt"] - pd.to_timedelta(df["created_dt"].dt.dayofweek, unit="D")).dt.floor("D")

    weekly = df.pivot_table(
        index="week_start",
        columns="group",
        values="id",
        aggfunc="count",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"noise": "#3498db", "road": "#e67e22", "flood": "#2ecc71", "sewer_backup": "#9b59b6"}

    for grp in GROUPS:
        if grp in weekly.columns:
            ax.plot(weekly.index, weekly[grp], label=grp.capitalize(), color=colors.get(grp, "#333333"), linewidth=1.8)

    split_dt = pd.to_datetime(split_date_str)
    ax.axvline(split_dt, color="#c0392b", linestyle="--", linewidth=2.0, label=f"Train/Test Split ({split_date_str})")

    ax.set_title("Weekly Complaint Volume by Group with Train/Test Split Boundary", fontsize=12, fontweight="bold")
    ax.set_xlabel("Week Starting Date", fontsize=10)
    ax.set_ylabel("Weekly Service Requests", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

