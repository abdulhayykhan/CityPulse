"""
CityPulse - Part 3, Step 2: Multi-Borough Comparative Analysis Helpers
======================================================================
Supplemental analysis and visualization utilities for borough selection:
- Data loading and rolling precipitation feature engineering
- Bootstrap confidence intervals for Spearman correlation (resampling days)
- Confidence interval overlap evaluation
- Scatter panels of daily flood count vs precipitation
- Bootstrap CI visualization
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BOROUGH_ORDER = ["BROOKLYN", "QUEENS", "BRONX", "MANHATTAN", "STATEN ISLAND"]


def load_and_prepare_daily_data(flood_rain_csv: Path) -> pd.DataFrame:
    """
    Load flood_vs_rain_all_boroughs.csv and compute lagged/rolling precipitation features:
    - rain_prev_day: precipitation shifted by 1 day per borough (0-filled on day 1)
    - rain_3d_sum: 3-day rolling precipitation sum per borough
    """
    df = pd.read_csv(flood_rain_csv)
    processed_dfs = []
    for b_name in BOROUGH_ORDER:
        b_df = df[df["borough"] == b_name].sort_values("date").copy()
        b_df["rain_prev_day"] = b_df["rain_mm"].shift(1).fillna(0.0)
        b_df["rain_3d_sum"] = b_df["rain_mm"].rolling(3, min_periods=1).sum()
        processed_dfs.append(b_df)
    return pd.concat(processed_dfs, ignore_index=True)


def compute_bootstrap_ci(
    daily_df: pd.DataFrame,
    boroughs: Optional[List[str]] = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute 95% bootstrap confidence interval of Spearman 3-day-sum correlation
    per borough by resampling calendar days with replacement.
    """
    if boroughs is None:
        boroughs = BOROUGH_ORDER

    rng = np.random.default_rng(seed)
    results = []

    for b in boroughs:
        b_sub = daily_df[daily_df["borough"] == b].copy()
        fc = b_sub["flood_count"].values
        r3 = b_sub["rain_3d_sum"].values
        n_days = len(fc)

        point_corr, _ = spearmanr(fc, r3)

        boot_corrs = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n_days, size=n_days)
            r_sample, _ = spearmanr(fc[idx], r3[idx])
            boot_corrs.append(r_sample)

        ci_low, ci_high = np.percentile(boot_corrs, [2.5, 97.5])
        results.append({
            "borough": b,
            "spearman_3d_sum": float(point_corr),
            "ci_lower_95": float(ci_low),
            "ci_upper_95": float(ci_high),
            "margin_error": float((ci_high - ci_low) / 2.0),
        })

    return pd.DataFrame(results)


def check_ci_overlap(ci_a: Tuple[float, float], ci_b: Tuple[float, float]) -> bool:
    """Check whether two closed intervals [low_a, high_a] and [low_b, high_b] overlap."""
    return (ci_a[0] <= ci_b[1]) and (ci_b[0] <= ci_a[1])


def plot_borough_flood_vs_rain(
    daily_df: pd.DataFrame,
    output_path: Path,
    boroughs: Optional[List[str]] = None,
) -> None:
    """
    Generate five scatter panels of daily flood count (log y) vs same-day rain.
    """
    if boroughs is None:
        boroughs = BOROUGH_ORDER

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#d35400"]

    for ax, b, color in zip(axes, boroughs, colors):
        b_sub = daily_df[daily_df["borough"] == b]
        # Use log1p / pseudo-log on y to handle zero counts cleanly
        y_val = np.maximum(b_sub["flood_count"], 0.5)
        ax.scatter(b_sub["rain_mm"], y_val, alpha=0.6, color=color, s=20, edgecolor="none")
        ax.set_yscale("log")
        ax.set_title(b, fontsize=11, fontweight="bold")
        ax.set_xlabel("Same-Day Rain (mm)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Daily Flood Count (Log Scale)", fontsize=10)
    plt.suptitle("Daily Flood Count vs. Same-Day Precipitation Across NYC Boroughs", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spearman_ci(
    ci_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot 95% bootstrap confidence intervals for Spearman correlation vs 3-day rain sum.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_pos = np.arange(len(ci_df))
    y_err = [
        ci_df["spearman_3d_sum"] - ci_df["ci_lower_95"],
        ci_df["ci_upper_95"] - ci_df["spearman_3d_sum"],
    ]

    ax.bar(
        x_pos,
        ci_df["spearman_3d_sum"],
        yerr=y_err,
        capsize=5,
        color="#3498db",
        edgecolor="#2980b9",
        alpha=0.85,
        ecolor="#2c3e50",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ci_df["borough"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Spearman Correlation (3-Day Rain Sum)", fontsize=10)
    ax.set_title("Spearman Correlation vs. 3-Day Rain Sum with 95% Bootstrap CI", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

