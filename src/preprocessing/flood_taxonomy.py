"""
CityPulse - Flood History & Taxonomy Drift Analysis (Step 1.3 / Chunk 2.6)
==========================================================================
Analysis and diagnostic utilities for:
- 16-year daily flood panel integrity and zero-fill verification
- Brooklyn cross-check vs complaints_clean and missing-coordinate accounting
- Multi-span precipitation coupling and rolling multi-day correlations
- Lead/lag reporting causality tests and dry-spike storm anticipation
- Storm day frequencies and time-based train/test split stability
- Administrative taxonomy drift (Sewer -> Sewer Maintenance), baseline dry-spell
  comparison, and municipal 311 volume controls.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing.groups import assign_group

BOROUGH_ORDER = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
GROUPS = ["flood", "noise", "road", "sewer_backup"]


def load_flood_panel(panel_path: Path) -> pd.DataFrame:
    """Load long-history daily flood panel."""
    df = pd.read_csv(panel_path)
    df["day"] = df["day"].astype(str)
    df["borough"] = df["borough"].astype(str)
    df["n"] = df["n"].astype(int)
    return df


def build_sewer_backup_daily(agg_2010_path: Path, agg_2020_path: Path) -> pd.DataFrame:
    """
    Build daily sewer_backup series per borough from raw aggregate tables (zero-filled).
    Grid spans 2010-01-01 to 2026-10-31 across the five NYC boroughs.
    """
    df1 = pd.read_csv(agg_2010_path)
    df2 = pd.read_csv(agg_2020_path)
    df_all = pd.concat([df1, df2], ignore_index=True)
    df_all["complaint_type"] = df_all["complaint_type"].fillna("").astype(str)
    df_all["descriptor"] = df_all["descriptor"].fillna("").astype(str)
    df_all["group"] = assign_group(df_all["complaint_type"], df_all["descriptor"])

    df_sb = df_all[df_all["group"] == "sewer_backup"]
    sb_daily = df_sb.groupby(["day", "borough"])["n"].sum().reset_index()

    all_days = pd.date_range("2010-01-01", "2026-10-31", freq="D").strftime("%Y-%m-%d")
    grid = pd.MultiIndex.from_product([all_days, BOROUGH_ORDER], names=["day", "borough"]).to_frame().reset_index(drop=True)

    merged = grid.merge(sb_daily, on=["day", "borough"], how="left")
    merged["n"] = merged["n"].fillna(0).astype(int)
    merged = merged.sort_values(by=["borough", "day"]).reset_index(drop=True)
    return merged


def check_panel_integrity(
    df_panel: pd.DataFrame,
    desc_path: Path,
    type_path: Path,
) -> Dict[str, Any]:
    """Evaluate panel row count, unique days, duplicates, totals, yearly trends, and boundary continuity."""
    n_rows = len(df_panel)
    n_days = df_panel["day"].nunique()
    n_dupes = int(df_panel.duplicated(subset=["day", "borough"]).sum())
    total_flood = int(df_panel["n"].sum())

    df_panel = df_panel.copy()
    df_panel["year"] = df_panel["day"].str[:4].astype(int)

    yearly_pivot = df_panel.pivot_table(index="year", columns="borough", values="n", aggfunc="sum")[BOROUGH_ORDER]

    # Flags where year is > 3x or < 1/3 of previous year
    ratios = yearly_pivot / yearly_pivot.shift(1)
    flags = (ratios > 3.0) | (ratios < (1.0 / 3.0))

    # Boundary table
    boundary_sub = df_panel[(df_panel["day"] >= "2019-12-25") & (df_panel["day"] <= "2020-01-05")]
    boundary_table = boundary_sub.pivot_table(index="day", columns="borough", values="n", aggfunc="sum")[BOROUGH_ORDER]

    # Descriptors first and last year
    df_desc = pd.read_csv(desc_path)
    desc_summary = df_desc.groupby("descriptor")["year"].agg(first_year="min", last_year="max", total_occurrences="sum").reset_index()
    desc_summary = desc_summary.sort_values(by="total_occurrences", ascending=False).reset_index(drop=True)

    # Complaint types discovered per year
    df_types = pd.read_csv(type_path)

    return {
        "n_rows": n_rows,
        "n_days": n_days,
        "n_dupes": n_dupes,
        "total_flood": total_flood,
        "yearly_pivot": yearly_pivot,
        "ratio_flags": flags,
        "boundary_table": boundary_table,
        "desc_summary": desc_summary,
        "df_types": df_types,
    }


def cross_check_brooklyn(
    df_panel: pd.DataFrame,
    clean_path: Path,
    gap_summary_path: Path,
    missing_desc_path: Path,
) -> Dict[str, Any]:
    """Cross-check Brooklyn daily flood counts between panel and clean dataset, and inspect missing coordinates."""
    panel_bk = df_panel[
        (df_panel["borough"] == "BROOKLYN") &
        (df_panel["day"] >= "2025-11-01") &
        (df_panel["day"] <= "2026-10-31")
    ].copy()
    panel_total = int(panel_bk["n"].sum())

    clean_df = pd.read_csv(clean_path, usecols=["group", "date"])
    clean_bk = clean_df[clean_df["group"] == "flood"].groupby("date").size().rename("clean_n").reset_index()
    clean_total = int(clean_bk["clean_n"].sum())

    cmp_df = panel_bk.merge(clean_bk, left_on="day", right_on="date", how="left")
    cmp_df["clean_n"] = cmp_df["clean_n"].fillna(0).astype(int)
    cmp_df["abs_diff"] = (cmp_df["n"] - cmp_df["clean_n"]).abs()

    days_differ = int((cmp_df["abs_diff"] > 0).sum())
    max_daily_diff = int(cmp_df["abs_diff"].max())
    pearson_r = float(cmp_df["n"].corr(cmp_df["clean_n"], method="pearson"))

    top_diff_days = cmp_df.sort_values(by=["abs_diff", "day"], ascending=[False, True]).head(10)[
        ["day", "n", "clean_n", "abs_diff"]
    ].reset_index(drop=True)

    gap_summary = pd.read_csv(gap_summary_path)
    missing_desc = pd.read_csv(missing_desc_path)

    return {
        "panel_total": panel_total,
        "clean_total": clean_total,
        "days_differ": days_differ,
        "max_daily_diff": max_daily_diff,
        "pearson_r": pearson_r,
        "top_diff_days": top_diff_days,
        "gap_summary": gap_summary,
        "missing_desc": missing_desc,
    }


def compute_rain_relationships(
    df_panel: pd.DataFrame,
    rain_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlations across three temporal spans:
    1. Full span: 2010-01-01 to 2026-10-31
    2. 2020 onward: 2020-01-01 to 2026-10-31
    3. Last 12 months: 2025-11-01 to 2026-10-31
    Also computes Brooklyn full span detailed correlations against mm, hours, and log1p counts.
    """
    df_rain = pd.read_csv(rain_path)
    df = df_panel.merge(df_rain, left_on=["day", "borough"], right_on=["date", "borough"]).sort_values(["borough", "day"]).reset_index(drop=True)

    df["rain_same"] = df["precip_mm"]
    df["rain_prev"] = df.groupby("borough")["precip_mm"].shift(1)
    df["rain_3d"] = df.groupby("borough")["precip_mm"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    df["hours_same"] = df["precip_hours"]
    df["hours_3d"] = df.groupby("borough")["precip_hours"].transform(lambda s: s.rolling(3, min_periods=3).sum())

    spans = {
        "full_span": ("2010-01-01", "2026-10-31"),
        "2020_onward": ("2020-01-01", "2026-10-31"),
        "last_12m": ("2025-11-01", "2026-10-31"),
    }

    rows_spans = []
    for span_name, (start_d, end_d) in spans.items():
        sub = df[(df["day"] >= start_d) & (df["day"] <= end_d)]
        for b in BOROUGH_ORDER:
            b_df = sub[sub["borough"] == b]
            r_same_p = float(b_df["n"].corr(b_df["rain_same"], method="pearson"))
            r_same_s = float(b_df["n"].corr(b_df["rain_same"], method="spearman"))
            r_prev_p = float(b_df["n"].corr(b_df["rain_prev"], method="pearson"))
            r_prev_s = float(b_df["n"].corr(b_df["rain_prev"], method="spearman"))
            r_3d_p = float(b_df["n"].corr(b_df["rain_3d"], method="pearson"))
            r_3d_s = float(b_df["n"].corr(b_df["rain_3d"], method="spearman"))

            rows_spans.append({
                "span": span_name,
                "borough": b,
                "pearson_same_day": r_same_p,
                "spearman_same_day": r_same_s,
                "pearson_prev_day": r_prev_p,
                "spearman_prev_day": r_prev_s,
                "pearson_3d_sum": r_3d_p,
                "spearman_3d_sum": r_3d_s,
            })

    df_spans = pd.DataFrame(rows_spans)

    # Brooklyn full-span detailed correlations
    bk_full = df[df["borough"] == "BROOKLYN"].copy()
    bk_full["log1p_n"] = np.log1p(bk_full["n"])

    predictors = [
        ("precip_mm", "rain_same"),
        ("precip_hours", "hours_same"),
        ("3-day sum precip_mm", "rain_3d"),
        ("3-day sum precip_hours", "hours_3d"),
    ]

    bk_rows = []
    for label, col in predictors:
        valid = bk_full.dropna(subset=[col, "n", "log1p_n"])
        bk_rows.append({
            "predictor": label,
            "flood_count_pearson": float(valid["n"].corr(valid[col], method="pearson")),
            "flood_count_spearman": float(valid["n"].corr(valid[col], method="spearman")),
            "log1p_flood_count_pearson": float(valid["log1p_n"].corr(valid[col], method="pearson")),
            "log1p_flood_count_spearman": float(valid["log1p_n"].corr(valid[col], method="spearman")),
        })

    df_bk_detailed = pd.DataFrame(bk_rows)
    return df_spans, df_bk_detailed


def compute_lead_tests(
    df_panel: pd.DataFrame,
    rain_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lead/lag test: Pearson and Spearman of flood count on day t vs rain on day t-3..t+3.
    Also computes dry spike days (flood >= 99th pctl, precip_mm < 5) and future storm anticipation.
    """
    df_rain = pd.read_csv(rain_path)
    df = df_panel.merge(df_rain, left_on=["day", "borough"], right_on=["date", "borough"]).sort_values(["borough", "day"]).reset_index(drop=True)

    for shift_k in [-3, -2, -1, 1, 2, 3]:
        df[f"rain_t{'+' if shift_k > 0 else ''}{shift_k}"] = df.groupby("borough")["precip_mm"].shift(-shift_k)

    lead_rows = []
    shifts = [-3, -2, -1, 1, 2, 3]
    for b in BOROUGH_ORDER:
        b_df = df[df["borough"] == b]
        for s in shifts:
            col = f"rain_t{'+' if s > 0 else ''}{s}"
            valid = b_df.dropna(subset=["n", col])
            lead_rows.append({
                "borough": b,
                "shift": f"t{'+' if s > 0 else ''}{s}",
                "offset_days": s,
                "pearson_r": float(valid["n"].corr(valid[col], method="pearson")),
                "spearman_r": float(valid["n"].corr(valid[col], method="spearman")),
            })

    df_leads = pd.DataFrame(lead_rows)

    # Next 3 days rain sum: t+1, t+2, t+3
    df["rain_next_3d"] = (
        df.groupby("borough")["precip_mm"].shift(-1) +
        df.groupby("borough")["precip_mm"].shift(-2) +
        df.groupby("borough")["precip_mm"].shift(-3)
    )

    spike_rows = []
    for b in BOROUGH_ORDER:
        b_df = df[df["borough"] == b].copy()
        p99 = float(b_df["n"].quantile(0.99))
        spikes = b_df[(b_df["n"] >= p99) & (b_df["precip_mm"] < 5.0)]
        n_spikes = len(spikes)
        share_ge20 = float((spikes["rain_next_3d"] >= 20.0).mean() * 100.0) if n_spikes > 0 else 0.0

        valid_base = b_df.dropna(subset=["rain_next_3d"])
        base_rate = float((valid_base["rain_next_3d"] >= 20.0).mean() * 100.0)

        spike_rows.append({
            "borough": b,
            "p99_threshold": p99,
            "dry_spike_days": n_spikes,
            "share_next_3d_ge_20mm_pct": share_ge20,
            "base_rate_all_days_ge_20mm_pct": base_rate,
        })

    df_spikes = pd.DataFrame(spike_rows)
    return df_leads, df_spikes


def compute_storm_days_and_split(
    df_panel: pd.DataFrame,
    rain_path: Path,
) -> Dict[str, Any]:
    """Compute storm days across spans, 80/20 temporal split metrics, and Brooklyn variance dominance."""
    df_rain = pd.read_csv(rain_path)
    df = df_panel.merge(df_rain, left_on=["day", "borough"], right_on=["date", "borough"]).sort_values(["borough", "day"]).reset_index(drop=True)

    storm_full = [int((df[(df["borough"] == b) & (df["precip_mm"] >= 20.0)]["precip_mm"]).count()) for b in BOROUGH_ORDER]

    df_12m = df[(df["day"] >= "2025-11-01") & (df["day"] <= "2026-10-31")]
    storm_12m = [int((df_12m[(df_12m["borough"] == b) & (df_12m["precip_mm"] >= 20.0)]["precip_mm"]).count()) for b in BOROUGH_ORDER]

    unique_days = sorted(df["day"].unique())
    n_days_total = len(unique_days)
    n_test = 1230
    n_train = n_days_total - n_test

    train_days = unique_days[:n_train]
    test_days = unique_days[-n_test:]
    test_start_date = test_days[0]

    df_test = df[df["day"].isin(test_days)]
    storm_test = [int((df_test[(df_test["borough"] == b) & (df_test["precip_mm"] >= 20.0)]["precip_mm"]).count()) for b in BOROUGH_ORDER]

    # Brooklyn time split preview
    bk_df = df[df["borough"] == "BROOKLYN"].sort_values("day").reset_index(drop=True)
    bk_train = bk_df[bk_df["day"].isin(train_days)]
    bk_test = bk_df[bk_df["day"].isin(test_days)]

    bk_storm_train = int((bk_train["precip_mm"] >= 20.0).sum())
    bk_storm_test = int((bk_test["precip_mm"] >= 20.0).sum())

    top5_test = bk_test.sort_values(by="n", ascending=False).head(5)[["day", "n", "precip_mm"]].reset_index(drop=True)

    y_test = bk_test["n"].values
    mean_y = float(np.mean(y_test))
    sq_dev = (y_test - mean_y)**2
    total_sq_dev = float(np.sum(sq_dev))
    top2_sq_dev = float(np.sum(np.sort(sq_dev)[::-1][:2]))
    share_top2_dev = float(top2_sq_dev / total_sq_dev * 100.0)

    return {
        "storm_full": storm_full,
        "storm_12m": storm_12m,
        "storm_test": storm_test,
        "test_length": n_test,
        "test_start_date": test_start_date,
        "train_length": n_train,
        "bk_storm_train": bk_storm_train,
        "bk_storm_test": bk_storm_test,
        "top5_test": top5_test,
        "total_sq_dev": total_sq_dev,
        "top2_sq_dev": top2_sq_dev,
        "share_top2_dev": share_top2_dev,
    }


def compute_taxonomy_drift(
    agg_2020_path: Path,
    rain_path: Path,
    df_panel: pd.DataFrame,
    df_sewer_backup: pd.DataFrame,
    cube_path: Path,
) -> Dict[str, Any]:
    """Compute monthly taxonomy transition, descriptor distributions, dry-spell benchmarks, and 311 cube controls."""
    df_raw_2020 = pd.read_csv(agg_2020_path)
    df_raw_2020["complaint_type"] = df_raw_2020["complaint_type"].fillna("").astype(str)
    df_raw_2020["descriptor"] = df_raw_2020["descriptor"].fillna("").astype(str)
    df_raw_2020["group"] = assign_group(df_raw_2020["complaint_type"], df_raw_2020["descriptor"])
    df_raw_2020["month"] = df_raw_2020["day"].str[:7]

    df_rain = pd.read_csv(rain_path)
    df_rain["month"] = df_rain["date"].str[:7]

    b_monthly_rain = df_rain.groupby(["borough", "month"])["precip_mm"].sum().reset_index()
    city_monthly_rain = b_monthly_rain.groupby("month")["precip_mm"].mean()

    months_drift = [f"2025-{m:02d}" for m in range(1, 13)] + [f"2026-{m:02d}" for m in range(1, 11)]

    drift_rows = []
    for m in months_drift:
        sub = df_raw_2020[df_raw_2020["month"] == m]
        sewer_cnt = int(sub[sub["complaint_type"] == "Sewer"]["n"].sum())
        sewer_maint_cnt = int(sub[sub["complaint_type"] == "Sewer Maintenance"]["n"].sum())
        flood_cnt = int(sub[sub["group"] == "flood"]["n"].sum())
        sb_cnt = int(sub[sub["group"] == "sewer_backup"]["n"].sum())
        tot_fsb = flood_cnt + sb_cnt
        f_share = float(flood_cnt / tot_fsb) if tot_fsb > 0 else 0.0
        rain_val = float(city_monthly_rain.get(m, 0.0))

        drift_rows.append({
            "month": m,
            "n_sewer": sewer_cnt,
            "n_sewer_maintenance": sewer_maint_cnt,
            "n_flood": flood_cnt,
            "n_sewer_backup": sb_cnt,
            "flood_share": f_share,
            "city_rain_sum": rain_val,
        })

    df_drift = pd.DataFrame(drift_rows)

    # Flood descriptors by month 2026-04 to 2026-10
    f_sub = df_raw_2020[
        (df_raw_2020["group"] == "flood") &
        (df_raw_2020["month"] >= "2026-04") &
        (df_raw_2020["month"] <= "2026-10")
    ]
    df_desc_monthly = f_sub.pivot_table(index="descriptor", columns="month", values="n", aggfunc="sum", fill_value=0).reset_index()
    month_cols = [c for c in df_desc_monthly.columns if c != "descriptor"]
    df_desc_monthly["total"] = df_desc_monthly[month_cols].sum(axis=1)
    df_desc_monthly = df_desc_monthly.sort_values(by="total", ascending=False).reset_index(drop=True)

    # Dry-day baseline per borough (precip_mm < 2)
    merged_fd = df_panel.merge(df_rain, left_on=["day", "borough"], right_on=["date", "borough"])
    dry_mask = merged_fd["precip_mm"] < 2.0
    period_A = (merged_fd["day"] >= "2025-01-01") & (merged_fd["day"] <= "2026-07-31")
    period_B = (merged_fd["day"] >= "2026-08-01") & (merged_fd["day"] <= "2026-10-31")

    rows_baseline = []
    for b in BOROUGH_ORDER:
        b_df = merged_fd[merged_fd["borough"] == b]
        s_A = b_df[dry_mask.loc[b_df.index] & period_A.loc[b_df.index]]["n"]
        s_B = b_df[dry_mask.loc[b_df.index] & period_B.loc[b_df.index]]["n"]
        mean_A = float(s_A.mean())
        mean_B = float(s_B.mean())
        ratio = float(mean_B / mean_A) if mean_A > 0 else np.nan

        rows_baseline.append({
            "borough": b,
            "A_days": len(s_A),
            "A_mean": mean_A,
            "B_days": len(s_B),
            "B_mean": mean_B,
            "B_over_A_ratio": ratio,
        })
    df_baseline = pd.DataFrame(rows_baseline)

    # Cleaner dry-spell test per borough
    # Day is dry spell if precip_mm < 1 and previous 3 days total precip_mm < 2
    merged_full = df_panel.merge(df_rain, left_on=["day", "borough"], right_on=["date", "borough"]).sort_values(["borough", "day"]).reset_index(drop=True)
    sb_merged = df_sewer_backup.rename(columns={"n": "sewer_backup_n"})
    merged_full = merged_full.merge(sb_merged, on=["day", "borough"], how="left")
    merged_full["sewer_backup_n"] = merged_full["sewer_backup_n"].fillna(0).astype(int)
    merged_full = merged_full.rename(columns={"n": "flood_n"})

    merged_full["prev_3d_rain"] = (
        merged_full.groupby("borough")["precip_mm"].shift(1) +
        merged_full.groupby("borough")["precip_mm"].shift(2) +
        merged_full.groupby("borough")["precip_mm"].shift(3)
    )
    merged_full["is_dry_spell"] = (merged_full["precip_mm"] < 1.0) & (merged_full["prev_3d_rain"] < 2.0)
    merged_full["year"] = merged_full["day"].str[:4].astype(int)
    merged_full["month_num"] = merged_full["day"].str[5:7].astype(int)

    years_test = [2023, 2024, 2025, 2026]
    aug_oct = merged_full[(merged_full["month_num"].isin([8, 9, 10])) & (merged_full["year"].isin(years_test))]

    dry_spell_rows = []
    for b in BOROUGH_ORDER:
        b_df = aug_oct[aug_oct["borough"] == b]
        stats = {}
        for y in years_test:
            y_df = b_df[(b_df["year"] == y) & (b_df["is_dry_spell"])]
            stats[f"{y}_days"] = len(y_df)
            stats[f"{y}_flood_mean"] = float(y_df["flood_n"].mean()) if len(y_df) > 0 else 0.0
            stats[f"{y}_sb_mean"] = float(y_df["sewer_backup_n"].mean()) if len(y_df) > 0 else 0.0

        prior_f = np.mean([stats[f"{y}_flood_mean"] for y in [2023, 2024, 2025]])
        prior_sb = np.mean([stats[f"{y}_sb_mean"] for y in [2023, 2024, 2025]])

        f_ratio = float(stats["2026_flood_mean"] / prior_f) if prior_f > 0 else np.nan
        sb_ratio = float(stats["2026_sb_mean"] / prior_sb) if prior_sb > 0 else np.nan

        dry_spell_rows.append({
            "borough": b,
            "days_2023": stats["2023_days"],
            "flood_2023": stats["2023_flood_mean"],
            "sb_2023": stats["2023_sb_mean"],
            "days_2024": stats["2024_days"],
            "flood_2024": stats["2024_flood_mean"],
            "sb_2024": stats["2024_sb_mean"],
            "days_2025": stats["2025_days"],
            "flood_2025": stats["2025_flood_mean"],
            "sb_2025": stats["2025_sb_mean"],
            "days_2026": stats["2026_days"],
            "flood_2026": stats["2026_flood_mean"],
            "sb_2026": stats["2026_sb_mean"],
            "flood_ratio_2026_to_prior": f_ratio,
            "sb_ratio_2026_to_prior": sb_ratio,
        })
    df_dry_spell = pd.DataFrame(dry_spell_rows)

    # 311 Cube Controls
    cube = pd.read_csv(cube_path)
    sub_cube = cube[(cube["year_month"] >= "2025-10") & (cube["year_month"] <= "2026-10")]
    monthly_tot = sub_cube.groupby("year_month")["n"].sum()
    monthly_c1 = sub_cube[sub_cube["complaint_type"] == 'Root/Sewer/Sidewalk Condition'].groupby("year_month")["n"].sum()
    monthly_c2 = sub_cube[sub_cube["complaint_type"] == 'Street Condition'].groupby("year_month")["n"].sum()

    drift_idx = df_drift.set_index("month")
    monthly_flood = drift_idx["n_flood"]
    monthly_sb = drift_idx["n_sewer_backup"]

    months_post = ["2026-08", "2026-09", "2026-10"]
    months_pre = [f"2026-{m:02d}" for m in range(1, 7)]

    def calc_post_pre_ratio(s: pd.Series) -> float:
        return float(s.loc[months_post].mean() / s.loc[months_pre].mean())

    controls_summary = pd.DataFrame([
        {"series": "Flood", "ratio_post_to_pre": calc_post_pre_ratio(monthly_flood)},
        {"series": "Sewer Backup", "ratio_post_to_pre": calc_post_pre_ratio(monthly_sb)},
        {"series": "Root/Sewer/Sidewalk Condition", "ratio_post_to_pre": calc_post_pre_ratio(monthly_c1)},
        {"series": "Street Condition", "ratio_post_to_pre": calc_post_pre_ratio(monthly_c2)},
        {"series": "Total 311 Volume", "ratio_post_to_pre": calc_post_pre_ratio(monthly_tot)},
    ])

    return {
        "df_drift": df_drift,
        "df_desc_monthly": df_desc_monthly,
        "df_baseline": df_baseline,
        "df_dry_spell": df_dry_spell,
        "controls_summary": controls_summary,
    }


def plot_taxonomy_drift(df_drift: pd.DataFrame, out_path: Path) -> None:
    """Plot monthly flood and sewer_backup counts alongside precipitation bars."""
    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=150)

    months = df_drift["month"].values
    x = np.arange(len(months))
    width = 0.35

    ax1.bar(x - width / 2, df_drift["n_flood"], width, label="Flood Group", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width / 2, df_drift["n_sewer_backup"], width, label="Sewer Backup Group", color="#ff7f0e", alpha=0.85)
    ax1.set_ylabel("Monthly Complaint Count", fontsize=11)
    ax1.set_xlabel("Month", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, df_drift["city_rain_sum"], color="#2ca02c", marker="o", linewidth=2.0, label="City Rain Sum (mm)")
    ax2.set_ylabel("City-wide Mean Rain (mm)", color="#2ca02c", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("City-wide Monthly Flood and Sewer Backup Volume vs Rainfall (2025-01 to 2026-10)", fontsize=12, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dry_spell_comparison(df_dry_spell: pd.DataFrame, out_path: Path) -> None:
    """Plot dry-spell mean daily flood count across years 2023-2026 per borough."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    boroughs = df_dry_spell["borough"].values
    x = np.arange(len(boroughs))
    width = 0.2

    ax.bar(x - 1.5 * width, df_dry_spell["flood_2023"], width, label="2023", color="#aec7e8")
    ax.bar(x - 0.5 * width, df_dry_spell["flood_2024"], width, label="2024", color="#7bafd4")
    ax.bar(x + 0.5 * width, df_dry_spell["flood_2025"], width, label="2025", color="#1f77b4")
    ax.bar(x + 1.5 * width, df_dry_spell["flood_2026"], width, label="2026", color="#d62728")

    ax.set_ylabel("Mean Daily Flood Complaints", fontsize=11)
    ax.set_xlabel("Borough", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(boroughs)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="August - October Window", loc="upper left")

    plt.title("Dry-Spell Mean Daily Flood Complaints by Borough and Year (Aug to Oct)", fontsize=12, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

