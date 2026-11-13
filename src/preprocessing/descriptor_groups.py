"""
CityPulse - Part 3, Step 2, Chunk 2.3: Descriptor Group Assignment
=================================================================
Assigns fine-grained complaint descriptors into standardized descriptor groups
using vectorized hierarchical rules across the four primary groups:
- noise
- road
- flood
- sewer_backup

Rules (first match wins, case-insensitive):
- noise:
    1. complaint_type startswith 'Noise - Helicopter' -> helicopter
    2. complaint_type startswith 'Noise - Vehicle' OR descriptor matches car/truck|engine idling|horn -> vehicle
    3. descriptor matches music|party -> music_party
    4. banging|pounding -> banging
    5. talking|television -> talking_tv
    6. construction|jack ?hammer -> construction
    7. air cond|ventilation|manufactur|lawn|carting|garbage truck -> machinery_hvac
    8. dog|animal|barking -> animal
    9. alarm -> alarm
    10. else other
- road:
    1. pothole -> pothole
    2. cave-in -> cave_in
    3. rough|pitted|cracked -> cracked_rough
    4. failed street repair -> failed_repair
    5. else other
- flood:
    1. manhole overflow -> manhole_overflow
    2. catch basin -> catch_basin_clogged
    3. flooding -> street_flooding
    4. else other
- sewer_backup:
    1. backup
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def assign_descriptor_group(
    group: Union[pd.Series, np.ndarray],
    complaint_type: Union[pd.Series, np.ndarray],
    descriptor: Union[pd.Series, np.ndarray],
) -> pd.Series:
    """
    Vectorized hierarchical assignment of descriptor_group.
    Strips whitespace first and applies case-insensitive rules.
    First matching rule wins.
    """
    # Ensure Series inputs with identical index
    if not isinstance(group, pd.Series):
        g = pd.Series(group)
    else:
        g = group.copy()

    if not isinstance(complaint_type, pd.Series):
        c = pd.Series(complaint_type, index=g.index)
    else:
        c = complaint_type.copy()
        c.index = g.index

    if not isinstance(descriptor, pd.Series):
        d = pd.Series(descriptor, index=g.index)
    else:
        d = descriptor.copy()
        d.index = g.index

    # Strip whitespace and convert to string
    g_clean = g.fillna("").astype(str).str.strip().str.lower()
    c_clean = c.fillna("").astype(str).str.strip()
    c_lower = c_clean.str.lower()
    d_clean = d.fillna("").astype(str).str.strip()

    # Pre-compute boolean masks per group
    is_sewer = g_clean == "sewer_backup"
    is_flood = g_clean == "flood"
    is_road = g_clean == "road"
    is_noise = g_clean == "noise"

    # 1. Sewer Backup
    # Always 'backup'
    # 2. Flood rules
    flood_mo = is_flood & d_clean.str.contains(r"manhole overflow", case=False, regex=True)
    flood_cb = is_flood & d_clean.str.contains(r"catch basin", case=False, regex=True)
    flood_fl = is_flood & d_clean.str.contains(r"flooding", case=False, regex=True)

    # 3. Road rules
    road_pot = is_road & d_clean.str.contains(r"pothole", case=False, regex=True)
    road_cave = is_road & d_clean.str.contains(r"cave-in", case=False, regex=True)
    road_rpc = is_road & d_clean.str.contains(r"rough|pitted|cracked", case=False, regex=True)
    road_fsr = is_road & d_clean.str.contains(r"failed street repair", case=False, regex=True)

    # 4. Noise rules
    noise_heli = is_noise & c_lower.str.startswith("noise - helicopter")
    noise_veh = is_noise & (
        c_lower.str.startswith("noise - vehicle")
        | d_clean.str.contains(r"car/truck|engine idling|horn", case=False, regex=True)
    )
    noise_music = is_noise & d_clean.str.contains(r"music|party", case=False, regex=True)
    noise_bang = is_noise & d_clean.str.contains(r"banging|pounding", case=False, regex=True)
    noise_talk = is_noise & d_clean.str.contains(r"talking|television", case=False, regex=True)
    noise_const = is_noise & d_clean.str.contains(r"construction|jack ?hammer", case=False, regex=True)
    noise_mach = is_noise & d_clean.str.contains(
        r"air cond|ventilation|manufactur|lawn|carting|garbage truck", case=False, regex=True
    )
    noise_anim = is_noise & d_clean.str.contains(r"dog|animal|barking", case=False, regex=True)
    noise_alarm = is_noise & d_clean.str.contains(r"alarm", case=False, regex=True)

    # Ordered conditions: first match wins
    conditions = [
        # Sewer backup
        is_sewer,
        # Flood
        flood_mo,
        flood_cb,
        flood_fl,
        is_flood,  # fallback for flood -> other
        # Road
        road_pot,
        road_cave,
        road_rpc,
        road_fsr,
        is_road,  # fallback for road -> other
        # Noise
        noise_heli,
        noise_veh,
        noise_music,
        noise_bang,
        noise_talk,
        noise_const,
        noise_mach,
        noise_anim,
        noise_alarm,
        is_noise,  # fallback for noise -> other
    ]

    choices = [
        # Sewer backup
        "backup",
        # Flood
        "manhole_overflow",
        "catch_basin_clogged",
        "street_flooding",
        "other",
        # Road
        "pothole",
        "cave_in",
        "cracked_rough",
        "failed_repair",
        "other",
        # Noise
        "helicopter",
        "vehicle",
        "music_party",
        "banging",
        "talking_tv",
        "construction",
        "machinery_hvac",
        "animal",
        "alarm",
        "other",
    ]

    result = np.select(conditions, choices, default="other")
    return pd.Series(result, index=g.index, dtype="object")


def build_descriptor_groups_table(
    df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Computes distinct combinations of (group, complaint_type, descriptor),
    assigns descriptor_group, and aggregates counts n.
    Sorted by group ascending, then n descending.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "descriptor_groups.csv"
    else:
        output_path = Path(output_path)

    # Calculate descriptor groups
    desc_grp = assign_descriptor_group(df["group"], df["complaint_type"], df["descriptor"])
    df_work = df[["group", "complaint_type", "descriptor"]].copy()
    df_work["descriptor_group"] = desc_grp

    table = (
        df_work.groupby(["group", "complaint_type", "descriptor", "descriptor_group"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    # Sort by group then n descending
    table = table.sort_values(by=["group", "n"], ascending=[True, False]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    return table


if __name__ == "__main__":
    from typing import Optional
    stage2_path = PROCESSED_DIR / "brooklyn_stage2.csv.gz"
    print(f"Reading stage 2 data from {stage2_path}...")
    df_stage2 = pd.read_csv(
        stage2_path,
        dtype=str,
        keep_default_na=False,
    )
    table = build_descriptor_groups_table(df_stage2)
    print(f"Saved {len(table)} combinations to {PROCESSED_DIR / 'descriptor_groups.csv'}")

    print("\n=== FULL DESCRIPTOR GROUPS TABLE ===")
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)
    print(table.to_string())

    print("\n=== SHARE OF 'OTHER' PER GROUP ===")
    df_stage2["descriptor_group"] = assign_descriptor_group(
        df_stage2["group"], df_stage2["complaint_type"], df_stage2["descriptor"]
    )
    for grp_name, grp_df in df_stage2.groupby("group"):
        total_n = len(grp_df)
        other_n = (grp_df["descriptor_group"] == "other").sum()
        pct = (other_n / total_n) * 100.0
        flag = " (> 10% ALERT!)" if pct > 10.0 else " (OK <= 10%)"
        print(f"Group: {grp_name:12s} | Total: {total_n:7d} | Other: {other_n:6d} ({pct:6.2f}%){flag}")
