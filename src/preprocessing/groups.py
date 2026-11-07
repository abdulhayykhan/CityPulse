"""
CityPulse - Complaint Groups Assignment Module
==============================================
Part 3, Step 1.2: Implements assign_group(complaint_type, descriptor) -> pd.Series
"""

import pandas as pd
import numpy as np


def assign_group(complaint_type: pd.Series, descriptor: pd.Series) -> pd.Series:
    """
    Vectorised classification of 311 complaints into candidate groups.

    Returns:
        pd.Series with values 'noise', 'road', 'flood', 'sewer_backup', or '' (empty string).

    Rules:
    - noise: complaint_type (case-insensitive) starts with 'noise'
    - road: (type == 'Street Condition' AND descriptor in {'Pothole','Cave-in','Rough, Pitted or Cracked','Failed Street Repair'})
            OR (type == 'Highway Condition' AND descriptor in {'Pothole - Highway','Cave-in'})
            (handles both 'Rough, Pitted or Cracked' and 'Rough, Pitted or Cracked Roads')
    - flood: type in {'Sewer','Sewer Maintenance'}
             AND descriptor matches regex (?i)flood|manhole overflow|catch basin clogged
             AND NOT matches (?i)rain garden
    - sewer_backup: type in {'Sewer','Sewer Maintenance'} AND descriptor matches (?i)backup
    - everything else ''
    """
    ct = complaint_type.fillna("").astype(str).str.strip()
    desc = descriptor.fillna("").astype(str).str.strip()

    ct_lower = ct.str.lower()

    # 1. Noise
    is_noise = ct_lower.str.startswith("noise")

    # 2. Road
    sc_desc_set = {
        "Pothole",
        "Cave-in",
        "Rough, Pitted or Cracked",
        "Rough, Pitted or Cracked Roads",
        "Failed Street Repair",
    }
    hw_desc_set = {"Pothole - Highway", "Cave-in"}

    is_road = (
        (ct == "Street Condition") & (desc.isin(sc_desc_set))
    ) | (
        (ct == "Highway Condition") & (desc.isin(hw_desc_set))
    )

    # Sewer types
    is_sewer_type = ct.isin({"Sewer", "Sewer Maintenance"})

    # 3. Sewer backup
    is_sewer_backup = is_sewer_type & desc.str.contains(r"(?i)backup", regex=True)

    # 4. Flood
    has_flood_kw = desc.str.contains(r"(?i)flood|manhole overflow|catch basin clogged", regex=True)
    has_rain_garden = desc.str.contains(r"(?i)rain garden", regex=True)
    is_flood = is_sewer_type & has_flood_kw & (~has_rain_garden) & (~is_sewer_backup)

    conditions = [is_noise, is_road, is_sewer_backup, is_flood]
    choices = ["noise", "road", "sewer_backup", "flood"]

    result = np.select(conditions, choices, default="")
    return pd.Series(result, index=complaint_type.index, dtype=str)

