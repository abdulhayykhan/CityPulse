"""
CityPulse - Part 3, Step 2, Chunk 2.3: Build Clean Dataset
==========================================================
Builds the final clean dataset for Brooklyn complaints:
- data/processed/complaints_clean.csv
- data/processed/complaints_clean_columns.csv

Output schema (25 columns in exact order):
1. id (=unique_key)
2. lat (=latitude)
3. lon (=longitude)
4. timestamp (created_date as ISO 'YYYY-MM-DDTHH:MM:SS')
5. group
6. descriptor
7. zone_id (empty for every row; filled in a later step)
8. descriptor_group
9. complaint_type
10. borough
11. status
12. closed_date (ISO or empty)
13. time_to_close_h
14. hour
15. weekday
16. month
17. date (YYYY-MM-DD)
18. is_midnight_timestamp (True/False)
19. channel
20. location_type
21. incident_address
22. address_norm
23. resolution_id (string, empty when missing)
24. x_m
25. y_m
"""

import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.descriptor_groups import assign_descriptor_group
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STAGE2_GZ = PROCESSED_DIR / "brooklyn_stage2.csv.gz"
CLEAN_CSV = PROCESSED_DIR / "complaints_clean.csv"
COLUMNS_CSV = PROCESSED_DIR / "complaints_clean_columns.csv"

EXACT_COLUMNS = [
    "id",
    "lat",
    "lon",
    "timestamp",
    "group",
    "descriptor",
    "zone_id",
    "descriptor_group",
    "complaint_type",
    "borough",
    "status",
    "closed_date",
    "time_to_close_h",
    "hour",
    "weekday",
    "month",
    "date",
    "is_midnight_timestamp",
    "channel",
    "location_type",
    "incident_address",
    "address_norm",
    "resolution_id",
    "x_m",
    "y_m",
]


def build_clean_dataset(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    columns_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Construct complaints_clean.csv and complaints_clean_columns.csv.
    """
    if input_path is None:
        input_path = STAGE2_GZ
    else:
        input_path = Path(input_path)

    if output_path is None:
        output_path = CLEAN_CSV
    else:
        output_path = Path(output_path)

    if columns_path is None:
        columns_path = COLUMNS_CSV
    else:
        columns_path = Path(columns_path)

    print(f"Loading stage 2 data from {input_path}...")
    df = pd.read_csv(
        input_path,
        dtype={
            "unique_key": str,
            "resolution_id": str,
            "status": str,
            "channel": str,
            "location_type": str,
            "complaint_type": str,
            "descriptor": str,
            "group": str,
            "borough": str,
            "incident_address": str,
            "address_norm": str,
            "date": str,
        },
        keep_default_na=False,
    )

    # Parse datetimes
    c_dt = pd.to_datetime(df["created_date"], errors="coerce")
    cl_dt = pd.to_datetime(df["closed_date"], errors="coerce")

    # Numeric coordinates
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["x_m"] = df["x_m"].astype(float)
    df["y_m"] = df["y_m"].astype(float)

    # Compute time_to_close_h (float hours; NaN when closed_date is NaT)
    diff_sec = (cl_dt - c_dt).dt.total_seconds()
    time_to_close_h = diff_sec / 3600.0

    print("Assembling clean dataframe with 25 standardized columns...")
    clean = pd.DataFrame(index=df.index)

    # 1. id
    clean["id"] = df["unique_key"].astype(str)
    # 2. lat
    clean["lat"] = df["latitude"]
    # 3. lon
    clean["lon"] = df["longitude"]
    # 4. timestamp (ISO YYYY-MM-DDTHH:MM:SS)
    clean["timestamp"] = c_dt.dt.strftime("%Y-%m-%dT%H:%M:%S")
    # 5. group
    clean["group"] = df["group"].astype(str)
    # 6. descriptor
    clean["descriptor"] = df["descriptor"].astype(str)
    # 7. zone_id (empty for every row)
    clean["zone_id"] = ""
    # 8. descriptor_group
    print("Assigning descriptor groups...")
    clean["descriptor_group"] = assign_descriptor_group(
        df["group"], df["complaint_type"], df["descriptor"]
    )
    # 9. complaint_type
    clean["complaint_type"] = df["complaint_type"].astype(str)
    # 10. borough
    clean["borough"] = df["borough"].astype(str)
    # 11. status
    clean["status"] = df["status"].astype(str)
    # 12. closed_date (ISO or empty)
    clean["closed_date"] = cl_dt.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    # 13. time_to_close_h
    clean["time_to_close_h"] = time_to_close_h
    # 14. hour
    clean["hour"] = df["hour"].astype(int)
    # 15. weekday
    clean["weekday"] = df["weekday"].astype(int)
    # 16. month
    clean["month"] = df["month"].astype(int)
    # 17. date
    clean["date"] = df["date"].astype(str)
    # 18. is_midnight_timestamp (True/False)
    # Convert bool string/type to bool
    if df["is_midnight_timestamp"].dtype == object:
        clean["is_midnight_timestamp"] = df["is_midnight_timestamp"].astype(str).str.lower() == "true"
    else:
        clean["is_midnight_timestamp"] = df["is_midnight_timestamp"].astype(bool)
    # 19. channel
    clean["channel"] = df["channel"].astype(str)
    # 20. location_type
    clean["location_type"] = df["location_type"].astype(str)
    # 21. incident_address
    clean["incident_address"] = df["incident_address"].astype(str)
    # 22. address_norm
    clean["address_norm"] = df["address_norm"].astype(str)
    # 23. resolution_id (string, empty when missing)
    clean["resolution_id"] = df["resolution_id"].astype(str)
    # 24. x_m
    clean["x_m"] = df["x_m"]
    # 25. y_m
    clean["y_m"] = df["y_m"]

    # Verify column count and order
    assert list(clean.columns) == EXACT_COLUMNS, f"Columns mismatch: {list(clean.columns)}"

    print(f"Writing {len(clean)} rows to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    # Write column metadata
    print(f"Writing column metadata to {columns_path}...")
    columns_meta = [
        {"column": "id", "dtype": "string", "description": "Unique 311 service request identifier", "safe_as_trust_feature": "yes", "notes": "Source: unique_key"},
        {"column": "lat", "dtype": "float", "description": "WGS84 latitude coordinate", "safe_as_trust_feature": "yes", "notes": "Verified within Brooklyn bounding box [40.55, 40.74]"},
        {"column": "lon", "dtype": "float", "description": "WGS84 longitude coordinate", "safe_as_trust_feature": "yes", "notes": "Verified within Brooklyn bounding box [-74.06, -73.83]"},
        {"column": "timestamp", "dtype": "string", "description": "Creation timestamp in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)", "safe_as_trust_feature": "yes", "notes": "Original request submission time"},
        {"column": "group", "dtype": "string", "description": "Primary complaint classification group", "safe_as_trust_feature": "yes", "notes": "Subset of {noise, road, flood, sewer_backup}"},
        {"column": "descriptor", "dtype": "string", "description": "Specific complaint descriptor from 311 agency", "safe_as_trust_feature": "yes", "notes": "Raw agency descriptor"},
        {"column": "zone_id", "dtype": "string", "description": "Spatial cluster zone identifier", "safe_as_trust_feature": "yes", "notes": "empty until K-Means step"},
        {"column": "descriptor_group", "dtype": "string", "description": "Standardized functional descriptor category", "safe_as_trust_feature": "yes", "notes": "Hierarchical rule-based assignment"},
        {"column": "complaint_type", "dtype": "string", "description": "Agency complaint type category", "safe_as_trust_feature": "yes", "notes": "Original 311 complaint type"},
        {"column": "borough", "dtype": "string", "description": "NYC Borough name", "safe_as_trust_feature": "yes", "notes": "All rows are BROOKLYN"},
        {"column": "status", "dtype": "string", "description": "Current ticket lifecycle status", "safe_as_trust_feature": "no", "notes": "exists only after the agency acts (label leakage)"},
        {"column": "closed_date", "dtype": "string", "description": "Resolution timestamp in ISO 8601 format", "safe_as_trust_feature": "no", "notes": "exists only after the agency acts (label leakage)"},
        {"column": "time_to_close_h", "dtype": "float", "description": "Duration to resolve request in elapsed hours", "safe_as_trust_feature": "no", "notes": "exists only after the agency acts (label leakage)"},
        {"column": "hour", "dtype": "int", "description": "Hour of day (0-23) when request was created", "safe_as_trust_feature": "yes", "notes": "Extracted from timestamp"},
        {"column": "weekday", "dtype": "int", "description": "Day of week (0=Monday to 6=Sunday) when created", "safe_as_trust_feature": "yes", "notes": "Extracted from timestamp"},
        {"column": "month", "dtype": "int", "description": "Month of year (1-12) when created", "safe_as_trust_feature": "yes", "notes": "Extracted from timestamp"},
        {"column": "date", "dtype": "string", "description": "Calendar date (YYYY-MM-DD) when created", "safe_as_trust_feature": "yes", "notes": "Extracted from timestamp"},
        {"column": "is_midnight_timestamp", "dtype": "bool", "description": "Boolean flag indicating exact midnight (00:00:00) timestamp", "safe_as_trust_feature": "yes", "notes": "Batch ingestion flag"},
        {"column": "channel", "dtype": "string", "description": "Submission channel (e.g. Mobile, Online, Phone)", "safe_as_trust_feature": "yes", "notes": "Open data channel type"},
        {"column": "location_type", "dtype": "string", "description": "Type of location where incident occurred", "safe_as_trust_feature": "yes", "notes": "Reported location environment"},
        {"column": "incident_address", "dtype": "string", "description": "Raw incident street address", "safe_as_trust_feature": "yes", "notes": "Reported location address"},
        {"column": "address_norm", "dtype": "string", "description": "Standardized uppercase alphanumeric street address", "safe_as_trust_feature": "yes", "notes": "Normalized address string"},
        {"column": "resolution_id", "dtype": "string", "description": "Agency resolution action code", "safe_as_trust_feature": "no", "notes": "exists only after the agency acts (label leakage)"},
        {"column": "x_m", "dtype": "float", "description": "Local planar x-coordinate in meters relative to Brooklyn centroid", "safe_as_trust_feature": "yes", "notes": "Equirectangular projection using fixed lat0, lon0"},
        {"column": "y_m", "dtype": "float", "description": "Local planar y-coordinate in meters relative to Brooklyn centroid", "safe_as_trust_feature": "yes", "notes": "Equirectangular projection using fixed lat0, lon0"},
    ]
    df_meta = pd.DataFrame(columns_meta)
    df_meta.to_csv(columns_path, index=False, encoding="utf-8", lineterminator="\n")

    return clean


if __name__ == "__main__":
    build_clean_dataset()
    print("Done building clean dataset.")
