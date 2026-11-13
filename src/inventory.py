"""
CityPulse - Inventory and Exploration Helpers
=============================================
Functions supporting thin exploratory and data inventory notebooks:
- get_environment_info(): collects software and system versions.
- get_raw_inventory(raw_dir): collects metadata on raw files.
- inspect_raw_csv_sample(csv_path): reads header, 5 sample rows, and binary tail.
- check_socrata_fields(dataset_ids): samples field names from Socrata API.
- get_cube_coverage(cube_path): extracts date range from precomputed cube.
"""

import sys
import shutil
import subprocess
from pathlib import Path
import csv
from typing import Dict, Any, List, Tuple
import pandas as pd
import requests


def get_environment_info(repo_root: Path) -> Dict[str, Any]:
    """Collect Python environment, library versions, git branch, and free disk."""
    import numpy as np
    import scipy
    import sklearn
    import librosa
    import torch
    import fastapi

    try:
        import osmnx
        ox_ver = osmnx.__version__
    except Exception as e:
        ox_ver = f"unavailable ({e})"

    try:
        ffmpeg_res = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        ffmpeg_line = ffmpeg_res.stdout.splitlines()[0] if ffmpeg_res.stdout else "unknown"
    except Exception as e:
        ffmpeg_line = f"unavailable ({e})"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except Exception as e:
        branch = f"unknown ({e})"

    total_disk, used_disk, free_disk = shutil.disk_usage("C:")
    free_gb = round(free_disk / (1024**3), 2)

    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "librosa": librosa.__version__,
        "torch": torch.__version__,
        "fastapi": fastapi.__version__,
        "osmnx": ox_ver,
        "ffmpeg": ffmpeg_line,
        "git_branch": branch,
        "free_disk_gb": free_gb,
    }


def get_raw_inventory(raw_dir: Path) -> List[Dict[str, Any]]:
    """List raw directory files with size in GB and modified times."""
    raw_dir = Path(raw_dir)
    items = []
    if not raw_dir.exists():
        return items

    for p in sorted(raw_dir.iterdir()):
        if p.is_file():
            stat = p.stat()
            size_gb = round(stat.st_size / (1024**3), 2)
            mtime = pd.to_datetime(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
            items.append({
                "filename": p.name,
                "size_gb": size_gb,
                "modified": mtime,
            })
    return items


def inspect_raw_csv_sample(csv_path: Path) -> Dict[str, Any]:
    """
    Inspect raw CSV without scanning full file:
    - Read header and count fields
    - Verify concept mappings
    - Read 5 sample rows and test date parsing
    - Read last 3 lines in binary mode and give completion verdict
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path.name}"}

    # 1. Read header line
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        header_fields = next(csv.reader([header_line]))
    header_count = len(header_fields)

    concept_mappings = {
        "created date": "Created Date",
        "closed date": "Closed Date",
        "complaint type": "Problem (formerly Complaint Type)",
        "descriptor": "Problem Detail (formerly Descriptor)",
        "status": "Status",
        "resolution description": "Resolution Description",
        "borough": "Borough",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "incident address": "Incident Address",
        "location type": "Location Type",
        "open data channel type": "Open Data Channel Type",
    }
    mapping_status = {k: (v in header_fields) for k, v in concept_mappings.items()}

    # 2. 5 Sample rows and date parse
    sample_df = pd.read_csv(csv_path, nrows=5, dtype=str)
    date_col = "Created Date"
    date_parse_failures = 0
    if date_col in sample_df.columns:
        parsed = pd.to_datetime(sample_df[date_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        date_parse_failures = int(parsed.isna().sum())

    # 3. Binary tail inspection
    chunk_tail = 65536
    with open(csv_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - chunk_tail))
        tail_bytes = f.read()

    tail_lines = [l for l in tail_bytes.decode("utf-8", errors="replace").splitlines() if l.strip()]
    last_3 = tail_lines[-3:] if len(tail_lines) >= 3 else tail_lines

    # Verdict
    is_json_error = any("500" in l or "Internal error" in l for l in last_3)
    if is_json_error:
        verdict = "INCOMPLETE / SERVER ERROR (ends with Socrata 500 error JSON payload)"
    else:
        try:
            last_line_fields = next(csv.reader([last_3[-1]]))
            if len(last_line_fields) == header_count:
                verdict = "COMPLETE CSV"
            else:
                verdict = "TRUNCATED CSV"
        except Exception:
            verdict = "UNKNOWN FORMAT"

    return {
        "header_count": header_count,
        "concept_mappings": concept_mappings,
        "mapping_status": mapping_status,
        "sample_rows": sample_df,
        "date_parse_failures": date_parse_failures,
        "last_3_lines": last_3,
        "verdict": verdict,
    }


def check_socrata_fields(dataset_ids: List[str]) -> Dict[str, Any]:
    """Light API check (1 sample record per dataset) verifying expected fields."""
    required_fields = ["created_date", "complaint_type", "descriptor", "borough"]
    results = {}
    for ds_id in dataset_ids:
        try:
            url = f"https://data.cityofnewyork.us/resource/{ds_id}.json"
            resp = requests.get(url, params={"$limit": 1}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                keys = list(data[0].keys()) if data else []
                field_presence = {f: (f in keys) for f in required_fields}
                results[ds_id] = {
                    "status_code": 200,
                    "fields_present": field_presence,
                    "sample_field_count": len(keys),
                    "error": None,
                }
            else:
                results[ds_id] = {
                    "status_code": resp.status_code,
                    "fields_present": {f: False for f in required_fields},
                    "sample_field_count": 0,
                    "error": f"HTTP {resp.status_code}",
                }
        except Exception as e:
            results[ds_id] = {
                "status_code": 0,
                "fields_present": {f: False for f in required_fields},
                "sample_field_count": 0,
                "error": str(e),
            }
    return results


def get_cube_coverage(cube_path: Path) -> Dict[str, Any]:
    """Extract month coverage details from 311_cube.csv."""
    cube_path = Path(cube_path)
    df = pd.read_csv(cube_path, usecols=["year_month", "n"], dtype={"year_month": str, "n": int})
    ym_valid = df[df["year_month"] != "UNKNOWN"]["year_month"].unique()
    ym_sorted = sorted(ym_valid)
    first_month = ym_sorted[0] if ym_sorted else "N/A"
    last_month = ym_sorted[-1] if ym_sorted else "N/A"
    return {
        "first_month": first_month,
        "last_month": last_month,
        "all_months": ym_sorted,
        "partial_months": ["2025-10", "2026-11"],
    }

