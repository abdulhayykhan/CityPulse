"""
CityPulse - FastAPI backend (Part 4, Zainab - Full-Stack Lead)

Wires the team's real models and data into six HTTP endpoints:

    POST /report    upload a clip/photo -> classify + intensity + trust score
    GET  /incidents merged incident hotspots (ranked / limited for the map)
    GET  /zones     K-Means hotspot zones + per-group counts
    POST /predict   rainfall -> flood risk, hour/weekday -> noise (per zone)
    POST /route     calm-and-safe route (stub until Step 8 / osmnx)
    GET  /admin      incidents ranked by report count x severity

Run from the repo root (so `src` and `models/` resolve):

    uvicorn backend.main:app --reload

Then open http://localhost:8000/docs
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file, so the working dir does not matter)
# --------------------------------------------------------------------------- #
_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent
_DATA = _REPO_ROOT / "data" / "processed" / "text"

INCIDENTS_CSV = _DATA / "incidents.csv"
COMPLAINTS_CSV = _DATA / "complaints_clean.csv"
ZONES_JSON = _DATA / "zones.json"
DB_PATH = _REPO_ROOT / "data" / "citypulse.db"

FRONTEND_DIR = _REPO_ROOT / "frontend"
DEMO_DIR = _REPO_ROOT / "demo_samples"

# Severity weights per complaint group, used by /admin ranking.
SEVERITY = {"flood": 3.0, "sewer_backup": 2.5, "road": 2.0, "noise": 1.0}

# --------------------------------------------------------------------------- #
# App + CORS (open, because the frontend is served as a static file / other port)
# --------------------------------------------------------------------------- #
_ZONES: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load incidents into SQLite and zones into memory.
    global _ZONES
    _init_db()
    _ZONES = _load_zones()
    yield


app = FastAPI(title="CityPulse API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Lazy model loaders  (import inside functions so the server still starts even
# if one model has an issue; the error then surfaces per-request, not at boot)
# --------------------------------------------------------------------------- #
def _predict_audio(path: str) -> dict:
    from src.models.audio import predict_audio
    return predict_audio(path)


def _predict_image(path: str) -> dict:
    from src.models.image import predict_image
    return predict_image(path)


def _predict_image_proba(path: str) -> dict:
    from src.models.image import predict_image_proba
    return predict_image_proba(path)


def _predict_flood_risk(rain_mm: float) -> float:
    from src.models.regression import predict_flood_risk
    return predict_flood_risk(rain_mm)


def _predict_noise(hour: int, weekday: int) -> float:
    from src.models.regression import predict_noise
    return predict_noise(hour, weekday)


def _score_trust(features: dict) -> float:
    from src.models.trust import score_trust
    return float(score_trust(features))


# --------------------------------------------------------------------------- #
# SQLite:  load incidents once; add a reports table for demo submissions
# --------------------------------------------------------------------------- #
@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def _init_db() -> None:
    """Load incidents into SQLite on first run; create the reports table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        have = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='incidents'"
        ).fetchone()
        if not have:
            if not INCIDENTS_CSV.is_file():
                raise FileNotFoundError(f"incidents.csv not found at {INCIDENTS_CSV}")
            df = pd.read_csv(INCIDENTS_CSV)
            df.to_sql("incidents", con, if_exists="replace", index=False)
            con.execute("CREATE INDEX IF NOT EXISTS ix_inc_count ON incidents(count)")
            con.execute("CREATE INDEX IF NOT EXISTS ix_inc_group ON incidents('group')")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id  TEXT PRIMARY KEY,
                timestamp  TEXT,
                lat        REAL,
                lon        REAL,
                modality   TEXT,
                type       TEXT,
                confidence REAL,
                intensity  REAL,
                trust      REAL
            )
            """
        )
        con.commit()


def _load_zones() -> List[Dict[str, Any]]:
    if not ZONES_JSON.is_file():
        raise FileNotFoundError(f"zones.json not found at {ZONES_JSON}")
    with open(ZONES_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Health / root
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    """Serve the dashboard if built, otherwise a small API banner."""
    index = FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"service": "CityPulse API", "docs": "/docs", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "zones": len(_ZONES), "db": DB_PATH.exists()}


@app.get("/demo_list")
def demo_list() -> Dict[str, List[dict]]:
    """List bundled demo clips/photos so the UI dropdown can offer them.
    Files are served statically under /demo_samples/..."""
    out: Dict[str, List[dict]] = {"audio": [], "image": []}

    audio_dir = DEMO_DIR / "audio"
    classes: Dict[str, str] = {}
    manifest = audio_dir / "manifest.csv"
    if manifest.is_file():
        try:
            mdf = pd.read_csv(manifest)
            classes = dict(zip(mdf["file"], mdf["true_class"]))
        except Exception:
            classes = {}
    if audio_dir.is_dir():
        for p in sorted(audio_dir.glob("*.wav")):
            out["audio"].append({
                "name": p.name,
                "cls": classes.get(p.name, p.stem.split("_")[0]),
                "url": f"/demo_samples/audio/{p.name}",
                "modality": "audio",
            })

    image_dir = DEMO_DIR / "images"
    if image_dir.is_dir():
        for p in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))):
            out["image"].append({
                "name": p.name,
                "cls": p.stem.split("_")[0],
                "url": f"/demo_samples/images/{p.name}",
                "modality": "image",
            })
    return out


# --------------------------------------------------------------------------- #
# GET /incidents
# --------------------------------------------------------------------------- #
@app.get("/incidents")
def get_incidents(
    limit: int = 500,
    group: Optional[str] = None,
    min_count: int = 1,
) -> List[dict]:
    """Return incident hotspots, ranked by report count (busiest first).

    154k incidents is far too many to draw at once, so this is capped by
    `limit` and filterable by `group` and `min_count`.
    """
    limit = max(1, min(limit, 5000))
    sql = "SELECT incident_id, lat, lon, count, \"group\", zone_id FROM incidents WHERE count >= ?"
    args: List[Any] = [min_count]
    if group:
        sql += " AND \"group\" = ?"
        args.append(group)
    sql += " ORDER BY count DESC LIMIT ?"
    args.append(limit)
    with _conn() as con:
        rows = con.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# GET /zones
# --------------------------------------------------------------------------- #
@app.get("/zones")
def get_zones() -> List[dict]:
    """Return the K-Means hotspot zones with per-group counts (for livability)."""
    return _ZONES


# --------------------------------------------------------------------------- #
# POST /predict  (rain -> flood risk, hour/weekday -> noise), recolored per zone
# --------------------------------------------------------------------------- #
class PredictIn(BaseModel):
    rain_mm: Optional[float] = None
    hour: Optional[int] = None
    weekday: Optional[int] = None


@app.post("/predict")
def predict(body: PredictIn) -> dict:
    """Forecast flood and/or noise and spread the city-wide number across zones
    in proportion to each zone's share of that complaint group."""
    out: Dict[str, Any] = {}

    if body.rain_mm is not None:
        total_flood = _predict_flood_risk(float(body.rain_mm))
        denom = sum(z["counts_by_group"].get("flood", 0) for z in _ZONES) or 1
        out["flood"] = {
            "rain_mm": float(body.rain_mm),
            "citywide_expected": round(total_flood, 2),
            "zones": [
                {
                    "zone_id": z["zone_id"],
                    "risk": round(total_flood * z["counts_by_group"].get("flood", 0) / denom, 3),
                }
                for z in _ZONES
            ],
        }

    if body.hour is not None and body.weekday is not None:
        total_noise = _predict_noise(int(body.hour), int(body.weekday))
        denom = sum(z["counts_by_group"].get("noise", 0) for z in _ZONES) or 1
        out["noise"] = {
            "hour": int(body.hour),
            "weekday": int(body.weekday),
            "citywide_expected": round(total_noise, 2),
            "zones": [
                {
                    "zone_id": z["zone_id"],
                    "risk": round(total_noise * z["counts_by_group"].get("noise", 0) / denom, 3),
                }
                for z in _ZONES
            ],
        }

    if not out:
        raise HTTPException(
            status_code=422,
            detail="Provide rain_mm for flood risk and/or hour+weekday for noise.",
        )
    return out


# --------------------------------------------------------------------------- #
# POST /report  (the heart of demo mode)
# --------------------------------------------------------------------------- #
def _to_wav(src: Path) -> Path:
    """Convert any audio container (e.g. browser .webm) to 22,050 Hz mono wav
    via ffmpeg. If it is already a .wav, use it as-is."""
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "22050", "-ac", "1", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg webm->wav conversion failed: {proc.stderr[-400:]}",
        )
    return dst


def _best_effort_trust(lat: float, lon: float, group: str) -> Optional[float]:
    """Try the real trust scorer with a minimal submission-time feature dict.
    Returns None if the scorer cannot build features from this input."""
    now = datetime.now()
    features = {
        "complaint_type": group,
        "descriptor_group": group,
        "group": group,
        "hour": now.hour,
        "weekday": now.weekday(),
        "channel": "MOBILE",
        "location_type": "Street",
        "lat": lat,
        "lon": lon,
        "nearby_prior_24h_200m": 0,
    }
    try:
        return round(_score_trust(features), 3)
    except Exception:
        return None


# Map a model's predicted class to one of the complaint groups used elsewhere.
_TYPE_TO_GROUP = {
    "construction": "noise",
    "traffic_horn": "noise",
    "music_loudspeaker": "noise",
    "siren": "noise",
    "hvac": "noise",
    "pothole": "road",
    "crack": "road",
    "flood": "flood",
    "normal": "road",
}


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    modality: str = Form(...),
) -> dict:
    """Receive a demo/live submission, classify it with the real models,
    trust-score it, persist it, and return a result card payload."""
    modality = modality.lower().strip()
    if modality not in {"audio", "image"}:
        raise HTTPException(status_code=422, detail="modality must be 'audio' or 'image'")

    # 1. Save the upload to a temp file, preserving its extension.
    suffix = Path(file.filename or "").suffix or (".webm" if modality == "audio" else ".jpg")
    tmp_dir = Path(tempfile.mkdtemp(prefix="citypulse_"))
    raw_path = tmp_dir / f"upload{suffix}"
    with open(raw_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        # 2. Run the right model.
        intensity: Optional[float] = None
        if modality == "audio":
            wav = _to_wav(raw_path)
            pred = _predict_audio(str(wav))
            pred_type = pred["type"]
            confidence = pred["confidence"]
            intensity = pred.get("relative_intensity")
        else:
            pred = _predict_image(str(raw_path))
            pred_type = pred["type"]
            confidence = pred["confidence"]

        # 3. Trust score (best effort).
        group = _TYPE_TO_GROUP.get(pred_type, "noise")
        trust = _best_effort_trust(lat, lon, group)

        # 4. Persist as a demo report.
        report_id = f"RPT-{uuid.uuid4().hex[:8]}"
        ts = datetime.now().isoformat(timespec="seconds")
        with _conn() as con:
            con.execute(
                "INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
                (report_id, ts, lat, lon, modality, pred_type,
                 float(confidence), intensity, trust),
            )
            con.commit()

        return {
            "report_id": report_id,
            "timestamp": ts,
            "lat": lat,
            "lon": lon,
            "modality": modality,
            "type": pred_type,
            "group": group,
            "confidence": round(float(confidence), 4),
            "intensity": intensity,
            "trust": trust,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/reports")
def list_reports() -> List[dict]:
    """All demo submissions so far (so the map can redraw pins on reload)."""
    with _conn() as con:
        rows = con.execute("SELECT * FROM reports ORDER BY timestamp DESC").fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# POST /route  -- STUB until Step 8 (osmnx drive network + weighted shortest path)
# --------------------------------------------------------------------------- #
class RouteIn(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


@app.post("/route")
def route(body: RouteIn) -> dict:
    """Calm & safe routing (Step 8): osmnx drive graph penalised near incidents
    and noisy zones, then networkx shortest_path. Live reports count as hazards,
    so a report dropped on the normal route makes the safe route detour."""
    from backend import routing

    # Current demo reports become extra hazards.
    with _conn() as con:
        rows = con.execute("SELECT lat, lon FROM reports").fetchall()
    extra = [(r["lat"], r["lon"]) for r in rows]

    try:
        result = routing.plan(
            (body.start_lat, body.start_lon),
            (body.end_lat, body.end_lon),
            extra_hazards=extra,
        )
    except Exception as exc:  # graph download / routing failure
        raise HTTPException(status_code=503, detail=f"routing unavailable: {exc}")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# --------------------------------------------------------------------------- #
# GET /admin  -- incidents ranked by report count x severity weight
# --------------------------------------------------------------------------- #
@app.get("/admin")
def admin(limit: int = 50) -> List[dict]:
    """Priority view: incidents sorted by count * severity(group)."""
    limit = max(1, min(limit, 1000))
    with _conn() as con:
        rows = con.execute(
            "SELECT incident_id, lat, lon, count, \"group\", zone_id FROM incidents"
        ).fetchall()
    ranked = []
    for r in rows:
        d = dict(r)
        d["priority"] = round(d["count"] * SEVERITY.get(d["group"], 1.0), 2)
        ranked.append(d)
    ranked.sort(key=lambda x: x["priority"], reverse=True)
    return ranked[:limit]


# --------------------------------------------------------------------------- #
# Static mounts  (demo files + frontend assets). Declared last so they never
# shadow the API routes above.
# --------------------------------------------------------------------------- #
if DEMO_DIR.is_dir():
    app.mount("/demo_samples", StaticFiles(directory=str(DEMO_DIR)), name="demo_samples")
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
