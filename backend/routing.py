"""
routing.py - Calm & Safe route planner (Part 4, Step 8).

Builds an OpenStreetMap drive network with osmnx once (cached to graphml),
penalises road segments that run near incidents or through noisy zones, and
returns both the plain shortest route and the hazard-avoiding "safe" route.

    weight(edge) = length * (1 + penalty)
    penalty is high within ~100 m of an incident, lower within ~250 m,
    plus a bump for edges inside a noise-dominated zone.

Newly submitted reports are passed in as extra hazards, so dropping a report
on the normal route makes the safe route detour (the Step 8 "Done when").
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent
_DATA = _ROOT / "data" / "processed" / "text"
INCIDENTS_CSV = _DATA / "incidents.csv"
ZONES_JSON = _DATA / "zones.json"
GRAPH_PATH = _ROOT / "data" / "brooklyn_drive.graphml"

# Radius of the drive network around the study centre (guide suggests ~3,000 m).
ROUTE_DIST = int(os.environ.get("CITYPULSE_ROUTE_DIST", "3500"))

_G = None                    # cached networkx graph
_INC_XY: Optional[np.ndarray] = None   # incident coordinates in metres
_ZONES: Optional[list] = None


# --------------------------------------------------------------------------- #
# Local equirectangular projection to metres (matches the data team's formula)
# --------------------------------------------------------------------------- #
def _to_xy(lat: float, lon: float) -> Tuple[float, float]:
    return lon * 111320.0 * math.cos(math.radians(lat)), lat * 110540.0


def _zones() -> list:
    global _ZONES
    if _ZONES is None:
        with open(ZONES_JSON, encoding="utf-8") as fh:
            _ZONES = json.load(fh)
    return _ZONES


def _study_centre() -> Tuple[float, float]:
    z = _zones()
    return (sum(x["centre_lat"] for x in z) / len(z),
            sum(x["centre_lon"] for x in z) / len(z))


# Public Overpass mirrors to try in order (the default one is often at capacity).
_OVERPASS_MIRRORS = [
    os.environ.get("CITYPULSE_OVERPASS", "").strip(),
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]


def _download_graph():
    """Download the drive network, trying several Overpass mirrors with backoff."""
    import time
    import osmnx as ox

    ox.settings.requests_timeout = 180
    lat, lon = _study_centre()
    last_err = None
    for url in [m for m in _OVERPASS_MIRRORS if m]:
        ox.settings.overpass_url = url
        for attempt in range(2):
            try:
                return ox.graph_from_point((lat, lon), dist=ROUTE_DIST,
                                           network_type="drive", simplify=True)
            except Exception as exc:  # noqa: BLE001 - try next mirror/attempt
                last_err = exc
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Overpass download failed on all mirrors: {last_err}")


def get_graph():
    """Load the cached drive graph, or download+cache it on first use."""
    global _G
    if _G is not None:
        return _G
    import osmnx as ox
    if GRAPH_PATH.is_file():
        _G = ox.load_graphml(GRAPH_PATH)
    else:
        _G = _download_graph()
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(_G, GRAPH_PATH)
    return _G


def _incident_xy() -> np.ndarray:
    """All incident coordinates projected to metres (cached in memory)."""
    global _INC_XY
    if _INC_XY is None:
        df = pd.read_csv(INCIDENTS_CSV, usecols=["lat", "lon"])
        lat = df["lat"].to_numpy(float)
        lon = df["lon"].to_numpy(float)
        x = lon * 111320.0 * np.cos(np.radians(lat))
        y = lat * 110540.0
        _INC_XY = np.column_stack([x, y])
    return _INC_XY


def _noisy_zone_centres() -> List[Tuple[float, float, float]]:
    """Return (x, y, radius_m) for zones dominated by noise complaints."""
    out = []
    for z in _zones():
        cb = z.get("counts_by_group", {})
        total = sum(cb.values()) or 1
        noise_share = cb.get("noise", 0) / total
        if z.get("dominant_group") == "noise" or noise_share > 0.6:
            x, y = _to_xy(z["centre_lat"], z["centre_lon"])
            out.append((x, y, float(z.get("radius_m_p90", 1500))))
    return out


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan(start: Sequence[float], end: Sequence[float],
         extra_hazards: Optional[List[Sequence[float]]] = None) -> dict:
    """Return normal and safe routes between start and end (each [lat, lon])."""
    import networkx as nx
    import osmnx as ox
    from scipy.spatial import cKDTree

    G = get_graph()

    # Edge midpoints + lengths (metres).
    edges = list(G.edges(keys=True, data=True))
    mids = np.empty((len(edges), 2))
    lengths = np.empty(len(edges))
    for i, (u, v, k, data) in enumerate(edges):
        xu, yu = _to_xy(G.nodes[u]["y"], G.nodes[u]["x"])
        xv, yv = _to_xy(G.nodes[v]["y"], G.nodes[v]["x"])
        mids[i] = ((xu + xv) / 2.0, (yu + yv) / 2.0)
        lengths[i] = float(data.get("length", 1.0))

    # 1. Background penalty from incident DENSITY (count within 100 m), so
    #    hotspot cores cost more than the incident-blanketed background.
    inc = _incident_xy()
    counts = cKDTree(inc).query_ball_point(mids, r=100.0, return_length=True)
    penalty = np.minimum(6.0, 1.2 * np.log1p(counts))

    # 2. Live reports act as strong, local barriers -> safe route detours.
    n_extra = 0
    if extra_hazards:
        ex = np.array([_to_xy(la, lo) for la, lo in extra_hazards], dtype=float)
        n_extra = len(ex)
        if n_extra:
            dex, _ = cKDTree(ex).query(mids, k=1)
            penalty = penalty + np.where(dex < 130.0, 40.0, np.where(dex < 260.0, 8.0, 0.0))

    # 3. Extra bump for edges inside a noise-dominated zone.
    for cx, cy, r in _noisy_zone_centres():
        inside = (mids[:, 0] - cx) ** 2 + (mids[:, 1] - cy) ** 2 < r * r
        penalty = penalty + inside * 2.0

    safe_w = lengths * (1.0 + penalty)
    for i, (u, v, k, data) in enumerate(edges):
        data["safe"] = float(safe_w[i])
    n_haz = len(inc) + n_extra

    # 3. Snap endpoints to nearest graph nodes and run both shortest paths.
    o = ox.distance.nearest_nodes(G, X=float(start[1]), Y=float(start[0]))
    d = ox.distance.nearest_nodes(G, X=float(end[1]), Y=float(end[0]))

    try:
        normal_nodes = nx.shortest_path(G, o, d, weight="length")
        safe_nodes = nx.shortest_path(G, o, d, weight="safe")
    except nx.NetworkXNoPath:
        return {"error": "No drive path between those points within the study area."}

    def coords(ns):
        return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in ns]

    def length_m(ns):
        tot = 0.0
        for a, b in zip(ns[:-1], ns[1:]):
            tot += min(G[a][b][kk].get("length", 0.0) for kk in G[a][b])
        return tot

    return {
        "normal_route": coords(normal_nodes),
        "safe_route": coords(safe_nodes),
        "normal_len_m": round(length_m(normal_nodes)),
        "safe_len_m": round(length_m(safe_nodes)),
        "detour_m": round(length_m(safe_nodes) - length_m(normal_nodes)),
        "hazards": int(n_haz),
    }
