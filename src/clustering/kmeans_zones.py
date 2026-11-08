"""
CityPulse - Part 3, Step 3.1: K-Means Spatial Zones
===================================================
Provides modular functions for spatial clustering of 311 municipal complaints:
- distinct_locations(df): Aggregate complaints to distinct (group, x_m, y_m) coordinates.
- build_weights(locs, variant): Generate sample weights under raw, capped, and capped_balanced variants.
- fit_kmeans(X, w, k, seed): Fit scikit-learn KMeans with sample weights.
- weighted_silhouette_sample(X, w, labels, n=10000, seed): Compute silhouette score on a weighted sample.
- zone_table(locs, labels, centers): Generate zone summary table with coordinates, counts, shares, lifts, and dominant groups.

Clustering Weight Variants:
---------------------------
1. "raw":
   Every complaint row has weight 1. Since distinct_locations aggregates identical
   (group, x_m, y_m) records, each distinct coordinate is weighted by its raw complaint
   count (locs['count']). The sum of weights equals the total number of raw complaints
   (236,507 in the Brooklyn dataset).
   
2. "capped":
   Weight is min(count, p99 of rows-per-coordinate within that group). Truncates
   extreme spatial hotspots (e.g. repeat callers or persistent administrative reporting
   at single addresses) to the 99th percentile of coordinate counts within each group.
   This dampens the pulling effect of massive coordinate outliers while preserving
   broad geographic incident density.
   
3. "capped_balanced":
   Starts with capped weights, then rescales each complaint group's weights so that
   each group has an equal total weight (rescaled to target = total_capped_weight / num_groups).
   This ensures equal representation across all complaint groups, preventing high-volume
   groups (such as noise, which comprises ~89.5% of raw complaints) from monopolizing
   cluster center placements at the expense of sparser hydrological or infrastructure
   categories (such as flood and sewer backup).
"""

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.preprocessing.features import load_projection_constants, unproject

GROUPS: List[str] = ["flood", "noise", "road", "sewer_backup"]


def distinct_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract distinct spatial locations per complaint group.

    Parameters
    ----------
    df : pd.DataFrame
        Clean complaints dataframe containing columns 'group', 'x_m', 'y_m'.

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per (group, x_m, y_m) and column 'count'
        representing the number of complaint rows at that coordinate for that group.
    """
    locs = (
        df.groupby(["group", "x_m", "y_m"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return locs


def build_weights(locs: pd.DataFrame, variant: str) -> np.ndarray:
    """
    Build sample weights for distinct coordinate locations across clustering variants.

    Variants:
    ---------
    "raw":
        Every complaint record has weight 1. Since locs is aggregated to distinct
        (group, x_m, y_m) coordinates, each distinct coordinate's weight equals its
        raw complaint count (locs['count']). The sum of weights equals the total number
        of raw complaints (236,507 in the Brooklyn dataset).

    "capped":
        Weight is min(count, p99 of rows-per-coordinate within that group). Truncates
        extreme spatial hotspots (e.g. repeat callers or persistent administrative reporting
        at single buildings) to the 99th percentile of coordinate counts within each group.
        This dampens the pulling effect of massive coordinate outliers while preserving
        broad geographic incident density.

    "capped_balanced":
        Starts with capped weights, then rescales each complaint group's weights so that
        each group has an equal total weight (rescaled to target = total_capped_weight / num_groups).
        This ensures equal representation across all complaint groups, preventing high-volume
        groups (such as noise, which comprises ~89.5% of raw complaints) from monopolizing
        cluster center placements at the expense of sparser hydrological or infrastructure
        categories (such as flood and sewer backup).

    Parameters
    ----------
    locs : pd.DataFrame
        Dataframe returned by distinct_locations(df).
    variant : str
        One of "raw", "capped", "capped_balanced".

    Returns
    -------
    np.ndarray
        1D array of non-negative float weights aligned with rows of locs.
    """
    if variant == "raw":
        return locs["count"].values.astype(float)

    elif variant == "capped":
        w = locs["count"].astype(float).copy()
        for g in locs["group"].unique():
            idx = locs["group"] == g
            p99 = float(locs.loc[idx, "count"].quantile(0.99))
            w.loc[idx] = locs.loc[idx, "count"].clip(upper=p99)
        return w.values

    elif variant == "capped_balanced":
        w = locs["count"].astype(float).copy()
        for g in locs["group"].unique():
            idx = locs["group"] == g
            p99 = float(locs.loc[idx, "count"].quantile(0.99))
            w.loc[idx] = locs.loc[idx, "count"].clip(upper=p99)

        total_capped = float(w.sum())
        n_groups = locs["group"].nunique()
        target_per_group = total_capped / n_groups if n_groups > 0 else 1.0

        for g in locs["group"].unique():
            idx = locs["group"] == g
            g_sum = float(w.loc[idx].sum())
            if g_sum > 0:
                w.loc[idx] = (w.loc[idx] / g_sum) * target_per_group
        return w.values

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Expected 'raw', 'capped', or 'capped_balanced'."
        )


def fit_kmeans(X: np.ndarray, w: np.ndarray, k: int, seed: int) -> KMeans:
    """
    Fit scikit-learn KMeans with sample weights.

    Parameters
    ----------
    X : np.ndarray
        (N, 2) array of projected coordinates (x_m, y_m).
    w : np.ndarray
        1D array of sample weights.
    k : int
        Number of clusters / zones.
    seed : int
        Random state seed.

    Returns
    -------
    KMeans
        Fitted KMeans estimator with n_init=10 and random_state=seed.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(X, sample_weight=w)
    return km


def weighted_silhouette_sample(
    X: np.ndarray,
    w: np.ndarray,
    labels: np.ndarray,
    n: int = 10000,
    seed: int = 42,
) -> float:
    """
    Compute silhouette score on a sample drawn with probability proportional to weights.

    Parameters
    ----------
    X : np.ndarray
        (N, 2) array of coordinates.
    w : np.ndarray
        1D array of sample weights.
    labels : np.ndarray
        1D array of cluster labels.
    n : int, default=10000
        Sample size to draw.
    seed : int, default=42
        Random state seed.

    Returns
    -------
    float
        Silhouette score on the weighted sample.
    """
    rng = np.random.default_rng(seed)
    w_arr = np.asarray(w, dtype=float)
    total_w = w_arr.sum()
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive.")
    p = w_arr / total_w

    sample_indices = rng.choice(len(X), size=n, replace=True, p=p)
    sample_X = X[sample_indices]
    sample_labels = labels[sample_indices]

    unique_labels = np.unique(sample_labels)
    if len(unique_labels) < 2:
        return float("nan")

    return float(silhouette_score(sample_X, sample_labels))


def davies_bouldin_sample(
    X: np.ndarray,
    w: np.ndarray,
    labels: np.ndarray,
    n: int = 10000,
    seed: int = 42,
) -> float:
    """
    Compute Davies-Bouldin index on a sample drawn with probability proportional to weights.

    Parameters
    ----------
    X : np.ndarray
        (N, 2) array of coordinates.
    w : np.ndarray
        1D array of sample weights.
    labels : np.ndarray
        1D array of cluster labels.
    n : int, default=10000
        Sample size to draw.
    seed : int, default=42
        Random state seed.

    Returns
    -------
    float
        Davies-Bouldin score on the weighted sample. Lower values indicate better
        clustering separation and compactness.
    """
    rng = np.random.default_rng(seed)
    w_arr = np.asarray(w, dtype=float)
    total_w = w_arr.sum()
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive.")
    p = w_arr / total_w

    sample_indices = rng.choice(len(X), size=n, replace=True, p=p)
    sample_X = X[sample_indices]
    sample_labels = labels[sample_indices]

    unique_labels = np.unique(sample_labels)
    if len(unique_labels) < 2:
        return float("nan")

    return float(davies_bouldin_score(sample_X, sample_labels))


def relabel_by_latitude(
    centers_latlon: Union[np.ndarray, Sequence[Sequence[float]]],
    labels: np.ndarray,
    return_order: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Relabel cluster labels 0..K-1 sorted by centre latitude descending (north to south).
    Ties are broken deterministically by centre longitude ascending (west to east).

    Parameters
    ----------
    centers_latlon : array-like of shape (K, 2)
        Cluster centre geographic coordinates (latitude, longitude).
    labels : np.ndarray
        Array of cluster labels (0..K-1).
    return_order : bool, default=False
        If True, also returns the permutation array `order` mapping new index -> old index.

    Returns
    -------
    new_labels : np.ndarray
        Array of relabeled cluster labels (0..K-1).
    order : np.ndarray, optional
        Permutation indices used to sort the centers (returned if return_order=True).
    """
    centers_arr = np.asarray(centers_latlon, dtype=float)
    k = len(centers_arr)
    # Sort key: latitude descending (-lat), longitude ascending (+lon)
    order = np.array(
        sorted(range(k), key=lambda i: (-centers_arr[i, 0], centers_arr[i, 1])),
        dtype=int,
    )

    # Mapping old cluster index -> new zone id
    old_to_new = {old: new for new, old in enumerate(order)}
    new_labels = np.array([old_to_new[int(lbl)] for lbl in labels], dtype=int)

    if return_order:
        return new_labels, order
    return new_labels


def assign_zones(
    x_m: Union[np.ndarray, pd.Series, Sequence[float]],
    y_m: Union[np.ndarray, pd.Series, Sequence[float]],
    centers_xy: np.ndarray,
) -> np.ndarray:
    """
    Assign points to the nearest zone centre in projected metres.

    Parameters
    ----------
    x_m : array-like of shape (N,)
        Projected X coordinates in metres.
    y_m : array-like of shape (N,)
        Projected Y coordinates in metres.
    centers_xy : np.ndarray of shape (K, 2)
        Zone center coordinates (x_m, y_m).

    Returns
    -------
    np.ndarray of shape (N,) with dtype int
        Index of the nearest zone centre for each coordinate pair.
    """
    x = np.asarray(x_m, dtype=float).reshape(-1, 1)
    y = np.asarray(y_m, dtype=float).reshape(-1, 1)
    centers = np.asarray(centers_xy, dtype=float)
    cx = centers[:, 0].reshape(1, -1)
    cy = centers[:, 1].reshape(1, -1)

    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.argmin(d2, axis=1).astype(int)


def zone_table(
    locs: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
) -> pd.DataFrame:
    """
    Construct comprehensive zone summary table.

    For each zone, computes:
    - Centre coordinates in meters (centre_x_m, centre_y_m)
    - Centre geographic coordinates (centre_lat, centre_lon) via unproject
    - Total complaint row count in the zone (count, total_count)
    - Per-group row count (count_<group>)
    - Share of each group in the zone (share_<group> = count_<group> / total_count)
    - Lift of each group (lift_<group> = group_share_in_zone / group_share_city_wide)
    - radius_m_p90: 90th percentile distance in metres of the zone's complaints
      to its centre, complaint-weighted.
    - dominant_lift: Maximum lift value among all groups in the zone.
    - dominant_group: The complaint group achieving maximum lift. If dominant_lift < 1.15,
      the zone is labeled "mixed".

    Why label dominant_lift < 1.15 as "mixed":
    -------------------------------------------
    Noise complaints constitute ~89.5% of all complaints city-wide in Brooklyn.
    Because lift is defined as (group_share_in_zone / group_share_city_wide), a group
    that already comprises 89.5% of the total dataset can achieve a maximum possible
    theoretical lift of 1.0 / 0.895 ≈ 1.117 (even in a hypothetical zone containing 100%
    noise complaints). Consequently, noise lift cannot exceed ~1.12.
    If a zone's highest lift is below 1.15, no group shows a pronounced spatial
    concentration above city-wide background rates, meaning the zone reflects baseline
    multi-category municipal demand and is appropriately categorized as "mixed".

    Parameters
    ----------
    locs : pd.DataFrame
        Distinct locations with 'group', 'x_m', 'y_m', and 'count'.
    labels : np.ndarray
        Cluster label per row in locs.
    centers : np.ndarray
        (K, 2) array of cluster center coordinates (x_m, y_m).
    lat0, lon0 : float, optional
        Projection origin. If None, loaded from data/processed/projection.json.

    Returns
    -------
    pd.DataFrame
        Zone summary table.
    """
    if lat0 is None or lon0 is None:
        lat0, lon0 = load_projection_constants()

    locs_copy = locs.copy()
    locs_copy["_zone"] = labels

    groups = sorted(locs["group"].unique())
    total_city = float(locs["count"].sum())
    city_shares = {}
    for g in groups:
        g_city_count = float(locs.loc[locs["group"] == g, "count"].sum())
        city_shares[g] = g_city_count / total_city if total_city > 0 else 0.0

    k = len(centers)
    rows = []
    for z in range(k):
        z_locs = locs_copy[locs_copy["_zone"] == z]
        z_total = int(z_locs["count"].sum())
        cx, cy = float(centers[z, 0]), float(centers[z, 1])
        clat, clon = unproject(cx, cy, lat0, lon0)

        # Compute complaint-weighted 90th percentile distance to centre
        if z_total > 0 and len(z_locs) > 0:
            dists = np.hypot(z_locs["x_m"].values - cx, z_locs["y_m"].values - cy)
            w_comp = z_locs["count"].values.astype(float)
            sorter = np.argsort(dists)
            dists_sorted = dists[sorter]
            cum_w = np.cumsum(w_comp[sorter])
            target_w = 0.90 * cum_w[-1]
            idx = np.searchsorted(cum_w, target_w)
            radius_p90 = float(dists_sorted[min(idx, len(dists_sorted) - 1)])
        else:
            radius_p90 = 0.0

        row: Dict[str, Any] = {
            "zone": z,
            "zone_id": z,
            "centre_x_m": cx,
            "centre_y_m": cy,
            "centre_lat": float(clat),
            "centre_lon": float(clon),
            "total_count": z_total,
            "count": z_total,
            "radius_m_p90": radius_p90,
        }

        lifts = {}
        for g in groups:
            g_count = int(z_locs.loc[z_locs["group"] == g, "count"].sum())
            g_share = float(g_count / z_total) if z_total > 0 else 0.0
            g_lift = float(g_share / city_shares[g]) if city_shares[g] > 0 else 0.0

            row[f"count_{g}"] = g_count
            row[f"share_{g}"] = g_share
            row[f"lift_{g}"] = g_lift
            lifts[g] = g_lift

        if lifts:
            dominant_candidate = max(groups, key=lambda g: lifts[g])
            dominant_lift_val = float(lifts[dominant_candidate])
        else:
            dominant_candidate = ""
            dominant_lift_val = 0.0

        row["dominant_lift"] = dominant_lift_val
        if dominant_lift_val < 1.15:
            row["dominant_group"] = "mixed"
        else:
            row["dominant_group"] = dominant_candidate

        rows.append(row)

    return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def evaluate_k_range(
    locs: pd.DataFrame,
    variant: str,
    k_min: int = 2,
    k_max: int = 15,
    seed: int = 42,
    n_samples: int = 10000,
) -> pd.DataFrame:
    """
    Evaluate weighted inertia and weighted silhouette across K = k_min..k_max.
    """
    X = locs[["x_m", "y_m"]].values
    w = build_weights(locs, variant)

    records = []
    for k in range(k_min, k_max + 1):
        km = fit_kmeans(X, w, k, seed)
        sil = weighted_silhouette_sample(X, w, km.labels_, n=n_samples, seed=seed)
        records.append({
            "variant": variant,
            "K": k,
            "inertia": float(km.inertia_),
            "silhouette": float(sil),
        })

    return pd.DataFrame(records)


def evaluate_stability(
    locs: pd.DataFrame,
    variants: Sequence[str] = ("raw", "capped", "capped_balanced"),
    k_values: Sequence[int] = (4, 6, 8, 10, 12),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """
    Evaluate cluster assignment stability via mean pairwise ARI across seeds.
    """
    X = locs[["x_m", "y_m"]].values

    rows = []
    for k in k_values:
        row: Dict[str, Any] = {"K": k}
        for v in variants:
            w = build_weights(locs, v)
            label_sets = []
            for s in seeds:
                km = fit_kmeans(X, w, k, s)
                label_sets.append(km.labels_)
            aris = [
                adjusted_rand_score(l1, l2)
                for l1, l2 in itertools.combinations(label_sets, 2)
            ]
            row[v] = float(np.mean(aris))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_elbow(df_metrics: pd.DataFrame, variant: str, out_path: Path) -> None:
    """Plot weighted inertia elbow curve for a specific variant."""
    sub = df_metrics[df_metrics["variant"] == variant]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(sub["K"], sub["inertia"], marker="o", color="#1f77b4", linewidth=2.0, markersize=5)
    ax.set_title(f"K-Means Weighted Inertia Elbow Plot (Variant: {variant})", fontsize=11, pad=10)
    ax.set_xlabel("Number of Zones (K)", fontsize=10)
    ax.set_ylabel("Weighted Inertia (Sum of Squared Distances)", fontsize=10)
    ax.set_xticks(sub["K"].values)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_silhouette(df_metrics: pd.DataFrame, variant: str, out_path: Path) -> None:
    """Plot weighted silhouette score curve for a specific variant."""
    sub = df_metrics[df_metrics["variant"] == variant]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(sub["K"], sub["silhouette"], marker="s", color="#2ca02c", linewidth=2.0, markersize=5)
    ax.set_title(f"K-Means Weighted Silhouette Score (Variant: {variant})", fontsize=11, pad=10)
    ax.set_xlabel("Number of Zones (K)", fontsize=10)
    ax.set_ylabel("Weighted Silhouette Score", fontsize=10)
    ax.set_xticks(sub["K"].values)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

