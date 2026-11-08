"""
CityPulse - Hierarchical Spatio-Temporal Deduplication Module
=============================================================
Provides functionality for evidence-gathering and hierarchical clustering
of municipal 311 service requests:
- weak_labels: Baseline address + complaint_type weak clusters ('anchor' and 'chain' modes).
- make_blocks: Spatio-temporal grid partition (group, calendar day, 500m grid cells).
- cluster_block: Agglomerative hierarchical clustering within a single block.
- cluster_all: Full-dataset blocking and clustering returning global incident labels.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster


def weak_labels(
    df: pd.DataFrame,
    window_hours: float = 24.0,
    mode: str = "anchor",
) -> np.ndarray:
    """
    Generate weak cluster labels based on (address_norm, complaint_type).

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe containing 'address_norm', 'complaint_type', and 'timestamp'.
    window_hours : float, default=24.0
        Temporal window threshold in hours.
    mode : str, default='anchor'
        'anchor': Sort by time within key; a new weak cluster starts when the time
                  since the FIRST report of the current cluster exceeds window_hours.
        'chain' : Sort by time within key; a new cluster starts only when the gap
                  since the PREVIOUS report exceeds window_hours (transitive chain).

    Returns
    -------
    np.ndarray
        Integer labels of shape (len(df),). Rows with address_norm == '' or null
        receive label -1 (excluded from ARI evaluation). Remaining rows receive
        0-indexed cluster IDs.
    """
    if mode not in ("anchor", "chain"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'anchor' or 'chain'.")

    n = len(df)
    labels = np.full(n, -1, dtype=np.int64)

    # Empty address check (handles empty string and NaN)
    valid_mask = (df["address_norm"] != "") & (df["address_norm"].notna())
    valid_idx = np.where(valid_mask.values)[0]

    if len(valid_idx) == 0:
        return labels

    # Extract timestamps in epoch seconds
    ts = pd.to_datetime(df["timestamp"])
    ts_sec = ts.values.astype("datetime64[s]").astype(np.int64)

    # Group by (address_norm, complaint_type)
    keys = pd.MultiIndex.from_arrays([
        df.loc[valid_mask, "address_norm"].astype(str),
        df.loc[valid_mask, "complaint_type"].astype(str),
    ])
    codes, _ = pd.factorize(keys)

    # Sort valid records by key and timestamp
    sort_order = np.lexsort((ts_sec[valid_idx], codes))
    sorted_idx = valid_idx[sort_order]
    sorted_codes = codes[sort_order]
    sorted_ts = ts_sec[sorted_idx]

    window_sec = float(window_hours * 3600.0)

    if mode == "chain":
        # Vectorized chain: new cluster if key changes or gap > window_sec
        new_cluster = np.empty(len(sorted_codes), dtype=bool)
        new_cluster[0] = True
        new_cluster[1:] = (sorted_codes[1:] != sorted_codes[:-1]) | (
            (sorted_ts[1:] - sorted_ts[:-1]) > window_sec
        )
        labels_sorted = np.cumsum(new_cluster) - 1
    else:
        # Anchor mode: new cluster if key changes or (t - cluster_first_t) > window_sec
        new_cluster = np.empty(len(sorted_codes), dtype=bool)
        new_cluster[0] = True
        curr_c = sorted_codes[0]
        curr_anchor = sorted_ts[0]
        for i in range(1, len(sorted_codes)):
            c = sorted_codes[i]
            t = sorted_ts[i]
            if c != curr_c or (t - curr_anchor) > window_sec:
                new_cluster[i] = True
                curr_c = c
                curr_anchor = t
            else:
                new_cluster[i] = False
        labels_sorted = np.cumsum(new_cluster) - 1

    labels[sorted_idx] = labels_sorted
    return labels


def make_blocks(df: pd.DataFrame, cell_m: float = 500.0) -> pd.Series:
    """
    Assign complaints to mutually exclusive spatio-temporal-group blocks.

    Block ID = (group, day, floor(x_m / cell_m), floor(y_m / cell_m)).
    No overlap between blocks.

    Note:
    Blocking reduces pairwise candidate comparisons from O(N^2) to manageable
    block sizes, but can split true duplicate reports if they cross grid cell
    boundaries or span across midnight.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe containing 'group', 'x_m', 'y_m', and either 'date'
        or 'timestamp'.
    cell_m : float, default=500.0
        Grid cell dimension in projected metres.

    Returns
    -------
    pd.Series
        Series of 4-tuples representing block IDs, aligned with df.index.
    """
    if "date" in df.columns:
        days = df["date"].astype(str).values
    else:
        days = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").values

    groups = df["group"].astype(str).values
    gx = np.floor(df["x_m"].values / cell_m).astype(int)
    gy = np.floor(df["y_m"].values / cell_m).astype(int)

    block_tuples = [
        (str(groups[i]), str(days[i]), int(gx[i]), int(gy[i]))
        for i in range(len(df))
    ]
    return pd.Series(block_tuples, index=df.index, name="block_id")


def cluster_block(
    X: Union[pd.DataFrame, np.ndarray],
    T: float,
    model: Optional[AgglomerativeClustering] = None,
) -> np.ndarray:
    """
    Cluster records within a single block using AgglomerativeClustering.

    Uses AgglomerativeClustering(n_clusters=None, distance_threshold=T, linkage='average')
    on features [x_m, y_m, t_hours * 50].
    Blocks of size 1 skip the model.
    Rows are sorted by (x_m, y_m, timestamp, id) first so results are deterministic.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Records in the block. If pd.DataFrame, expected columns include 'x_m', 'y_m',
        'timestamp', and optionally 'id' or 't_hours'. If np.ndarray of shape (N, 3),
        columns are assumed to be [x_m, y_m, t_hours * 50].
    T : float
        Distance threshold for merging clusters.
    model : Optional[AgglomerativeClustering], default=None
        Optional pre-instantiated sklearn model for performance reuse.

    Returns
    -------
    np.ndarray
        Integer cluster labels of shape (len(X),) from 0 to K_block-1.
    """
    n = len(X)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    if isinstance(X, pd.DataFrame):
        sort_cols = [c for c in ["x_m", "y_m", "timestamp", "id"] if c in X.columns]
        if sort_cols:
            order = X.sort_values(sort_cols).index
            pos_map = {orig_idx: i for i, orig_idx in enumerate(X.index)}
            sorted_pos = np.array([pos_map[idx] for idx in order])
            rev_pos = np.empty(n, dtype=int)
            rev_pos[sorted_pos] = np.arange(n)
        else:
            sorted_pos = np.arange(n)
            rev_pos = np.arange(n)

        if "t_hours" in X.columns:
            t_h = X["t_hours"].values.astype(float)
        elif "timestamp" in X.columns:
            ts = pd.to_datetime(X["timestamp"])
            t_h = ts.values.astype("datetime64[s]").astype(float) / 3600.0
        else:
            t_h = np.zeros(n, dtype=float)

        feats = np.column_stack([
            X["x_m"].values.astype(float),
            X["y_m"].values.astype(float),
            t_h * 50.0,
        ])
        feats_sorted = feats[sorted_pos]

        if model is None:
            clf = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=T,
                linkage="average",
            )
        else:
            clf = model
            clf.distance_threshold = T

        lbls_sorted = clf.fit_predict(feats_sorted)
        return lbls_sorted[rev_pos].astype(np.int64)

    else:
        # np.ndarray shape (N, 3)
        order = np.lexsort((X[:, 2], X[:, 1], X[:, 0]))
        rev_order = np.empty(n, dtype=int)
        rev_order[order] = np.arange(n)
        feats_sorted = X[order]

        if model is None:
            clf = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=T,
                linkage="average",
            )
        else:
            clf = model
            clf.distance_threshold = T

        lbls_sorted = clf.fit_predict(feats_sorted)
        return lbls_sorted[rev_order].astype(np.int64)


def cluster_all(
    df: pd.DataFrame,
    T: float,
    cell_m: float = 500.0,
    precomputed_trees: Optional[Dict[Any, Any]] = None,
    precomputed_blocks: Optional[Dict[Any, np.ndarray]] = None,
    sort_info: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster all complaints across spatio-temporal blocks.

    Clusters complaints inside each block defined by make_blocks(df, cell_m).
    Deterministic sorting by (x_m, y_m, timestamp, id) is enforced.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe containing 'group', 'x_m', 'y_m', 'timestamp', 'id',
        and optionally 'date'.
    T : float
        Distance threshold for hierarchical clustering.
    cell_m : float, default=500.0
        Grid cell size in metres.
    precomputed_trees : Optional[Dict[Any, Any]], default=None
        Precomputed scipy linkage trees per block to accelerate multiple T sweeps.
        If provided, fcluster is used on precomputed linkage trees.
    precomputed_blocks : Optional[Dict[Any, np.ndarray]], default=None
        Precomputed block index dictionary mapping block_id to sorted row indices.
    sort_info : Optional[Tuple[np.ndarray, np.ndarray]], default=None
        Tuple of (sort_idx, rev_pos) to avoid re-sorting across multiple calls.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        global_labels : np.ndarray of shape (len(df),) with unique global cluster IDs.
        block_sizes : np.ndarray of block sizes (counts of rows per block).
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if sort_info is not None and precomputed_blocks is not None:
        sort_idx, rev_pos = sort_info
        block_groups = precomputed_blocks
    else:
        # Sort deterministically upfront
        sort_cols = [c for c in ["x_m", "y_m", "timestamp", "id"] if c in df.columns]
        if sort_cols:
            sort_idx = df.sort_values(sort_cols).index
            df_sorted = df.loc[sort_idx]
        else:
            sort_idx = df.index
            df_sorted = df

        block_series = make_blocks(df_sorted, cell_m=cell_m)
        block_groups = df_sorted.groupby(block_series, sort=False).indices

        pos_map = {orig_idx: i for i, orig_idx in enumerate(df.index)}
        sorted_pos = np.array([pos_map[idx] for idx in df_sorted.index])
        rev_pos = np.empty(n, dtype=int)
        rev_pos[sorted_pos] = np.arange(n)

    block_sizes = np.array([len(indices) for indices in block_groups.values()], dtype=np.int64)

    global_labels_sorted = np.empty(n, dtype=np.int64)
    next_label = 0

    if precomputed_trees is not None:
        for b_key, b_idx in block_groups.items():
            k_len = len(b_idx)
            if k_len == 1:
                global_labels_sorted[b_idx[0]] = next_label
                next_label += 1
            else:
                tree = precomputed_trees[b_key]
                lbls = fcluster(tree, t=T, criterion="distance") - 1
                global_labels_sorted[b_idx] = lbls + next_label
                next_label += int(lbls.max() + 1)
    else:
        # Prepare features if trees not precomputed
        df_sorted = df.loc[sort_idx]
        if "t_hours" in df_sorted.columns:
            t_h = df_sorted["t_hours"].values.astype(float)
        else:
            ts = pd.to_datetime(df_sorted["timestamp"])
            t_h = ts.values.astype("datetime64[s]").astype(float) / 3600.0
        t_feat = t_h * 50.0

        feats = np.column_stack([
            df_sorted["x_m"].values.astype(float),
            df_sorted["y_m"].values.astype(float),
            t_feat,
        ])

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=T,
            linkage="average",
        )
        for b_idx in block_groups.values():
            k_len = len(b_idx)
            if k_len == 1:
                global_labels_sorted[b_idx[0]] = next_label
                next_label += 1
            else:
                lbls = model.fit_predict(feats[b_idx])
                global_labels_sorted[b_idx] = lbls + next_label
                next_label += int(lbls.max() + 1)

    global_labels = global_labels_sorted[rev_pos]
    return global_labels, block_sizes


def precompute_block_linkages(
    df: pd.DataFrame,
    cell_m: float = 500.0,
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute linkage matrices for all blocks with size > 1.
    Allows rapid multi-threshold sweeps (seconds instead of minutes).

    Returns
    -------
    Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        trees : Dict mapping block_id to scipy linkage matrix.
        block_groups : Dict mapping block_id to sorted row indices.
        sort_info : Tuple of (sort_idx, rev_pos) for mapping back.
    """
    n = len(df)
    sort_cols = [c for c in ["x_m", "y_m", "timestamp", "id"] if c in df.columns]
    if sort_cols:
        sort_idx = df.sort_values(sort_cols).index
        df_sorted = df.loc[sort_idx].copy()
    else:
        sort_idx = df.index
        df_sorted = df.copy()

    block_series = make_blocks(df_sorted, cell_m=cell_m)
    block_groups = df_sorted.groupby(block_series, sort=False).indices

    pos_map = {orig_idx: i for i, orig_idx in enumerate(df.index)}
    sorted_pos = np.array([pos_map[idx] for idx in df_sorted.index])
    rev_pos = np.empty(n, dtype=int)
    rev_pos[sorted_pos] = np.arange(n)
    sort_info = (sort_idx, rev_pos)

    ts = pd.to_datetime(df_sorted["timestamp"])
    t_h = ts.values.astype("datetime64[s]").astype(float) / 3600.0
    feats = np.column_stack([
        df_sorted["x_m"].values.astype(float),
        df_sorted["y_m"].values.astype(float),
        t_h * 50.0,
    ])

    trees = {}
    for b_key, b_idx in block_groups.items():
        if len(b_idx) > 1:
            trees[b_key] = linkage(feats[b_idx], method="average")

    return trees, block_groups, sort_info
