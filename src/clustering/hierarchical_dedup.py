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
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import coo_matrix

from src.preprocessing.features import project, unproject, load_projection_constants


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


def make_time_chunks(
    df: pd.DataFrame,
    cell_m: float = 500.0,
    t_gap_hours: float = 12.0,
    max_chunk: int = 12000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Partition complaints into spatio-temporal time-gap chunks.

    chunk = rows of the same (group, floor(x_m/cell_m), floor(y_m/cell_m)),
    sorted by (timestamp, id), split where the gap to the previous row exceeds t_gap_hours.
    If a chunk has more than max_chunk rows, split it recursively at its largest
    internal time gap until every part <= max_chunk.

    Docstring: splitting at gaps > Tmax/time_scale hours is lossless under average
    linkage because every cross-gap pair distance exceeds Tmax; forced splits are lossy
    and are counted; spatial cell edges remain lossy and cannot be measured with
    address-based weak labels.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe with 'group', 'x_m', 'y_m', 'timestamp', and 'id'.
    cell_m : float, default=500.0
        Grid cell size in metres.
    t_gap_hours : float, default=12.0
        Maximum temporal gap in hours before splitting into a new chunk.
    max_chunk : int, default=12000
        Maximum allowed chunk size. Chunks exceeding this size are recursively
        split at their largest internal time gap.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        chunk_ids : np.ndarray of shape (len(df),) with integer chunk IDs aligned with df.
        report : Dict with 'n_chunks', 'n_forced_splits', 'rows_in_forced_splits'.
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.int64), {
            "n_chunks": 0,
            "n_forced_splits": 0,
            "rows_in_forced_splits": 0,
        }

    groups = df["group"].astype(str).values
    gx = np.floor(df["x_m"].values / cell_m).astype(int)
    gy = np.floor(df["y_m"].values / cell_m).astype(int)
    ts = pd.to_datetime(df["timestamp"])
    ts_sec = ts.values.astype("datetime64[s]").astype(np.int64)
    ids = df["id"].astype(str).values

    # Deterministic sorting by cell, timestamp, id
    keys = pd.MultiIndex.from_arrays([groups, gx, gy])
    codes, _ = pd.factorize(keys)
    order = np.lexsort((ids, ts_sec, codes))

    sorted_codes = codes[order]
    sorted_ts = ts_sec[order]

    t_gap_sec = float(t_gap_hours * 3600.0)
    new_chunk = np.empty(n, dtype=bool)
    new_chunk[0] = True
    new_chunk[1:] = (sorted_codes[1:] != sorted_codes[:-1]) | (
        (sorted_ts[1:] - sorted_ts[:-1]) > t_gap_sec
    )

    chunk_starts = np.where(new_chunk)[0]
    chunk_ends = np.append(chunk_starts[1:], n)

    # Check for oversized chunks needing forced recursive splits
    n_forced_splits = 0
    rows_in_forced_splits = 0

    def _split_oversized(st_pos: int, en_pos: int) -> List[Tuple[int, int]]:
        nonlocal n_forced_splits
        if (en_pos - st_pos) <= max_chunk:
            return [(st_pos, en_pos)]
        sub_gaps = sorted_ts[st_pos + 1 : en_pos] - sorted_ts[st_pos : en_pos - 1]
        split_rel = int(np.argmax(sub_gaps)) + 1
        split_pos = st_pos + split_rel
        n_forced_splits += 1
        return _split_oversized(st_pos, split_pos) + _split_oversized(split_pos, en_pos)

    final_chunks: List[Tuple[int, int]] = []
    for st, en in zip(chunk_starts, chunk_ends):
        chunk_len = en - st
        if chunk_len <= max_chunk:
            final_chunks.append((st, en))
        else:
            rows_in_forced_splits += chunk_len
            parts = _split_oversized(st, en)
            final_chunks.extend(parts)

    chunk_ids_sorted = np.empty(n, dtype=np.int64)
    for c_id, (st, en) in enumerate(final_chunks):
        chunk_ids_sorted[st:en] = c_id

    chunk_ids = np.empty(n, dtype=np.int64)
    chunk_ids[order] = chunk_ids_sorted

    report = {
        "n_chunks": len(final_chunks),
        "n_forced_splits": n_forced_splits,
        "rows_in_forced_splits": rows_in_forced_splits,
    }
    return chunk_ids, report


def linkage_per_chunk(
    df: pd.DataFrame,
    chunk_ids: np.ndarray,
    time_scale_m_per_h: float = 50.0,
) -> Dict[str, Any]:
    """
    Precompute scipy average linkage trees per chunk. Chunks of size 1 skip the model.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe containing 'x_m', 'y_m', and 'timestamp'.
    chunk_ids : np.ndarray
        Integer chunk assignments of shape (len(df),).
    time_scale_m_per_h : float, default=50.0
        Temporal conversion factor in metres per hour.

    Returns
    -------
    Dict[str, Any]
        linkages cache containing:
        - 'trees': Dict[int, np.ndarray] of scipy linkage matrices for chunks > 1.
        - 'chunk_groups': Dict[int, np.ndarray] of row indices per chunk.
        - 'size_1_indices': np.ndarray of indices for singleton chunks.
        - 'time_scale': float conversion factor.
        - 'n_rows': total number of rows.
    """
    n = len(df)
    ts = pd.to_datetime(df["timestamp"])
    t_hours = ts.values.astype("datetime64[s]").astype(float) / 3600.0

    feats = np.column_stack([
        df["x_m"].values.astype(float),
        df["y_m"].values.astype(float),
        t_hours * float(time_scale_m_per_h),
    ])

    chunk_series = pd.Series(np.arange(n))
    chunk_groups = chunk_series.groupby(chunk_ids, sort=True).indices

    trees: Dict[int, np.ndarray] = {}
    size_1_list: List[int] = []

    for c_id, idxs in chunk_groups.items():
        if len(idxs) == 1:
            size_1_list.append(idxs[0])
        else:
            trees[c_id] = linkage(feats[idxs], method="average")

    return {
        "trees": trees,
        "chunk_groups": chunk_groups,
        "size_1_indices": np.array(size_1_list, dtype=np.int64),
        "time_scale": time_scale_m_per_h,
        "n_rows": n,
    }


def cut_incidents(
    linkages: Union[Dict[str, Any], Any],
    T: float,
    chunk_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Cut precomputed hierarchical linkages at distance threshold T to form global incident clusters.

    Parameters
    ----------
    linkages : Dict[str, Any]
        Output of linkage_per_chunk.
    T : float
        Distance threshold in projected spatio-temporal units (metres).
    chunk_ids : Optional[np.ndarray], default=None
        Optional chunk IDs if raw trees dict is provided.

    Returns
    -------
    np.ndarray
        Global 0-indexed integer incident labels of shape (n_rows,).
    """
    n = linkages["n_rows"]
    trees = linkages["trees"]
    chunk_groups = linkages["chunk_groups"]
    size_1_indices = linkages.get("size_1_indices", None)

    global_labels = np.empty(n, dtype=np.int64)
    next_label = 0

    if size_1_indices is not None and len(size_1_indices) > 0:
        global_labels[size_1_indices] = np.arange(len(size_1_indices))
        next_label = len(size_1_indices)
        for c_id, tree in trees.items():
            idxs = chunk_groups[c_id]
            lbls = fcluster(tree, t=T, criterion="distance") - 1
            global_labels[idxs] = lbls + next_label
            next_label += int(lbls.max() + 1)
    else:
        for c_id, idxs in chunk_groups.items():
            k_len = len(idxs)
            if k_len == 1:
                global_labels[idxs[0]] = next_label
                next_label += 1
            else:
                tree = trees[c_id]
                lbls = fcluster(tree, t=T, criterion="distance") - 1
                global_labels[idxs] = lbls + next_label
                next_label += int(lbls.max() + 1)

    return global_labels


def pair_prf(
    pred: np.ndarray,
    true: np.ndarray,
) -> Dict[str, Union[int, float]]:
    """
    Compute exact pairwise Precision, Recall, and F1 via contingency table.
    Evaluated strictly on rows where true != -1.

    Parameters
    ----------
    pred : np.ndarray
        Predicted cluster labels of shape (N,).
    true : np.ndarray
        Ground truth cluster labels of shape (N,). Rows with true == -1
        are excluded from evaluation.

    Returns
    -------
    Dict[str, Union[int, float]]
        tp, fp, fn, precision, recall, f1
    """
    valid_mask = true != -1
    t_valid = true[valid_mask]
    p_valid = pred[valid_mask]

    if len(t_valid) <= 1:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 0.0,
        }

    t_codes, _ = pd.factorize(t_valid)
    p_codes, _ = pd.factorize(p_valid)

    counts = np.ones(len(t_codes), dtype=np.int64)
    mat = coo_matrix((counts, (t_codes, p_codes))).tocsr()

    data = mat.data.astype(np.int64)
    tp = int(np.sum(data * (data - 1) // 2))

    row_sums = np.array(mat.sum(axis=1), dtype=np.int64).ravel()
    tp_plus_fn = int(np.sum(row_sums * (row_sums - 1) // 2))

    col_sums = np.array(mat.sum(axis=0), dtype=np.int64).ravel()
    tp_plus_fp = int(np.sum(col_sums * (col_sums - 1) // 2))

    fp = int(tp_plus_fp - tp)
    fn = int(tp_plus_fn - tp)

    precision = float(tp / tp_plus_fp) if tp_plus_fp > 0 else 1.0
    recall = float(tp / tp_plus_fn) if tp_plus_fn > 0 else 1.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_incidents(
    df: pd.DataFrame,
    labels: np.ndarray,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
    return_map: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build consolidated incidents table from complaints and cluster labels.

    Deterministic incident_id formatted as 'INC-%06d' numbered by
    (first_timestamp, group, min row id).
    lat/lon is unprojected mean of member x_m/y_m via the existing projection (never refit).
    zone_id is the most frequent member zone_id with ties broken by smallest id.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe with 'id', 'timestamp', 'group', 'x_m', 'y_m', 'zone_id'.
    labels : np.ndarray
        Cluster labels of shape (len(df),).
    lat0 : Optional[float], default=None
        Projection origin latitude. If None, loaded from projection.json.
    lon0 : Optional[float], default=None
        Projection origin longitude. If None, loaded from projection.json.
    return_map : bool, default=False
        If True, returns (incidents_df, map_df).
        If False, returns incidents_df (with map_df available in incidents_df.attrs['map']).

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        incidents_df : Consolidated incidents DataFrame with columns:
            incident_id, lat, lon, count, group, first_timestamp, last_timestamp, zone_id.
    """
    if lat0 is None or lon0 is None:
        lat0, lon0 = load_projection_constants()

    n = len(df)
    order = np.argsort(labels)
    s_lbls = labels[order]
    s_ts = df["timestamp"].astype(str).values[order]
    s_id = df["id"].astype(str).values[order]
    s_grp = df["group"].astype(str).values[order]
    s_x = df["x_m"].values.astype(float)[order]
    s_y = df["y_m"].values.astype(float)[order]
    s_zone = df["zone_id"].values.astype(int)[order]

    uniq_lbls, first_idx, counts = np.unique(s_lbls, return_index=True, return_counts=True)
    n_inc = len(uniq_lbls)

    is_single = (counts == 1)
    single_pos = first_idx[is_single]

    first_ts = np.empty(n_inc, dtype=object)
    last_ts = np.empty(n_inc, dtype=object)
    min_id = np.empty(n_inc, dtype=object)
    grp_arr = np.empty(n_inc, dtype=object)
    x_mean = np.empty(n_inc, dtype=float)
    y_mean = np.empty(n_inc, dtype=float)
    zone_arr = np.empty(n_inc, dtype=int)

    first_ts[is_single] = s_ts[single_pos]
    last_ts[is_single] = s_ts[single_pos]
    min_id[is_single] = s_id[single_pos]
    grp_arr[is_single] = s_grp[single_pos]
    x_mean[is_single] = s_x[single_pos]
    y_mean[is_single] = s_y[single_pos]
    zone_arr[is_single] = s_zone[single_pos]

    multi_indices = np.where(~is_single)[0]
    for idx in multi_indices:
        st = first_idx[idx]
        cnt = counts[idx]
        en = st + cnt
        first_ts[idx] = s_ts[st:en].min()
        last_ts[idx] = s_ts[st:en].max()
        min_id[idx] = s_id[st:en].min()
        grp_arr[idx] = s_grp[st]
        x_mean[idx] = s_x[st:en].mean()
        y_mean[idx] = s_y[st:en].mean()
        # mode zone_id with smallest tie
        z_slice = s_zone[st:en]
        z_bincount = np.bincount(z_slice, minlength=8)
        zone_arr[idx] = int(np.argmax(z_bincount))

    # Unproject mean planar coordinates to lat/lon
    mean_lat, mean_lon = unproject(x_mean, y_mean, lat0=lat0, lon0=lon0)

    raw_df = pd.DataFrame({
        "_orig_lbl": uniq_lbls,
        "lat": mean_lat,
        "lon": mean_lon,
        "count": counts,
        "group": grp_arr,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "zone_id": zone_arr,
        "_min_id": min_id,
    })

    # Deterministic sorting: (first_timestamp, group, min row id)
    raw_df = raw_df.sort_values(by=["first_timestamp", "group", "_min_id"], ascending=True).reset_index(drop=True)

    # Assign incident_id format INC-%06d
    raw_df["incident_id"] = [f"INC-{i:06d}" for i in range(len(raw_df))]

    # Build mapping
    lbl_to_inc = dict(zip(raw_df["_orig_lbl"], raw_df["incident_id"]))
    incident_ids_per_row = np.array([lbl_to_inc[lbl] for lbl in labels], dtype=object)

    map_df = pd.DataFrame({
        "id": df["id"].astype(str).values,
        "incident_id": incident_ids_per_row,
    })

    incidents_df = raw_df[[
        "incident_id",
        "lat",
        "lon",
        "count",
        "group",
        "first_timestamp",
        "last_timestamp",
        "zone_id",
    ]].copy()

    incidents_df.attrs["map"] = map_df

    if return_map:
        return incidents_df, map_df
    return incidents_df


def build_complaint_incident_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    incidents_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build complaint-to-incident mapping table with columns id and incident_id.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe.
    labels : np.ndarray
        Cluster labels of shape (len(df),).
    incidents_df : pd.DataFrame
        Incidents dataframe produced by build_incidents.

    Returns
    -------
    pd.DataFrame
        Mapping DataFrame with columns ['id', 'incident_id'].
    """
    if "map" in incidents_df.attrs:
        return incidents_df.attrs["map"].copy()

    # Fallback if map not attached
    _, map_df = build_incidents(df, labels, return_map=True)
    return map_df


def online_assign(
    x_m: float,
    y_m: float,
    t: Any,
    group: str,
    incidents_xy: pd.DataFrame,
    time_scale: float = 50.0,
    T: float = 200.0,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
) -> Optional[str]:
    """
    Assign a single incoming complaint to an existing incident in real time.

    Docstring: simplified single-point rule for the live demo, not identical to average linkage.

    Parameters
    ----------
    x_m : float
        Projected x-coordinate in metres.
    y_m : float
        Projected y-coordinate in metres.
    t : Any
        Complaint timestamp (string or datetime).
    group : str
        Complaint group ('noise', 'road', 'flood', 'sewer_backup').
    incidents_xy : pd.DataFrame
        DataFrame of existing incidents. Must have 'incident_id', 'group',
        'first_timestamp', 'last_timestamp', and either ('x_m', 'y_m') or ('lat', 'lon').
    time_scale : float, default=50.0
        Temporal conversion factor in metres per hour.
    T : float, default=200.0
        Distance threshold in metres.
    lat0, lon0 : Optional[float]
        Projection origin constants (if x_m/y_m must be computed from lat/lon).

    Returns
    -------
    Optional[str]
        incident_id of the nearest same-group incident with distance <= T; None otherwise.
    """
    same_group = incidents_xy[incidents_xy["group"] == group].copy()
    if len(same_group) == 0:
        return None

    if "x_m" in same_group.columns and "y_m" in same_group.columns:
        inc_x = same_group["x_m"].values.astype(float)
        inc_y = same_group["y_m"].values.astype(float)
    elif "lat" in same_group.columns and "lon" in same_group.columns:
        if lat0 is None or lon0 is None:
            lat0, lon0 = load_projection_constants()
        inc_x, inc_y = project(
            same_group["lat"].values.astype(float),
            same_group["lon"].values.astype(float),
            lat0=lat0,
            lon0=lon0,
        )
    else:
        raise ValueError(
            "incidents_xy must contain either ('x_m', 'y_m') or ('lat', 'lon') columns."
        )

    t_target = pd.to_datetime(t)
    first_ts = pd.to_datetime(same_group["first_timestamp"])
    last_ts = pd.to_datetime(same_group["last_timestamp"])

    t_sec = t_target.value / 1e9
    f_sec = first_ts.values.astype("datetime64[s]").astype(float)
    l_sec = last_ts.values.astype("datetime64[s]").astype(float)

    # dt_hours is 0 if inside [first_timestamp, last_timestamp], else gap to nearest end
    dt_sec = np.maximum(0.0, np.maximum(f_sec - t_sec, t_sec - l_sec))
    dt_hours = dt_sec / 3600.0

    dx = float(x_m) - inc_x
    dy = float(y_m) - inc_y
    dist = np.sqrt(dx**2 + dy**2 + (float(time_scale) * dt_hours) ** 2)

    min_idx = np.argmin(dist)
    if dist[min_idx] <= T:
        return str(same_group["incident_id"].iloc[min_idx])
    return None

