from __future__ import annotations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, List

import numpy as np
from scipy.spatial import cKDTree


def _as_xyz(a: np.ndarray | Sequence[Sequence[float]] | None) -> np.ndarray:
    r"""
    Coerce input to a contiguous ``(N, 3)`` array of ``float64`` XYZ positions.

    Parameters
    ----------
    a : array_like or sequence of sequence or None
        Input positions. If ``a`` has at least three columns, only the first
        three are used; otherwise an empty array is returned.

    Returns
    -------
    ndarray, shape (N, 3)
        XYZ positions in meters (no unit conversion is applied here). If ``a``
        is ``None`` or is not at least 2-D with 3 columns, returns an empty
        array of shape ``(0, 3)``.

    Notes
    -----
    The function is intentionally permissive and never raises on malformed
    inputs; downstream metrics treat empty arrays as "no data".
    """
    if a is None:
        return np.empty((0, 3), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return a[:, :3]


def _pairwise_min_dists_small(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""
    Compute :math:`d_i=\min_j\|A_i-B_j\|_2` by broadcasting (small inputs).

    Parameters
    ----------
    A : ndarray, shape (NA, 3)
        Query points.
    B : ndarray, shape (NB, 3)
        Reference points.

    Returns
    -------
    ndarray, shape (NA,)
        For each row of ``A``, the Euclidean distance to the nearest row in ``B``.
        If ``B`` is empty, distances are ``+∞``. If ``A`` is empty, returns
        an empty array.

    Notes
    -----
    This is an :math:`\mathcal{O}(N_A N_B)` method using
    broadcasting; it is faster than building a KD-tree only for small sizes.
    """
    if A.size == 0:
        return np.empty((0,), dtype=np.float64)
    if B.size == 0:
        return np.full((len(A),), np.inf, dtype=np.float64)
    diff = A[:, None, :] - B[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    return np.sqrt(d2.min(axis=1))


def _nn_dists(
    A: np.ndarray,
    B: np.ndarray,
    *,
    tree_B: Optional[cKDTree] = None,
    return_tree: bool = False,
    n_jobs: Optional[int] = None,
    switch_small: int = 64,  # if min(len(A),len(B)) <= this -> use broadcast path
) -> Tuple[np.ndarray, Optional[cKDTree]] | np.ndarray:
    r"""
    Nearest-neighbor distances from each point in ``A`` to the set ``B``.

    Let :math:`A=\{a_i\}_{i=1}^{N_A}`, :math:`B=\{b_j\}_{j=1}^{N_B}`.
    This function returns

    .. math::

        d_i \;=\; \min_{1\le j\le N_B}\ \|a_i-b_j\|_2,\qquad i=1,\dots,N_A.

    Two computation paths are used:

    1. **Small problem fast path** (when ``min(NA, NB) <= switch_small``):
       direct broadcasting (quadratic in size).
    2. **KD-tree path**: a :class:`scipy.spatial.cKDTree` is built on ``B``
       (or reused if provided) and one-NN queries are issued.

    Parameters
    ----------
    A : ndarray, shape (NA, 3)
        Query points.
    B : ndarray, shape (NB, 3)
        Reference points to search.
    tree_B : cKDTree, optional
        Prebuilt KD-tree on ``B`` to reuse.
    return_tree : bool, optional
        If ``True``, also return the KD-tree (possibly the input one).
    n_jobs : int or None, optional
        Number of worker threads for KD-tree queries. Forwarded as
        ``workers=...`` to :meth:`cKDTree.query` when available.
    switch_small : int, optional
        Threshold to use the broadcasting fast path.

    Returns
    -------
    d : ndarray, shape (NA,)
        Nearest distances from each ``A[i]`` to ``B``. ``+∞`` if ``B`` is empty.
    tree : cKDTree or None
        Returned only if ``return_tree=True``.

    Notes
    -----
    - If ``A`` is empty, returns an empty array (and the tree if requested).
    - The KD-tree path is typically superior once either set exceeds ~``switch_small``.
    """
    nA, nB = len(A), len(B)
    if nA == 0:
        out = np.empty((0,), dtype=np.float64)
        return (out, tree_B) if return_tree else out
    if nB == 0:
        out = np.full((nA,), np.inf, dtype=np.float64)
        return (out, tree_B) if return_tree else out

    # Small-problem fast path (avoid KD-tree overhead)
    if min(nA, nB) <= switch_small:
        d = _pairwise_min_dists_small(A, B)
        return (d, tree_B) if return_tree else d

    # KD-tree path
    if tree_B is None:
        tree_B = cKDTree(B, balanced_tree=True, compact_nodes=True, copy_data=False)

    # SciPy >=1.6 supports workers
    kwargs = {}
    if n_jobs is not None:
        kwargs["workers"] = int(n_jobs)

    d, _ = tree_B.query(A, k=1, **kwargs)
    if return_tree:
        return np.asarray(d, dtype=np.float64), tree_B
    return np.asarray(d, dtype=np.float64)


def compute_metrics(
    xs: np.ndarray,
    true_points: np.ndarray,
    tol: float = 0.005,
    *,
    names: Optional[Iterable[str]] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    r"""
    Compute geometric track-quality metrics (fast, extensible).

    Let :math:`P=\{p_i\}_{i=1}^{N_p}\subset\mathbb{R}^3` be predicted points and
    :math:`T=\{t_j\}_{j=1}^{N_t}\subset\mathbb{R}^3` be ground truth. Define
    nearest-neighbor distances

    .. math::
        d^{(p)}_i = \min_j \|p_i - t_j\|_2,\qquad
        d^{(t)}_j = \min_i \|t_j - p_i\|_2.

    The following metrics are returned (subset controllable by ``names``):

    - **mse**:
      :math:`\operatorname{MSE}=\frac{1}{N_p}\sum_i (d^{(p)}_i)^2`
      (``inf`` if ``N_p=0``).
    - **rmse**:
      :math:`\sqrt{\operatorname{MSE}}`.
    - **recall** (%) at tolerance ``tol``:
      :math:`100\cdot \frac{\#\{j:\ d^{(t)}_j\le \text{tol}\}}{N_t}`.
    - **precision** (%) at tolerance ``tol``:
      :math:`100\cdot \frac{\#\{i:\ d^{(p)}_i\le \text{tol}\}}{N_p}`.
    - **f1** (%) harmonic mean of precision and recall:
      :math:`\frac{2PR}{P+R}` (with :math:`P,R` in percent).
    - **tp**, **fp**, **fn**:
      counts at threshold ``tol`` (``tp`` is the **min** of truth-side and
      pred-side positives to be conservative).
    - **n_pred**, **n_true**, **tol**:
      basic counts and the tolerance repeated for convenience.

    Parameters
    ----------
    xs : array_like, shape (N_p, >=3)
        Predicted positions. Only the first three columns are used.
    true_points : array_like, shape (N_t, >=3)
        Ground-truth positions. Only the first three columns are used.
    tol : float, optional
        Distance threshold (meters) used for precision/recall/F1.
    names : iterable of str or None, optional
        If provided, compute and return only these metrics. Valid names:
        ``{"mse","rmse","recall","precision","f1","tp","fp","fn","n_pred","n_true","tol"}``.
        Requesting a subset skips unnecessary KD-tree computations.
    n_jobs : int or None, optional
        Forwarded to :meth:`scipy.spatial.cKDTree.query` as ``workers=...``.

    Returns
    -------
    dict
        Mapping from metric name to value (``float``). If ``names`` is given,
        only those keys are present (others are omitted).

    Examples
    --------
    >>> P = np.array([[0,0,0],[1,0,0],[0,1,0]], float)
    >>> T = np.array([[0,0,0],[1,1,0]], float)
    >>> m = compute_metrics(P, T, tol=0.75)
    >>> round(m["rmse"], 3), round(m["recall"], 1)
    (0.408, 50.0)
    """
    P = _as_xyz(xs)
    T = _as_xyz(true_points)

    # Decide which distances we actually need to compute
    default_keys = ("mse", "rmse", "recall", "precision", "f1", "tp", "fp", "fn", "n_pred", "n_true", "tol")
    wanted = set(default_keys) if names is None else set(names)

    need_pred_d = any(k in wanted for k in ("mse", "rmse", "precision", "tp", "fp", "n_pred"))
    need_true_d = any(k in wanted for k in ("recall", "f1", "fn", "n_true"))

    d_pred = None
    d_true = None
    pred_tree = None
    true_tree = None

    # Compute pred->truth once (and get truth KD-tree), if needed
    if need_pred_d:
        d_pred, true_tree = _nn_dists(P, T, return_tree=True, n_jobs=n_jobs)

    # Compute truth->pred once (reuse pred KD-tree if we already built it), if needed
    if need_true_d:
        d_true, pred_tree = _nn_dists(T, P, return_tree=True, n_jobs=n_jobs)

    out: Dict[str, float] = {}

    # Counts
    if "n_pred" in wanted:
        out["n_pred"] = float(len(P))
    if "n_true" in wanted:
        out["n_true"] = float(len(T))
    if "tol" in wanted:
        out["tol"] = float(tol)

    # Regression metrics
    if "mse" in wanted or "rmse" in wanted:
        if d_pred is None:  # can happen if only rmse requested but not mse explicitly
            d_pred = _nn_dists(P, T, n_jobs=n_jobs)
        if len(d_pred):
            mse = float(np.mean(d_pred * d_pred))
        else:
            mse = float("inf")
        if "mse" in wanted:
            out["mse"] = mse
        if "rmse" in wanted:
            out["rmse"] = float(np.sqrt(mse)) if np.isfinite(mse) else float("inf")

    # Efficiency metrics (percentages + TP/FP/FN)
    if any(k in wanted for k in ("recall", "precision", "f1", "tp", "fp", "fn")):
        # Truth-side (recall)
        if d_true is None:
            d_true = _nn_dists(T, P, n_jobs=n_jobs)
        tp_truth = int(np.sum(d_true <= tol))
        fn = int(len(d_true) - tp_truth)
        recall = (100.0 * tp_truth / len(d_true)) if len(d_true) else 0.0

        # Pred-side (precision)
        if d_pred is None:
            d_pred = _nn_dists(P, T, n_jobs=n_jobs)
        tp_pred = int(np.sum(d_pred <= tol))
        fp = int(len(d_pred) - tp_pred)
        precision = (100.0 * tp_pred / len(d_pred)) if len(d_pred) else 0.0

        # Conservative TP count (min of the two sides)
        if "tp" in wanted:
            out["tp"] = float(min(tp_truth, tp_pred))
        if "fp" in wanted:
            out["fp"] = float(fp)
        if "fn" in wanted:
            out["fn"] = float(fn)
        if "recall" in wanted:
            out["recall"] = float(recall)
        if "precision" in wanted:
            out["precision"] = float(precision)
        if "f1" in wanted:
            denom = precision + recall
            out["f1"] = float(2.0 * precision * recall / denom) if denom > 0 else 0.0

    # If a subset was requested, return only those keys in order-independent dict
    if names is not None:
        return {k: out[k] for k in names if k in out}
    return out


def unpack(metrics: Mapping[str, float], *keys: str) -> Tuple[float, ...]:
    r"""
    Convenience extractor for metric dicts.

    Parameters
    ----------
    metrics : mapping
        Dictionary returned by :func:`compute_metrics`.
    *keys : str
        Keys to extract in order.

    Returns
    -------
    tuple of float
        Values in the requested order; missing keys yield ``np.nan``.

    Examples
    --------
    >>> m = {"mse": 0.1, "recall": 80.0}
    >>> unpack(m, "recall", "precision", "mse")
    (80.0, nan, 0.1)
    """
    return tuple(float(metrics.get(k, np.nan)) for k in keys)


def compute_metrics_compat(xs: np.ndarray, true_points: np.ndarray, tol: float = 0.005) -> Tuple[float, float]:
    r"""
    Backwards-compatible interface returning ``(mse, pct_recovered)``.

    Parameters
    ----------
    xs : array_like, shape (N_p, >=3)
        Predicted positions.
    true_points : array_like, shape (N_t, >=3)
        Ground-truth positions.
    tol : float, optional
        Distance threshold (meters) for the recovered-hit percentage.

    Returns
    -------
    mse : float
        Mean squared nearest-truth distance of predictions.
    pct_recovered : float
        Alias of **recall** (%) at the given ``tol``.

    Notes
    -----
    This mirrors earlier project code paths that consumed a two-tuple. It is a
    thin wrapper around :func:`compute_metrics`.
    """
    m = compute_metrics(xs, true_points, tol=tol, names=("mse", "recall"))
    return m["mse"], m["recall"]


def branch_mse(branch: Mapping, true_xyz: np.ndarray) -> float:
    r"""
    MSE of a branch trajectory against the nearest ground-truth hits.

    Parameters
    ----------
    branch : mapping
        Object with key ``"traj"`` storing an iterable of 3D points.
    true_xyz : array_like, shape (N, >=3)
        Ground-truth hit positions.

    Returns
    -------
    float
        :math:`\frac{1}{N_p}\sum_i d_i^2` where :math:`d_i` is the distance from
        branch point :math:`i` to its nearest truth. Returns ``+∞`` if the
        branch trajectory is empty.
    """
    traj = _as_xyz(np.asarray(branch.get("traj"), dtype=np.float64))
    if traj.size == 0:
        return float("inf")
    return compute_metrics(traj, _as_xyz(true_xyz), names=("mse",))["mse"]


def branch_hit_stats(branch: Mapping, true_xyz: np.ndarray, threshold: float = 1.0) -> Tuple[float, int]:
    r"""
    Recall (%) and misses (FN) for a branch versus ground truth.

    Parameters
    ----------
    branch : mapping
        Object with key ``"traj"`` storing an iterable of 3D points.
    true_xyz : array_like, shape (N, >=3)
        Ground-truth positions.
    threshold : float, optional
        Distance threshold (meters) for a "hit match".

    Returns
    -------
    recall : float
        Percentage of truth hits within ``threshold`` of some branch point.
    fn : int
        Number of truth hits **not** matched within ``threshold``.

    Notes
    -----
    Uses the truth-side distances :math:`d^{(t)}_j` defined in
    :func:`compute_metrics` with ``tol=threshold``.
    """
    traj = _as_xyz(np.asarray(branch.get("traj"), dtype=np.float64))
    met = compute_metrics(traj, _as_xyz(true_xyz), tol=threshold, names=("recall", "fn"))
    return float(met["recall"]), int(met["fn"])
