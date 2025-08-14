from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def build_layer_trees(
    hits: pd.DataFrame,
) -> Tuple[Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]], List[Tuple[int, int]]]:
    r"""
    Build **per-layer** KD-trees from a hits table without mutating ``hits``.

    Given a DataFrame with detector hits and their layer identifiers
    ``(volume_id, layer_id)``, the function performs a **single lexicographic
    sort** by ``(volume_id, layer_id)`` and then constructs a compact
    :class:`scipy.spatial.cKDTree` for each contiguous layer segment. Arrays are
    materialized with tight, contiguous dtypes (``float64`` for coordinates,
    ``int64`` for ids) and **no temporary columns** are added to ``hits``.

    Mathematical outline
    --------------------
    Let the sorted layer keys be :math:`(v_i,\ell_i)` for rows :math:`i=0,\dots,N-1`.
    Segment boundaries are detected by

    .. math::

    b_0 = 0,\qquad
    b_{i} = \mathbf{1}\!\left[(v_i,\ell_i)\neq(v_{i-1},\ell_{i-1})\right]
    \ \ (i\ge 1),

    and layer segments are the half-open intervals between the ``True`` indices of
    :math:`b` (with the final end at :math:`N`). For a segment with indices
    :math:`[a,b)`, the point cloud is :math:`P=\{(x_j,y_j,z_j)\}_{j=a}^{b-1}` and a
    KD-tree :math:`\mathcal{T}_{v,\ell}` is built over :math:`P`.

    Complexity
    ----------
    - Sorting: :math:`\mathcal{O}(N\log N)` for :func:`numpy.lexsort`.
    - Per layer :math:`k\in\{1,\dots,K\}` with size :math:`n_k`:
    KD-tree build cost ~ :math:`\mathcal{O}(n_k \log n_k)`.
    - Memory: one contiguous ``(N,3)`` float64 block for points (per layer) and one
    ``(N,)`` int64 block for hit ids (per layer). The input ``hits`` is untouched.

    Parameters
    ----------
    hits : pandas.DataFrame
        Table with at least the columns
        ``'volume_id'``, ``'layer_id'``, ``'hit_id'``, ``'x'``, ``'y'``, ``'z'``.
        Coordinate units are passed through unchanged; ensure consistency across rows.

    Returns
    -------
    trees : dict[(int, int) -> (cKDTree, ndarray, ndarray)]
        Mapping from ``(volume_id, layer_id)`` to a 3-tuple:
        ``(tree, points, hit_ids)`` where

        - ``tree`` is a :class:`scipy.spatial.cKDTree` built on ``points``,
        - ``points`` is ``(N,3)`` ``float64`` with C-contiguous layout
        storing ``(x,y,z)``, and
        - ``hit_ids`` is ``(N,)`` ``int64`` aligned with ``points``.

    layers : list[(int, int)]
        Sorted unique layer keys found in ``hits`` (ascending by volume, then layer).

    Raises
    ------
    KeyError
        If any required column is missing from ``hits``.
    ValueError
        If the required columns have mismatched lengths.

    Notes
    -----
    - Trees are built with ``balanced_tree=True`` and ``compact_nodes=True`` for
    cache-friendly queries; ``copy_data=False`` avoids duplicating the point array.
    - When ``len(hits) == 0``, returns ``({}, [])``.
    - The function copies only what is necessary for numerical kernels (points and
    ids) and never writes back to ``hits`` (no pandas CoW warnings).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from scipy.spatial import cKDTree
    >>> hits = pd.DataFrame({
    ...     "volume_id":[8,8,8,13], "layer_id":[2,2,3,1],
    ...     "hit_id":[10,11,12,13],
    ...     "x":[0.1,0.2,0.3,-0.1], "y":[0.0,0.1,0.1, 0.2], "z":[1.0,1.1,1.2,-0.5]
    ... })
    >>> trees, layers = build_layer_trees(hits)
    >>> layers
    [(8, 2), (8, 3), (13, 1)]
    >>> tree, pts, ids = trees[(8, 2)]
    >>> isinstance(tree, cKDTree), pts.shape, ids.dtype
    (True, (2, 3), dtype('int64'))
    """
    # Fast column materialization with tight dtypes & contiguous layout
    try:
        vol = hits["volume_id"].to_numpy(dtype=np.int64,  copy=False)
        lay = hits["layer_id"].to_numpy(dtype=np.int64,   copy=False)
        hid = hits["hit_id"].to_numpy(dtype=np.int64,     copy=False)
        x   = hits["x"].to_numpy(dtype=np.float64,        copy=False)
        y   = hits["y"].to_numpy(dtype=np.float64,        copy=False)
        z   = hits["z"].to_numpy(dtype=np.float64,        copy=False)
    except KeyError as e:
        raise KeyError(f"Missing required column: {e.args[0]}") from e

    n = hid.size
    if not (vol.size == lay.size == x.size == y.size == z.size == n):
        raise ValueError("Mismatched column lengths in `hits` DataFrame.")

    if n == 0:
        return {}, []

    # Single stable sort by (volume_id, layer_id)
    order = np.lexsort((lay, vol))  # sorts by vol, then lay (ascending)
    vol_s = vol[order]
    lay_s = lay[order]
    hid_s = np.ascontiguousarray(hid[order], dtype=np.int64)
    x_s   = np.ascontiguousarray(x[order], dtype=np.float64)
    y_s   = np.ascontiguousarray(y[order], dtype=np.float64)
    z_s   = np.ascontiguousarray(z[order], dtype=np.float64)

    # Segment boundaries where (vol, lay) changes
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (vol_s[1:] != vol_s[:-1]) | (lay_s[1:] != lay_s[:-1])
    starts = np.flatnonzero(change)
    # Ends are the next starts, plus final n
    ends = np.empty(starts.size, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = n

    trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]] = {}
    layers: List[Tuple[int, int]] = []

    # Build a compact KD-tree for each contiguous layer segment
    for a, b in zip(starts, ends):
        if a == b:
            continue
        vol_i = int(vol_s[a])
        lay_i = int(lay_s[a])

        # Points as a single contiguous (N,3) array
        pts = np.empty((b - a, 3), dtype=np.float64)
        pts[:, 0] = x_s[a:b]
        pts[:, 1] = y_s[a:b]
        pts[:, 2] = z_s[a:b]

        ids = hid_s[a:b].copy()  # keep a private view for safety

        tree = cKDTree(pts, copy_data=False, balanced_tree=True, compact_nodes=True)
        key = (vol_i, lay_i)
        trees[key] = (tree, pts, ids)
        layers.append(key)

    # Already sorted by (vol, lay) due to lexsort order
    # (defensive: ensure ascending lexicographic order)
    layers.sort(key=lambda t: (t[0], t[1]))
    return trees, layers
