from __future__ import annotations

from typing import Dict, List, Tuple, Set, Iterable, Optional, Mapping, Iterator, Any
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class _TreesView(Mapping):
    r"""
    Read-only mapping view over :class:`HitPool`'s internal tree store that exposes
    **3-tuples** ``(tree, points, ids)`` for compatibility with branchers.

    Notes
    -----
    Internally, :class:`HitPool` keeps **4-tuples**
    ``(tree, points, ids, frozenset(ids))`` where the final element is an
    immutable set used for fast membership tests. This view **does not copy**
    arrays; it simply returns a sliced tuple so that downstream consumers that
    expect exactly three items continue to work.

    The mapping interface is lightweight:

    - Keys are layer identifiers ``(volume_id, layer_id)``.
    - Values are:

      * ``tree`` – :class:`scipy.spatial.cKDTree` built on ``points``,
      * ``points`` – ``(N,3)`` array of hit positions (meters, ``float64``),
      * ``ids`` – ``(N,)`` array of hit IDs (``int64``).
    """
    __slots__ = ("_base",)

    def __init__(self, base: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray, frozenset]]) -> None:
        self._base = base

    def __getitem__(self, key: Tuple[int, int]) -> Tuple[cKDTree, np.ndarray, np.ndarray]:
        r"""
        Get the **3-tuple** view for a layer.

        Parameters
        ----------
        key : tuple of int
            Layer key ``(volume_id, layer_id)``.

        Returns
        -------
        (tree, points, ids) : tuple
            Slices of the stored 4-tuple (no copies of the arrays).
        """
        entry = self._base[key]
        # return a sliced tuple view (no copies of arrays)
        return entry[0], entry[1], entry[2]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        r"""
        Iterate over layer keys.

        Yields
        ------
        tuple of int
            Keys ``(volume_id, layer_id)``.
        """
        return iter(self._base)

    def __len__(self) -> int:
        r"""
        Number of layers in the view.

        Returns
        -------
        int
        """
        return len(self._base)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        r"""
        Membership test.

        Parameters
        ----------
        key : object
            Candidate key.

        Returns
        -------
        bool
            ``True`` if the key exists.
        """
        return key in self._base

    def get(self, key: Tuple[int, int], default: Any = None) -> Any:
        r"""
        ``dict.get``-style accessor returning a 3-tuple.

        Parameters
        ----------
        key : tuple of int
            Layer key.
        default : Any, optional
            Value to return if ``key`` is absent.

        Returns
        -------
        Any
            ``(tree, points, ids)`` or ``default``.
        """
        if key in self._base:
            e = self._base[key]
            return e[0], e[1], e[2]
        return default

    def items(self):
        r"""
        Iterate over ``(key, value)`` items with **3-tuple** values.

        Yields
        ------
        ((volume_id, layer_id), (tree, points, ids))
        """
        # generator that yields (key, (tree, pts, ids))
        for k, e in self._base.items():
            yield k, (e[0], e[1], e[2])

    def values(self):
        r"""
        Iterate over **3-tuples** ``(tree, points, ids)``.

        Yields
        ------
        (tree, points, ids)
        """
        for e in self._base.values():
            yield (e[0], e[1], e[2])

    def keys(self):
        r"""
        View of layer keys.

        Returns
        -------
        KeysView
        """
        return self._base.keys()


class HitPool:
    r"""
    High-performance hit pool with per-layer KD-trees and fast assignment bookkeeping.

    Design goals
    ------------
    - **Zero mutation of the input DataFrames** (no CoW warnings).
    - **No prints**; keep the hot path quiet.
    - **Cache-friendly layer builds** using a vectorized unique/sort pass (no groupby).
    - **All arrays are contiguous NumPy** (``float64`` coords, ``int64`` ids).
    - **Set-based assignment** with bulk ops for :math:`\mathcal{O}(H)` adds/removes.
    - **Compatibility layer**: exposes 3-tuples to branchers via a read-only view.
    - Ready-made helpers for both *kNN* and *radius* queries filtered by assignment.

    Geometry and queries
    --------------------
    For a predicted position :math:`p\in\mathbb{R}^3` and a layer point set
    :math:`\{q_j\}_{j=1}^N`, we use Euclidean distance
    :math:`d_j = \|q_j-p\|_2`. The utilities provide:

    - **kNN**: choose indices of the :math:`k` smallest :math:`d_j`.
    - **Ball**: choose indices with :math:`d_j \le r`.

    In both cases, **assigned** hits are filtered out using a global set.

    Attributes
    ----------
    hits : :class:`pandas.DataFrame`
        All hits (columns include ``hit_id, x, y, z, volume_id, layer_id``).
    pt_cut_hits : :class:`pandas.DataFrame`
        Truth-matched & :math:`p_T`-filtered view retained for downstream usage.
    _assigned_hits : set of int
        Globally reserved hit IDs.
    _trees : dict[(int,int) -> (cKDTree, ndarray(N,3), ndarray(N,), frozenset)]
        Per-layer spatial indexes and cached immutable id sets.
    _trees_view : :class:`_TreesView`
        Read-only mapping exposing **3-tuples** for brancher compatibility.
    _layers_sorted : list[(int,int)]
        Cached sorted layer keys (stable across calls).
    """

    __slots__ = ("hits", "pt_cut_hits", "_assigned_hits", "_trees", "_trees_view", "_layers_sorted")

    def __init__(self, hits: pd.DataFrame, pt_cut_hits: pd.DataFrame) -> None:
        self.hits = hits
        self.pt_cut_hits = pt_cut_hits
        self._assigned_hits: Set[int] = set()
        self._trees: Dict[
            Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray, frozenset]
        ] = self._build_layer_trees_fast(self.hits)
        self._trees_view = _TreesView(self._trees)  # ← compatibility view (3-tuples)
        self._layers_sorted: List[Tuple[int, int]] = sorted(self._trees.keys(), key=lambda x: (x[0], x[1]))

    @property
    def trees(self) -> Mapping[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]]:
        r"""
        Per-layer KD-trees and backing arrays as **3-tuples** ``(tree, points, ids)``.

        Returns
        -------
        mapping
            Read-only mapping compatible with branchers. No array duplication.
        """
        return self._trees_view

    @property
    def layers(self) -> List[Tuple[int, int]]:
        r"""
        Sorted list of layer keys.

        Returns
        -------
        list of tuple of int
            ``[(volume_id, layer_id), ...]`` ordered lexicographically.
        """
        return self._layers_sorted

    def _build_layer_trees_fast(
        self, hits: pd.DataFrame
    ) -> Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray, frozenset]]:
        r"""
        Build per-layer KD-trees using a single **stable sort** on an integer-encoded key.

        Avoids ``pandas.groupby`` overhead and does not add temporary columns.

        Parameters
        ----------
        hits : :class:`pandas.DataFrame`
            Must provide columns ``x,y,z,volume_id,layer_id,hit_id``.

        Returns
        -------
        dict
            Mapping ``(vol, lay) -> (tree, points, hit_ids, frozenset(hit_ids))``, where
            ``points`` is ``(N,3)`` ``float64`` and ``hit_ids`` is ``(N,)`` ``int64``.

        Notes
        -----
        We encode a 64-bit key

        .. math::

            k \;=\; (\text{volume\_id} \ll 32) \;|\; (\text{layer\_id} \;\&\; 2^{32}-1),

        perform a single **stable mergesort**, and then take contiguous segments
        of identical keys as layers. Complexity is
        :math:`\mathcal{O}(N \log N)` for sorting plus linear passes.

        The KD-tree is built with ``balanced_tree=True`` and ``compact_nodes=True``
        for cache-friendly queries.
        """
        # Materialize necessary columns once, with tight dtypes & contiguity.
        x = np.asarray(hits["x"].to_numpy(dtype=np.float64, copy=False), order="C")
        y = np.asarray(hits["y"].to_numpy(dtype=np.float64, copy=False), order="C")
        z = np.asarray(hits["z"].to_numpy(dtype=np.float64, copy=False), order="C")
        vol = np.asarray(hits["volume_id"].to_numpy(dtype=np.int64, copy=False), order="C")
        lay = np.asarray(hits["layer_id"].to_numpy(dtype=np.int64, copy=False), order="C")
        hid = np.asarray(hits["hit_id"].to_numpy(dtype=np.int64, copy=False), order="C")

        n = hid.size
        if not (x.size == y.size == z.size == vol.size == lay.size == n):
            raise ValueError("Mismatched column lengths in hits DataFrame.")

        # Encode layer key as 64-bit int: (vol << 32) | (lay & 0xffffffff)
        key = (vol << 32) | (lay & np.int64(0xFFFFFFFF))

        # Stable sort once by key; contiguous segments are layers.
        order = np.argsort(key, kind="mergesort")
        key_s = key[order]
        x_s, y_s, z_s, hid_s, vol_s, lay_s = x[order], y[order], z[order], hid[order], vol[order], lay[order]

        # Find segment boundaries
        # boundaries indices include 0 and n; segments are [b[i], b[i+1])
        change = np.empty(n + 1, dtype=bool)
        change[0] = True
        change[-1] = True
        if n > 1:
            change[1:-1] = key_s[1:] != key_s[:-1]
        bounds = np.flatnonzero(change)

        trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray, frozenset]] = {}
        for i in range(bounds.size - 1):
            a, b = bounds[i], bounds[i + 1]
            if a == b:
                continue
            vol_i, lay_i = int(vol_s[a]), int(lay_s[a])
            pts = np.column_stack((x_s[a:b], y_s[a:b], z_s[a:b]))
            ids = hid_s[a:b].copy()
            # Build tree
            tree = cKDTree(pts, copy_data=False, balanced_tree=True, compact_nodes=True)
            trees[(vol_i, lay_i)] = (tree, pts, ids, frozenset(ids.tolist()))
        return trees

    def get_candidates_knn(
        self,
        predicted_position: np.ndarray,
        layer: Tuple[int, int],
        max_candidates: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        k-nearest **unassigned** neighbors in a layer.

        Given a prediction :math:`p\in\mathbb{R}^3` and layer point set
        :math:`\{q_j\}`, returns up to ``max_candidates`` hits minimizing
        :math:`\|q_j - p\|_2`.

        Parameters
        ----------
        predicted_position : ndarray, shape (3,)
            Query position :math:`p` (meters).
        layer : tuple of int
            Layer key ``(volume_id, layer_id)``.
        max_candidates : int, optional
            Maximum :math:`k` to return (capped at layer size).

        Returns
        -------
        positions : ndarray, shape (M, 3), dtype float64
            Candidate positions, **unassigned only**.
        hit_ids : ndarray, shape (M,), dtype int64
            Corresponding hit IDs.

        Notes
        -----
        When ``k==1`` the underlying :meth:`cKDTree.query` returns scalars;
        we normalize via :func:`numpy.atleast_1d`. Assigned hits are filtered
        out post-query. If all nearest neighbors are assigned, returns empty arrays.
        """
        entry = self._trees.get(layer, None)
        if entry is None:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

        tree, points, hit_ids, _ = entry
        k = min(int(max_candidates), int(hit_ids.size))
        if k <= 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

        # cKDTree.query returns scalars for k==1; normalize with np.atleast_1d
        _, idx = tree.query(np.asarray(predicted_position, dtype=np.float64), k=k)
        idx = np.atleast_1d(idx)
        cand_ids = hit_ids[idx]

        # Filter out assigned (fast small-k and vectorized large-k)
        assigned = self._assigned_hits
        if idx.size <= 8:
            mask = np.fromiter((int(h) not in assigned for h in cand_ids), dtype=bool, count=idx.size)
        else:
            # Convert set to array once; vectorized isin is faster for larger k
            mask = ~np.isin(cand_ids, np.fromiter((h for h in assigned), dtype=np.int64, count=len(assigned)))
        if not mask.any():
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

        idx_keep = idx[mask]
        return points[idx_keep], hit_ids[idx_keep]

    def get_candidates_in_gate(
        self,
        predicted_position: np.ndarray,
        layer: Tuple[int, int],
        radius: float,
        max_return: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Radius gate: **unassigned** hits within distance ``radius`` of a prediction.

        Parameters
        ----------
        predicted_position : ndarray, shape (3,)
            Query position :math:`p`.
        layer : tuple of int
            Layer key.
        radius : float
            Gate radius :math:`r` (meters). Hits satisfy :math:`\|q_j-p\|_2 \le r`.
        max_return : int, optional
            If given, truncate the output to at most this many hits.

        Returns
        -------
        positions : ndarray, shape (M, 3), dtype float64
        hit_ids : ndarray, shape (M,), dtype int64

        Notes
        -----
        Uses :meth:`cKDTree.query_ball_point`. Assigned hits are removed. If no
        points lie inside the ball, returns empty arrays.
        """
        entry = self._trees.get(layer, None)
        if entry is None or radius <= 0.0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

        tree, points, hit_ids, _ = entry
        idxs = tree.query_ball_point(np.asarray(predicted_position, dtype=np.float64), r=float(radius))
        if not idxs:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

        idxs = np.asarray(idxs, dtype=np.int64)
        ids = hit_ids[idxs]

        # Unassigned filter
        assigned = self._assigned_hits
        if idxs.size <= 16:
            mask = np.fromiter((int(h) not in assigned for h in ids), dtype=bool, count=idxs.size)
        else:
            mask = ~np.isin(ids, np.fromiter((h for h in assigned), dtype=np.int64, count=len(assigned)))
        if not mask.any():
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)

        idxs = idxs[mask]
        if max_return is not None and idxs.size > int(max_return):
            idxs = idxs[: int(max_return)]
        return points[idxs], hit_ids[idxs]

    def assign_hit(self, hit_id: int) -> bool:
        r"""
        Reserve a hit globally.

        Parameters
        ----------
        hit_id : int
            Hit identifier.

        Returns
        -------
        bool
            ``True`` if the hit was newly assigned, ``False`` if it was already reserved.
        """
        if hit_id in self._assigned_hits:
            return False
        self._assigned_hits.add(int(hit_id))
        return True

    def assign_hits(self, hit_ids: Iterable[int]) -> int:
        r"""
        Reserve many hits at once.

        Parameters
        ----------
        hit_ids : iterable of int
            Hit identifiers.

        Returns
        -------
        int
            Number of **newly** assigned hits.
        """
        before = len(self._assigned_hits)
        self._assigned_hits.update(int(h) for h in hit_ids)
        return len(self._assigned_hits) - before

    def release_hit(self, hit_id: int) -> bool:
        r"""
        Release a hit reservation.

        Parameters
        ----------
        hit_id : int

        Returns
        -------
        bool
            ``True`` if the hit was previously assigned and is now released.
        """
        try:
            self._assigned_hits.remove(int(hit_id))
            return True
        except KeyError:
            return False

    def release_hits(self, hit_ids: Iterable[int]) -> int:
        r"""
        Release many hits.

        Parameters
        ----------
        hit_ids : iterable of int

        Returns
        -------
        int
            Count of hits actually removed from the assignment set.
        """
        cnt = 0
        s = self._assigned_hits
        for h in hit_ids:
            try:
                s.remove(int(h))
                cnt += 1
            except KeyError:
                pass
        return cnt

    def reset(self) -> None:
        r"""
        Release **all** reservations.

        Notes
        -----
        Equivalent to clearing the internal assignment set.
        """
        self._assigned_hits.clear()

    def is_hit_available(self, hit_id: int) -> bool:
        r"""
        Test whether a hit is **not** currently assigned.

        Parameters
        ----------
        hit_id : int

        Returns
        -------
        bool
        """
        return int(hit_id) not in self._assigned_hits

    def get_available_hit_count(self) -> int:
        r"""
        Number of unassigned hits in the whole pool.

        Returns
        -------
        int
        """
        return int(len(self.hits) - len(self._assigned_hits))

    def get_assignment_ratio(self) -> float:
        r"""
        Fraction of assigned hits in :math:`[0,1]`.

        Returns
        -------
        float
            ``len(assigned) / len(hits)`` (``0.0`` if ``hits`` is empty).
        """
        total = len(self.hits)
        return (len(self._assigned_hits) / total) if total else 0.0

    def get_layer_statistics(self) -> Dict[Tuple[int, int], Dict[str, float | int]]:
        r"""
        Vectorized per-layer assignment statistics.

        Returns
        -------
        dict
            ``layer -> {'total_hits', 'assigned_hits', 'available_hits', 'assignment_ratio'}``.

        Notes
        -----
        Uses a single vectorized :func:`numpy.isin` against the layer's ``ids``;
        complexity per layer is :math:`\mathcal{O}(N_\text{layer} + A)`, where
        :math:`A` is the number of assigned hits (converted once to an array).
        """
        stats: Dict[Tuple[int, int], Dict[str, float | int]] = {}
        assigned = np.fromiter((h for h in self._assigned_hits), dtype=np.int64, count=len(self._assigned_hits))
        for layer, (_, _, ids, _) in self._trees.items():
            total = int(ids.size)
            if total == 0:
                stats[layer] = {
                    "total_hits": 0,
                    "assigned_hits": 0,
                    "available_hits": 0,
                    "assignment_ratio": 0.0,
                }
                continue
            if assigned.size == 0:
                assigned_cnt = 0
            else:
                assigned_cnt = int(np.isin(ids, assigned, assume_unique=False).sum())
            stats[layer] = {
                "total_hits": total,
                "assigned_hits": assigned_cnt,
                "available_hits": total - assigned_cnt,
                "assignment_ratio": (assigned_cnt / total) if total else 0.0,
            }
        return stats

    def get_unassigned_hits_in_layer(self, layer: Tuple[int, int]) -> List[int]:
        r"""
        List **unassigned** hit IDs for a given layer.

        Parameters
        ----------
        layer : tuple of int

        Returns
        -------
        list of int
        """
        entry = self._trees.get(layer, None)
        if entry is None:
            return []
        _, _, ids, _ = entry
        if not self._assigned_hits:
            return ids.tolist()
        mask = ~np.isin(ids, np.fromiter((h for h in self._assigned_hits), dtype=np.int64, count=len(self._assigned_hits)))
        return ids[mask].tolist()

    def get_assigned_hits_in_layer(self, layer: Tuple[int, int]) -> List[int]:
        r"""
        List **assigned** hit IDs for a given layer.

        Parameters
        ----------
        layer : tuple of int

        Returns
        -------
        list of int
        """
        entry = self._trees.get(layer, None)
        if entry is None:
            return []
        _, _, ids, _ = entry
        if not self._assigned_hits:
            return []
        mask = np.isin(ids, np.fromiter((h for h in self._assigned_hits), dtype=np.int64, count=len(self._assigned_hits)))
        return ids[mask].tolist()

    def has_layer(self, layer: Tuple[int, int]) -> bool:
        r"""
        Check whether the pool contains a layer.

        Parameters
        ----------
        layer : tuple of int

        Returns
        -------
        bool
        """
        return layer in self._trees

    def layer_size(self, layer: Tuple[int, int]) -> int:
        r"""
        Number of hits in a layer.

        Parameters
        ----------
        layer : tuple of int

        Returns
        -------
        int
            ``0`` if the layer is missing.
        """
        entry = self._trees.get(layer, None)
        return int(entry[2].size) if entry is not None else 0
