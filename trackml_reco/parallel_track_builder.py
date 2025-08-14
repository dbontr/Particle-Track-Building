from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Type, Tuple, Optional, Any, Sequence, Iterable

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from trackml_reco.branchers.brancher import Brancher
from trackml_reco.track_builder import TrackBuilder, TrackCandidate, TrackStatus
import trackml_reco.hit_pool as trk_hit_pool

logger = logging.getLogger(__name__)


class SharedHitBook:
    r"""
    Thread-safe best-score ledger for per-hit ownership.

    This object maintains a mapping

    .. math::

        \mathcal{B} \;:\; h \in \mathbb{Z} \longmapsto (s_h,\;o_h),

    where each hit id :math:`h` is associated with the **best known** branch score
    :math:`s_h \in \mathbb{R}\cup\{+\infty\}` and the owning track id
    :math:`o_h \in \mathbb{Z}`. Lower scores are better.

    A branch with score :math:`s` may **claim** a set of hits
    :math:`H = \{h_1,\dots,h_m\}` iff, at the moment of claiming,

    .. math::

        \forall h\in H:\quad s < s_h - \Delta,

    where :math:`\Delta\ge 0` is the *claim margin* (a small non-negative slack).
    The check and the subsequent commit are performed atomically under an internal
    mutex to avoid race conditions among threads.

    Notes
    -----
    - All methods are **thread-safe** unless documented otherwise.
    - Complexity: lookups and updates are :math:`\mathcal{O}(|H|)` for a claim.
    - The ledger only stores **winners**; losers need not be recorded.
    """

    __slots__ = ("_best", "_lock")

    def __init__(self) -> None:
        self._best: Dict[int, Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def snapshot_deny_hits(self, hits: Sequence[int]) -> List[int]:
        r"""
        Return the subset of ``hits`` that are already owned by *someone*.

        Parameters
        ----------
        hits : Sequence[int]
            Candidate hit ids that a branch intends to use.

        Returns
        -------
        list[int]
            Those ids in ``hits`` that currently appear in the internal best map.

        Notes
        -----
        This is a **read-only** snapshot taken without locking the whole map; it is
        safe because each lookup is independent. In the (rare) presence of concurrent
        commits, a caller should still treat this as a *best-effort* deny list.
        Complexity: :math:`\mathcal{O}(|\text{hits}|)`.
        """
        if not hits:
            return []
        best = self._best  # local ref; read-only
        # list comprehension is fastest here for small/medium lists
        return [hid for hid in hits if hid in best]

    def try_claim_branch(
        self,
        hit_ids: Sequence[int],
        branch_score: float,
        margin: float,
        owner_track_id: int,
    ) -> bool:
        r"""
        Attempt to atomically claim a set of hits for a branch.

        The operation succeeds iff the branch score is **strictly** better than the
        current best (by at least ``margin``) for *every* requested hit:

        .. math::

        \forall h\in \text{hit\_ids}:\quad
        \text{branch\_score} < s_h - \text{margin}.

        If successful, the method assigns all hits to ``owner_track_id`` with
        score ``branch_score`` in one critical section.

        Parameters
        ----------
        hit_ids : Sequence[int]
            Hits the branch intends to use. Empty is always admissible (returns True).
        branch_score : float
            Cumulative cost/score for the branch (lower is better).
        margin : float
            Non-negative slack applied to the comparison (:math:`\Delta` in the
            inequality above).
        owner_track_id : int
            Id of the track that will own these hits if the claim succeeds.

        Returns
        -------
        bool
            ``True`` if the claim succeeded and ownership was recorded; ``False`` if a
            conflict was detected.

        Notes
        -----
        - The method holds an internal lock around the *check-then-commit* sequence,
        making the operation **atomic**.
        - Complexity: :math:`\mathcal{O}(|\text{hit\_ids}|)`.
        """
        if not hit_ids:
            return True
        with self._lock:
            # check requirement first
            for hid in hit_ids:
                best = self._best.get(hid, (np.inf, -1))[0]
                if branch_score >= best - margin:
                    return False
            # commit ownership
            score_owner = (branch_score, owner_track_id)
            for hid in hit_ids:
                self._best[hid] = score_owner
            return True


class CollaborativeParallelTrackBuilder(TrackBuilder):
    r"""
    Parallel, *collaborative* track builder with shared hit-claiming.

    This builder distributes per-seed work across a thread pool and coordinates
    between threads using a global :class:`SharedHitBook`. For each seed group
    (typically one per ``particle_id``), we:

    1. Form a deny list combining **already assigned** hits and **currently
    claimed** hits from other threads (snapshot).
    2. Run the configured brancher on the seed to obtain candidate branches and
    their cumulative scores.
    3. Sort branches by ascending score and, for each branch, attempt to
    **claim** its hits via the shared book. A claim with score :math:`s`
    succeeds iff, for every hit :math:`h` in the branch,

    .. math:: s < s_h - \Delta,

    where :math:`s_h` is the best recorded score for :math:`h` and
    :math:`\Delta=\)` ``claim_margin``.
    4. On success, we *commit* the track: assign its hits in the
    :class:`~trackml_reco.hit_pool.HitPool`, store the candidate, and (optionally)
    merge the brancher's debug graph.

    The per-seed loop can be **time-bounded** by ``per_seed_time_budget_s``; when
    the elapsed time exceeds the budget we stop considering further branches for
    that seed.

    Parameters
    ----------
    hit_pool : HitPool
        Source of hits and per-layer KD-trees.
    brancher_cls : Type[Brancher]
        Brancher class to instantiate per seed (e.g. EKF, PSO, A* variants).
    brancher_config : dict, optional
        Keyword arguments forwarded to ``brancher_cls(...)``.
    max_workers : int, optional
        Size of the :class:`concurrent.futures.ThreadPoolExecutor`.
    claim_margin : float, optional
        Margin :math:`\Delta` used in the claim inequality above.
    per_seed_time_budget_s : float or None, optional
        Optional time budget (seconds) per seed/particle.
    enable_plot, overlay_predicted, show_truth : bool, optional
        Plotting toggles for the optional R–Z visualization.
    record_graph : bool, optional
        If ``True``, merges debug graphs returned by branchers into a global graph.

    Notes
    -----
    - **Thread safety**: inter-thread coordination happens solely through
    :class:`SharedHitBook` and the hit assignments in :class:`HitPool`.
    - **Determinism**: due to parallel evaluation and time budgets, exact output
    ordering and selection can vary run-to-run unless all sources of randomness
    are fixed and budgets are disabled.
    """

    def __init__(
        self,
        hit_pool: trk_hit_pool.HitPool,
        brancher_cls: Type[Brancher],
        brancher_config: Optional[Dict] = None,
        max_workers: int = 8,
        claim_margin: float = 1.0,
        per_seed_time_budget_s: Optional[float] = None,
        *,
        # toggles (zero-cost when False)
        enable_plot: bool = False,
        overlay_predicted: bool = False,
        show_truth: bool = False,
        record_graph: bool = False,
    ) -> None:
        super().__init__(hit_pool, brancher_cls, brancher_config or {})
        self.max_workers = int(max_workers)
        self.claim_margin = float(claim_margin)
        self.per_seed_time_budget_s = per_seed_time_budget_s

        # plotting / graph controls
        self.enable_plot = bool(enable_plot)
        self.overlay_predicted = bool(overlay_predicted)
        self.show_truth = bool(show_truth)
        self.record_graph = bool(record_graph)

        self.graph = nx.DiGraph() if self.record_graph else nx.DiGraph()  # kept for API compatibility
        self._measured_traj: Dict[int, np.ndarray] = {}

        # shared score book
        self._book = SharedHitBook()

        # Build layer→hit_ids map once from HitPool.trees (fast, NumPy only)
        # HitPool.trees values are (cKDTree, points, ids, [maybe frozenset])
        self._layer_to_hit_ids: Dict[Tuple[int, int], np.ndarray] = {}
        for layer, pack in self.hit_pool.trees.items():
            # tolerate either 3- or 4-tuple (backward/forward compatibility)
            ids = pack[2]
            self._layer_to_hit_ids[layer] = np.asarray(ids, dtype=np.int64, order="C")

    def build_tracks_from_truth(
        self,
        max_seeds: Optional[int] = None,
        max_tracks_per_seed: int = 30,
        max_branches: int = 12,
        jitter_sigma: float = 1e-3,
    ) -> List[TrackCandidate]:
        r"""
        Build tracks from truth-seeded triplets in **parallel**.

        Seeds are grouped by ``particle_id``; each group that contains at least
        three ordered seed points spawns a task. Tasks run concurrently in a thread
        pool, each invoking the brancher once and committing any branches that
        successfully claim their hits.

        Parameters
        ----------
        max_seeds : int or None, optional
            If provided, limit the number of seeds used to initialize the build.
        max_tracks_per_seed : int, optional
            Upper bound requested from the brancher (a *soft* limit).
        max_branches : int, optional
            Maximum number of **committed** tracks per particle.
        jitter_sigma : float, optional
            Small noise added during seed building (meters).

        Returns
        -------
        list[TrackCandidate]
            All committed tracks (across all particles/seeds).

        Notes
        -----
        - The optional per-seed time budget is enforced within each task.
        - The global debug graph (if enabled) is the union of per-seed brancher graphs.

        Examples
        --------
        >>> builder.build_tracks_from_truth(max_seeds=100, max_branches=8)  # doctest: +SKIP
        """
        seeds_df = self.build_seeds_from_truth(max_seeds, jitter_sigma)

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = []
            # iterate by particle_id; we only need 3 seed points per particle
            for pid, group in seeds_df.groupby("particle_id", sort=False):
                if len(group) < 3:
                    continue
                futures.append(
                    exe.submit(
                        self._process_seed_group_parallel,
                        pid,
                        group,
                        max_tracks_per_seed,
                        max_branches,
                    )
                )

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.exception("Seed task failed: %s", e)

        # plotting is optional and never triggered unless enabled
        if self.enable_plot:
            self._plot_tracks_rz(
                overlay_predicted=self.overlay_predicted,
                show_truth=self.show_truth,
            )
        return self.completed_tracks

    def _process_seed_group_parallel(
        self,
        pid: int,
        seed_group: pd.DataFrame,
        max_tracks: int,
        max_branches: int,
    ) -> None:
        r"""
        Process a single seed group (one particle) in a worker thread.

        Workflow
        --------
        1. Sort the group's three seed points by their seed index to build
        ``seed_xyz`` and the ordered list of future ``layers``.
        2. Form a deny list as the union of:
        (a) a snapshot of currently **claimed** hits (from :class:`SharedHitBook`),
        and (b) globally **assigned** hits in :class:`HitPool`.
        3. Instantiate the brancher and run it once to obtain candidate branches.
        4. Optionally merge the returned graph.
        5. Iterate branches in ascending score and attempt to **claim** their hits.
        On success, assign hits in the pool and commit the track. Stop if either
        ``max_branches`` is reached or the per-seed time budget is exceeded.

        Parameters
        ----------
        pid : int
            Ground-truth particle id for this seed group.
        seed_group : pandas.DataFrame
            DataFrame holding the seed points (must include ``x,y,z``,
            ``seed_point_index`` and ``future_layers``).
        max_tracks : int
            Maximum number of tracks to request from the brancher.
        max_branches : int
            Maximum number of **committed** tracks for this particle.

        Returns
        -------
        None
        """
        start_time = time.perf_counter()
        brancher = self.brancher_cls(trees=self.hit_pool.trees, **self.brancher_config)

        sg = seed_group.sort_values("seed_point_index", kind="stable")
        seed_xyz = sg[["x", "y", "z"]].to_numpy()[:3]
        future_layers: Sequence[Tuple[int, int]] = sg.iloc[0]["future_layers"]
        t = np.linspace(0.0, 1.0, len(future_layers) + 3, dtype=np.float64)

        # deny-list: already-claimed + already-assigned on these layers
        candidate_layer_hits = self._hits_on_layers_fast(future_layers)
        deny_claimed = set(self._book.snapshot_deny_hits(candidate_layer_hits))
        deny_assigned = self.hit_pool._assigned_hits  # set[int], read-only here
        deny_hits = list(deny_claimed.union(deny_assigned))

        # brancher run
        try:
            branches, G = brancher.run(
                seed_xyz, future_layers, t, plot_tree=self.record_graph, deny_hits=deny_hits
            )
        except TypeError:
            branches, G = brancher.run(
                seed_xyz, future_layers, t, plot_tree=self.record_graph
            )

        if self.record_graph and isinstance(G, nx.DiGraph) and G.number_of_edges() > 0:
            self._merge_brancher_graph(G)

        if not branches:
            return

        # best first
        branches.sort(key=lambda b: float(b.get("score", np.inf)))

        committed = 0
        # hard per-seed time budget (if any)
        budget = self.per_seed_time_budget_s

        for br in branches:
            if budget is not None and (time.perf_counter() - start_time) > budget:
                logger.info("[pid=%s] time budget exceeded; committed=%d", pid, committed)
                break

            hit_ids = list(map(int, br.get("hit_ids", ())))
            if not hit_ids:
                continue

            score = float(br.get("score", np.inf))
            tentative_id = self.next_track_id  # optimistic id used for claim ownership

            if not self._book.try_claim_branch(hit_ids, score, self.claim_margin, tentative_id):
                continue  # lost the claim

            # claim succeeded → commit
            track = TrackCandidate(
                id=tentative_id,
                particle_id=pid,
                trajectory=br.get("traj", []),
                hit_ids=hit_ids,
                state=br.get("state", None),
                covariance=br.get("cov", None),
                score=score,
                status=(
                    TrackStatus.COMPLETED
                    if len(hit_ids) == len(future_layers)
                    else TrackStatus.ACTIVE
                ),
            )
            self.next_track_id += 1

            self.hit_pool.assign_hits(track.hit_ids)
            self.completed_tracks.append(track)
            self.track_candidates_by_particle.setdefault(pid, {})[track.id] = track
            self._measured_traj[track.id] = self._hit_ids_to_xyz(track.hit_ids)
            committed += 1

            if committed >= max_branches:
                break

        # keep at most max_branches per particle locally
        self._prune_tracks(max_branches)

    def _hits_on_layers_fast(self, layers: Sequence[Tuple[int, int]]) -> List[int]:
        r"""
        Collect the **union** of hit ids over the requested layers.

        Parameters
        ----------
        layers : Sequence[tuple[int,int]]
            Sequence of ``(volume_id, layer_id)`` keys.

        Returns
        -------
        list[int]
            Concatenation of per-layer id arrays (no deduplication).

        Notes
        -----
        This is a pure NumPy fast path that avoids any pandas calls. The result can be
        used to pre-compute deny lists or candidate id sets.
        """
        if not layers:
            return []
        out: List[np.ndarray] = []
        get = self._layer_to_hit_ids.get
        for key in layers:
            arr = get(key)
            if arr is not None and arr.size:
                out.append(arr)
        if not out:
            return []
        return np.concatenate(out, axis=0).tolist()

    def _merge_brancher_graph(self, G: nx.DiGraph) -> None:
        r"""
        Merge a brancher-provided graph into the global builder graph.

        Edges are copied verbatim. If node attributes contain a ``"pos"`` entry, it is
        propagated; otherwise the node is inserted without attributes.

        Parameters
        ----------
        G : nx.DiGraph
            Source graph produced by a brancher.

        Returns
        -------
        None

        Notes
        -----
        The merge is **lightweight** and does not attempt to deduplicate attributes
        beyond the node identity. It is safe to call repeatedly from multiple tasks.
        """
        # Avoid node attribute churn; only copy edges and minimal pos if present
        graph = self.graph
        npos = G.nodes
        for u, v, data in G.edges(data=True):
            # copy nodes with optional position
            pu = npos[u].get("pos") if u in npos else None
            pv = npos[v].get("pos") if v in npos else None
            if pu is not None:
                graph.add_node(u, pos=tuple(float(x) for x in pu))
            else:
                graph.add_node(u)
            if pv is not None:
                graph.add_node(v, pos=tuple(float(x) for x in pv))
            else:
                graph.add_node(v)
            graph.add_edge(u, v, **data)

    def _hit_ids_to_xyz(self, hit_ids: Iterable[int]) -> np.ndarray:
        r"""
        Map hit ids → positions as an ordered array.

        The lookup preserves the order of ``hit_ids`` using an index-based
        ``reindex`` and returns a dense array of shape :math:`(N,3)`.

        Parameters
        ----------
        hit_ids : Iterable[int]
            Hit ids to resolve.

        Returns
        -------
        ndarray, shape (N, 3), dtype float64
            Cartesian positions ``(x, y, z)`` in **meters**. Any ids not present in
            the table are skipped (rows with NaNs are dropped).

        Notes
        -----
        This function performs a single pandas ``set_index`` + ``reindex`` and then
        converts the result to a NumPy array for speed.
        """
        hit_ids = list(hit_ids)
        if not hit_ids:
            return np.empty((0, 3), dtype=np.float64)
        df = self.hit_pool.hits
        # Order-preserving lookup via index and reindex (fast, safe)
        xyz = (
            df.set_index("hit_id", drop=False)[["x", "y", "z"]]
            .reindex(hit_ids)
            .to_numpy(dtype=np.float64, copy=False)
        )
        # If any ids were missing (shouldn't), drop NaNs defensively
        if np.isnan(xyz).any():
            mask = ~np.isnan(xyz).any(axis=1)
            xyz = xyz[mask]
        return xyz

    def _plot_tracks_rz(
        self,
        *,
        overlay_predicted: bool = False,
        show_truth: bool = False,
    ) -> None:
        r"""
        Plot reconstructed tracks in the :math:`(r,z)` plane.

        For each layer, draw an axis-aligned rectangle bounding its hits in the
        :math:`(z,r)` projection (cheap visual cue). For each track, plot either the
        measured trajectory (when available) or the filtered trajectory. Optionally
        overlay ground-truth curves.

        Parameters
        ----------
        overlay_predicted : bool, optional
            If ``True``, overlay predicted (filtered) positions with a dotted line.
        show_truth : bool, optional
            If ``True``, overlay truth polylines.

        Returns
        -------
        None

        Notes
        -----
        The radial coordinate is computed as

        .. math:: r \,=\, \sqrt{x^2 + y^2}.

        This function is only called when plotting is explicitly enabled; it imports
        and uses :mod:`matplotlib` locally to avoid incurring GUI overhead otherwise.
        """
        hits_df = self.hit_pool.hits

        # Bulk precomputation
        x = hits_df["x"].to_numpy(dtype=np.float64, copy=False)
        y = hits_df["y"].to_numpy(dtype=np.float64, copy=False)
        z = hits_df["z"].to_numpy(dtype=np.float64, copy=False)
        r = np.sqrt(x * x + y * y)

        zmin, zmax = float(np.min(z)), float(np.max(z))
        rmin, rmax = float(np.min(r)), float(np.max(r))
        pad_z = 0.05 * (zmax - zmin) if zmax > zmin else 1.0
        pad_r = 0.05 * (rmax - rmin) if rmax > rmin else 1.0

        fig, ax = plt.subplots(figsize=(11, 8))

        # layer rectangles (cheap bounds using boolean masks once per layer)
        # Pull layer keys directly from HitPool.trees
        for (vol, lay), pack in self.hit_pool.trees.items():
            ids = pack[2]
            if ids.size == 0:
                continue
            # select rows for these hit_ids
            sub = hits_df.loc[hits_df.hit_id.isin(ids), ["x", "y", "z"]]
            if sub.empty:
                continue
            zz = sub["z"].to_numpy(dtype=np.float64, copy=False)
            rr = np.sqrt(
                sub["x"].to_numpy(dtype=np.float64, copy=False) ** 2
                + sub["y"].to_numpy(dtype=np.float64, copy=False) ** 2
            )
            rect = patches.Rectangle(
                (float(np.min(zz)), float(np.min(rr))),
                float(np.max(zz) - np.min(zz)),
                float(np.max(rr) - np.min(rr)),
                linewidth=1.0,
                edgecolor="black",
                facecolor="none",
                alpha=0.35,
            )
            ax.add_patch(rect)

        # legend stubs
        meas_handle, = ax.plot([], [], "-o", linewidth=1.1, label="measured hits", markersize=3)
        pred_with_hits_handle, = ax.plot([], [], ":", linewidth=0.9, label="filtered (with hits)")
        pred_no_hits_handle, = ax.plot([], [], "--o", linewidth=1.0, label="filtered (no hits)", markersize=3)
        truth_handle = None

        if show_truth and getattr(self.hit_pool, "pt_cut_hits", None) is not None:
            truth_df = self.hit_pool.pt_cut_hits
            first = True
            for _, grp in truth_df.groupby("particle_id", sort=False):
                xt = grp["x"].to_numpy(dtype=np.float64, copy=False)
                yt = grp["y"].to_numpy(dtype=np.float64, copy=False)
                zt = grp["z"].to_numpy(dtype=np.float64, copy=False)
                rt = np.sqrt(xt * xt + yt * yt)
                if first:
                    truth_handle, = ax.plot(zt, rt, "--k", linewidth=1.0, alpha=0.6, label="truth")
                    first = False
                else:
                    ax.plot(zt, rt, "--k", linewidth=1.0, alpha=0.6)

        # plot reconstructed tracks
        for tc in self.completed_tracks:
            meas = self._measured_traj.get(tc.id)
            if meas is not None and meas.size:
                zc = meas[:, 2]
                rc = np.sqrt(meas[:, 0] ** 2 + meas[:, 1] ** 2)
                ax.plot(zc, rc, "-o", markersize=3, alpha=0.9, linewidth=1.1)
                if overlay_predicted and tc.trajectory:
                    traj = np.asarray(tc.trajectory, dtype=np.float64)
                    zt = traj[:, 2]
                    rt = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
                    ax.plot(zt, rt, ":", alpha=0.6, linewidth=0.9)
            else:
                if tc.trajectory:
                    traj = np.asarray(tc.trajectory, dtype=np.float64)
                    zt = traj[:, 2]
                    rt = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
                    ax.plot(zt, rt, "--o", markersize=3, alpha=0.75, linewidth=1.0)

        ax.set_xlim(zmin - pad_z, zmax + pad_z)
        ax.set_ylim(rmin - pad_r, rmax + pad_r)
        ax.set_xlabel("z (m)")
        ax.set_ylabel("r (m)")
        ax.set_title("Tracks over detector layers — r vs z")
        handles = [meas_handle, pred_with_hits_handle, pred_no_hits_handle]
        if truth_handle is not None:
            handles.append(truth_handle)
        ax.legend(handles=handles, loc="best", framealpha=0.85)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
