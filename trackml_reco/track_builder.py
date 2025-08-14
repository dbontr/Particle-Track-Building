from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

import trackml_reco.hit_pool as trk_hit_pool
import trackml_reco.plotting as trk_plot
from trackml_reco.branchers.brancher import Brancher

logger = logging.getLogger(__name__)

# Global switch for per-seed branch plotting via trk_plot.plot_branches(...)
PLOT_BRANCHES = True

class TrackStatus(Enum):
    """Lifecycle state of a track candidate."""
    ACTIVE = "active"
    COMPLETED = "completed"
    DEAD = "dead"


@dataclass(slots=True)
class TrackCandidate:
    r"""
    Lightweight container for a single track hypothesis.

    A candidate aggregates the evolving estimate of a particle track produced by
    a branching EKF-based search: the ordered **measured** points, the consumed
    ``hit_ids`` in the same order, the latest EKF state and covariance, an
    accumulated scalar score, and bookkeeping metadata.

    Notation
    --------
    Let the EKF state be :math:`x\in\mathbb{R}^7` with layout
    ``[x, y, z, v_x, v_y, v_z, \kappa]`` and covariance :math:`P\in\mathbb{R}^{7\times7}`.
    Each accepted hit contributes a (Mahalanobis) incremental cost
    :math:`\chi^2_i\ge 0`; the candidate's score is the running sum

    .. math::

    S \;=\; \sum_{i=1}^{m} \chi^2_i,

    where :math:`m` is the number of post-seed measurements.

    Attributes
    ----------
    id : int
        Unique identifier assigned by the builder.
    particle_id : int
        Ground-truth particle id used to group seeds and evaluate tracks.
    trajectory : list of (3,) ndarray
        Ordered list of **measured** points accepted by the brancher
        (includes the three seed points first, followed by layer hits).
    hit_ids : list of int
        Hit identifiers in the same order as ``trajectory``.
    state : (7,) ndarray
        Latest EKF state vector.
    covariance : (7,7) ndarray
        Latest EKF covariance matrix.
    score : float
        Accumulated :math:`\sum \chi^2` cost.
    status : TrackStatus
        Lifecycle status (``ACTIVE``/``COMPLETED``/``DEAD``).
    parent_id : int, optional
        Optional parent track id (not used by the default builder).
    seed_row : pandas.Series, optional
        One of the seed rows from which the track originated (cached for debug).

    See Also
    --------
    TrackBuilder : Produces and manages :class:`TrackCandidate` objects.
    """
    id: int
    particle_id: int
    trajectory: List[np.ndarray]   # list of (3,) points
    hit_ids: List[int]             # consumed hits in same order (after seeds)
    state: np.ndarray              # (7,)
    covariance: np.ndarray         # (7,7)
    score: float                   # accumulated chi^2
    status: TrackStatus
    parent_id: Optional[int] = None
    seed_row: Optional[pd.Series] = None

    def add_hit(
        self,
        position: np.ndarray,
        hit_id: int,
        new_state: np.ndarray,
        new_covariance: np.ndarray,
        chi2_contribution: float,
    ) -> None:
        r"""
        Append one accepted measurement to the candidate and update bookkeeping.

        Parameters
        ----------
        position : (3,) array_like
            Measured Cartesian position :math:`z_i = (x,y,z)` appended to ``trajectory``.
        hit_id : int
            Identifier of the consumed hit (appended to ``hit_ids``).
        new_state : (7,) array_like
            EKF state after applying the measurement update.
        new_covariance : (7,7) array_like
            EKF covariance after the update.
        chi2_contribution : float
            Incremental measurement cost :math:`\chi^2_i` added to ``score``.

        Notes
        -----
        The cumulative score is updated as

        .. math::

        S_{\text{new}} \leftarrow S_{\text{old}} + \chi^2_i.

        This method does **not** perform any gating or EKF math; it assumes the
        caller has executed a predict/update and supplies the post-update state.
        """
        self.trajectory.append(position)
        self.hit_ids.append(int(hit_id))
        self.state = new_state
        self.covariance = new_covariance
        self.score += float(chi2_contribution)


class TrackBuilder:
    r"""
    Build tracks from truth-derived seeds using a branching EKF strategy.

    Pipeline
    --------
    1. **Seed extraction** from truth hits (three closest-in-radius layers per
    particle). The cylindrical radius is

    .. math:: r = \sqrt{x^2 + y^2}.

    2. **Branching EKF** search using a configurable :class:`~trackml_reco.branchers.brancher.Brancher`
    subclass (e.g. EKF + A* / GA / PSO / SA / Hungarian). The brancher returns
    one or more branches (ordered by cumulative cost), each carrying
    ``traj``, ``hit_ids``, ``state``, ``cov``, and a scalar ``score``
    :math:`S=\sum_i \chi^2_i`.

    3. **Track conversion & pruning**:
    branches are converted into :class:`TrackCandidate` instances, their
    hits are reserved in the :class:`~trackml_reco.hit_pool.HitPool`,
    and per-particle candidates are pruned to the top-``k`` lowest scores.

    Time Grid
    ---------
    For a seed with :math:`L` future layers, a simple normalized time vector
    is used with :math:`L+3` samples (seed points first):

    .. math::

    t_j \;=\; \frac{j}{L+2}, \qquad j=0,\ldots,L+2.

    Parameters
    ----------
    hit_pool : trackml_reco.hit_pool.HitPool
        Source of hits, truth-filtered hits, and per-layer KD-trees.
    brancher_cls : Type[Brancher]
        Brancher class to instantiate for each build (e.g. :class:`HelixEKFBrancher`).
    brancher_config : dict, optional
        Keyword arguments forwarded to ``brancher_cls(...)`` (e.g. noise, beam size).

    Attributes
    ----------
    hit_pool : HitPool
        Data and spatial indices.
    brancher_cls : Type[Brancher]
        Selected branching strategy class.
    brancher_config : dict
        Configuration used to construct branchers.
    track_candidates_by_particle : dict[int, dict[int, TrackCandidate]]
        Per-particle map of candidate tracks keyed by ``track_id``.
    completed_tracks : list[TrackCandidate]
        Tracks whose length reached the number of expected future layers.
    next_track_id : int
        Monotonic identifier for newly created candidates.
    seeds_df : pandas.DataFrame
        Cache of constructed seeds with columns ``x,y,z,particle_id,volume_id,
        layer_id,seed_point_index,future_layers,future_hit_ids``.

    Notes
    -----
    - The builder does not run any plotting by itself. If
    ``trackml_reco.plotting.PLOT_BRANCHES`` is set ``True`` in your process,
    :func:`trackml_reco.plotting.plot_branches` may be invoked for quick visuals.
    - All heavy EKF mathematics (gating, prediction, updates) lives in the
    brancher classes and their kernels; the builder orchestrates data flow
    and bookkeeping.

    See Also
    --------
    trackml_reco.branchers.brancher.Brancher
    trackml_reco.hit_pool.HitPool
    """

    def __init__(
        self,
        hit_pool: trk_hit_pool.HitPool,
        brancher_cls: Type[Brancher],
        brancher_config: Dict | None = None,
    ) -> None:
        self.hit_pool = hit_pool
        self.brancher_cls = brancher_cls
        self.brancher_config = dict(brancher_config or {})

        # Builder state
        self.track_candidates_by_particle: Dict[int, Dict[int, TrackCandidate]] = {}
        self.completed_tracks: List[TrackCandidate] = []
        self.next_track_id: int = 0

        # Seeds
        self.seeds_df: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _ensure_r_column(df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Ensure a cylindrical radius column ``r`` exists in a hits DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain float columns ``x`` and ``y`` (meters).

        Returns
        -------
        pandas.DataFrame
            A copy with an added ``r`` column if absent; otherwise the original
            object is returned unchanged.

        Notes
        -----
        The radius is computed in float64 as

        .. math:: r = \sqrt{x^2 + y^2},

        to match the precision used by downstream numerical routines.
        """
        if "r" in df.columns:
            return df
        out = df.copy()
        # Use float64 here to match numerical routines elsewhere
        xy = out[["x", "y"]].to_numpy(dtype=np.float64, copy=False)
        out["r"] = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
        return out

    def build_seeds_from_truth(
        self,
        max_seeds: int | None = None,
        jitter_sigma: float = 1e-4,
    ) -> pd.DataFrame:
        r"""
        Construct 3-point seeds per particle from truth-matched hits (vectorized).

        Algorithm
        ---------
        Given the truth-filtered hits (``hit_pool.pt_cut_hits``):

        1. Compute / ensure :math:`r=\sqrt{x^2+y^2}` and **stably** sort by ascending
        radius (inner → outer).
        2. For each ``(particle_id, volume_id, layer_id)`` triple keep the first
        occurrence (closest in ``r``) to avoid duplicates within a physical layer.
        3. Take the first three entries per particle as seed points; the remainder
        are considered future layers/hits.
        4. Optionally add i.i.d. Gaussian jitter :math:`\mathcal{N}(0,\sigma^2)` to
        the seed ``x,y,z`` (useful for stress-testing robustness).
        5. Attach per-particle lists:

        - ``future_layers`` : list of ``(volume_id, layer_id)`` tuples (order preserved)
        - ``future_hit_ids`` : list of hit ids for those future layers

        Parameters
        ----------
        max_seeds : int or None, default: None
            If set, keep only the first ``max_seeds`` particles (after sorting).
        jitter_sigma : float, default: 1e-4
            Standard deviation of Gaussian noise added to seed coordinates (meters).
            Set to ``0`` to disable.

        Returns
        -------
        pandas.DataFrame
            Seed rows with columns
            ``x,y,z,r,particle_id,volume_id,layer_id,seed_point_index,future_layers,future_hit_ids``.

        Warnings
        --------
        If a particle has fewer than three distinct layer hits after de-duplication,
        it is skipped.

        Examples
        --------
        >>> seeds = builder.build_seeds_from_truth(max_seeds=10, jitter_sigma=0.0)
        >>> seeds.head()[["particle_id","seed_point_index","future_layers"]]
        """
        th = getattr(self.hit_pool, "pt_cut_hits", None)
        if th is None or th.empty:
            logger.warning("No truth hits available to build seeds.")
            self.seeds_df = pd.DataFrame()
            return self.seeds_df

        # Ensure r exists
        th = self._ensure_r_column(th)

        # Stable sort by r (so cumcount is deterministic)
        th_sorted = th.sort_values("r", kind="mergesort")

        # Keep closest-in-radius hit per (pid, vol, layer)
        dedup = th_sorted.drop_duplicates(
            subset=["particle_id", "volume_id", "layer_id"], keep="first", ignore_index=True
        )

        # Rank within particle after sorting (0,1,2,...)
        dedup = dedup.copy()  # CoW-safe: ensure we own the frame
        ranks = dedup.groupby("particle_id", sort=False).cumcount().to_numpy(dtype=np.int64, copy=False)
        dedup = dedup.assign(_rank=ranks)


        # Split into seeds (first 3) and future (rest)
        seeds = dedup.loc[dedup["_rank"] < 3].copy()     # copy: we may mutate xyz
        future = dedup.loc[dedup["_rank"] >= 3].copy()

        # Optionally limit to first N particles (by appearance after sort)
        if max_seeds is not None:
            keep_pids = (
                seeds["particle_id"].drop_duplicates().iloc[: int(max_seeds)]
            )
            seeds = seeds[seeds["particle_id"].isin(keep_pids)]
            future = future[future["particle_id"].isin(keep_pids)]

        if seeds.empty:
            logger.warning("No seed groups with >=3 distinct layers.")
            self.seeds_df = pd.DataFrame()
            return self.seeds_df

        # Optional jitter (dtype-safe; avoids CoW/chained warnings)
        cols = ("x", "y", "z")
        if jitter_sigma and float(jitter_sigma) > 0.0:
            # Perform in float64, then cast back if original was float32
            xyz64 = seeds.loc[:, cols].to_numpy(dtype=np.float64, copy=False)
            noise = np.random.normal(scale=float(jitter_sigma), size=xyz64.shape)
            xyz64 += noise

            # If any of x/y/z were float32, cast all three back to float32
            wants_f32 = any(seeds[c].dtype == np.float32 for c in cols)
            seeds.loc[:, cols] = xyz64.astype(np.float32 if wants_f32 else np.float64, copy=False)

        # Precompute per-particle future layers and ids with fast tuple creation
        # Use to_records(...).tolist() to get List[Tuple[int,int]] quickly
        future_layers_map = {
            pid: g[["volume_id", "layer_id"]]
                .to_records(index=False)
                .tolist()
            for pid, g in future.groupby("particle_id", sort=False)
        }
        future_ids_map = {
            pid: g["hit_id"].astype(np.int64, copy=False).tolist()
            for pid, g in future.groupby("particle_id", sort=False)
        }

        # Order seeds by (pid, _rank) and attach seed_point_index + future lists
        seeds.sort_values(["particle_id", "_rank"], kind="mergesort", inplace=True)

        out_parts: list[pd.DataFrame] = []
        for pid, grp in seeds.groupby("particle_id", sort=False):
            if len(grp) < 3:
                # Skip particles without 3 distinct seed hits
                continue
            g = grp.copy()
            g["seed_point_index"] = g["_rank"].astype(np.int64, copy=False)
            flist = future_layers_map.get(pid, [])
            fidlist = future_ids_map.get(pid, [])
            # Broadcast shared lists across the three seed rows
            g["future_layers"] = [flist] * len(g)
            g["future_hit_ids"] = [fidlist] * len(g)
            out_parts.append(g)

        if not out_parts:
            logger.warning("No particles had at least 3 seed points.")
            self.seeds_df = pd.DataFrame()
            return self.seeds_df

        self.seeds_df = pd.concat(out_parts, ignore_index=True)
        if "_rank" in self.seeds_df.columns:
            self.seeds_df.drop(columns=["_rank"], inplace=True)

        logger.info(
            "Built %d seed points from %d particles",
            len(self.seeds_df),
            self.seeds_df["particle_id"].nunique(),
        )
        return self.seeds_df


    def build_tracks_from_truth(
        self,
        max_seeds: int | None = None,
        max_tracks_per_seed: int = 30,
        max_branches: int = 12,
        jitter_sigma: float = 1e-3,
    ) -> List[TrackCandidate]:
        r"""
        End-to-end convenience: build seeds then construct tracks.

        Parameters
        ----------
        max_seeds : int or None, default: None
            Forwarded to :meth:`build_seeds_from_truth`.
        max_tracks_per_seed : int, default: 30
            Upper bound on the number of branches the brancher may return per seed.
        max_branches : int, default: 12
            Per-particle pruning budget kept after conversion (top-k by lowest score).
        jitter_sigma : float, default: 1e-3
            Jitter passed to :meth:`build_seeds_from_truth`.

        Returns
        -------
        list[TrackCandidate]
            The list of **completed** tracks accumulated so far.

        See Also
        --------
        build_seeds_from_truth
        build_tracks_from_seeds
        """
        seeds_df = self.build_seeds_from_truth(max_seeds, jitter_sigma)
        return self.build_tracks_from_seeds(seeds_df, max_tracks_per_seed, max_branches)

    def build_tracks_from_seeds(
        self,
        seeds_df: pd.DataFrame,
        max_tracks_per_seed: int = 30,
        max_branches: int = 12,
    ) -> List[TrackCandidate]:
        r"""
        Build tracks by running the brancher for each 3-point seed group.

        Parameters
        ----------
        seeds_df : pandas.DataFrame
            Output of :meth:`build_seeds_from_truth`. Must contain per-particle groups
            with three seed rows and attached ``future_layers``.
        max_tracks_per_seed : int, default: 30
            Maximum branches considered per seed (forwarded to the brancher, if used).
        max_branches : int, default: 12
            Per-particle cap on candidates kept after pruning.

        Returns
        -------
        list[TrackCandidate]
            The growing list of **completed** tracks.

        Notes
        -----
        - A single brancher instance is constructed for this call and reused for all
        particles to avoid rebuilding per-layer KD-trees.
        - The builder does **not** plot here; optional visuals are handled elsewhere.
        """
        if seeds_df.empty:
            logger.warning("No seeds to build tracks from.")
            return self.completed_tracks

        # One brancher instance for this build (fast path, reuses trees)
        self.brancher = self.brancher_cls(trees=self.hit_pool.trees, **self.brancher_config)

        # Iterate per particle (three rows per seed group)
        for pid, group in seeds_df.groupby("particle_id", sort=False):
            if len(group) < 3:
                continue
            self._process_seed_group(group, max_tracks_per_seed, max_branches)

        return self.completed_tracks

    def _process_seed_group(
        self,
        seed_group: pd.DataFrame,
        max_tracks: int,
        max_branches: int,
    ) -> None:
        r"""
        Process one particle's seed group: invoke brancher, convert, and prune.

        Parameters
        ----------
        seed_group : pandas.DataFrame
            Exactly three seed rows for a single ``particle_id`` with attached
            ``future_layers``.
        max_tracks : int
            Brancher-side bound on how many branches to evaluate/return (if applicable).
        max_branches : int
            Keep at most this many lowest-score candidates for the particle.

        Notes
        -----
        The time grid has length :math:`L+3` where :math:`L=\text{len}(future\_layers)`.
        Branches returned by the brancher are sorted by cumulative cost
        :math:`S=\sum_i \chi^2_i` before conversion.

        If ``PLOT_BRANCHES`` is enabled in :mod:`trackml_reco.plotting`, a lightweight
        R-Z overlay is emitted for quick inspection.
        """
        sg = seed_group.sort_values("seed_point_index", kind="mergesort")
        # contiguous, float64 seed points (3,3)
        seed_points = sg[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)[:3]

        future_layers: List[Tuple[int, int]] = list(sg.iloc[0]["future_layers"])
        # Time grid for EKF: 3 seed points + L future layers
        t = np.linspace(0.0, 1.0, len(future_layers) + 3, dtype=np.float64)

        # Run brancher (plot_tree kept False for speed; plotting is handled elsewhere)
        branches, _ = self.brancher.run(seed_points, future_layers, t, plot_tree=False)

        pid = int(sg.iloc[0]["particle_id"])
        if not branches:
            logger.debug("No branches returned for PID=%s", pid)
            return

        self._convert_branches_to_tracks(branches, pid, sg.iloc[0])

        before = len(self.get_track_candidates_by_particle(pid))
        self._prune_tracks(max_branches)
        after = len(self.get_track_candidates_by_particle(pid))
        logger.debug("PID=%s: tracks before prune=%d, after=%d", pid, before, after)

        # Optional quick visual (keeps logic isolated and cheap when disabled)
        if PLOT_BRANCHES:
            trk_plot.plot_branches(
                branches=self.get_track_candidates_by_particle(pid),
                seed_points=seed_points,
                future_layers=future_layers,
                hits_df=self.hit_pool.hits,
                truth_hits=self.hit_pool.pt_cut_hits,
                particle_id=pid,
            )

    def _convert_branches_to_tracks(
        self,
        branches: List[Dict],
        particle_id: int,
        seed_row: pd.Series,
    ) -> None:
        r"""
        Convert branch dictionaries to :class:`TrackCandidate` objects and register them.

        Parameters
        ----------
        branches : list[dict]
            Each branch is expected to provide keys: ``"traj"`` (list/array of 3D
            points), ``"hit_ids"``, ``"state"`` (7,), ``"cov"`` (7,7), and ``"score"``.
        particle_id : int
            Particle id (copied to each candidate).
        seed_row : pandas.Series
            Any of the seed rows for this particle (cached for debug/metadata).

        Side Effects
        ------------
        - Reserves the branch hits in the :class:`~trackml_reco.hit_pool.HitPool`.
        - Appends completed candidates (length ≥ number of future layers) to
        :attr:`completed_tracks`.
        - Updates :attr:`track_candidates_by_particle` and :attr:`next_track_id`.

        Warnings
        --------
        If some hits were already reserved, only the unreserved ones are claimed;
        a debug log entry records the discrepancy.
        """
        # Precompute completion threshold once
        complete_len = len(seed_row["future_layers"])

        for br in branches:
            track_id = self.next_track_id
            self.next_track_id += 1

            traj = br.get("traj", [])
            hit_ids = list(map(int, br.get("hit_ids", [])))
            state = np.asarray(br["state"], dtype=np.float64, order="C")
            cov = np.asarray(br["cov"], dtype=np.float64, order="C")
            score = float(br["score"])

            status = (
                TrackStatus.COMPLETED
                if len(traj) >= complete_len
                else TrackStatus.ACTIVE
            )

            tc = TrackCandidate(
                id=track_id,
                particle_id=int(particle_id),
                trajectory=traj,
                hit_ids=hit_ids,
                state=state,
                covariance=cov,
                score=score,
                status=status,
                seed_row=seed_row,
            )

            # Assign hits (bulk). If some are already assigned, just log it.
            assigned = self.hit_pool.assign_hits(tc.hit_ids)
            if assigned != len(tc.hit_ids):
                logger.debug(
                    "Track %d: only %d/%d hits assigned (some already reserved)",
                    track_id, assigned, len(tc.hit_ids)
                )

            # Register
            d = self.track_candidates_by_particle.setdefault(int(particle_id), {})
            d[track_id] = tc
            if status is TrackStatus.COMPLETED:
                self.completed_tracks.append(tc)

    def _prune_tracks(self, max_branches: int) -> None:
        r"""
        Keep only the top-``k`` (lowest score) candidates per particle.

        Parameters
        ----------
        max_branches : int
            Per-particle retention budget. If ``<=0``, **all** candidates are dropped.

        Notes
        -----
        For a particle with :math:`n` candidates and scores
        :math:`S_1,\ldots,S_n`, the retained set is the indices of the
        :math:`k=\text{max\_branches}` smallest :math:`S_i`. All dropped candidates
        have their hits released back to the pool via :meth:`_remove_track`.

        Complexity
        ----------
        Sorting per particle is :math:`\mathcal{O}(n\log n)`; typical ``n`` is small.
        """
        if max_branches <= 0:
            # Extreme case: drop everything
            for pid, cand_map in list(self.track_candidates_by_particle.items()):
                to_remove_ids = list(cand_map.keys())
                for tid in to_remove_ids:
                    self._remove_track(pid, tid)
            return

        for pid, cand_map in list(self.track_candidates_by_particle.items()):
            n = len(cand_map)
            if n <= max_branches:
                continue

            tracks = list(cand_map.values())
            tracks.sort(key=lambda t: t.score)
            keep = {t.id for t in tracks[:max_branches]}
            drop = [t for t in tracks if t.id not in keep]

            logger.debug("PID=%s: pruning %d → keep %d, drop %d", pid, n, len(keep), len(drop))

            for t in drop:
                self._remove_track(pid, t.id)

    def _remove_track(self, particle_id: int, track_id: int) -> None:
        r"""
        Remove a candidate: release its hits, drop maps, and update completed list.

        Parameters
        ----------
        particle_id : int
            Particle id owning the candidate.
        track_id : int
            Candidate identifier to remove.

        Notes
        -----
        Hit ids referenced by the candidate are released back to the
        :class:`~trackml_reco.hit_pool.HitPool`. If the candidate was in
        :attr:`completed_tracks`, it is removed from that list as well.
        """
        cand_map = self.track_candidates_by_particle.get(int(particle_id))
        if not cand_map:
            return
        tc = cand_map.pop(track_id, None)
        if tc is None:
            return

        # Release hits back to pool
        released = self.hit_pool.release_hits(tc.hit_ids)
        if released != len(tc.hit_ids):
            logger.debug(
                "Track %d: only %d/%d hits released (some may be shared/duplicate)",
                track_id, released, len(tc.hit_ids)
            )

        # Remove from completed list if present
        if tc in self.completed_tracks:
            self.completed_tracks.remove(tc)

        # If particle has no more candidates, drop the entry
        if not cand_map:
            del self.track_candidates_by_particle[int(particle_id)]

    def get_best_tracks(self, n: int | None = None) -> List[TrackCandidate]:
        r"""
        Return the best (lowest-score) completed tracks.

        Parameters
        ----------
        n : int or None, optional
            If ``None``, return **all** completed tracks sorted by score ascending.
            Otherwise return the first ``n``.

        Returns
        -------
        list[TrackCandidate]
            Sorted subset of :attr:`completed_tracks`.
        """
        tracks = sorted(self.completed_tracks, key=lambda t: t.score)
        return tracks if n is None else tracks[: int(n)]

    def get_track_statistics(self) -> Dict:
        r"""
        Aggregate high-level statistics about the current builder state.

        Returns
        -------
        dict
            Keys include:
            ``total_tracks_created``, ``completed_tracks``, ``active_tracks``,
            ``seeds_built``, ``unique_particles``, ``assigned_hits``,
            ``available_hits``, ``total_hits``, ``assignment_ratio``.

        Notes
        -----
        The assignment ratio is

        .. math::

        \text{ratio} \;=\; \frac{\#\text{assigned hits}}{\#\text{total hits}}
        \;\in\; [0,1].
        """
        active_tracks = sum(
            1 for pm in self.track_candidates_by_particle.values() for t in pm.values()
            if t.status is TrackStatus.ACTIVE
        )
        seeds_built = len(self.seeds_df)
        uniq_p = int(self.seeds_df["particle_id"].nunique()) if seeds_built else 0
        return {
            "total_tracks_created": self.next_track_id,
            "completed_tracks": len(self.completed_tracks),
            "active_tracks": active_tracks,
            "seeds_built": seeds_built,
            "unique_particles": uniq_p,
            "assigned_hits": len(self.hit_pool._assigned_hits),
            "available_hits": self.hit_pool.get_available_hit_count(),
            "total_hits": len(self.hit_pool.hits),
            "assignment_ratio": self.hit_pool.get_assignment_ratio(),
        }

    def get_seeds_dataframe(self) -> pd.DataFrame:
        r"""
        Return a defensive copy of the cached seeds DataFrame.

        Returns
        -------
        pandas.DataFrame
            See :meth:`build_seeds_from_truth` for column semantics.
        """
        return self.seeds_df.copy()

    def get_tracks_by_seed(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Get all **completed** tracks corresponding to a given ``particle_id``.

        Parameters
        ----------
        particle_id : int
            The particle identifier used to group seeds.

        Returns
        -------
        list[TrackCandidate]
            Completed tracks whose ``particle_id`` matches the argument.
        """
        pid = int(particle_id)
        return [t for t in self.completed_tracks if t.particle_id == pid]

    def get_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Get **all** candidates (active and completed) for a particle.

        Parameters
        ----------
        particle_id : int
            The particle identifier.

        Returns
        -------
        list[TrackCandidate]
            Values of :attr:`track_candidates_by_particle[particle_id]`
            (empty list if none).
        """
        return list(self.track_candidates_by_particle.get(int(particle_id), {}).values())

    def get_active_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Filter active (non-completed) candidates for a particle.

        Parameters
        ----------
        particle_id : int
            The particle identifier.

        Returns
        -------
        list[TrackCandidate]
            Candidates with ``status == TrackStatus.ACTIVE``.
        """
        return [t for t in self.get_track_candidates_by_particle(particle_id) if t.status is TrackStatus.ACTIVE]

    def get_completed_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Filter completed candidates for a particle.

        Parameters
        ----------
        particle_id : int
            The particle identifier.

        Returns
        -------
        list[TrackCandidate]
            Candidates with ``status == TrackStatus.COMPLETED``.
        """
        return [t for t in self.get_track_candidates_by_particle(particle_id) if t.status is TrackStatus.COMPLETED]

    def get_all_particle_ids(self) -> List[int]:
        r"""
        Return the list of particle ids that currently have candidates.

        Returns
        -------
        list[int]
            Keys of :attr:`track_candidates_by_particle`.
        """
        return list(self.track_candidates_by_particle.keys())

    def get_track_by_id(self, track_id: int) -> Optional[TrackCandidate]:
        r"""
        Look up a candidate by its global ``track_id``.

        Parameters
        ----------
        track_id : int
            Candidate identifier assigned by the builder.

        Returns
        -------
        TrackCandidate or None
            The matching candidate if present; otherwise ``None``.
        """
        tid = int(track_id)
        for cand_map in self.track_candidates_by_particle.values():
            if tid in cand_map:
                return cand_map[tid]
        return None

    def get_particle_statistics(self, particle_id: int) -> Dict:
        r"""
        Per-particle summary of candidate counts, lengths, and best score.

        Parameters
        ----------
        particle_id : int
            The particle identifier.

        Returns
        -------
        dict
            Fields:
            ``particle_id``, ``total_candidates``, ``active_candidates``,
            ``completed_candidates``, ``best_score``, ``avg_length``,
            ``min_length``, ``max_length``.

        Notes
        -----
        - ``best_score`` is :math:`\min_i S_i` over all candidates for the particle.
        - Lengths refer to the number of points in each candidate's ``trajectory``.
        """
        cands = self.get_track_candidates_by_particle(particle_id)
        if not cands:
            return {
                "particle_id": int(particle_id),
                "total_candidates": 0,
                "active_candidates": 0,
                "completed_candidates": 0,
                "best_score": None,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
            }
        active = sum(1 for t in cands if t.status is TrackStatus.ACTIVE)
        completed = sum(1 for t in cands if t.status is TrackStatus.COMPLETED)
        scores = [t.score for t in cands]
        lengths = [len(t.trajectory) for t in cands]
        return {
            "particle_id": int(particle_id),
            "total_candidates": len(cands),
            "active_candidates": active,
            "completed_candidates": completed,
            "best_score": float(min(scores)) if scores else None,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "min_length": int(min(lengths)) if lengths else 0,
            "max_length": int(max(lengths)) if lengths else 0,
        }

    def get_pruning_statistics(self) -> Dict:
        r"""
        Summarize the result of pruning across all particles.

        Returns
        -------
        dict
            Top-level keys:
            ``particles_with_tracks``, ``total_tracks_after_pruning``,
            and ``per_particle_stats`` (a nested dict with
            ``tracks_kept``, ``best_score``, ``avg_length`` per particle).

        See Also
        --------
        _prune_tracks
        """
        stats: Dict[int, Dict[str, float | int | None]] = {}
        total_after = 0
        for pid in self.get_all_particle_ids():
            cands = self.get_track_candidates_by_particle(pid)
            total_after += len(cands)
            stats[int(pid)] = {
                "tracks_kept": len(cands),
                "best_score": (float(min(t.score for t in cands)) if cands else None),
                "avg_length": (float(np.mean([len(t.trajectory) for t in cands])) if cands else 0.0),
            }
        return {
            "particles_with_tracks": len(stats),
            "total_tracks_after_pruning": total_after,
            "per_particle_stats": stats,
        }

    def reset(self) -> None:
        r"""
        Reset builder state and release all reserved hits.

        Side Effects
        ------------
        - Clears candidate maps and :attr:`completed_tracks`.
        - Empties :attr:`seeds_df` and resets :attr:`next_track_id` to zero.
        - Calls :meth:`trackml_reco.hit_pool.HitPool.reset` to clear assignment state.
        """
        self.track_candidates_by_particle.clear()
        self.completed_tracks.clear()
        self.seeds_df = pd.DataFrame()
        self.next_track_id = 0
        self.hit_pool.reset()
