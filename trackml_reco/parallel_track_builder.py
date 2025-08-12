import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Type, Tuple, Optional, Any, Sequence

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
    Thread-safe best-score table for per-hit ownership in collaborative track building.

    This data structure keeps, for each hit ID :math:`h`, the best score
    :math:`s_h` obtained so far and the ID of the track candidate that owns it.

    Notes
    -----
    A *score* in this context is typically a trajectory cost or residual
    distance, where **lower is better**. When a new branch attempts to claim
    hits, it must improve the score by at least the `margin` value.

    Internally uses a `threading.Lock` to ensure atomic updates across
    multiple threads.
    """
    def __init__(self) -> None:
        self._best: Dict[int, Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def snapshot_deny_hits(self, hits: Sequence[int]) -> List[int]:
        r"""
        Get the subset of hits that are already claimed by some track.

        Parameters
        ----------
        hits : sequence of int
            Candidate hit IDs to check.

        Returns
        -------
        list of int
            The subset of `hits` that are already present in the best-score table.

        Notes
        -----
        This method is lock-free and safe because it only reads keys from
        the dictionary without mutating state.
        """
        if not hits:
            return []
        # local ref (no lock needed for read-only pattern of dict keys)
        best = self._best
        return [hid for hid in hits if hid in best]

    def try_claim_branch(self,
                         hit_ids: Sequence[int],
                         branch_score: float,
                         margin: float,
                         owner_track_id: int) -> bool:
        r"""
        Attempt to claim all given hits for a branch if it improves the
        best score by at least a given margin.

        A claim succeeds only if:

        .. math::

            \forall h \in \text{hit\_ids}:\quad
            s_\mathrm{branch} < s_h - \text{margin}

        where:
          * :math:`s_\mathrm{branch}` is the score of the current branch.
          * :math:`s_h` is the best score recorded for hit :math:`h`.

        If any hit fails the condition, **no hits are claimed**.

        Parameters
        ----------
        hit_ids : sequence of int
            Hit IDs to claim.
        branch_score : float
            The branch's score (lower is better).
        margin : float
            Minimum required improvement over the best score for all hits.
        owner_track_id : int
            The track ID that will own these hits if the claim succeeds.

        Returns
        -------
        bool
            ``True`` if all hits were claimed successfully, ``False`` otherwise.
        """
        if not hit_ids:
            return True
        with self._lock:
            # check
            for hid in hit_ids:
                best = self._best.get(hid, (np.inf, -1))[0]
                if branch_score >= best - margin:
                    return False
            # commit
            for hid in hit_ids:
                self._best[hid] = (branch_score, owner_track_id)
            return True

class CollaborativeParallelTrackBuilder(TrackBuilder):
    r"""
    Parallel, collaborative track builder using shared hit-claiming logic.

    This builder:

    * Runs branchers (`Brancher.run`) in parallel threads using seeds from truth.
    * Claims hits **atomically** across threads based on a score + margin rule.
    * Passes per-seed deny-lists to branchers to avoid conflicts.
    * Merges per-thread brancher debug graphs into one.
    * Optionally enforces a per-seed **time budget** to avoid stragglers.
    * Can plot reconstructed tracks in :math:`(r, z)` space.

    Parameters
    ----------
    hit_pool : HitPool
        Object containing all hits, KD-trees, and assigned hit bookkeeping.
    brancher_cls : Type[Brancher]
        Brancher class implementing the propagation and gating strategy.
    brancher_config : dict, optional
        Configuration parameters for the brancher.
    max_workers : int, optional
        Number of parallel threads to use. Default is 8.
    claim_margin : float, optional
        Minimum required improvement in score to claim a hit (see
        :meth:`SharedHitBook.try_claim_branch`). Default is 1.0.
    per_seed_time_budget_s : float or None, optional
        If set, maximum allowed seconds for processing one seed group.
    """

    def __init__(self,
                 hit_pool: trk_hit_pool.HitPool,
                 brancher_cls: Type[Brancher],
                 brancher_config: Dict = None,
                 max_workers: int = 8,
                 claim_margin: float = 1.0,
                 per_seed_time_budget_s: Optional[float] = None) -> None:
        super().__init__(hit_pool, brancher_cls, brancher_config)
        self.max_workers = max_workers
        self.claim_margin = float(claim_margin)
        self.per_seed_time_budget_s = per_seed_time_budget_s

        self.graph = nx.DiGraph()
        self._measured_traj: Dict[int, np.ndarray] = {}

        # shared “score book” (hit_id -> best score, owner)
        self._book = SharedHitBook()

        # cache: (volume_id, layer_id) -> np.ndarray(hit_ids) for fast deny snapshots
        hits = self.hit_pool.hits
        hits = hits.assign(layer_key=list(zip(hits.volume_id, hits.layer_id)))
        self._layer_to_hit_ids: Dict[Tuple[int, int], np.ndarray] = {
            key: grp['hit_id'].astype(int).to_numpy()
            for key, grp in hits.groupby('layer_key')
        }

    def build_tracks_from_truth(self,
                                max_seeds: Optional[int] = None,
                                max_tracks_per_seed: int = 30,
                                max_branches: int = 12,
                                jitter_sigma: float = 1e-3) -> List[TrackCandidate]:
        r"""
        Build tracks from truth seeds in parallel, enforcing collaborative hit-claiming.

        Each truth seed group is processed by
        :meth:`_process_seed_group_parallel` in a separate thread.

        Parameters
        ----------
        max_seeds : int or None
            If given, limit the number of seeds used.
        max_tracks_per_seed : int
            Maximum number of tracks brancher may generate per seed.
        max_branches : int
            Maximum number of branches to keep per particle after claiming.
        jitter_sigma : float
            Gaussian noise sigma (in coordinate units) applied to seeds.

        Returns
        -------
        list of TrackCandidate
            The list of completed tracks after parallel building.
        """
        seeds_df = self.build_seeds_from_truth(max_seeds, jitter_sigma)

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = []
            for pid, group in seeds_df.groupby('particle_id'):
                if len(group) < 3:
                    continue
                futures.append(exe.submit(self._process_seed_group_parallel,
                                          pid, group, max_tracks_per_seed, max_branches))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.exception("Seed task failed: %s", e)

        self._plot_tracks_rz()
        return self.completed_tracks

    def _process_seed_group_parallel(self, pid: int, seed_group: pd.DataFrame,
                                     max_tracks: int, max_branches: int) -> None:
        r"""
        Process a single particle's seed group in parallel context.

        Steps
        -----
        1. Sort seed points by `seed_point_index`.
        2. Build per-seed deny-list from claimed and assigned hits.
        3. Run brancher to generate track branches.
        4. Claim hits for each branch in ascending score order.
        5. Commit successful branches to the completed track list.
        6. Prune tracks to keep at most `max_branches` per particle.

        Parameters
        ----------
        pid : int
            Particle ID for this seed group.
        seed_group : pd.DataFrame
            DataFrame of seed hits for the particle.
        max_tracks : int
            Maximum tracks per seed to request from brancher.
        max_branches : int
            Maximum branches to commit for this particle.
        """
        start_time = time.time()
        brancher = self.brancher_cls(trees=self.hit_pool.trees, **self.brancher_config)

        sg = seed_group.sort_values('seed_point_index')
        seed_xyz = sg[['x', 'y', 'z']].values[:3]
        future_layers = sg.iloc[0]['future_layers']
        t = np.linspace(0, 1, len(future_layers) + 3)

        # Per-seed deny list:
        #  - hits on the layers we plan to traverse, that are already claimed by other threads
        #  - PLUS hits already assigned in the pool (from prior completed tracks)
        candidate_layer_hits = self._hits_on_layers_fast(future_layers)
        deny_claimed = set(self._book.snapshot_deny_hits(candidate_layer_hits))
        deny_assigned = set(int(h) for h in self.hit_pool._assigned_hits)
        deny_hits = list(deny_claimed.union(deny_assigned))

        # Run brancher; pass deny_hits if supported
        try:
            branches, tree = brancher.run(seed_xyz, future_layers, t, plot_tree=False, deny_hits=deny_hits)
        except TypeError:
            branches, tree = brancher.run(seed_xyz, future_layers, t, plot_tree=False)

        self._merge_brancher_graph(tree)
        if not branches:
            return

        # Sort by total score ascending (best first).
        branches.sort(key=lambda b: float(b['score']))

        committed = 0
        for br in branches:
            # soft time budget
            if self.per_seed_time_budget_s is not None and (time.time() - start_time) > self.per_seed_time_budget_s:
                logger.info(f"[pid={pid}] time budget exceeded; committed={committed}")
                break

            hits = list(map(int, br.get('hit_ids', [])))
            score = float(br['score'])
            if not hits:
                continue

            # optimistic track id for claim ownership
            tentative_id = self.next_track_id

            if not self._book.try_claim_branch(hits, score, self.claim_margin, tentative_id):
                continue  # lost the claim

            # Claim succeeded → commit track
            track = TrackCandidate(
                id=tentative_id,
                particle_id=pid,
                trajectory=br['traj'],
                hit_ids=hits,
                state=br['state'],
                covariance=br['cov'],
                score=score,
                status=TrackStatus.COMPLETED if len(hits) == len(future_layers) else TrackStatus.ACTIVE
            )
            self.next_track_id += 1

            self.hit_pool.assign_hits(track.hit_ids)
            self.completed_tracks.append(track)
            self.track_candidates_by_particle.setdefault(pid, {})[track.id] = track
            self._measured_traj[track.id] = self._hit_ids_to_xyz(track.hit_ids)
            committed += 1

            if committed >= max_branches:
                break

        # local per-particle prune (keeps at most max_branches per particle)
        self._prune_tracks(max_branches)

    def _hits_on_layers_fast(self, layers: Sequence[Tuple[int,int]]) -> List[int]:
        r"""
        Get all hit IDs that lie on the specified layers.

        Parameters
        ----------
        layers : sequence of (volume_id, layer_id)
            Detector layer tuples.

        Returns
        -------
        list of int
            Hit IDs present on those layers.
        """
        if not layers:
            return []
        out = []
        get = self._layer_to_hit_ids.get
        for key in layers:
            arr = get(key)
            if arr is not None and arr.size:
                out.append(arr)
        if not out:
            return []
        return np.concatenate(out).tolist()

    def _merge_brancher_graph(self, G: nx.DiGraph) -> None:
        r"""
        Merge another brancher's debug graph into the global builder graph.

        Nodes preserve their `(x, y, z)` position attributes if available.
        """
        for u, v, data in G.edges(data=True):
            pu = self._extract_pos(G, u)
            pv = self._extract_pos(G, v)
            if pu is not None:
                self.graph.add_node(u, pos=tuple(pu))
            else:
                self.graph.add_node(u)
            if pv is not None:
                self.graph.add_node(v, pos=tuple(pv))
            else:
                self.graph.add_node(v)
            self.graph.add_edge(u, v, **data)

    def _extract_pos(self, G: nx.DiGraph, node: Any) -> Optional[Tuple[float, float, float]]:
        r"""
        Extract a 3D position tuple from a node, if available.

        Supports:
          * Node attributes `'pos'` with a (3,) array.
          * Tuple-form nodes `(id, (x, y, z))`.
          * Generic sequences of length 3.

        Returns
        -------
        tuple of float or None
            Extracted `(x, y, z)` or ``None`` if not found.
        """
        nd = G.nodes.get(node, {})
        if 'pos' in nd:
            pos = np.asarray(nd['pos'], dtype=float)
            if pos.shape == (3,):
                return float(pos[0]), float(pos[1]), float(pos[2])

        if isinstance(node, tuple) and len(node) >= 2:
            arr = np.asarray(node[1], dtype=float)
            if arr.shape == (3,):
                return float(arr[0]), float(arr[1]), float(arr[2])

        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            arr = np.asarray(node, dtype=float)
            if arr.shape == (3,):
                return float(arr[0]), float(arr[1]), float(arr[2])
        return None

    def _hit_ids_to_xyz(self, hit_ids: List[int]) -> np.ndarray:
        r"""
        Convert a list of hit IDs to an ordered `(x, y, z)` coordinate array.

        Order is preserved according to the given hit ID sequence.
        """
        if not hit_ids:
            return np.empty((0, 3), dtype=float)
        df = self.hit_pool.hits
        sub = df[df.hit_id.isin(hit_ids)][['hit_id', 'x', 'y', 'z']].copy()
        order = {hid: i for i, hid in enumerate(hit_ids)}
        sub['order'] = sub.hit_id.map(order)
        sub = sub.sort_values('order')
        return sub[['x', 'y', 'z']].to_numpy()

    def _plot_tracks_rz(self, overlay_predicted: bool = False, show_truth: bool = False) -> None:
        r"""
        Plot all reconstructed tracks in :math:`(r, z)` coordinates.

        Parameters
        ----------
        overlay_predicted : bool
            If True, overlay brancher-predicted trajectories.
        show_truth : bool
            If True, overlay truth particle trajectories.
        """
        hits_df = self.hit_pool.hits
        layer_tuples = sorted(
            set(zip(hits_df.volume_id, hits_df.layer_id)),
            key=lambda x: (x[0], x[1])
        )

        z_all = hits_df['z'].to_numpy()
        r_all = np.sqrt(hits_df['x'].to_numpy()**2 + hits_df['y'].to_numpy()**2)
        zmin, zmax = float(z_all.min()), float(z_all.max())
        rmin, rmax = float(r_all.min()), float(r_all.max())
        pad_z = 0.05 * (zmax - zmin)
        pad_r = 0.05 * (rmax - rmin)

        fig, ax = plt.subplots(figsize=(11, 8))

        # layer rectangles
        for vol, lay in layer_tuples:
            df = hits_df[(hits_df.volume_id == vol) & (hits_df.layer_id == lay)]
            if df.empty:
                continue
            z_vals = df['z'].to_numpy()
            r_vals = np.sqrt(df['x'].to_numpy()**2 + df['y'].to_numpy()**2)
            rect = patches.Rectangle(
                (float(z_vals.min()), float(r_vals.min())),
                float(z_vals.max() - z_vals.min()),
                float(r_vals.max() - r_vals.min()),
                linewidth=1.2, edgecolor='black', facecolor='none', alpha=0.45
            )
            ax.add_patch(rect)

        # legend handles
        meas_handle, = ax.plot([], [], '-o',  linewidth=1.2, label='measured hits')
        pred_with_hits_handle, = ax.plot([], [], ':',   linewidth=0.9, label='filtered (with hits)')
        pred_no_hits_handle, = ax.plot([], [], '--o', linewidth=1.0, label='filtered (no hits)')
        truth_handle = None

        if show_truth and hasattr(self.hit_pool, 'pt_cut_hits') and self.hit_pool.pt_cut_hits is not None:
            truth_df = self.hit_pool.pt_cut_hits
            first = True
            for _, grp in truth_df.groupby('particle_id'):
                zt = grp['z'].to_numpy()
                rt = np.sqrt(grp['x'].to_numpy()**2 + grp['y'].to_numpy()**2)
                if first:
                    truth_handle, = ax.plot(zt, rt, '--k', linewidth=1.0, alpha=0.6, label='truth')
                    first = False
                else:
                    ax.plot(zt, rt, '--k', linewidth=1.0, alpha=0.6)

        # plot tracks
        for tc in self.completed_tracks:
            meas = self._measured_traj.get(tc.id, np.empty((0, 3)))
            if meas.size:
                z = meas[:, 2]
                r = np.sqrt(meas[:, 0]**2 + meas[:, 1]**2)
                ax.plot(z, r, '-o', markersize=3, alpha=0.9, linewidth=1.2)
                if overlay_predicted and len(tc.trajectory) > 0:
                    traj = np.array(tc.trajectory)
                    zt = traj[:, 2]
                    rt = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
                    ax.plot(zt, rt, ':', alpha=0.6, linewidth=0.9)
            else:
                if len(tc.trajectory) > 0:
                    traj = np.array(tc.trajectory)
                    z = traj[:, 2]
                    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
                    ax.plot(z, r, '--o', markersize=3, alpha=0.75, linewidth=1.0)

        ax.set_xlim(zmin - pad_z, zmax + pad_z)
        ax.set_ylim(rmin - pad_r, rmax + pad_r)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('r (mm)')
        ax.set_title('Tracks over detector layers — r vs z')
        handles = [meas_handle, pred_with_hits_handle, pred_no_hits_handle]
        if truth_handle is not None:
            handles.append(truth_handle)
        ax.legend(handles=handles, loc='best', framealpha=0.85)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

