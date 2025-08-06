import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Type

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from trackml_reco.branchers.brancher import Brancher
from trackml_reco.track_builder import TrackBuilder, TrackCandidate, TrackStatus
import trackml_reco.hit_pool as trk_hit_pool

logger = logging.getLogger(__name__)

class CollaborativeParallelTrackBuilder(TrackBuilder):
    """
    Parallel, collaborative builder that:
      - delegates full EKF propagation + gating to Brancher.run()
      - shares hit-level best χ² in a thread-safe map
      - prunes any branch that tries to use a hit already claimed with a lower χ²
      - retains all nodes in the graph for post-mortem
      - plots all reconstructed tracks in R vs Z with layer boundaries
    """
    def __init__(self,
                 hit_pool: trk_hit_pool.HitPool,
                 brancher_cls: Type[Brancher],
                 brancher_config: Dict = None,
                 max_workers: int = 8):
        """
        Initialize the collaborative parallel track builder.

        Parameters
        ----------
        hit_pool : HitPool
            Pool of hits to build tracks from.
        brancher_cls : Type[Brancher]
            Brancher class used for EKF propagation.
        brancher_config : dict, optional
            Configuration parameters for the brancher.
        max_workers : int, optional
            Number of threads for parallel processing. Default is 8.
        """
        super().__init__(hit_pool, brancher_cls, brancher_config)
        self.max_workers = max_workers
        self.best_hit_map: Dict[int, float] = {}
        self.map_lock = threading.Lock()
        self.graph = nx.DiGraph()

    def build_tracks_from_truth(self,
                                max_seeds: int = None,
                                max_tracks_per_seed: int = 30,
                                max_branches: int = 12,
                                jitter_sigma: float = 1e-3) -> List[TrackCandidate]:
        """
        Build and prune tracks from truth seeds in parallel, then plot R-Z overlay.

        Parameters
        ----------
        max_seeds : int or None
            Maximum number of unique particles (seeds) to process. If None, use all.
        max_tracks_per_seed : int
            Maximum track candidates to attempt per seed (unused in this implementation).
        max_branches : int
            Maximum number of branches to retain per seed (pruning threshold).
        jitter_sigma : float, optional
            Standard deviation for seed jitter. Default is 1e-3.

        Returns
        -------
        completed_tracks : list of TrackCandidate
            List of successfully built and pruned track candidates.
        """
        # build seeds
        seeds_df = self.build_seeds_from_truth(max_seeds, jitter_sigma)

        # parallel dispatch of each seed‐group
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = []
            for pid, group in seeds_df.groupby('particle_id'):
                if len(group) < 3:
                    continue
                futures.append(exe.submit(self._process_seed_group_parallel,
                                         pid, group, max_tracks_per_seed, max_branches))
            for f in as_completed(futures):
                f.result()

        # after all seeds done, plot RZ overlay
        self._plot_tracks_rz()
        return self.completed_tracks

    def _process_seed_group_parallel(self, pid: int, seed_group: pd.DataFrame,
                                     max_tracks: int, max_branches: int) -> None:
        """
        Process one seed group: run EKF brancher, collect and prune branches.

        Parameters
        ----------
        pid : int
            Particle ID for this seed group.
        seed_group : pandas.DataFrame
            DataFrame containing the three seed points and future layers info.
        max_tracks : int
            Maximum track candidates per seed (unused here).
        max_branches : int
            Maximum number of branches to keep per particle after pruning.

        Returns
        -------
        None
        """
        # instantiate brancher for this seed
        brancher = self.brancher_cls(
            trees=self.hit_pool.trees,
            **self.brancher_config
        )

        # sort & extract seed points + times
        sg = seed_group.sort_values('seed_point_index')
        seed_xyz = sg[['x','y','z']].values[:3]
        future_layers = sg.iloc[0]['future_layers']
        t = np.linspace(0,1,len(future_layers)+3)

        # run full EKF tree
        branches, tree = brancher.run(seed_xyz, future_layers, t, plot_tree=False)

        # add all nodes into our master graph
        for u, v in tree.edges():
            pos_u = tree.nodes[u]['pos']
            pos_v = tree.nodes[v]['pos']
            self.graph.add_node(u, pos=tuple(pos_u))
            self.graph.add_node(v, pos=tuple(pos_v))
            self.graph.add_edge(u, v)

        # now convert each branch into a TrackCandidate only if none of its hits
        # have been claimed by a strictly better score so far
        kept = []
        for br in branches:
            hits = br.get('hit_ids', [])
            br_score = br['score']

            conflict = False
            with self.map_lock:
                for hid in hits:
                    best = self.best_hit_map.get(hid, np.inf)
                    if br_score >= best:
                        conflict = True
                        break
                if not conflict:
                    for hid in hits:
                        self.best_hit_map[hid] = br_score

            if not conflict:
                kept.append(br)

        # convert kept branches to TrackCandidate and assign
        for br in kept:
            track = TrackCandidate(
                id=self.next_track_id,
                particle_id=pid,
                trajectory=br['traj'],
                hit_ids=br.get('hit_ids', []),
                state=br['state'],
                covariance=br['cov'],
                score=br['score'],
                status=TrackStatus.COMPLETED if len(br.get('hit_ids', []))==len(future_layers) else TrackStatus.ACTIVE
            )
            self.next_track_id += 1

            self.hit_pool.assign_hits(track.hit_ids)
            self.completed_tracks.append(track)
            self.track_candidates_by_particle.setdefault(pid, {})[track.id] = track

        # prune per-particle top scoring
        self._prune_tracks(max_branches)

    def _plot_tracks_rz(self):
        """
        Overlay all completed tracks on R vs Z layer boundaries.

        Draws each detector layer as a rectangle in (z, r) space, then plots
        each reconstructed track’s (z, r) trajectory on top.

        Returns
        -------
        None
        """
        hits_df = self.hit_pool.hits
        layer_tuples = sorted(set(zip(hits_df.volume_id, hits_df.layer_id)), key=lambda x: (x[0], x[1]))
        z_all = hits_df['z']
        r_all = np.sqrt(hits_df['x']**2 + hits_df['y']**2)
        zmin, zmax = z_all.min(), z_all.max()
        rmin, rmax = r_all.min(), r_all.max()
        pad_z = (zmax - zmin) * 0.05
        pad_r = (rmax - rmin) * 0.05

        fig, ax = plt.subplots(figsize=(10, 8))
        # draw layer rectangles
        for vol, lay in layer_tuples:
            df = hits_df[(hits_df.volume_id == vol) & (hits_df.layer_id == lay)]
            if df.empty: continue
            z_vals = df['z']
            r_vals = np.sqrt(df.x**2 + df.y**2)
            rect = patches.Rectangle((z_vals.min(), r_vals.min()),
                                     z_vals.max() - z_vals.min(),
                                     r_vals.max() - r_vals.min(),
                                     linewidth=1.2, edgecolor='black', facecolor='none', alpha=0.5)
            ax.add_patch(rect)
        # plot track trajectories
        for tc in self.completed_tracks:
            traj = np.array(tc.trajectory)
            z_traj = traj[:,2]
            r_traj = np.sqrt(traj[:,0]**2 + traj[:,1]**2)
            ax.plot(z_traj, r_traj, '-o', markersize=3, alpha=0.7)
        ax.set_xlim(zmin - pad_z, zmax + pad_z)
        ax.set_ylim(rmin - pad_r, rmax + pad_r)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('r (mm)')
        ax.set_title('Reconstructed Tracks on Layer Boundaries (r vs z)')
        ax.grid(True)
        plt.show()