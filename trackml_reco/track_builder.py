import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

import trackml_reco.metrics as trk_metrics
import trackml_reco.hit_pool as trk_hit_pool
import trackml_reco.utils as trk_utils
import trackml_reco.plotting as trk_plot
from trackml_reco.branchers.brancher import Brancher

logger = logging.getLogger(__name__)

PLOT_BRANCHES = True

class TrackStatus(Enum):
    r"""
    Enumeration for the lifecycle state of a track candidate.

    Members
    -------
    ACTIVE
        Track is still being extended (not all planned layers have been visited).
    COMPLETED
        Track reached its intended terminal layer set and is considered finished.
    DEAD
        Track can no longer be extended (e.g., ran out of valid hits).
    """
    ACTIVE = "active"      # Still being built
    COMPLETED = "completed"  # Reached final layer
    DEAD = "dead"          # No valid hits found


@dataclass
class TrackCandidate:
    r"""
    Container for a single track hypothesis during building.

    Attributes
    ----------
    id : int
        Global identifier for this track hypothesis.
    particle_id : int
        Ground-truth particle identifier associated to this track (if known).
    trajectory : list of ndarray
        Ordered 3D positions :math:`[\mathbf{x}_0,\ldots,\mathbf{x}_k]` used for this track
        (measured hit positions in most branchers).
    hit_ids : list of int
        Hit identifiers consumed by this track in the same order as ``trajectory`` (after seeds).
    state : ndarray, shape (7,)
        Current EKF state vector.
    covariance : ndarray, shape (7, 7)
        Current EKF state covariance.
    score : float
        Accumulated data-term, typically a sum of Mahalanobis distances:

        .. math::

            \text{score} \;\equiv\; \sum_{i} \left(\mathbf{z}_i - H \hat{\mathbf{x}}_i\right)^\top
            S_i^{-1} \left(\mathbf{z}_i - H \hat{\mathbf{x}}_i\right),

        where :math:`S_i = H P_i H^\top + R`.
    status : TrackStatus
        Current status flag (``ACTIVE``, ``COMPLETED``, or ``DEAD``).
    parent_id : int, optional
        Parent track ID if this candidate was created by branching.
    seed_row : pandas.Series, optional
        Reference to the seed row used to start this track (for debugging/plotting).
    """
    id: int
    particle_id: int
    trajectory: List[np.ndarray]  # List of 3D positions
    hit_ids: List[int]           # List of assigned hit IDs
    state: np.ndarray            # Current EKF state (7D)
    covariance: np.ndarray       # Current EKF covariance (7x7)
    score: float                 # Accumulated chi2 score
    status: TrackStatus
    parent_id: Optional[int] = None
    seed_row: Optional[pd.Series] = None  # Reference to the seed row that started this track
    
    def add_hit(self, position: np.ndarray, hit_id: int, 
                new_state: np.ndarray, new_covariance: np.ndarray, 
                chi2_contribution: float) -> None:
        r"""
        Append a new measurement to the track and update EKF state/score.

        Parameters
        ----------
        position : ndarray, shape (3,)
            Measured hit position :math:`\mathbf{z}` appended to the trajectory.
        hit_id : int
            Identifier of the consumed hit.
        new_state : ndarray, shape (7,)
            EKF posterior state after incorporating the measurement.
        new_covariance : ndarray, shape (7, 7)
            EKF posterior covariance.
        chi2_contribution : float
            Incremental Mahalanobis distance :math:`(\mathbf{z}-H\hat{\mathbf{x}})^\top S^{-1} (\mathbf{z}-H\hat{\mathbf{x}})`.

        Returns
        -------
        None
        """
        self.trajectory.append(position)
        self.hit_ids.append(hit_id)
        self.state = new_state
        self.covariance = new_covariance
        self.score += chi2_contribution


class TrackBuilder:
    r"""
    Build tracks from seed hits using a branching EKF strategy.

    The builder orchestrates:

    1. **Seed extraction** from truth hits (three points per particle).
    2. **Brancher execution** (propagation, gating, updates) via :class:`~trackml_reco.branchers.brancher.Brancher`.
    3. **Hit assignment** and candidate bookkeeping.
    4. **Pruning** to keep only the best hypotheses.

    Notes
    -----
    Cylindrical radius is used frequently:

    .. math::

        r \;=\; \sqrt{x^2 + y^2}.
    """
    
    def __init__(self, 
                 hit_pool: trk_hit_pool.HitPool,
                 brancher_cls: Type[Brancher],
                 brancher_config: Dict = None):
        self.hit_pool = hit_pool
        self.brancher_cls = brancher_cls
        self.brancher_config = brancher_config or {}
        
        # Track building state - organized by particle ID for efficient access
        self.track_candidates_by_particle: Dict[int, Dict[int, TrackCandidate]] = {}  # particle_id -> {track_id -> TrackCandidate}
        self.completed_tracks: List[TrackCandidate] = []
        self.next_track_id: int = 0
        
        # Seed management - now using DataFrame
        self.seeds_df: pd.DataFrame = pd.DataFrame()
        
    def build_seeds_from_truth(self,
                              max_seeds: int = None,
                              jitter_sigma: float = 1e-4) -> pd.DataFrame:
        r"""
        Construct 3-point seeds from truth hits (one triplet per particle).

        Procedure
        ---------
        1. Sort truth hits by radius :math:`r=\sqrt{x^2+y^2}`.
        2. Deduplicate by ``(particle_id, layer_id, volume_id)`` keeping the closest in radius.
        3. Take the first three hits per particle as seed points.
        4. (Optionally) apply Gaussian jitter with std ``jitter_sigma``.

        Parameters
        ----------
        max_seeds : int, optional
            If provided, limit to the first ``max_seeds`` unique particles.
        jitter_sigma : float, optional
            Standard deviation for positional jitter (same units as coordinates).
            Default is ``1e-4``.

        Returns
        -------
        pandas.DataFrame
            Tidy seed table containing per-seed coordinates, seed index
            (``seed_point_index``), and derived lists like ``future_layers`` and
            ``future_hit_ids`` for each particle.
        """
        print(f"Building seeds from {len(self.hit_pool.pt_cut_hits)} truth hits...")
        
        # Sort by radius to get hits from inside out
        truth_hits_sorted = self.hit_pool.pt_cut_hits.sort_values('r')
        
        # Drop duplicates to keep only the closest hit per (particle_id, layer_id)
        deduped = truth_hits_sorted.drop_duplicates(subset=['particle_id', 'layer_id', 'volume_id'], keep='first')
        
        # Group by particle_id and take the first 3 hits for seeds
        seed_data = deduped.groupby('particle_id').head(3)
        
        # Limit seeds if requested
        if max_seeds is not None:
            unique_particles = seed_data['particle_id'].unique()[:max_seeds]
            seed_data = seed_data[seed_data['particle_id'].isin(unique_particles)]
        
        # Add jitter to seed points
        #jittered_points = trk_utils.jitter_seed_points(seed_data[['x', 'y', 'z']].values, sigma=jitter_sigma)
        jittered_points = seed_data[['x', 'y', 'z']].values
        seed_data = seed_data.copy()
        seed_data[['x', 'y', 'z']] = jittered_points
        
        # Get the remaining hits (after the first 3) for each particle
        remaining_hits = deduped.groupby('particle_id').tail(-3)
        
        # Create seeds DataFrame with future layer information
        seeds_list = []
        print("there are ", seed_data.particle_id.nunique(), " particles")
        for particle_id, group in seed_data.groupby('particle_id'):
            if len(group) >= 3:  # Need at least 3 points for a seed
                # Get future hits for this particle
                future_hits = remaining_hits[remaining_hits['particle_id'] == particle_id]
                future_layers = list(zip(future_hits['volume_id'].values, future_hits['layer_id'].values))
                # Create a row for each seed point with the same future layer information
                for i, (_, row) in enumerate(group.iterrows()):
                    seed_row = row.copy()
                    seed_row['seed_point_index'] = i  # 0, 1, or 2
                    seed_row['future_layers'] = future_layers
                    seed_row['future_hit_ids'] = future_hits['hit_id'].tolist()
                    seeds_list.append(seed_row)
        
        self.seeds_df = pd.DataFrame(seeds_list)
        print(f"Built {len(self.seeds_df)} seed points from {len(seed_data['particle_id'].unique())} particles")
        return self.seeds_df
    
    def build_tracks_from_truth(self, 
                               max_seeds: int = None,
                               max_tracks_per_seed: int = 30,
                               max_branches: int = 12,
                               jitter_sigma: float = 1e-3) -> List[TrackCandidate]:
        r"""
        Full pipeline: build seeds then construct tracks from those seeds.

        Parameters
        ----------
        max_seeds : int, optional
            Maximum number of seed particles to process.
        max_tracks_per_seed : int
            Upper bound on concurrent hypotheses per seed inside the brancher.
        max_branches : int
            Maximum number of branches kept per layer (beam width / prune size).
        jitter_sigma : float
            Positional jitter applied at seed building time.

        Returns
        -------
        list of TrackCandidate
            All completed tracks produced by the pipeline.
        """
        # Build seeds first
        seeds_df = self.build_seeds_from_truth(max_seeds, jitter_sigma)
        
        # Then build tracks from seeds
        return self.build_tracks_from_seeds(seeds_df, max_tracks_per_seed, max_branches)
    
    def build_tracks_from_seeds(self, seeds_df: pd.DataFrame, 
                               max_tracks_per_seed: int = 30,
                               max_branches: int = 12) -> List[TrackCandidate]:
        r"""
        Build tracks by running the brancher on a seeds table.

        Parameters
        ----------
        seeds_df : pandas.DataFrame
            Output of :meth:`build_seeds_from_truth`.
        max_tracks_per_seed : int
            Upper bound on track hypotheses per seed used by the brancher.
        max_branches : int
            Beam size: number of best hypotheses retained after each expansion.

        Returns
        -------
        list of TrackCandidate
            Completed tracks (status ``COMPLETED``) accumulated during building.
        """
        # Create brancher with current hit pool state
        self.brancher = self.brancher_cls(
            trees=self.hit_pool.trees,
            **self.brancher_config
        )
        
        # Group seeds by particle_id to get 3-point seeds
        for particle_id, group in seeds_df.groupby('particle_id'):
            if len(group) >= 3:  # Need at least 3 points for a seed
                self._process_seed_group(group, max_tracks_per_seed, max_branches)
            
        return self.completed_tracks
    
    def _process_seed_group(self, seed_group: pd.DataFrame, max_tracks: int, max_branches: int) -> None:
        r"""
        Process one particle's 3-point seed and run the brancher to build tracks.

        Steps
        -----
        1. Sort the three seed points by ``seed_point_index``.
        2. Build the layer sequence (``future_layers``) and a corresponding time grid
           :math:`t = \operatorname{linspace}(0, 1, L+3)`.
        3. Run ``brancher.run(seed_xyz, layers, t)`` to obtain candidate branches.
        4. Convert branches to :class:`TrackCandidate` and assign their hits.
        5. Prune to at most ``max_branches`` per particle.

        Parameters
        ----------
        seed_group : pandas.DataFrame
            Three seed rows for a single particle (with shared ``future_layers``).
        max_tracks : int
            Maximum number of brancher hypotheses per seed (forwarded to brancher if relevant).
        max_branches : int
            Per-particle cap when pruning resulting tracks.

        Returns
        -------
        None
        """
        # Sort by seed_point_index to ensure correct order
        seed_group = seed_group.sort_values('seed_point_index')
        
        # Extract the 3 seed points
        seed_points = seed_group[['x', 'y', 'z']].values[:3]  # Take first 3 points
        
        # Get future layer information (same for all points in the group)
        future_layers = seed_group.iloc[0]['future_layers']
        
        # Create time array for EKF
        t = np.linspace(0, 1, len(future_layers) + 3)
        
        # Run the branching EKF
        branches, graph = self.brancher.run(seed_points, future_layers, t, plot_tree=False)
        # Convert branches to track candidates and assign hits
        print("len branches", len(branches))
        particle_id = seed_group.iloc[0]['particle_id']
        self._convert_branches_to_tracks(branches, particle_id, seed_group.iloc[0])
        
        # Keep only the best tracks
        print(f"Before pruning: {len(self.get_track_candidates_by_particle(particle_id))} tracks for particle {particle_id}")
        self._prune_tracks(max_branches)
        branches_list = self.get_track_candidates_by_particle(particle_id)
        print(f"After pruning: {len(branches_list)} tracks for particle {particle_id}")
        
        # Ensure we keep at least one track per particle
        if len(branches_list) == 0:
            print(f"⚠️  Warning: All tracks pruned for particle {particle_id}")
            # Restore the best track if all were pruned
            if particle_id in self.track_candidates_by_particle:
                # This shouldn't happen with per-particle pruning, but just in case
                print(f"  Restoring best track for particle {particle_id}")
        if PLOT_BRANCHES: 
            trk_plot.plot_branches(branches_list, seed_points, future_layers, hits_df=self.hit_pool.hits, truth_hits=self.hit_pool.pt_cut_hits, particle_id=particle_id)
    
    def _convert_branches_to_tracks(self, branches: List[Dict], particle_id: int, seed_row: pd.Series) -> None:
        r"""
        Convert brancher outputs into :class:`TrackCandidate` objects and assign hits.

        Parameters
        ----------
        branches : list of dict
            Each branch dictionary must contain
            ``'traj'``, ``'hit_ids'``, ``'state'``, ``'cov'``, and ``'score'``.
        particle_id : int
            Particle identifier used to index candidates.
        seed_row : pandas.Series
            Seed metadata for this particle; used to determine completion status.

        Returns
        -------
        None
        """
        for branch in branches:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # Create track candidate
            track = TrackCandidate(
                id=track_id,
                particle_id=particle_id,
                trajectory=branch['traj'],
                hit_ids=branch.get('hit_ids', []),
                state=branch['state'],
                covariance=branch['cov'],
                score=branch['score'],
                status=TrackStatus.COMPLETED if len(branch['traj']) >= len(seed_row['future_layers']) else TrackStatus.ACTIVE,
                seed_row=seed_row  # Store reference to the seed row
            )
            
            # Assign hits to the pool (batch operation)
            assigned_count = self.hit_pool.assign_hits(track.hit_ids)
            if assigned_count != len(track.hit_ids):
                print(f"Warning: Only {assigned_count}/{len(track.hit_ids)} hits assigned for track {track_id}")
            
            # Store track candidate organized by particle ID
            if particle_id not in self.track_candidates_by_particle:
                self.track_candidates_by_particle[particle_id] = {}
            self.track_candidates_by_particle[particle_id][track_id] = track
            
            if track.status == TrackStatus.COMPLETED:
                self.completed_tracks.append(track)
    
    def _prune_tracks(self, max_branches: int) -> None:
        r"""
        Keep only the top-scoring candidates per particle (beam pruning).

        For each particle, sort by accumulated score (lower is better) and keep
        the first ``max_branches`` items. Releases hits from removed candidates.

        Parameters
        ----------
        max_branches : int
            Per-particle cap.

        Returns
        -------
        None
        """
        # Prune each particle separately to ensure each keeps some tracks
        for particle_id, particle_tracks in self.track_candidates_by_particle.items():
            if len(particle_tracks) <= max_branches:
                continue  # No pruning needed for this particle
                
            # Sort tracks for this particle by score and keep the best
            sorted_tracks = sorted(particle_tracks.values(), key=lambda t: t.score)
            tracks_to_keep = sorted_tracks[:max_branches]
            tracks_to_remove = sorted_tracks[max_branches:]
            
            print(f"Particle {particle_id}: keeping {len(tracks_to_keep)}/{len(particle_tracks)} tracks")
            
            # Remove tracks that didn't make the cut for this particle
            for track in tracks_to_remove:
                track_id = track.id
                
                # Free up the hits for other tracks
                released_count = self.hit_pool.release_hits(track.hit_ids)
                if released_count != len(track.hit_ids):
                    print(f"Warning: Only {released_count}/{len(track.hit_ids)} hits released for track {track_id}")
                
                # Remove from particle organization
                del self.track_candidates_by_particle[particle_id][track_id]
    
    def get_best_tracks(self, n: int = None) -> List[TrackCandidate]:
        r"""
        Return the top-``n`` completed tracks by ascending score.

        Parameters
        ----------
        n : int, optional
            Number of tracks to return. If ``None``, return all completed tracks.

        Returns
        -------
        list of TrackCandidate
            Sorted by increasing ``score``.
        """
        sorted_tracks = sorted(self.completed_tracks, key=lambda t: t.score)
        if n is None:
            return sorted_tracks
        return sorted_tracks[:n]
    
    def get_track_statistics(self) -> Dict:
        r"""
        Summarize global building statistics.

        Returns
        -------
        dict
            Keys include:

            * ``total_tracks_created`` — total IDs allocated (including pruned).
            * ``completed_tracks`` — number with status ``COMPLETED``.
            * ``active_tracks`` — number with status ``ACTIVE``.
            * ``seeds_built`` — number of seed rows in ``seeds_df``.
            * ``unique_particles`` — distinct particle IDs in seeds.
            * ``assigned_hits`` — count of hits currently assigned in pool.
            * ``available_hits`` — unassigned hits remaining.
            * ``total_hits`` — size of hit table.
            * ``assignment_ratio`` — assigned / total.
        """
        # Count active tracks across all particles
        active_tracks = 0
        for particle_tracks in self.track_candidates_by_particle.values():
            active_tracks += len([t for t in particle_tracks.values() 
                                if t.status == TrackStatus.ACTIVE])
        
        return {
            'total_tracks_created': self.next_track_id,
            'completed_tracks': len(self.completed_tracks),
            'active_tracks': active_tracks,
            'seeds_built': len(self.seeds_df),
            'unique_particles': self.seeds_df['particle_id'].nunique() if not self.seeds_df.empty else 0,
            'assigned_hits': len(self.hit_pool._assigned_hits),
            'available_hits': self.hit_pool.get_available_hit_count(),
            'total_hits': len(self.hit_pool.hits),
            'assignment_ratio': self.hit_pool.get_assignment_ratio()
        } # 'layer_stats': self.hit_pool.get_layer_statistics()
    
    def get_seeds_dataframe(self) -> pd.DataFrame:
        r"""
        Return a defensive copy of the current seeds table.

        Returns
        -------
        pandas.DataFrame
            Copy of the internal ``seeds_df``.
        """
        return self.seeds_df.copy()
    
    def get_tracks_by_seed(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Retrieve completed tracks derived from a given seed (by particle ID).

        Parameters
        ----------
        particle_id : int
            Seed particle identifier.

        Returns
        -------
        list of TrackCandidate
            Completed tracks whose ``particle_id`` equals the query.
        """
        return [track for track in self.completed_tracks if track.particle_id == particle_id]
    
    def get_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Retrieve all track candidates (completed and active) for a particle.

        Parameters
        ----------
        particle_id : int
            Identifier to query.

        Returns
        -------
        list of TrackCandidate
            All candidates currently kept for the particle.
        """
        if particle_id not in self.track_candidates_by_particle:
            return []
        return list(self.track_candidates_by_particle[particle_id].values())
    
    def get_active_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Retrieve *active* candidates for a particle.

        Parameters
        ----------
        particle_id : int
            Identifier to query.

        Returns
        -------
        list of TrackCandidate
            Candidates with status ``ACTIVE``.
        """
        candidates = self.get_track_candidates_by_particle(particle_id)
        return [track for track in candidates if track.status == TrackStatus.ACTIVE]
    
    def get_completed_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        r"""
        Retrieve *completed* candidates for a particle.

        Parameters
        ----------
        particle_id : int
            Identifier to query.

        Returns
        -------
        list of TrackCandidate
            Candidates with status ``COMPLETED``.
        """
        candidates = self.get_track_candidates_by_particle(particle_id)
        return [track for track in candidates if track.status == TrackStatus.COMPLETED]
    
    def get_all_particle_ids(self) -> List[int]:
        r"""
        List all particle IDs present in the current builder state.

        Returns
        -------
        list of int
            Keys of the per-particle candidate map.
        """
        return list(self.track_candidates_by_particle.keys())
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackCandidate]:
        r"""
        Look up a track candidate by its global ID.

        Parameters
        ----------
        track_id : int
            Global track identifier allocated by the builder.

        Returns
        -------
        TrackCandidate or None
            The candidate if present, otherwise ``None``.
        """
        for particle_tracks in self.track_candidates_by_particle.values():
            if track_id in particle_tracks:
                return particle_tracks[track_id]
        return None
    
    def get_particle_statistics(self, particle_id: int) -> Dict:
        r"""
        Per-particle candidate summary.

        For the set :math:`\mathcal{T}` of candidates of a particle, we report:

        * counts of ``ACTIVE`` and ``COMPLETED``
        * best (minimum) score
        * summary of trajectory lengths

        Parameters
        ----------
        particle_id : int
            Particle ID to summarize.

        Returns
        -------
        dict
            ``{'particle_id', 'total_candidates', 'active_candidates',
            'completed_candidates', 'best_score', 'avg_length',
            'min_length', 'max_length'}``
        """
        candidates = self.get_track_candidates_by_particle(particle_id)
        if not candidates:
            return {
                'particle_id': particle_id,
                'total_candidates': 0,
                'active_candidates': 0,
                'completed_candidates': 0,
                'best_score': None,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0
            }
        
        active_count = len([t for t in candidates if t.status == TrackStatus.ACTIVE])
        completed_count = len([t for t in candidates if t.status == TrackStatus.COMPLETED])
        scores = [t.score for t in candidates]
        lengths = [len(t.trajectory) for t in candidates]
        
        return {
            'particle_id': particle_id,
            'total_candidates': len(candidates),
            'active_candidates': active_count,
            'completed_candidates': completed_count,
            'best_score': min(scores) if scores else None,
            'avg_length': np.mean(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0
        }
    
    def get_pruning_statistics(self) -> Dict:
        r"""
        Summary after pruning per particle.

        Returns
        -------
        dict
            Includes total counts and, for each particle, the number of
            tracks kept and simple aggregates (best score, average length).
        """
        stats = {}
        total_tracks_before = 0
        total_tracks_after = 0
        
        for particle_id in self.get_all_particle_ids():
            candidates = self.get_track_candidates_by_particle(particle_id)
            stats[particle_id] = {
                'tracks_kept': len(candidates),
                'best_score': min([t.score for t in candidates]) if candidates else None,
                'avg_length': np.mean([len(t.trajectory) for t in candidates]) if candidates else 0
            }
            total_tracks_after += len(candidates)
        
        return {
            'particles_with_tracks': len(stats),
            'total_tracks_after_pruning': total_tracks_after,
            'per_particle_stats': stats
        }
    
    def reset(self) -> None:
        r"""
        Clear all internal state and release all assigned hits.

        Effects
        -------
        * Empties per-particle candidate maps and completed list.
        * Resets seed table and next track ID counter.
        * Calls :meth:`HitPool.reset` to mark all hits as unassigned.

        Returns
        -------
        None
        """
        self.track_candidates_by_particle.clear()
        self.completed_tracks.clear()
        self.seeds_df = pd.DataFrame()
        self.next_track_id = 0
        self.hit_pool.reset() 