from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import trackml_reco.brancher as trk_brancher
import trackml_reco.metrics as trk_metrics
import trackml_reco.hit_pool as trk_hit_pool
import trackml_reco.utils as trk_utils
import trackml_reco.plotting as trk_plot


PLOT_BRANCHES = True

class TrackStatus(Enum):
    """Status of a track candidate"""
    ACTIVE = "active"      # Still being built
    COMPLETED = "completed"  # Reached final layer
    DEAD = "dead"          # No valid hits found


@dataclass
class TrackCandidate:
    """Represents a single track candidate during building"""
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
        """Add a hit to this track candidate"""
        self.trajectory.append(position)
        self.hit_ids.append(hit_id)
        self.state = new_state
        self.covariance = new_covariance
        self.score += chi2_contribution


class TrackBuilder:
    """Main class for building tracks from seeds"""
    
    def __init__(self, 
                 hit_pool: trk_hit_pool.HitPool,
                 brancher_config: Dict = None):
        """
        Initialize track builder
        
        Parameters
        ----------
        hit_pool : HitPool
            Pool of available hits
        brancher_config : Dict, optional
            Configuration for the EKF brancher
        """
        self.hit_pool = hit_pool
        self.brancher_config = brancher_config or {}
        
        # Track building state - organized by particle ID for efficient access
        self.track_candidates_by_particle: Dict[int, Dict[int, TrackCandidate]] = {}  # particle_id -> {track_id -> TrackCandidate}
        self.completed_tracks: List[TrackCandidate] = []
        self.next_track_id = 0
        
        # Seed management - now using DataFrame
        self.seeds_df: pd.DataFrame = pd.DataFrame()
        
    def build_seeds_from_truth(self,
                              max_seeds: int = None,
                              jitter_sigma: float = 0.0001) -> pd.DataFrame:
        """
        Build seeds from truth hits for track building
        
        Parameters
        ----------
        max_seeds : int, optional
            Maximum number of seeds to build (for debugging)
        jitter_sigma : float
            Standard deviation of Gaussian noise to add to seed points
            
        Returns
        -------
        pd.DataFrame
            DataFrame with seed information including future layer IDs
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
                print(future_layers)
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
                               jitter_sigma: float = 0.001) -> List[TrackCandidate]:
        """
        Complete pipeline: build seeds from truth and then build tracks
        
        Parameters
        ----------
        max_seeds : int, optional
            Maximum number of seeds to build
        max_tracks_per_seed : int
            Maximum number of track candidates per seed
        max_branches : int
            Maximum number of branches to keep at each layer
        jitter_sigma : float
            Standard deviation of jitter for seed points
            
        Returns
        -------
        List[TrackCandidate]
            List of completed tracks
        """
        # Build seeds first
        seeds_df = self.build_seeds_from_truth(max_seeds, jitter_sigma)
        
        # Then build tracks from seeds
        return self.build_tracks_from_seeds(seeds_df, max_tracks_per_seed, max_branches)
    
    def build_tracks_from_seeds(self, seeds_df: pd.DataFrame, 
                               max_tracks_per_seed: int = 30,
                               max_branches: int = 12) -> List[TrackCandidate]:
        """
        Build tracks from a seeds DataFrame
        
        Parameters
        ----------
        seeds_df : pd.DataFrame
            DataFrame with seed information
        max_tracks_per_seed : int
            Maximum number of track candidates to maintain per seed
        max_branches : int
            Maximum number of branches to keep at each layer
            
        Returns
        -------
        List[TrackCandidate]
            List of completed tracks
        """
        # Create brancher with current hit pool state
        self.brancher = trk_brancher.HelixEKFBrancher(
            trees=self.hit_pool.trees,
            **self.brancher_config
        )
        
        # Group seeds by particle_id to get 3-point seeds
        for particle_id, group in seeds_df.groupby('particle_id'):
            if len(group) >= 3:  # Need at least 3 points for a seed
                self._process_seed_group(group, max_tracks_per_seed, max_branches)
            
        return self.completed_tracks
    
    def _process_seed_group(self, seed_group: pd.DataFrame, max_tracks: int, max_branches: int) -> None:
        """Process a group of 3 seed points for one particle"""
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
            trk_plot.plot_branches(branches_list, seed_points, future_layers, truth_hits=self.hit_pool.pt_cut_hits, particle_id=particle_id)
    
    def _convert_branches_to_tracks(self, branches: List[Dict], particle_id: int, seed_row: pd.Series) -> None:
        """Convert EKF branches to track candidates"""
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
        """Keep only the best track candidates per particle"""
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
        """Get the best completed tracks by score"""
        sorted_tracks = sorted(self.completed_tracks, key=lambda t: t.score)
        if n is None:
            return sorted_tracks
        return sorted_tracks[:n]
    
    def get_track_statistics(self) -> Dict:
        """Get statistics about the track building process"""
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
            'assignment_ratio': self.hit_pool.get_assignment_ratio(),
            'layer_stats': self.hit_pool.get_layer_statistics()
        }
    
    def get_seeds_dataframe(self) -> pd.DataFrame:
        """Get the seeds DataFrame for analysis or plotting"""
        return self.seeds_df.copy()
    
    def get_tracks_by_seed(self, particle_id: int) -> List[TrackCandidate]:
        """Get all tracks that were built from a specific seed"""
        return [track for track in self.completed_tracks if track.particle_id == particle_id]
    
    def get_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        """Get all track candidates (active and completed) for a specific particle ID"""
        if particle_id not in self.track_candidates_by_particle:
            return []
        
        return list(self.track_candidates_by_particle[particle_id].values())
    
    def get_active_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        """Get only active track candidates for a specific particle ID"""
        candidates = self.get_track_candidates_by_particle(particle_id)
        return [track for track in candidates if track.status == TrackStatus.ACTIVE]
    
    def get_completed_track_candidates_by_particle(self, particle_id: int) -> List[TrackCandidate]:
        """Get only completed track candidates for a specific particle ID"""
        candidates = self.get_track_candidates_by_particle(particle_id)
        return [track for track in candidates if track.status == TrackStatus.COMPLETED]
    
    def get_all_particle_ids(self) -> List[int]:
        """Get all particle IDs that have track candidates"""
        return list(self.track_candidates_by_particle.keys())
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackCandidate]:
        """Get a specific track candidate by its ID"""
        for particle_tracks in self.track_candidates_by_particle.values():
            if track_id in particle_tracks:
                return particle_tracks[track_id]
        return None
    
    def get_particle_statistics(self, particle_id: int) -> Dict:
        """Get statistics for a specific particle"""
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
        """Get statistics about pruning across all particles"""
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
        """Reset the track builder state"""
        self.track_candidates_by_particle.clear()
        self.completed_tracks.clear()
        self.seeds_df = pd.DataFrame()
        self.next_track_id = 0
        self.hit_pool.reset() 