from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class HitPool:
    """Manages the pool of available hits for track building"""
    
    def __init__(self, hits: pd.DataFrame, pt_cut_hits: pd.DataFrame):
        """
        Initialize hit pool with all hits and spatial trees
        
        Parameters
        ----------
        hits : pd.DataFrame
            DataFrame containing all hits with columns ['hit_id', 'x', 'y', 'z', ...]
        trees : Dict
            Dictionary mapping (volume_id, layer_id) to (cKDTree, points, hit_ids)
        """
        self.hits = hits
        self.pt_cut_hits = pt_cut_hits
        self._assigned_hits: Set[int] = set()
        self._trees = self.build_layer_trees()
        
    @property
    def trees(self) -> Dict:
        """Get the spatial trees (read-only)"""
        return self._trees
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        """Get the sorted list of layer keys"""
        return sorted(self._trees.keys(), key=lambda x: (x[0], x[1]))
        
    
    def build_layer_trees(self) -> Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]]:
        """
        Builds KD-trees for each detector layer to enable fast hit lookup.

        Returns
        -------
        Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]]
            Dictionary mapping layer keys to KD-tree tuples (tree, points, hit_ids)
        """
        self.hits['layer_key'] = list(zip(self.hits.volume_id, self.hits.layer_id))
        trees = {}
        for key, grp in self.hits.groupby('layer_key'):
            points = grp[['x', 'y', 'z']].values
            ids = grp['hit_id'].values
            trees[key] = (cKDTree(points), points, ids)
            print(f"Layer {key}: {len(points)} hits")
        return trees


    def get_candidates(self, predicted_position: np.ndarray, 
                      layer: Tuple[int, int], 
                      max_candidates: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get candidate hits for a given layer and predicted position
        
        Parameters
        ----------
        predicted_position : np.ndarray
            3D predicted position
        layer : Tuple[int, int]
            (volume_id, layer_id) tuple
        max_candidates : int
            Maximum number of candidates to return
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (candidate_positions, candidate_hit_ids) where hit_ids are filtered to unassigned hits
        """
        if layer not in self.trees:
            return np.array([]), np.array([])
            
        tree, points, hit_ids = self.trees[layer]
        
        # Find nearest candidates
        distances, indices = tree.query(predicted_position, k=max_candidates)
        indices = np.atleast_1d(indices)
        
        # Filter to only unassigned hits
        unassigned_mask = ~np.isin(hit_ids[indices], list(self._assigned_hits))
        unassigned_indices = indices[unassigned_mask]
        
        if len(unassigned_indices) == 0:
            return np.array([]), np.array([])
            
        return points[unassigned_indices], hit_ids[unassigned_indices]
    
    def assign_hit(self, hit_id: int) -> bool:
        """
        Mark a hit as assigned to a track
        
        Parameters
        ----------
        hit_id : int
            The hit ID to assign
            
        Returns
        -------
        bool
            True if hit was successfully assigned, False if already assigned
        """
        if hit_id in self._assigned_hits:
            return False
        self._assigned_hits.add(hit_id)
        return True
    
    def assign_hits(self, hit_ids: List[int]) -> int:
        """
        Assign multiple hits at once
        
        Parameters
        ----------
        hit_ids : List[int]
            List of hit IDs to assign
            
        Returns
        -------
        int
            Number of hits successfully assigned
        """
        assigned_count = 0
        for hit_id in hit_ids:
            if self.assign_hit(hit_id):
                assigned_count += 1
        return assigned_count
    
    def release_hit(self, hit_id: int) -> bool:
        """
        Release a hit back to the pool
        
        Parameters
        ----------
        hit_id : int
            The hit ID to release
            
        Returns
        -------
        bool
            True if hit was successfully released, False if not assigned
        """
        if hit_id in self._assigned_hits:
            self._assigned_hits.remove(hit_id)
            return True
        return False
    
    def release_hits(self, hit_ids: List[int]) -> int:
        """
        Release multiple hits back to the pool
        
        Parameters
        ----------
        hit_ids : List[int]
            List of hit IDs to release
            
        Returns
        -------
        int
            Number of hits successfully released
        """
        released_count = 0
        for hit_id in hit_ids:
            if self.release_hit(hit_id):
                released_count += 1
        return released_count
    
    def is_hit_available(self, hit_id: int) -> bool:
        """Check if a hit is still available for assignment"""
        return hit_id not in self._assigned_hits
    
    def get_available_hit_count(self) -> int:
        """Get count of remaining unassigned hits"""
        return len(self.hits) - len(self._assigned_hits)
    
    def get_assignment_ratio(self) -> float:
        """Get the ratio of assigned hits to total hits"""
        return len(self._assigned_hits) / len(self.hits) if len(self.hits) > 0 else 0.0
    
    def get_layer_statistics(self) -> Dict[Tuple[int, int], Dict]:
        """Get statistics about hit assignment per layer"""
        stats = {}
        for layer, (tree, points, hit_ids) in self.trees.items():
            layer_hits = set(hit_ids)
            assigned_in_layer = layer_hits.intersection(self._assigned_hits)
            stats[layer] = {
                'total_hits': len(layer_hits),
                'assigned_hits': len(assigned_in_layer),
                'available_hits': len(layer_hits) - len(assigned_in_layer),
                'assignment_ratio': len(assigned_in_layer) / len(layer_hits) if len(layer_hits) > 0 else 0.0
            }
        return stats
    
    def reset(self) -> None:
        """Reset the hit pool, releasing all assigned hits"""
        self._assigned_hits.clear()
    
    def get_unassigned_hits_in_layer(self, layer: Tuple[int, int]) -> List[int]:
        """
        Get all unassigned hit IDs in a specific layer
        
        Parameters
        ----------
        layer : Tuple[int, int]
            (volume_id, layer_id) tuple
            
        Returns
        -------
        List[int]
            List of unassigned hit IDs in the layer
        """
        if layer not in self.trees:
            return []
            
        _, _, hit_ids = self.trees[layer]
        return [hid for hid in hit_ids if hid not in self._assigned_hits]
    
    def get_assigned_hits_in_layer(self, layer: Tuple[int, int]) -> List[int]:
        """
        Get all assigned hit IDs in a specific layer
        
        Parameters
        ----------
        layer : Tuple[int, int]
            (volume_id, layer_id) tuple
            
        Returns
        -------
        List[int]
            List of assigned hit IDs in the layer
        """
        if layer not in self.trees:
            return []
            
        _, _, hit_ids = self.trees[layer]
        return [hid for hid in hit_ids if hid in self._assigned_hits] 