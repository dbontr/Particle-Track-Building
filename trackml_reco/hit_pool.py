from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class HitPool:
    r"""
    Manage the pool of detector hits for track building, including fast spatial
    lookup via per-layer KD-trees and book-keeping of assigned/unassigned hits.

    Attributes
    ----------
    hits : pd.DataFrame
        All hits with columns such as ``['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', ...]``.
    pt_cut_hits : pd.DataFrame
        Subset of hits that pass a transverse-momentum selection (for downstream use).
    _assigned_hits : Set[int]
        Set of hit IDs currently reserved/assigned to tracks.
    _trees : Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]]
        Spatial indexes per layer; each value is ``(tree, points, hit_ids)`` where
        ``points`` has shape ``(N, 3)`` and ``hit_ids`` has shape ``(N,)``.

    Notes
    -----
    * Layer keys are tuples ``(volume_id, layer_id)``.
    * Nearest-neighbor queries are performed in Euclidean 3D space.
    """
    
    def __init__(self, hits: pd.DataFrame, pt_cut_hits: pd.DataFrame):
        r"""
        Initialize the hit pool and build per-layer KD-trees.

        Parameters
        ----------
        hits : pd.DataFrame
            Full hit table. Must include at least:
            ``'hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id'``.
        pt_cut_hits : pd.DataFrame
            Hits that pass a :math:`p_T` selection (kept for convenience).
        """
        self.hits = hits
        self.pt_cut_hits = pt_cut_hits
        self._assigned_hits: Set[int] = set()
        self._trees = self.build_layer_trees()
        
    @property
    def trees(self) -> Dict:
        r"""
        KD-trees for fast spatial lookup, keyed by layer.

        Returns
        -------
        dict
            ``{(volume_id, layer_id): (cKDTree, points, hit_ids)}``.
        """
        return self._trees
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        r"""
        Sorted list of detector layer keys.

        Returns
        -------
        list of tuple(int, int)
            Sorted ``(volume_id, layer_id)`` pairs, ascending in both fields.
        """
        return sorted(self._trees.keys(), key=lambda x: (x[0], x[1]))
        
    
    def build_layer_trees(self) -> Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]]:
        r"""
        Build a KD-tree per detector layer for fast nearest-neighbor queries.

        Returns
        -------
        dict
            Maps layer key ``(volume_id, layer_id)`` → ``(tree, points, hit_ids)`` where
            ``points`` is ``(N, 3)`` and ``hit_ids`` is ``(N,)``.

        Notes
        -----
        Each tree is built over the coordinates :math:`(x, y, z)`.
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
        r"""
        Return nearest unassigned hit candidates in a given layer.

        Parameters
        ----------
        predicted_position : ndarray, shape (3,)
            Predicted Cartesian position :math:`\hat{x} = (x,y,z)`.
        layer : tuple(int, int)
            Layer key ``(volume_id, layer_id)`` to search within.
        max_candidates : int, optional
            Maximum number of nearest neighbors to retrieve. Default ``10``.

        Returns
        -------
        candidate_positions : ndarray, shape (M, 3)
            Candidate hit positions (unassigned only), with :math:`M \le \text{max\_candidates}`.
        candidate_hit_ids : ndarray, shape (M,)
            Corresponding hit IDs (unassigned only).

        Notes
        -----
        Nearest neighbors are computed in Euclidean 3D distance relative to
        ``predicted_position``; already-assigned hits are filtered out.
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
        r"""
        Mark a hit as assigned.

        Parameters
        ----------
        hit_id : int
            Unique hit identifier to reserve.

        Returns
        -------
        bool
            ``True`` if the hit was newly assigned; ``False`` if it was already assigned.
        """
        if hit_id in self._assigned_hits:
            return False
        self._assigned_hits.add(hit_id)
        return True
    
    def assign_hits(self, hit_ids: List[int]) -> int:
        r"""
        Assign multiple hits at once.

        Parameters
        ----------
        hit_ids : list of int
            Hit identifiers to reserve.

        Returns
        -------
        int
            Number of hits successfully assigned (i.e., not previously assigned).
        """
        assigned_count = 0
        for hit_id in hit_ids:
            if self.assign_hit(hit_id):
                assigned_count += 1
        return assigned_count
    
    def release_hit(self, hit_id: int) -> bool:
        r"""
        Release a previously assigned hit back to the pool.

        Parameters
        ----------
        hit_id : int
            Hit identifier to release.

        Returns
        -------
        bool
            ``True`` if the hit was released; ``False`` if it was not assigned.
        """
        if hit_id in self._assigned_hits:
            self._assigned_hits.remove(hit_id)
            return True
        return False
    
    def release_hits(self, hit_ids: List[int]) -> int:
        r"""
        Release multiple hits back to the pool.

        Parameters
        ----------
        hit_ids : list of int
            Hit identifiers to release.

        Returns
        -------
        int
            Number of hits successfully released.
        """
        released_count = 0
        for hit_id in hit_ids:
            if self.release_hit(hit_id):
                released_count += 1
        return released_count
    
    def is_hit_available(self, hit_id: int) -> bool:
        r"""
        Check if a hit is currently unassigned (available).

        Parameters
        ----------
        hit_id : int
            Hit identifier to query.

        Returns
        -------
        bool
            ``True`` if unassigned, ``False`` otherwise.
        """
        return hit_id not in self._assigned_hits
    
    def get_available_hit_count(self) -> int:
        r"""
        Number of unassigned hits remaining.

        Returns
        -------
        int
            Count of available (not assigned) hits.
        """
        return len(self.hits) - len(self._assigned_hits)
    
    def get_assignment_ratio(self) -> float:
        r"""
        Fraction of assigned hits.

        Returns
        -------
        float
            Ratio :math:`\rho \in [0,1]` defined as

            .. math::

               \rho = \frac{\lvert \mathcal{H}_{\text{assigned}} \rvert}
                           {\lvert \mathcal{H}_{\text{total}} \rvert},

            with the convention :math:`\rho=0` if there are no hits.
        """
        return len(self._assigned_hits) / len(self.hits) if len(self.hits) > 0 else 0.0
    
    def get_layer_statistics(self) -> Dict[Tuple[int, int], Dict]:
        r"""
        Per-layer assignment statistics.

        Returns
        -------
        dict
            For each layer key ``(volume_id, layer_id)``, a dictionary with:

            * ``'total_hits'`` : int
            * ``'assigned_hits'`` : int
            * ``'available_hits'`` : int
            * ``'assignment_ratio'`` : float in :math:`[0,1]`.

        Notes
        -----
        Ratios are computed as :math:`\text{assigned} / \text{total}` per layer,
        with 0 if the layer has no hits.
        """
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
        r"""
        Release all assigned hits (clear reservations).

        Returns
        -------
        None
        """
        self._assigned_hits.clear()
    
    def get_unassigned_hits_in_layer(self, layer: Tuple[int, int]) -> List[int]:
        r"""
        List all **unassigned** hit IDs in a given layer.

        Parameters
        ----------
        layer : tuple(int, int)
            Layer key ``(volume_id, layer_id)``.

        Returns
        -------
        list of int
            Unassigned hit IDs in the specified layer. Empty list if the layer
            is unknown or fully assigned.
        """
        if layer not in self.trees:
            return []
            
        _, _, hit_ids = self.trees[layer]
        return [hid for hid in hit_ids if hid not in self._assigned_hits]
    
    def get_assigned_hits_in_layer(self, layer: Tuple[int, int]) -> List[int]:
        r"""
        List all **assigned** hit IDs in a given layer.

        Parameters
        ----------
        layer : tuple(int, int)
            Layer key ``(volume_id, layer_id)``.

        Returns
        -------
        list of int
            Assigned hit IDs in the specified layer. Empty list if the layer
            is unknown or has no assignments.
        """
        if layer not in self.trees:
            return []
            
        _, _, hit_ids = self.trees[layer]
        return [hid for hid in hit_ids if hid in self._assigned_hits] 