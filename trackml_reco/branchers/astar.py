import numpy as np
import heapq
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
from scipy.optimize import newton
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFAStarBrancher(Brancher):
    """
    EKF-based track finder using A* search instead of breadth-first branching.
    Defines a dynamic endpoint via physics-informed propagation to the final layer.
    """
    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: List[dict],
                 noise_std: float=2.0,
                 B_z: float=0.002,
                 max_cands: int=10,
                 step_candidates: int=5):
        super().__init__(trees=trees,
                         layers=list(range(len(layer_surfaces))),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.step_candidates = step_candidates
        self.state_dim = 7

    def _solve_goal_position(self, x0: np.ndarray, goal_layer: Tuple[int,int]) -> np.ndarray:
        """
        Propagate initial state to specified goal layer surface to define the A* goal.
        """
        surf = self.layer_surfaces[goal_layer]
        dt_goal = self._solve_dt_to_surface(x0, surf, dt_init=1.0)
        return self.propagate(x0, dt_goal)[:3]

    def _heuristic(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """
        Heuristic function: Euclidean distance from current position to goal.
        """
        return np.linalg.norm(pos - goal)

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int,int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        """
        Execute A* search: nodes are hit indices, costs are KF chi2 residuals.
        Returns list of one best branch (dict) and the search graph.

        Parameters
        ----------
        seed_xyz : ndarray of shape (3,3)
            Initial three hit coordinates.
        layers : list of layer keys
            Ordered list of (volume_id, layer_id) tuples to traverse.
        t : ndarray
            Time points associated with each seed positions.
        plot_tree : bool, optional
            Ignored for A*, kept for API compatibility.
        """
        # Estimate initial state from seed
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, k0])
        P0 = np.eye(self.state_dim) * 0.1

        # Determine dynamic goal as intersection with last layer
        goal_layer = layers[-1]
        goal_pos = self._solve_goal_position(x0, goal_layer)

        # Prepare A* open set
        open_heap = []
        count = 0
        start_node = {
            'layer_idx': 0,
            'state': x0,
            'cov': P0,
            'pos': seed_xyz[2],
            'traj': [seed_xyz[0], seed_xyz[1], seed_xyz[2]],
            'g': 0.0,
            'h': self._heuristic(seed_xyz[2], goal_pos),
            'hit_ids': []
        }
        heapq.heappush(open_heap, (start_node['g'] + start_node['h'], count, start_node))
        closed = set()
        G = nx.DiGraph()

        # A* main loop
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            layer_idx = current['layer_idx']
            # Goal reached when layer index beyond last
            if layer_idx >= len(layers):
                # build branch dict
                branch = {
                    'traj': current['traj'],
                    'state': current['state'],
                    'cov': current['cov'],
                    'score': current['g'],
                    'hit_ids': current['hit_ids']
                }
                return [branch], G

            layer_key = layers[layer_idx]
            state, P = current['state'], current['cov']

            # Predict to this layer surface
            surf = self.layer_surfaces[layer_key]
            try:
                dt = self._solve_dt_to_surface(state, surf)
            except Exception:
                continue
            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ P @ F.T + self.Q0 * dt

            # Measurement update
            H = self.H_jac(None)
            S = H @ P_pred @ H.T + self.R
            gate_r = 3.0 * np.sqrt(np.max(np.linalg.eigvalsh(S)))
            pts, ids = self._get_candidates_in_gate(x_pred[:3], layer_key, gate_r)

            # Expand neighbors
            for z, hid in zip(pts, ids):
                res = z - x_pred[:3]
                chi2 = float(res @ np.linalg.solve(S, res))
                if chi2 < np.inf:
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    x_upd = x_pred + K @ res
                    P_upd = (np.eye(self.state_dim) - K @ H) @ P_pred

                    g_new = current['g'] + chi2
                    h_new = self._heuristic(z, goal_pos)
                    next_node = {
                        'layer_idx': layer_idx + 1,
                        'state': x_upd,
                        'cov': P_upd,
                        'pos': z,
                        'traj': current['traj'] + [z],
                        'g': g_new,
                        'h': h_new,
                        'hit_ids': current['hit_ids'] + [int(hid)]
                    }
                    key = (layer_idx + 1, int(hid))
                    if key in closed:
                        continue
                    closed.add(key)
                    count += 1
                    heapq.heappush(open_heap, (g_new + h_new, count, next_node))
                    G.add_edge((layer_idx, tuple(current['pos'])),
                               (layer_idx+1, tuple(z)), cost=chi2)

        # No valid path found
        return [], G