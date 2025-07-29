import numpy as np
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
from scipy.optimize import newton
import networkx as nx
from scipy.optimize import linear_sum_assignment
from trackml_reco.branchers.brancher import Brancher

class HelixEKFHungarianBrancher(Brancher):
    """
    EKF-based track finder using Hungarian (assignment) algorithm.
    At each layer, compute a cost matrix of residual χ² between all active tracks and all candidate hits,
    then solve via Hungarian in one global optimization per layer.
    """
    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int,int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 max_cands: int = 20):
        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.state_dim = 7

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int,int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        # initialize single branch from seed
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj = [seed_xyz[2]]
        hit_ids: List[int] = []
        G = nx.DiGraph()

        for i, layer in enumerate(layers):
            # propagate prediction to this layer
            surf = self.layer_surfaces[layer]
            try:
                dt = self._solve_dt_to_surface(state, surf)
            except Exception:
                break
            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            H = self.H_jac(None)
            S = H @ P_pred @ H.T + self.R

            # gather candidates
            tree, pts, ids = self.trees[layer]
            idxs = tree.query_ball_point(x_pred[:3], r=np.inf)
            cand_pts = pts[idxs]
            cand_ids = ids[idxs]
            if len(cand_ids) == 0:
                break

            # compute cost (χ²) for each candidate
            cost_vec = np.array([float((p - x_pred[:3]) @ np.linalg.solve(S, p - x_pred[:3])) for p in cand_pts])

            # solve assignment: choose minimal cost
            # since single track, pick the min cost candidate:
            best_idx = np.argmin(cost_vec)
            chosen_pt = cand_pts[best_idx]
            chosen_id = int(cand_ids[best_idx])

            # EKF update
            res = chosen_pt - x_pred[:3]
            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ res
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            # record
            traj.append(state[:3])
            hit_ids.append(chosen_id)
            G.add_edge((i, tuple(traj[-2])), (i+1, tuple(state[:3])), cost=float(res @ np.linalg.solve(S, res)))

        # return as single branch
        branch = {'traj': traj, 'hit_ids': hit_ids, 'state': state, 'cov': cov, 'score': 0.0}
        return [branch], G
