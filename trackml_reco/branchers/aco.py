import numpy as np
import random
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFACOBrancher(Brancher):
    """
    EKF-based track finder using Ant Colony Optimization (ACO) instead of branching.
    Each "ant" stochastically picks one candidate per layer based on pheromone + heuristic,
    then pheromones are updated to reinforce low-chi2 paths.
    """
    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int,int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 n_ants: int = 20,
                 evap_rate: float = 0.5,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 max_cands: int = 10,
                 step_candidates: int = 5):
        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.n_ants = n_ants
        self.evap_rate = evap_rate
        self.alpha = alpha
        self.beta = beta
        self.step_candidates = step_candidates
        self.state_dim = 7

    def _ant_walk(self,
                  seed_xyz: np.ndarray,
                  t: np.ndarray,
                  layers: List[Tuple[int,int]],
                  pheromone: Dict) -> Dict:
        """
        Single ant path through the specified candidate layers.
        """
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        layers_used: List[Tuple[int,int]] = []
        total_score = 0.0
        G = nx.DiGraph()

        for i, layer in enumerate(layers):
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
            gate_r = 3.0 * np.sqrt(np.max(np.linalg.eigvalsh(S)))

            pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, gate_r)
            if len(ids) == 0:
                break

            costs = np.array([float((p - x_pred[:3]) @ np.linalg.solve(S, p - x_pred[:3])) + 1e-6 for p in pts])
            tau = np.array([pheromone.get((layer, int(hid)), 1.0) for hid in ids])
            eta = 1.0 / costs
            probs = (tau**self.alpha) * (eta**self.beta)
            probs = probs / np.sum(probs)

            idx = np.random.choice(len(ids), p=probs)
            chosen_pt, chosen_id, chi2 = pts[idx], ids[idx], costs[idx]

            # EKF update
            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            hit_ids.append(int(chosen_id))
            layers_used.append(layer)
            total_score += chi2
            G.add_edge((i, tuple(traj[-2])), (i+1, tuple(chosen_pt)), cost=chi2)

        return {
            'traj': traj,
            'hit_ids': hit_ids,
            'layers': layers_used,
            'state': state,
            'cov': cov,
            'score': total_score,
            'graph': G
        }

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int,int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        """
        Run ACO over the specified future_layers sequence, not every detector layer.
        """
        # Initialize pheromone on each (layer, hit)
        pheromone: Dict[Tuple[Tuple[int,int], int], float] = {}
        for layer in layers:
            _, _, ids = self.trees[layer]
            for hid in ids:
                pheromone[(layer, int(hid))] = 1.0

        best_branch = None
        best_score = np.inf
        global_graph = nx.DiGraph()

        # Each ant walks only through the `layers` list
        for _ in range(self.n_ants):
            branch = self._ant_walk(seed_xyz, t, layers, pheromone)
            delta = 1.0 / (branch['score'] + 1e-6)

            # Update pheromone only on used (layer, hit)
            for layer, hid in zip(branch['layers'], branch['hit_ids']):
                key = (layer, int(hid))
                pheromone[key] = (
                    (1 - self.evap_rate) * pheromone.get(key, 0.0)
                    + self.evap_rate * delta
                )

            if branch['score'] < best_score:
                best_score = branch['score']
                best_branch = branch
            global_graph = nx.compose(global_graph, branch['graph'])

        # Format for TrackBuilder compatibility
        result = {
            'traj': best_branch['traj'],
            'hit_ids': best_branch['hit_ids'],
            'state': best_branch['state'],
            'cov': best_branch['cov'],
            'score': best_branch['score']
        }
        return [result], global_graph
