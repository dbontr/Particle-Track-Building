import numpy as np
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
from scipy.optimize import newton
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFSABrancher(Brancher):
    """
    EKF-based track finder using Simulated Annealing (SA).
    Each iteration, we propose a slight mutation to the current path (swap, replace one hit)
    and accept based on Metropolis criterion with a cooling temperature.
    """
    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int,int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 initial_temp: float = 1.0,
                 cooling_rate: float = 0.95,
                 n_iters: int = 1000,
                 max_cands: int = 10,
                 step_candidates: int = 5):
        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_iters = n_iters
        self.step_candidates = step_candidates
        self.state_dim = 7

    def _initial_path(self, seed_xyz: np.ndarray, t: np.ndarray, layers: List[Tuple[int,int]]) -> Dict:
        """Build a greedy initial path using lowest chi2 at each layer."""
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
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

            costs = np.array([float((p - x_pred[:3]) @ np.linalg.solve(S, p - x_pred[:3])) for p in pts])
            idx = np.argmin(costs)
            chosen_pt, chosen_id, chi2 = pts[idx], ids[idx], costs[idx]

            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            hit_ids.append(int(chosen_id))
            total_score += chi2
            G.add_edge((i, tuple(traj[-2])), (i+1, tuple(chosen_pt)), cost=chi2)

        return {'traj': traj, 'hit_ids': hit_ids, 'state': state, 'cov': cov, 'score': total_score, 'graph': G}

    def _mutate_path(self, path: Dict, seed_xyz: np.ndarray, t: np.ndarray, layers: List[Tuple[int,int]]) -> Dict:
        """Mutate one hit in the existing path and recompute the full score via EKF."""
        traj, hit_ids = list(path['traj']), list(path['hit_ids'])
        if len(hit_ids) == 0:
            return path
        idx = np.random.randint(len(hit_ids))

        # rebuild from seed with possible mutation
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        new_traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        new_hit_ids: List[int] = []
        new_score = 0.0
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

            costs = np.array([float((p - x_pred[:3]) @ np.linalg.solve(S, p - x_pred[:3])) for p in pts])
            if i == idx:
                # choose alternative candidate
                alt = [j for j, hid in enumerate(ids) if hid != hit_ids[i]]
                if not alt:
                    return path
                j = np.random.choice(alt)
            else:
                # try original, fallback to best
                try:
                    j = ids.tolist().index(hit_ids[i])
                except ValueError:
                    j = np.argmin(costs)

            chosen_pt, chosen_id, chi2 = pts[j], ids[j], costs[j]

            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            new_traj.append(chosen_pt)
            new_hit_ids.append(int(chosen_id))
            new_score += chi2
            G.add_edge((i, tuple(new_traj[-2])), (i+1, tuple(chosen_pt)), cost=chi2)

        return {'traj': new_traj, 'hit_ids': new_hit_ids, 'state': state, 'cov': cov, 'score': new_score, 'graph': G}

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int,int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        """Run SA: start from greedy path, then iteratively mutate & accept/reject."""
        current = self._initial_path(seed_xyz, t, layers)
        best = current
        temp = self.initial_temp
        global_graph = current['graph']

        for _ in range(self.n_iters):
            candidate = self._mutate_path(current, seed_xyz, t, layers)
            delta = candidate['score'] - current['score']
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current = candidate
            if current['score'] < best['score']:
                best = current
            temp *= self.cooling_rate
            global_graph = nx.compose(global_graph, candidate['graph'])

        result = {
            'traj': best['traj'],
            'hit_ids': best['hit_ids'],
            'state': best['state'],
            'cov': best['cov'],
            'score': best['score']
        }
        return [result], global_graph
