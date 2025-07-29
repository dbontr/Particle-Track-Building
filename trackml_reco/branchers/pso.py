import numpy as np
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
from scipy.optimize import newton
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFPSOBrancher(Brancher):
    """
    EKF-based track finder using Particle Swarm Optimization (PSO).
    Each particle maintains a personal best path, and the swarm converges
    toward a global best trajectory through velocity updates on hit-selection scores.
    """
    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int,int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 n_particles: int = 20,
                 n_iters: int = 10,
                 w: float = 0.5,
                 c1: float = 1.0,
                 c2: float = 2.0,
                 max_cands: int = 10,
                 step_candidates: int = 5):
        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.step_candidates = step_candidates
        self.state_dim = 7

        # velocity scores for each particle: list of dict(layer->hit_id->velocity)
        self.velocities: List[Dict[Tuple[int,int], Dict[int, float]]] = []
        # personal best for each particle
        self.pbest: List[Dict] = []
        # global best
        self.gbest: Dict = {'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}

    def _init_particles(self, layers: List[Tuple[int,int]]):
        """Initialize velocities and personal/global best placeholders."""
        self.velocities = []
        self.pbest = []
        for _ in range(self.n_particles):
            vel = {layer: {int(h): 0.0 for h in self.trees[layer][2]} for layer in layers}
            self.velocities.append(vel)
            self.pbest.append({'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None})
        self.gbest = {'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}

    def _build_path(self,
                    seed_xyz: np.ndarray,
                    t: np.ndarray,
                    layers: List[Tuple[int,int]],
                    velocity: Dict[Tuple[int,int], float]) -> Dict:
        """
        Construct a single trajectory by sampling candidates influenced by velocity scores.
        """
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

            # compute chi2 costs
            costs = np.array([float((p - x_pred[:3]) @ np.linalg.solve(S, p - x_pred[:3])) + 1e-6 for p in pts])
            # velocity contributions
            vel_scores = np.array([velocity[layer].get(int(h), 0.0) for h in ids])
            # combine inverse cost with velocity
            weights = (1.0 / costs) + vel_scores
            weights = weights / np.sum(weights)

            idx = np.random.choice(len(ids), p=weights)
            chosen_pt, chosen_id, chi2 = pts[idx], ids[idx], costs[idx]

            # EKF update
            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            hit_ids.append(int(chosen_id))
            total_score += chi2
            G.add_edge((i, tuple(traj[-2])), (i+1, tuple(chosen_pt)), cost=chi2)

        return {'traj': traj, 'hit_ids': hit_ids, 'state': state, 'cov': cov, 'score': total_score, 'graph': G}

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int,int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        """
        Run PSO: particles sample paths, update personal & global bests, adjust velocities.
        """
        self._init_particles(layers)
        global_graph = nx.DiGraph()

        for _ in range(self.n_iters):
            for p in range(self.n_particles):
                path = self._build_path(seed_xyz, t, layers, self.velocities[p])
                # update personal best
                if path['score'] < self.pbest[p]['score']:
                    self.pbest[p] = path
                # update global best
                if path['score'] < self.gbest['score']:
                    self.gbest = path
                global_graph = nx.compose(global_graph, path['graph'])

            # update velocities
            for p in range(self.n_particles):
                for layer in layers:
                    for hid in list(self.velocities[p][layer].keys()):
                        y_p = 1.0 if hid in self.pbest[p]['hit_ids'] else 0.0
                        y_g = 1.0 if hid in self.gbest['hit_ids'] else 0.0
                        v_old = self.velocities[p][layer][hid]
                        r1, r2 = np.random.rand(), np.random.rand()
                        v_new = (self.w * v_old
                                 + self.c1 * r1 * (y_p - v_old)
                                 + self.c2 * r2 * (y_g - v_old))
                        self.velocities[p][layer][hid] = v_new

        result = {
            'traj': self.gbest['traj'],
            'hit_ids': self.gbest['hit_ids'],
            'state': self.gbest['state'],
            'cov': self.gbest['cov'],
            'score': self.gbest['score']
        }
        return [result], global_graph
