import numpy as np
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree
from scipy.optimize import newton
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFGABrancher(Brancher):
    """
    EKF-based track finder using a Genetic Algorithm (GA).
    Evolves a population of candidate hit-sequences via selection, crossover, and mutation.
    """
    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int,int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 pop_size: int = 30,
                 n_gens: int = 20,
                 cx_rate: float = 0.7,
                 mut_rate: float = 0.1,
                 max_cands: int = 10,
                 step_candidates: int = 5):
        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.step_candidates = step_candidates
        self.state_dim = 7

    def _build_path(self,
                    seed_xyz: np.ndarray,
                    t: np.ndarray,
                    layers: List[Tuple[int,int]],
                    hit_sequence: List[int]=None) -> Dict:
        """
        Construct a path and score via EKF propagating through `layers`.
        If `hit_sequence` is provided, try to use those hits in order where valid,
        otherwise pick cheapest candidates.
        """
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        total_score = 0.0
        G = nx.DiGraph()
        seq = hit_sequence or []
        new_sequence: List[int] = []

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
            tree, pts, ids = self.trees[layer]
            # restrict to gate
            idxs = tree.query_ball_point(x_pred[:3], r=gate_r)
            if not idxs:
                break
            cand_pts = pts[idxs]
            cand_ids = ids[idxs]

            # choose hit: from sequence if valid
            if i < len(seq) and seq[i] in cand_ids:
                chosen_idx = int(np.where(cand_ids == seq[i])[0][0])
            else:
                costs = np.array([float((p - x_pred[:3]) @ np.linalg.solve(S, p - x_pred[:3])) for p in cand_pts])
                chosen_idx = int(np.argmin(costs))
            chosen_pt = cand_pts[chosen_idx]
            chosen_id = int(cand_ids[chosen_idx])
            chi2 = float((chosen_pt - x_pred[:3]) @ np.linalg.solve(S, chosen_pt - x_pred[:3]))

            # EKF update
            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            new_sequence.append(chosen_id)
            total_score += chi2
            G.add_edge((i, tuple(traj[-2])), (i+1, tuple(chosen_pt)), cost=chi2)

        return {
            'traj': traj,
            'hit_ids': new_sequence,
            'state': state,
            'cov': cov,
            'score': total_score,
            'graph': G,
            'sequence': new_sequence
        }

    def _select(self, population: List[Dict]) -> List[Dict]:
        new_pop = []
        for _ in population:
            a, b = np.random.choice(population, 2, replace=False)
            new_pop.append(a if a['score'] < b['score'] else b)
        return new_pop

    def _crossover(self, p1_seq: List[int], p2_seq: List[int]) -> Tuple[List[int], List[int]]:
        if np.random.rand() > self.cx_rate or len(p1_seq) < 2 or len(p2_seq) < 2:
            return p1_seq.copy(), p2_seq.copy()
        pt = np.random.randint(1, min(len(p1_seq), len(p2_seq)))
        return p1_seq[:pt] + p2_seq[pt:], p2_seq[:pt] + p1_seq[pt:]

    def _mutate(self,
                sequence: List[int],
                layers: List[Tuple[int,int]]) -> List[int]:
        if np.random.rand() > self.mut_rate or not sequence:
            return sequence
        i = np.random.randint(len(sequence))
        layer = layers[i]
        _, _, all_ids = self.trees[layer]
        alts = [int(h) for h in all_ids if int(h) != sequence[i]]
        if alts:
            sequence[i] = int(np.random.choice(alts))
        return sequence

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int,int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        pop = [self._build_path(seed_xyz, t, layers, []) for _ in range(self.pop_size)]
        combined_graph = nx.DiGraph()

        for _ in range(self.n_gens):
            pop = self._select(pop)
            next_pop = []
            for i in range(0, self.pop_size, 2):
                seq1 = pop[i]['sequence']; seq2 = pop[(i+1)%self.pop_size]['sequence']
                c1, c2 = self._crossover(seq1, seq2)
                c1 = self._mutate(c1, layers)
                c2 = self._mutate(c2, layers)
                next_pop.append(self._build_path(seed_xyz, t, layers, c1))
                next_pop.append(self._build_path(seed_xyz, t, layers, c2))
            pop = next_pop
            for ind in pop:
                combined_graph = nx.compose(combined_graph, ind['graph'])

        best = min(pop, key=lambda x: x['score'])
        result = {k: best[k] for k in ('traj','hit_ids','state','cov','score')}
        return [result], combined_graph