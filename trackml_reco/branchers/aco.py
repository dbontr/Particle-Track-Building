import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy.spatial import cKDTree
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFACOBrancher(Brancher):
    r"""
    Ant Colony Optimization (ACO) brancher with EKF propagation (Turbo version).

    This variant is optimized for speed while keeping good physics fidelity.

    Design highlights
    -----------------
    - Vectorized :math:`\chi^2` per layer using a single :math:`\mathbf{S}^{-1}`.
    - No graph construction unless ``plot_tree=True``.
    - Single-batch ACO by default (``n_iters=1``).
    - Min–max pheromone (MMAS) with evaporation, all-ant, elite, and global updates.
    - Optional cheap physics penalties (angle and curvature).

    Parameters
    ----------
    trees : dict[(int, int) -> (cKDTree, ndarray of shape (N, 3), ndarray of shape (N,))]
        KD-trees and hit banks per layer.
    layer_surfaces : dict[(int, int) -> dict]
        Geometry per layer. Either ``{'type':'disk','n':(3,), 'p':(3,)}`` or
        ``{'type':'cylinder','R': float}``.
    noise_std : float, optional
        Measurement noise standard deviation in millimeters. Default is ``2.0``.
    B_z : float, optional
        Magnetic field (Tesla) along :math:`z`. Default is ``0.002``.
    n_ants : int, optional
        Number of ants per batch. Smaller values are faster. Default is ``12``.
    n_iters : int, optional
        Number of ACO batches (passes). Default is ``1``.
    evap_rate : float, optional
        Pheromone evaporation rate in ``(0, 1)``. Default is ``0.5``.
    alpha : float, optional
        Pheromone exponent :math:`\alpha`. Default is ``1.0``.
    beta : float, optional
        Heuristic exponent :math:`\beta` (with :math:`\eta\propto 1/\chi^2`). Default is ``2.0``.
    max_cands : int, optional
        Maximum neighbors to query per layer (passed to base). Default is ``10``.
    step_candidates : int, optional
        Keep at most this many in-gate candidates per layer (cheap beam). Default is ``5``.
    gate_multiplier : float, optional
        Gating constant; radius
        :math:`r_\mathrm{gate}=\text{gate\_multiplier}\,\sqrt{\lambda_{\max}(\mathbf{S})}`.
        Default is ``3.0``.
    tau_min : float, optional
        Minimum pheromone level (MMAS lower bound). Default is ``1e-3``.
    tau_max : float, optional
        Maximum pheromone level (MMAS upper bound). Default is ``5.0``.
    angle_weight : float, optional
        Angle penalty weight in :math:`\chi^2` units; uses :math:`(1-\cos\theta)`. Default ``0.0``.
    curvature_weight : float, optional
        Curvature-change penalty weight in :math:`\chi^2` units; uses :math:`(\Delta\kappa)^2`.
        Default ``0.0``.
    rng_seed : int or None, optional
        Random seed for reproducibility. ``None`` uses a random seed.

    Notes
    -----
    **Measurement model.** With state prediction :math:`\hat{\mathbf{x}}` and covariance
    :math:`\mathbf{P}_\text{pred}`, the innovation covariance
    :math:`\mathbf{S} = \mathbf{H}\mathbf{P}_\text{pred}\mathbf{H}^\top + \mathbf{R}` and
    :math:`\chi^2(\mathbf{z}) = (\mathbf{z}-\hat{\mathbf{x}})^\top \mathbf{S}^{-1} (\mathbf{z}-\hat{\mathbf{x}})`.

    **ACO selection.** For each gated candidate :math:`i`,
    :math:`\eta_i = 1/(\text{cost}_i+\varepsilon)`,
    :math:`\text{desir}_i = \tau_i^\alpha \eta_i^\beta`, and
    :math:`p_i = \text{desir}_i \Big/ \sum_j \text{desir}_j`.
    Here, :math:`\text{cost}=\chi^2 + C_\text{angle} + C_\kappa` (if enabled).

    **Pheromone update (MMAS).** After evaporation
    :math:`\tau \leftarrow (1-\rho)\,\tau`, deposits are applied from all ants,
    the batch elite, and the global-best path; the result is clamped to
    ``[tau_min, tau_max]``.

    See Also
    --------
    run : Execute ACO and return the best branch and (optional) graph.
    """

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 n_ants: int = 12,
                 n_iters: int = 1,
                 evap_rate: float = 0.5,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 max_cands: int = 10,
                 step_candidates: int = 5,
                 gate_multiplier: float = 3.0,
                 tau_min: float = 1e-3,
                 tau_max: float = 5.0,
                 angle_weight: float = 0.0,
                 curvature_weight: float = 0.0,
                 rng_seed: Optional[int] = None) -> None:

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)

        self.layer_surfaces = layer_surfaces
        self.n_ants = int(n_ants)
        self.n_iters = int(n_iters)
        self.evap_rate = float(evap_rate)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.step_candidates = int(step_candidates)
        self.gate_multiplier = float(gate_multiplier)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.angle_weight = float(angle_weight)
        self.curvature_weight = float(curvature_weight)
        self.state_dim = 7
        self.rng = np.random.default_rng(rng_seed)

        # internal holder
        self._gbest: Dict[str, Any] = {'score': np.inf}

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        r"""
        Normalize a vector to unit length.

        Parameters
        ----------
        v : ndarray, shape (n,)
            Input vector.

        Returns
        -------
        ndarray, shape (n,)
            Unit-normalized vector. If the vector norm is less than ``1e-12``,
            the original vector is returned unchanged.

        Notes
        -----
        This function avoids division by zero by using a small norm threshold.
        """
        n = np.linalg.norm(v)
        return v if n < 1e-12 else (v / n)

    def _angle_pen(self, dir_prev: np.ndarray, step_vec: np.ndarray) -> float:
        r"""
        Compute an angular penalty between two step directions.

        Parameters
        ----------
        dir_prev : ndarray, shape (n,)
            Previous step direction vector.
        step_vec : ndarray, shape (n,)
            Candidate step direction vector.

        Returns
        -------
        float
            Angular penalty value. Zero if ``angle_weight`` is less than or equal to zero.

        Notes
        -----
        The penalty is computed as::

            penalty = angle_weight * (1 - \cos(\theta))

        where ``\theta`` is the angle between the two unit-normalized vectors.
        """
        if self.angle_weight <= 0.0:
            return 0.0
        c = float(np.clip(self._unit(dir_prev).dot(self._unit(step_vec)), -1.0, 1.0))
        return self.angle_weight * (1.0 - c)

    def _curv_pen_arr(self, k_prev: float, k_new: np.ndarray) -> np.ndarray:
        r"""
        Compute a curvature change penalty for multiple candidates.

        Parameters
        ----------
        k_prev : float
            Previous curvature value.
        k_new : ndarray, shape (m,)
            New curvature values for each candidate.

        Returns
        -------
        ndarray, shape (m,)
            Curvature penalty for each candidate.

        Notes
        -----
        The penalty is computed as::

            penalty = curvature_weight * (k_new - k_prev)^2

        This discourages large curvature changes between successive track segments.
        """
        if self.curvature_weight <= 0.0:
            return np.zeros_like(k_new, dtype=float)
        dk2 = np.square(k_new - float(k_prev))
        return self.curvature_weight * dk2

    def _gate_r(self, S: np.ndarray) -> float:
        r"""
        Compute the gating radius from a covariance-like matrix.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance matrix.

        Returns
        -------
        float
            Gating radius computed as ``gate_multiplier * \sqrt{\lambda_{\max}}``,
            where ``\lambda_{\max}`` is the largest eigenvalue of ``S``.

        Notes
        -----
        This method provides a robust measure for spatial gating in track finding.
        If the eigenvalue computation fails, a default radius of ``gate_multiplier * 3.0`` is returned.
        """
        # radius = c * sqrt(max eigen) — cheap and robust
        try:
            return float(self.gate_multiplier * np.sqrt(np.max(np.linalg.eigvalsh(S))))
        except Exception:
            return float(self.gate_multiplier * 3.0)

    def _walk_one_ant(self,
                      seed_xyz: np.ndarray,
                      t: np.ndarray,
                      layers: List[Tuple[int, int]],
                      pher: Dict[Tuple[Tuple[int, int], int], float],
                      build_graph: bool) -> Dict[str, Any]:
        r"""
        Perform one ant walk pass through detector layers.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Initial 3D positions for the seed points (used to estimate the initial helix state).
        t : ndarray, shape (n,), optional
            Parameter values along the seed trajectory. If ``None``, a default spacing of 1.0 is used.
        layers : list of tuple of int
            Sequence of ``(volume_id, layer_id)`` pairs representing detector layers to traverse.
        pher : dict
            Pheromone values mapping ``((volume_id, layer_id), hit_id)`` to a float.
        build_graph : bool
            If True, constructs and returns a NetworkX directed graph of the path.

        Returns
        -------
        dict
            Dictionary containing:
            - 'traj' : list of ndarray
                Sequence of 3D points visited.
            - 'hit_ids' : list of int
                IDs of hits selected along the path.
            - 'layers' : list of tuple of int
                Layers corresponding to each selected hit.
            - 'state' : ndarray
                Final state vector of the filter.
            - 'cov' : ndarray
                Final state covariance matrix.
            - 'score' : float
                Cumulative physical cost for the path.
            - 'graph' : nx.DiGraph
                Graph representation of the path (empty if ``build_graph`` is False).

        Notes
        -----
        This implements a single iteration of an Ant Colony Optimization (ACO) style
        search for particle tracking, incorporating:

        - Extended Kalman Filter (EKF) propagation and update.
        - Spatial gating based on the predicted state and innovation covariance.
        - Physical penalties for curvature change and direction change.
        - Pheromone weighting and probabilistic selection of candidates.
        - Optional top-K candidate pruning for efficiency.

        The method proceeds layer-by-layer, updating the state and covariance
        for each chosen hit until no valid candidates remain or all layers are visited.
        """
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj: List[np.ndarray] = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        used_layers: List[Tuple[int, int]] = []
        score = 0.0

        G = nx.DiGraph() if build_graph else None
        prev_dir = seed_xyz[2] - seed_xyz[1]
        prev_k = float(k0)

        H = self.H_jac(None)  # constant 3x7

        for i, layer in enumerate(layers):
            surf = self.layer_surfaces[layer]
            try:
                dt = self._solve_dt_to_surface(state, surf)
            except Exception:
                break

            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R
            S_inv = np.linalg.inv(S)

            # gating
            gate_r = self._gate_r(S)
            pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, gate_r)
            if len(ids) == 0:
                break

            # vectorized χ² and update candidates
            diff = pts - x_pred[:3]                         # (m,3)
            chi2 = np.einsum('ij,jk,ik->i', diff, S_inv, diff)

            # EKF update for each candidate (vectorized)
            K = P_pred @ H.T @ S_inv                        # (7,3)
            x_upds = x_pred + (K @ diff.T).T               # (m,7)
            k_new = x_upds[:, 6]                            # (m,)
            # cheap penalties (optional)
            curv_pen = self._curv_pen_arr(prev_k, k_new)
            if self.angle_weight > 0.0:
                # angle penalty needs a loop (m is tiny: step_candidates <= 5..10)
                angl_pen = np.array([self._angle_pen(prev_dir, d) for d in diff], dtype=float)
            else:
                angl_pen = 0.0

            phys_cost = chi2 + curv_pen + angl_pen

            # keep only top-K by phys_cost to reduce randomness & speed
            if len(phys_cost) > self.step_candidates:
                topk_idx = np.argpartition(phys_cost, self.step_candidates)[:self.step_candidates]
                pts = pts[topk_idx]; ids = ids[topk_idx]; diff = diff[topk_idx]
                x_upds = x_upds[topk_idx]; phys_cost = phys_cost[topk_idx]

            eta = 1.0 / (phys_cost + 1e-6)
            tau = np.array([np.clip(pher.get((layer, int(h)), 1.0), self.tau_min, self.tau_max)
                            for h in ids], dtype=float)
            desir = np.power(tau, self.alpha) * np.power(eta, self.beta)

            if not np.isfinite(desir).any() or desir.sum() <= 0:
                idx = int(np.argmin(phys_cost))
            else:
                probs = desir / desir.sum()
                idx = int(self.rng.choice(len(ids), p=probs))

            # commit chosen
            chosen_pt = pts[idx]; chosen_id = int(ids[idx])
            state = x_upds[idx]
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred
            traj.append(chosen_pt)
            hit_ids.append(chosen_id)
            used_layers.append(layer)
            score += float(phys_cost[idx])
            prev_dir = chosen_pt - traj[-2]
            prev_k = float(state[6])

            if build_graph:
                G.add_edge((i, tuple(traj[-2])), (i + 1, tuple(chosen_pt)), cost=float(phys_cost[idx]))

        return {
            'traj': traj,
            'hit_ids': hit_ids,
            'layers': used_layers,
            'state': state,
            'cov': cov,
            'score': score,
            'graph': (G if build_graph else nx.DiGraph())
        }

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Run fast ACO and return the best single branch.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3,3)
            Three seed hits.
        layers : list of tuple
            Future layer keys (vol, layer) to traverse.
        t : ndarray
            Time stamps for seed points.
        plot_tree : bool, optional
            If True, builds/returns a composed graph; otherwise cheap empty graph.

        Returns
        -------
        branches : list of dict
            Single-element list containing best branch dict with keys:
            'traj', 'hit_ids', 'state', 'cov', 'score'.
        G : nx.DiGraph
            Graph (only populated if plot_tree=True).
        """

        if not layers:
            return [], nx.DiGraph()

        # init pheromone only on requested layers
        pher: Dict[Tuple[Tuple[int, int], int], float] = {}
        for layer in layers:
            _, _, ids = self.trees[layer]
            for hid in ids:
                pher[(layer, int(hid))] = 1.0

        build_graph = bool(plot_tree)
        global_graph = nx.DiGraph() if build_graph else nx.DiGraph()
        self._gbest = {'score': np.inf}

        for _ in range(self.n_iters):
            ants: List[Dict[str, Any]] = [
                self._walk_one_ant(seed_xyz, t, layers, pher, build_graph)
                for _ in range(self.n_ants)
            ]
            if build_graph:
                for a in ants:
                    global_graph = nx.compose(global_graph, a['graph'])

            iter_best = min(ants, key=lambda a: a['score'])
            if iter_best['score'] < self._gbest['score']:
                self._gbest = iter_best

            # evaporate
            for k in pher.keys():
                pher[k] = max(self.tau_min, (1.0 - self.evap_rate) * pher[k])

            # all ants deposit
            for a in ants:
                delta = 1.0 / (a['score'] + 1e-6)
                for layer, hid in zip(a['layers'], a['hit_ids']):
                    key = (layer, int(hid))
                    pher[key] = min(self.tau_max, pher[key] + delta)

            # elite
            delta_e = 1.0 / (iter_best['score'] + 1e-6)
            for layer, hid in zip(iter_best['layers'], iter_best['hit_ids']):
                key = (layer, int(hid))
                pher[key] = min(self.tau_max, pher[key] + delta_e)

            # global
            delta_g = 1.0 / (self._gbest['score'] + 1e-6)
            for layer, hid in zip(self._gbest['layers'], self._gbest['hit_ids']):
                key = (layer, int(hid))
                pher[key] = min(self.tau_max, pher[key] + delta_g)

        best = self._gbest
        result = {k: best[k] for k in ('traj', 'hit_ids', 'state', 'cov', 'score')}
        return [result], global_graph
