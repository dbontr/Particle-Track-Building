import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy.spatial import cKDTree
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFPSOBrancher(Brancher):
    r"""
    Fast EKF-based track finder using **Particle Swarm Optimization (PSO)**.

    Each particle in the swarm encodes a sequence of hit selections — one per
    detector layer. Paths are built using **EKF propagation** and **velocity-biased
    sampling** from gated hit candidates.

    Speedups in this implementation:
    
    * **Sparse velocities** — track velocities only for hits encountered in gates.
    * **Vectorized per-layer scoring** — single :math:`S^{-1}`, single gain matrix :math:`K`.
    * **No NetworkX overhead** unless ``plot_tree=True``.
    * **Top-K gating** per layer and :math:`\varepsilon`-greedy sampling.
    * **Early stopping** if the global best score stagnates.

    Parameters
    ----------
    trees : dict[(int, int), tuple(cKDTree, ndarray(N,3), ndarray(N,))]
        Mapping ``(volume_id, layer_id) → (tree, points, hit_ids)``.
    layer_surfaces : dict[(int, int), dict]
        Geometry per layer:
        * Disk: ``{'type': 'disk', 'n': normal_vec, 'p': point_on_plane}``
        * Cylinder: ``{'type': 'cylinder', 'R': radius}``
    noise_std : float, optional
        Isotropic measurement noise standard deviation (mm). Default is ``2.0``.
    B_z : float, optional
        Magnetic field along z-axis (Tesla). Default ``0.002``.
    n_particles : int, optional
        Swarm size (number of parallel hypotheses). Default ``16``.
    n_iters : int, optional
        Maximum iterations over the swarm. Default ``8``.
    w : float, optional
        PSO inertia weight. Default ``0.6``.
    c1 : float, optional
        PSO cognitive coefficient (personal influence). Default ``1.0``.
    c2 : float, optional
        PSO social coefficient (global influence). Default ``2.0``.
    max_cands : int, optional
        Max KD-tree neighbors to query before gating. Default ``10``.
    step_candidates : int, optional
        Top-K hits to retain per layer after gating. Default ``6``.
    gate_multiplier : float, optional
        Gate radius factor:

        .. math::

            r = \text{gate\_multiplier} \cdot \sqrt{\lambda_{\max}(S)}

        where :math:`S` is the innovation covariance.
    epsilon_greedy : float, optional
        With probability :math:`\varepsilon`, choose the minimum-\ :math:`\chi^2`
        candidate (exploit); otherwise sample proportionally to weights (explore).
        Default ``0.1``.
    patience : int, optional
        Early stop if the global best score does not improve for this many iterations.
        Default ``3``.
    rng_seed : int or None, optional
        Random number generator seed for reproducibility. Default ``None``.

    Notes
    -----
    * EKF state vector: :math:`[x, y, z, p_x, p_y, p_z, q]`.
    * Per-layer candidate gating uses the Mahalanobis distance:

      .. math::

         \chi^2 = (z - \hat{x})^\mathsf{T} S^{-1} (z - \hat{x})

      with :math:`z` the hit position and :math:`\hat{x}` the predicted position.
    """

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 n_particles: int = 16,
                 n_iters: int = 8,
                 w: float = 0.6,
                 c1: float = 1.0,
                 c2: float = 2.0,
                 max_cands: int = 10,
                 step_candidates: int = 6,
                 gate_multiplier: float = 3.0,
                 epsilon_greedy: float = 0.1,
                 patience: int = 3,
                 rng_seed: Optional[int] = None):

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.w, self.c1, self.c2 = float(w), float(c1), float(c2)
        self.step_candidates = int(step_candidates)
        self.gate_multiplier = float(gate_multiplier)
        self.epsilon_greedy = float(epsilon_greedy)
        self.patience = int(patience)
        self.state_dim = 7
        self.rng = np.random.default_rng(rng_seed)

        # Sparse velocities: list[particle] -> dict[layer] -> dict[hit_id] -> velocity
        self.velocities: List[Dict[Tuple[int, int], Dict[int, float]]] = []
        # Per-particle personal best
        self.pbest: List[Dict[str, Any]] = []
        # Global best
        self.gbest: Dict[str, Any] = {'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}

    def _gate_radius(self, S: np.ndarray) -> float:
        r"""
        Compute gate radius from innovation covariance.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance.

        Returns
        -------
        r : float
            Gate radius:

            .. math::

                r = \text{gate\_multiplier} \cdot \sqrt{\lambda_{\max}(S)}

        Notes
        -----
        Falls back to ``3.0 * gate_multiplier`` if eigenvalue computation fails.
        """
        try:
            return float(self.gate_multiplier * np.sqrt(np.max(np.linalg.eigvalsh(S))))
        except Exception:
            return float(self.gate_multiplier * 3.0)

    def _init_particles(self, layers: List[Tuple[int, int]]) -> None:
        r"""
        Initialize swarm particle data structures.

        Parameters
        ----------
        layers : list of tuple(int, int)
            Ordered layer keys.

        Notes
        -----
        * Velocities are stored sparsely:

          .. code-block:: text

              velocities[particle][layer][hit_id] = velocity_value

        * Personal best (pbest) and global best (gbest) scores start at ``np.inf``.
        """
        self.velocities = [{layer: {} for layer in layers} for _ in range(self.n_particles)]
        self.pbest = [{'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}
                      for _ in range(self.n_particles)]
        self.gbest = {'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}

    def _build_path(self,
                    seed_xyz: np.ndarray,
                    t: np.ndarray,
                    layers: List[Tuple[int, int]],
                    velocity: Dict[Tuple[int, int], Dict[int, float]],
                    build_graph: bool) -> Dict[str, Any]:
        r"""
        Build one track path using EKF updates and PSO velocity-biased sampling.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions :math:`(x, y, z)`.
        t : ndarray
            Seed time points (used for initial velocity estimation).
        layers : list of tuple(int, int)
            Ordered layer keys to traverse.
        velocity : dict
            Sparse velocity map for this particle:

            ``velocity[layer][hit_id] = v``
        build_graph : bool
            If True, return a :class:`networkx.DiGraph` of the chosen edges.

        Returns
        -------
        result : dict
            Contains:
            * ``'traj'`` : list of hit positions (ndarray, shape (3,))
            * ``'hit_ids'`` : list of int
            * ``'state'`` : final EKF state vector
            * ``'cov'`` : final EKF covariance matrix
            * ``'score'`` : accumulated :math:`\chi^2`
            * ``'graph'`` : :class:`networkx.DiGraph`
            * ``'touched'`` : dict mapping layer → list of hit IDs touched

        Notes
        -----
        Sampling rule for hit index :math:`j`:

        .. math::

            w_j &= \frac{1}{\chi^2_j + \epsilon} + \max(0, v_j) \\
            p(j) &= 
            \begin{cases}
                1 & \text{if $\varepsilon$-greedy and $j = \arg\min \chi^2$} \\
                \frac{w_j}{\sum_k w_k} & \text{otherwise}
            \end{cases}
        """
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj: List[np.ndarray] = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        score = 0.0
        touched: Dict[Tuple[int, int], List[int]] = {layer: [] for layer in layers}

        H = self.H_jac(None)
        G = nx.DiGraph() if build_graph else None

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

            # candidates (gated)
            gate_r = self._gate_radius(S)
            pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, gate_r)
            m = len(ids)
            if m == 0:
                break

            # vectorized chi2
            diff = pts - x_pred[:3]                                # (m,3)
            chi2 = np.einsum('ij,jk,ik->i', diff, S_inv, diff)     # (m,)

            # keep only best K by chi2
            if m > self.step_candidates:
                idx = np.argpartition(chi2, self.step_candidates)[:self.step_candidates]
                pts = pts[idx]; ids = ids[idx]; diff = diff[idx]; chi2 = chi2[idx]
                m = len(ids)

            # ensure velocity entries exist (sparse)
            vel_l = velocity[layer]
            vel = np.array([vel_l.get(int(h), 0.0) for h in ids], dtype=float)
            # sampling weights: inverse chi2 + velocity (shift to positive)
            inv = 1.0 / (chi2 + 1e-6)
            wts = inv + np.maximum(0.0, vel)
            if not np.isfinite(wts).any() or wts.sum() <= 0:
                j = int(np.argmin(chi2))
            else:
                if self.rng.random() < self.epsilon_greedy:
                    j = int(np.argmin(chi2))  # exploit best
                else:
                    wts = wts / wts.sum()
                    j = int(self.rng.choice(m, p=wts))

            chosen_pt = pts[j]; chosen_h = int(ids[j])
            # EKF update
            K = P_pred @ H.T @ S_inv
            state = x_pred + K @ diff[j]
            cov = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            hit_ids.append(chosen_h)
            score += float(chi2[j])

            touched[layer].extend(int(h) for h in ids)  # only these will be updated

            if build_graph:
                G.add_edge((i, tuple(traj[-2])), (i + 1, tuple(chosen_pt)), cost=float(chi2[j]))

        return {
            'traj': traj,
            'hit_ids': hit_ids,
            'state': state,
            'cov': cov,
            'score': score,
            'graph': (G if build_graph else nx.DiGraph()),
            'touched': touched
        }

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Run the PSO-based track finding.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions.
        layers : list of tuple(int, int)
            Ordered layer keys to traverse.
        t : ndarray
            Seed "times" for velocity estimation.
        plot_tree : bool, optional
            If True, aggregate per-particle graphs into one output graph.

        Returns
        -------
        branches : list of dict
            Best branch only (single element) with:
            * ``traj`` : list of 3D positions
            * ``hit_ids`` : list[int]
            * ``state`` : final EKF state
            * ``cov`` : final EKF covariance
            * ``score`` : accumulated :math:`\chi^2`
        G : :class:`networkx.DiGraph`
            Aggregated graph (empty if ``plot_tree=False``).

        Notes
        -----
        * Per-iteration steps:

          1. Build paths for all particles.
          2. Update personal bests (pbest) and global best (gbest).
          3. Early stopping if ``patience`` reached.
          4. Update sparse velocities for touched hits only:

             .. math::

                v_{p,h}^{(t+1)} =
                w \, v_{p,h}^{(t)}
                + c_1 r_1 \, (y_p - v_{p,h}^{(t)})
                + c_2 r_2 \, (y_g - v_{p,h}^{(t)})

             where:
             * :math:`y_p = 1` if hit in pbest, else 0
             * :math:`y_g = 1` if hit in gbest, else 0
             * :math:`r_1, r_2 \sim U(0,1)`
        """
        if not layers:
            return [], nx.DiGraph()

        self._init_particles(layers)
        build_graph = bool(plot_tree)
        global_graph = nx.DiGraph() if build_graph else nx.DiGraph()

        best_score_prev = np.inf
        not_improved = 0

        for it in range(self.n_iters):
            # 1) sample paths
            paths: List[Dict[str, Any]] = [
                self._build_path(seed_xyz, t, layers, self.velocities[p], build_graph)
                for p in range(self.n_particles)
            ]
            if build_graph:
                for pth in paths:
                    global_graph = nx.compose(global_graph, pth['graph'])

            # 2) update pbest/gbest
            for p, pth in enumerate(paths):
                if pth['score'] < self.pbest[p]['score']:
                    self.pbest[p] = pth
                if pth['score'] < self.gbest['score']:
                    self.gbest = pth

            # 3) early stop check
            if self.gbest['score'] + 1e-9 < best_score_prev:
                best_score_prev = self.gbest['score']
                not_improved = 0
            else:
                not_improved += 1
                if not_improved >= self.patience:
                    break

            # 4) velocity updates — sparse, only over touched ids this iteration
            for p in range(self.n_particles):
                vel_p = self.velocities[p]
                pbest_hits = set(self.pbest[p]['hit_ids'])
                gbest_hits = set(self.gbest['hit_ids'])
                touched = paths[p]['touched']  # dict[layer] -> list[hit ids]
                for layer, ids in touched.items():
                    if not ids:
                        continue
                    # decay only touched keys
                    for hid in set(ids):
                        v_old = vel_p[layer].get(hid, 0.0)
                        y_p = 1.0 if hid in pbest_hits else 0.0
                        y_g = 1.0 if hid in gbest_hits else 0.0
                        r1 = float(self.rng.random()); r2 = float(self.rng.random())
                        v_new = (self.w * v_old
                                 + self.c1 * r1 * (y_p - v_old)
                                 + self.c2 * r2 * (y_g - v_old))
                        # keep sparse: drop near-zero entries
                        if abs(v_new) < 1e-4:
                            if hid in vel_p[layer]:
                                del vel_p[layer][hid]
                        else:
                            vel_p[layer][hid] = v_new

        best = self.gbest
        result = {k: best[k] for k in ('traj', 'hit_ids', 'state', 'cov', 'score')}
        return [result], global_graph
