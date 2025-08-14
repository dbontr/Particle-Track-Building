import math
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
# networkx is imported lazily only if needed
from trackml_reco.branchers.brancher import Brancher, MathBackend


class HelixEKFACOBrancher(Brancher):
    r"""
    Ant Colony Optimization (ACO) brancher with EKF propagation (Turbo version).

    This brancher performs stochastic path construction across detector layers
    using Ant Colony Optimization (MMAS-style pheromone bounds) while
    propagating a 7-D helical state with an Extended Kalman Filter (EKF).
    Candidates at each layer are χ²-gated against the innovation covariance and
    scored with optional physics penalties before sampling via a stabilized
    softmax over pheromone-weighted desirabilities.

    **State convention.**
    The EKF state is
    :math:`\mathbf{x}=[z, v_x, v_y, v_z, \dots, \kappa]^\top`
    (7-D in total, with curvature :math:`\kappa` as the last component). The
    measurement is the Cartesian hit position :math:`\mathbf{z}\in\mathbb{R}^3`.

    **Measurement model.**
    With predicted state :math:`\hat{\mathbf{x}}` and covariance
    :math:`\mathbf{P}_\text{pred}`, a linearized measurement matrix
    :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}` and noise
    :math:`\mathbf{R}\in\mathbb{R}^{3\times 3}`, the usual EKF quantities are

    .. math::

        \mathbf{S} &= \mathbf{H}\mathbf{P}_\text{pred}\mathbf{H}^\top + \mathbf{R}, \\
        \chi^2(\mathbf{z}) &= (\mathbf{z}-\hat{\mathbf{x}}_{0:3})^\top
                              \mathbf{S}^{-1} (\mathbf{z}-\hat{\mathbf{x}}_{0:3}).

    Gating uses a Mahalanobis ball with radius

    .. math::

        r_\mathrm{gate} = \text{gate\_multiplier}\,\sqrt{\lambda_{\max}(\mathbf{S})}.

    **ACO selection.**
    For each in-gate candidate :math:`i` with "cost"
    :math:`c_i=\chi^2_i + C_{\text{angle},i} + C_{\kappa,i}`,
    the heuristic is :math:`\eta_i = 1/(c_i+\varepsilon)`. Using pheromone
    :math:`\tau_i`, exponents :math:`\alpha,\beta`, the desirability and
    sampling probability are

    .. math::

        \text{desir}_i = \tau_i^\alpha\,\eta_i^\beta, \qquad
        p_i = \frac{\text{desir}_i}{\sum_j \text{desir}_j}.

    **MMAS pheromone update.**
    After evaporation :math:`\tau\leftarrow (1-\rho)\tau`, deposits are applied
    by all ants, the iteration elite, and the global best, and the result is
    clamped to :math:`[\tau_{\min}, \tau_{\max}]`.

    Key optimizations
    -----------------
    - Uses :class:`Brancher` backend kernels (Cholesky) for χ² and Kalman gain
      (no explicit inverses).
    - Top-K gating via :meth:`Brancher._layer_topk_candidates`
      (vectorized, deny-aware).
    - Pheromones stored in per-layer NumPy arrays for fast evaporate/deposit.
    - Vectorized angle & curvature penalties; stable log-softmax sampling.
    - Optional multi-threaded ant walks (``n_jobs``) — NumPy releases the GIL.
    - Allocation minimization: cached :math:`\mathbf{H}`, :math:`\mathbf{I}`,
      dtypes, and small temporaries.

    Parameters
    ----------
    trees : dict[tuple[int, int], tuple(cKDTree, ndarray, ndarray)]
        Mapping from layer key ``(volume_id, layer_id)`` to a tuple
        ``(tree, points, ids)`` where ``tree`` is a :class:`scipy.spatial.cKDTree`
        built over ``points`` of shape ``(N, 3)`` and ``ids`` contains the
        corresponding integer hit identifiers of shape ``(N,)``.
    layer_surfaces : dict[tuple[int, int], dict]
        Geometry per layer. Each value is either
        ``{'type': 'disk', 'n': (3,), 'p': (3,)}`` or
        ``{'type': 'cylinder', 'R': float}``.
    noise_std : float, optional
        Measurement noise (mm). Sets :math:`\mathbf{R}=\sigma^2\mathbf{I}_3`.
        Default is ``2.0``.
    B_z : float, optional
        Magnetic field (Tesla) along :math:`z`. Default is ``0.002``.
    n_ants : int, optional
        Number of ants per iteration/batch. Default is ``12``.
    n_iters : int, optional
        Number of ACO iterations (batches). Default is ``1``.
    evap_rate : float, optional
        Pheromone evaporation rate :math:`\rho\in(0,1)`. Default is ``0.5``.
    alpha : float, optional
        Pheromone exponent :math:`\alpha`. Default is ``1.0``.
    beta : float, optional
        Heuristic exponent :math:`\beta`. Default is ``2.0``.
    max_cands : int, optional
        Maximum neighbors to query per layer (passed to base). Default ``10``.
    step_candidates : int, optional
        Beam size after gating/penalties (max kept per step). Default ``5``.
    gate_multiplier : float, optional
        See gating radius above. Default ``3.0``.
    tau_min : float, optional
        Minimum pheromone (MMAS lower bound). Default ``1e-3``.
    tau_max : float, optional
        Maximum pheromone (MMAS upper bound). Default ``5.0``.
    angle_weight : float, optional
        Angle penalty weight in χ² units, using :math:`(1-\cos\theta)`.
        Default ``0.0``.
    curvature_weight : float, optional
        Curvature-change penalty weight in χ² units, using
        :math:`(\Delta\kappa)^2`. Default ``0.0``.
    rng_seed : int or None, optional
        Random seed for reproducibility. ``None`` uses a random seed.
    n_jobs : int, optional
        Thread pool size for parallel ant walks. ``1`` disables threading.
        Default ``1``.
    backend : MathBackend or None, optional
        Optional backend overriding the one from :class:`Brancher`.

    Attributes
    ----------
    layer_surfaces : dict
        Cached geometry per layer (as passed in).
    _pher : dict[tuple[int, int], ndarray]
        Per-layer pheromone arrays aligned with hit ID order.
    _id2idx : dict[tuple[int, int], dict[int, int]]
        Per-layer maps from hit ID to row index in ``_pher[layer]``.
    _gbest : dict
        Best path observed so far in the current ``run``.

    See Also
    --------
    run : Execute ACO and return the best branch and (optional) a graph.

    Notes
    -----
    - The linearized EKF Jacobians and process noise :math:`\mathbf{Q}_0`
      are supplied by the :class:`Brancher` base via ``compute_F``,
      ``propagate`` and ``Q0``.
    - All per-candidate EKF updates use a Cholesky-based Kalman gain from the
      ``backend`` to avoid explicit matrix inversions.
    """

    __slots__ = (
        "layer_surfaces", "n_ants", "n_iters", "evap_rate", "alpha", "beta",
        "tau_min", "tau_max", "angle_weight", "curvature_weight", "n_jobs",
        "_gbest", "_id2idx", "_pher", "_H", "_I"
    )

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
                 rng_seed: Optional[int] = None,
                 n_jobs: int = 1,
                 backend: Optional[MathBackend] = None) -> None:

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands,
                         step_candidates=step_candidates,
                         gate_multiplier=gate_multiplier,
                         backend=backend)

        self.layer_surfaces = layer_surfaces
        self.n_ants = int(n_ants)
        self.n_iters = int(n_iters)
        self.evap_rate = float(evap_rate)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.angle_weight = float(angle_weight)
        self.curvature_weight = float(curvature_weight)
        self.n_jobs = max(1, int(n_jobs))

        # internal best
        self._gbest: Dict[str, Any] = {'score': np.inf}

        # fast access to H/I/dtype
        self._H = self._H_cached()
        self._I = self._I  # from base (cached identity)

        # Build fast pheromone store: per-layer arrays + id->index maps
        self._id2idx: Dict[Tuple[int, int], Dict[int, int]] = {}
        self._pher: Dict[Tuple[int, int], np.ndarray] = {}
        for layer, (tree, pts, ids) in self.trees.items():
            # map hit id -> row index
            self._id2idx[layer] = {int(h): int(i) for i, h in enumerate(ids)}
            self._pher[layer] = np.full(len(ids), 1.0, dtype=self.dtype)

        # RNG lives in base as self._rng
        self.set_rng(rng_seed)

    @staticmethod
    def _unit_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        r"""
        Normalize rows of a matrix to unit :math:`\ell_2` norm.

        Parameters
        ----------
        M : ndarray, shape (m, d)
            Input matrix; each row is normalized independently.
        eps : float, optional
            Minimum norm used to avoid division by zero. Default ``1e-12``.

        Returns
        -------
        ndarray, shape (m, d)
            Row-normalized matrix :math:`\tilde{M}` with
            :math:`\|\tilde{M}_{i,:}\|_2 = 1` up to ``eps`` floor.
        """
        n = np.linalg.norm(M, axis=1, keepdims=True)
        n = np.maximum(n, eps)
        return M / n

    def _angle_pen_vec(self, prev_dir: np.ndarray, diffs: np.ndarray) -> np.ndarray:
        r"""
        Vectorized angle penalty for multiple candidate step vectors.

        The angle :math:`\theta_i` is computed between the previous direction
        :math:`\mathbf{d}_\text{prev}` and each candidate step
        :math:`\mathbf{d}_i`. The penalty is

        .. math::

            C_{\text{angle},i} = w_\theta\,\bigl(1-\cos\theta_i\bigr), \qquad
            \cos\theta_i =
            \frac{\mathbf{d}_i^\top \mathbf{d}_\text{prev}}
                 {\|\mathbf{d}_i\|_2\,\|\mathbf{d}_\text{prev}\|_2}.

        Parameters
        ----------
        prev_dir : ndarray, shape (3,)
            Previous step direction.
        diffs : ndarray, shape (m, 3)
            Candidate step vectors :math:`\mathbf{d}_i`.

        Returns
        -------
        ndarray, shape (m,)
            Penalty values. Returns zeros if ``angle_weight <= 0``.
        """
        if self.angle_weight <= 0.0:
            return np.zeros(diffs.shape[0], dtype=self.dtype)
        pd = prev_dir.astype(self.dtype, copy=False)
        pdn = pd / max(np.linalg.norm(pd), 1e-12)
        dn = self._unit_rows(diffs.astype(self.dtype, copy=False))
        cos_t = np.clip(dn @ pdn, -1.0, 1.0)
        return self.angle_weight * (1.0 - cos_t)

    def _curv_pen_vec(self, k_prev: float, k_new: np.ndarray) -> np.ndarray:
        r"""
        Vectorized curvature-change penalty.

        Uses

        .. math::

            C_{\kappa,i} = w_\kappa\,(\kappa_i - \kappa_{\text{prev}})^2.

        Parameters
        ----------
        k_prev : float
            Previous curvature :math:`\kappa_{\text{prev}}`.
        k_new : ndarray, shape (m,)
            Candidate curvatures :math:`\kappa_i`.

        Returns
        -------
        ndarray, shape (m,)
            Penalty values. Returns zeros if ``curvature_weight <= 0``.
        """
        if self.curvature_weight <= 0.0:
            return np.zeros_like(k_new, dtype=self.dtype)
        dk = (k_new - float(k_prev)).astype(self.dtype, copy=False)
        return self.curvature_weight * (dk * dk)

    @staticmethod
    def _stable_softmax(logw: np.ndarray) -> np.ndarray:
        r"""
        Numerically stable softmax from log-weights.

        Computes

        .. math::

            p_i = \frac{\exp(\ell_i - \max_j \ell_j)}
                        {\sum_k \exp(\ell_k - \max_j \ell_j)}.

        Parameters
        ----------
        logw : ndarray, shape (m,)
            Log-weights :math:`\ell_i`.

        Returns
        -------
        ndarray, shape (m,)
            Probabilities :math:`\mathbf{p}`. If the sum underflows to zero,
            returns a uniform distribution.
        """
        m = np.max(logw)
        ex = np.exp(logw - m)
        s = ex.sum()
        return ex / s if s > 0 else np.full_like(ex, 1.0 / len(ex))

    def _walk_one_ant(self,
                      seed_xyz: np.ndarray,
                      t: Optional[np.ndarray],
                      layers: List[Tuple[int, int]],
                      rng: np.random.Generator,
                      build_graph: bool) -> Dict[str, Any]:
        r"""
        Single ant pass through ``layers``.

        At each layer:
        (1) geometrically advance to the layer surface by solving for
        :math:`\Delta t`, (2) EKF predict/update to evaluate χ² and produce
        per-candidate updated states, (3) add physics penalties, (4) keep the
        best ``step_candidates`` by cost, and (5) sample one by a stabilized
        softmax over :math:`\log\tau^\alpha + \log\eta^\beta`.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three positions used to initialize a helix (two segments). The
            method estimates an initial velocity and curvature from these.
        t : ndarray or None
            Optional timestamps aligned with ``seed_xyz``. If given, the
            initial :math:`\Delta t` is ``t[1]-t[0]``; otherwise uses ``1.0``.
        layers : list of tuple[int, int]
            Ordered layer keys to traverse.
        rng : numpy.random.Generator
            Random generator used for sampling candidates.
        build_graph : bool
            If ``True``, collect edges for visualization.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'traj'`` : list of ndarray(3,), visited 3D points.
            - ``'hit_ids'`` : list[int], selected hit IDs.
            - ``'layers'`` : list[tuple[int,int]], visited layers.
            - ``'state'`` : ndarray(7,), final EKF state.
            - ``'cov'`` : ndarray(7,7), final covariance.
            - ``'score'`` : float, accumulated cost.
            - ``'edges'`` : list of tuples, only if ``build_graph`` was ``True``;
              each is ``((depth, prev_xyz), (depth+1, xyz), cost)``.

        Notes
        -----
        - The Kalman gain :math:`\mathbf{K}` is computed via the backend
          using a Cholesky factor of :math:`\mathbf{S}` for numerical stability.
        - Covariance update uses the simplified form
          :math:`(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}_\text{pred}` since
          the gain is stable.
        - If no candidates are chosen (e.g., due to gating), the score is set
          to :math:`+\infty` so the path cannot become best.
        """
        dtype = self.dtype
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(np.asarray(seed_xyz, dtype=dtype), dt0, self.B_z)
        state = np.hstack([seed_xyz[2].astype(dtype), v0.astype(dtype), np.array([k0], dtype=dtype)])
        cov = np.eye(self.state_dim, dtype=dtype) * 0.1
        traj: List[np.ndarray] = [seed_xyz[0].astype(dtype), seed_xyz[1].astype(dtype), seed_xyz[2].astype(dtype)]
        hit_ids: List[int] = []
        used_layers: List[Tuple[int, int]] = []
        score = 0.0

        prev_dir = (seed_xyz[2] - seed_xyz[1]).astype(dtype)
        prev_k = float(k0)
        edges: List[Tuple[Any, Any, float]] = []

        H = self._H

        for depth, layer in enumerate(layers):
            # geometric time step to surface
            surf = self.layer_surfaces[layer]
            try:
                dt = self._solve_dt_to_surface(state, surf, dt_init=1.0)
            except Exception:
                break

            # EKF predict
            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R

            # Get top-K gated candidates using Brancher (denial-aware, χ² via Cholesky)
            depth_frac = depth / max(1, len(layers) - 1)
            pts, ids, chi2 = self._layer_topk_candidates(
                x_pred, S, layer, k=max(self.step_candidates * 2, self.step_candidates),
                depth_frac=depth_frac, deny_hits=None, apply_global_deny=True
            )
            if ids.size == 0:
                break

            # Vectorized EKF update for all candidates using fast Kalman gain
            diff = (pts - x_pred[:3]).astype(dtype, copy=False)             # (m,3)
            K = self._backend.kalman_gain(P_pred, H, S).astype(dtype)       # (7,3)
            x_upds = x_pred + (diff @ K.T)                                  # (m,7)
            k_new = x_upds[:, 6]

            # Physics penalties
            phys_cost = chi2.astype(dtype, copy=False)
            if self.curvature_weight > 0.0:
                phys_cost = phys_cost + self._curv_pen_vec(prev_k, k_new)
            if self.angle_weight > 0.0:
                phys_cost = phys_cost + self._angle_pen_vec(prev_dir, diff)

            # Reduce to step_candidates strongest (smallest cost)
            if ids.size > self.step_candidates:
                keep = np.argpartition(phys_cost, self.step_candidates - 1)[:self.step_candidates]
                keep = keep[np.argsort(phys_cost[keep])]
                ids = ids[keep]; pts = pts[keep]; phys_cost = phys_cost[keep]; x_upds = x_upds[keep]

            # MMAS desirability (log-space for stability): log τ^α + log η^β
            # τ comes from per-layer array, indexed by hit ids
            pher_arr = self._pher[layer]
            id2idx = self._id2idx[layer]
            idxs = np.fromiter((id2idx[int(h)] for h in ids), dtype=np.int64, count=len(ids))
            tau = np.clip(pher_arr[idxs], self.tau_min, self.tau_max)
            # η = 1 / (cost + eps)
            eta = 1.0 / (phys_cost + 1e-6)
            log_desir = self.alpha * np.log(tau) + self.beta * np.log(eta)

            # Sample a candidate
            probs = self._stable_softmax(log_desir)
            pick = int(rng.choice(len(ids), p=probs))

            # Commit chosen candidate
            chosen_pt = pts[pick]; chosen_id = int(ids[pick])
            state = x_upds[pick]
            KH = K @ H
            cov = (self._I - KH) @ P_pred  # Joseph not needed; K from Cholesky is stable
            traj.append(chosen_pt)
            hit_ids.append(chosen_id)
            used_layers.append(layer)
            score += float(phys_cost[pick])
            prev_dir = (chosen_pt - traj[-2]).astype(dtype)
            prev_k = float(state[6])

            if build_graph:
                edges.append(((depth, tuple(traj[-2])), (depth + 1, tuple(chosen_pt)), float(phys_cost[pick])))

        # no picks? penalize so it won't become best
        if len(hit_ids) == 0:
            score = float("inf")

        return {
            'traj': traj,
            'hit_ids': hit_ids,
            'layers': used_layers,
            'state': state,
            'cov': cov,
            'score': score,
            'edges': edges,
        }

    def _evaporate(self):
        r"""
        Apply evaporation to all pheromone arrays.

        Performs the MMAS decay

        .. math::

            \tau \leftarrow \max(\tau_{\min}, (1-\rho)\,\tau),

        element-wise for each layer, where :math:`\rho=\text{evap\_rate}`.
        """
        rho = float(self.evap_rate)
        if rho <= 0:
            return
        for layer, arr in self._pher.items():
            np.multiply(arr, (1.0 - rho), out=arr)
            np.maximum(arr, self.tau_min, out=arr)

    def _deposit_path(self, layers: List[Tuple[int, int]], hit_ids: List[int], amount: float):
        r"""
        Deposit pheromone along a path.

        Parameters
        ----------
        layers : list[tuple[int, int]]
            Per-step layer keys aligned with ``hit_ids``.
        hit_ids : list[int]
            Selected hit IDs for the path.
        amount : float
            Deposit amount added to each visited hit on its layer, before
            clamping to :math:`[\tau_{\min}, \tau_{\max}]`.

        Notes
        -----
        This is implemented as a vector scatter add into the per-layer arrays
        using the precomputed :attr:`_id2idx` maps.
        """
        # Vector scatter add into per-layer pheromone arrays
        for layer, hid in zip(layers, hit_ids):
            idx = self._id2idx[layer].get(int(hid), None)
            if idx is not None:
                self._pher[layer][idx] = min(self.tau_max, self._pher[layer][idx] + amount)

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: Optional[np.ndarray] = None,
            plot_tree: bool = False) -> Tuple[List[Dict[str, Any]], Any]:
        r"""
        Execute ACO and return the best branch and (optionally) a graph.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed points used to initialize helical motion and curvature.
        layers : list[tuple[int, int]]
            Ordered layer keys to traverse.
        t : ndarray or None, optional
            Optional timestamps aligned with ``seed_xyz``. See
            :meth:`_walk_one_ant` for how this is used.
        plot_tree : bool, optional
            If ``True``, also return a directed graph of sampled edges suitable
            for visualization with :mod:`networkx`.

        Returns
        -------
        results : list[dict]
            A singleton list containing the best result with keys
            ``'traj'``, ``'hit_ids'``, ``'state'``, ``'cov'``, ``'score'``.
        G : networkx.DiGraph or None
            If ``plot_tree`` is ``True``, a graph with edges
            ``u -> v`` annotated by ``cost``; otherwise ``None``.

        Notes
        -----
        - The algorithm runs for ``n_iters`` iterations. In each iteration,
          ``n_ants`` ant walks are executed (in parallel if ``n_jobs>1``).
        - Pheromone updates include: all ants (``1/score``), iteration-best,
          and global-best deposits, followed by global clamping to
          :math:`[\tau_{\min}, \tau_{\max}]`.
        """
        if not layers:
            if plot_tree:
                import networkx as nx  # lazy
                return [], nx.DiGraph()
            return [], None

        # reset global best
        self._gbest = {'score': np.inf}

        # run iterations
        build_graph = bool(plot_tree)
        all_edges: List[Tuple[Any, Any, float]] = []

        for _ in range(self.n_iters):
            # independent RNG for each ant for reproducibility
            ant_rngs = [np.random.default_rng(int(self._rng.integers(0, 2**32 - 1))) for _ in range(self.n_ants)]

            if self.n_jobs > 1 and self.n_ants > 1:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                    ants = list(ex.map(lambda args: self._walk_one_ant(*args),
                                       [(seed_xyz, t, layers, r, build_graph) for r in ant_rngs]))
            else:
                ants = [self._walk_one_ant(seed_xyz, t, layers, r, build_graph) for r in ant_rngs]

            if build_graph:
                all_edges.extend(e for a in ants for e in a['edges'])

            # pick iteration best and update global best
            iter_best = min(ants, key=lambda a: a['score'])
            if iter_best['score'] < self._gbest.get('score', float("inf")):
                self._gbest = iter_best

            # pheromone updates
            self._evaporate()

            # all ants deposit (1 / score)
            for a in ants:
                if math.isfinite(a['score']) and a['score'] > 0:
                    self._deposit_path(a['layers'], a['hit_ids'], 1.0 / (a['score'] + 1e-6))

            # elite & global deposits
            if math.isfinite(iter_best['score']) and iter_best['score'] > 0:
                self._deposit_path(iter_best['layers'], iter_best['hit_ids'], 1.0 / (iter_best['score'] + 1e-6))
            if math.isfinite(self._gbest['score']) and self._gbest['score'] > 0:
                self._deposit_path(self._gbest['layers'], self._gbest['hit_ids'], 1.0 / (self._gbest['score'] + 1e-6))

            # clamp τ to [tau_min, tau_max]
            for arr in self._pher.values():
                np.clip(arr, self.tau_min, self.tau_max, out=arr)

        best = self._gbest
        result = {k: best[k] for k in ('traj', 'hit_ids', 'state', 'cov', 'score')}

        if plot_tree:
            import networkx as nx  # lazy
            G = nx.DiGraph()
            for u, v, w in all_edges:
                G.add_edge(u, v, cost=w)
            return [result], G

        return [result], None