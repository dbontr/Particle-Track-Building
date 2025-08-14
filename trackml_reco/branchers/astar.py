import heapq
from typing import Tuple, Dict, List, Any, Optional
import numpy as np

# networkx is imported lazily only if plot_tree=True
from scipy.spatial import cKDTree
from trackml_reco.branchers.brancher import Brancher
from trackml_reco.ekf_kernels import kalman_gain  # Cholesky-based, no inverses


class HelixEKFAStarBrancher(Brancher):
    r"""
    EKF-based A* brancher with cross-track suppression and flip damping (Turbo).

    This brancher performs deterministic best-first (A*) search over ordered
    detector layers while propagating a 7-D helical state with an Extended
    Kalman Filter (EKF). At each layer, a vectorized candidate set is produced
    by χ²-gating of KD-tree neighbors; per-candidate updates use a **single**
    Kalman gain for the layer. Candidates are scored by measurement χ² plus
    physics-informed penalties (perpendicular, angle, curvature, monotonic
    trend, and flip damping), then pruned to a fixed **beam** before A* expansion.

    **State / measurement convention.**
    The EKF state is
    :math:`\mathbf{x}=[x, y, z, v_x, v_y, v_z, \kappa]^\top\in\mathbb{R}^7`.
    The measurement is the Cartesian hit position :math:`\mathbf{z}\in\mathbb{R}^3`
    with linearized measurement matrix :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}`.
    With predicted state :math:`\hat{\mathbf{x}}` and covariance
    :math:`\mathbf{P}_\text{pred}`,

    .. math::

        \mathbf{S} &= \mathbf{H}\mathbf{P}_\text{pred}\mathbf{H}^\top + \mathbf{R}, \\
        \chi^2(\mathbf{z}) &= (\mathbf{z}-\hat{\mathbf{x}}_{0:3})^\top
                              \mathbf{S}^{-1}(\mathbf{z}-\hat{\mathbf{x}}_{0:3}), \\
        \mathbf{K} &= \texttt{kalman\_gain}(\mathbf{P}_\text{pred},\mathbf{H},\mathbf{S}) \quad
                     \text{(Cholesky solve, no inverses)}.

    The update for a candidate residual :math:`\mathbf{r}=\mathbf{z}-\hat{\mathbf{x}}_{0:3}` is

    .. math::

        \mathbf{x}^+ = \hat{\mathbf{x}} + \mathbf{K}\,\mathbf{r}, \qquad
        \mathbf{P}^+ \approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}_\text{pred}.

    **Physics penalties.** Let :math:`\hat{\mathbf{t}}` be a smoothed tangent
    direction and :math:`\mathbf{r}` a candidate residual. The penalties are:

    - *Perpendicular* (cross-track) suppression:

      .. math:: C_\perp = \frac{w_\perp}{\sigma^2}\,\bigl\|\mathbf{r}_\perp\bigr\|_2^2, \quad
                \mathbf{r}_\perp = \mathbf{r} - (\mathbf{r}\!\cdot\!\hat{\mathbf{t}})\,\hat{\mathbf{t}}, \quad
                \sigma^2=\tfrac{1}{3}\mathrm{tr}(\mathbf{S}).

    - *Angle* penalty:

      .. math:: C_\theta = w_\theta\,\bigl(1-\cos\theta\bigr)^2,\quad
                \cos\theta = \frac{\mathbf{r}^\top\hat{\mathbf{t}}}{\|\mathbf{r}\|\,\|\hat{\mathbf{t}}\|}.

    - *Curvature-change* penalty:

      .. math:: C_\kappa = w_\kappa\,(\kappa^+ - \kappa_\text{prev})^2.

    - *Monotonic trend* penalty: encourage consistent progress along :math:`z`
      (for disks) or radius :math:`r=\sqrt{x^2+y^2}` (for cylinders):

      .. math:: C_\text{trend} = w_\text{trend}\,|\Delta| \cdot \mathbb{1}\{\text{wrong sign}\}.

    - *Flip damping*: discourage oscillation of the along-track component
      :math:`r_\parallel=\mathbf{r}\!\cdot\!\hat{\mathbf{t}}`:

      .. math:: C_\text{flip} = w_\text{flip}\,|r_\parallel| \cdot
                \mathbb{1}\{\operatorname{sign}(r_\parallel)\neq \operatorname{sign}(r_{\parallel,\text{prev}})\}.

    **A\* objective.** Each partial branch has
    :math:`f = g + h`, where :math:`g` is the accumulated local cost
    (χ² + penalties) and :math:`h` is a light-weight heuristic based on a
    one-step prediction to the goal surface (see :meth:`_heur_to_goal`).

    Optimizations
    -------------
    • Vectorized per-layer candidate scoring:
      χ² (from base) + perpendicular, angle, curvature, trend, flip penalties.  
    • Single-Gain EKF updates: compute :math:`\mathbf{K}` once per layer, update all candidates.  
    • Top-K via argpartition; small beam for fast, robust search.  
    • Tapered gate near the end of the layer list (less branching).  
    • Heuristic caching on quantized position to cut repeated surface solves.  
    • Deny-list respected (pass ``deny_hits`` to :meth:`run`).  

    Parameters
    ----------
    trees, layer_surfaces, noise_std, B_z : see :class:`Brancher` and notes below
    max_cands : int
        KD-tree preselect per layer (passed to base).
    step_candidates : int
        Max candidates retained *per layer* before the A* beam (default ``5``).
    gate_multiplier : float
        Gating radius multiplier used by :meth:`Brancher._layer_topk_candidates`
        (default ``3.0``).
    beam_width : int
        A* beam width (nodes expanded per layer; default ``8``).
    angle_weight, curvature_weight, trend_weight : float
        Physics penalty weights (χ² units).
    switch_margin : float
        Require at least this much :math:`g`-improvement to revisit the same
        (layer, hit) pair.
    perp_weight, flip_weight, ema_alpha, theta_max, taper_last_frac, min_gate_multiplier : float
        Extra physics knobs and gate tapering. See field docs below.

    Attributes
    ----------
    layer_surfaces : dict
        Geometry per layer; either ``{'type':'disk','n':(3,), 'p':(3,)}`` or
        ``{'type':'cylinder','R': float}``.
    step_candidates : int
        Max candidates kept per layer before beam pruning.
    beam_width : int
        Beam width for A*.
    ema_alpha : float
        Exponential smoothing factor for the tangent
        :math:`\hat{\mathbf{t}} \leftarrow (1-\alpha)\hat{\mathbf{t}} + \alpha\,\mathbf{v}`.
    theta_max : float
        Maximum substep turn angle (rad) used by :meth:`_substep_predict`.
    taper_last_frac : float
        Fraction of terminal layers over which the gating multiplier is tapered.
    min_gate_multiplier : float
        Floor for the tapered gating multiplier.
    _heur_cache : dict
        Cache for heuristic values keyed by quantized position and remaining depth.
    _dtype : numpy dtype
        Numeric dtype inferred from the first layer's points.

    See Also
    --------
    :class:`Brancher` : Gating, F/propagation, geometry utilities.  
    :func:`trackml_reco.ekf_kernels.kalman_gain` : Cholesky-based EKF gain.  
    run : Main A* search procedure.

    Notes
    -----
    - All χ² evaluations and gating come from :meth:`Brancher._layer_topk_candidates`,
      which uses a Cholesky solve (no inverses).
    - The single-layer :math:`\mathbf{K}` is reused for all candidate updates
      at that layer to reduce repeated linear algebra.
    - Covariance updates use :math:`(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}`
      (Joseph form omitted for speed; Cholesky gain is numerically stable).
    """

    __slots__ = (
        "layer_surfaces", "step_candidates", "gate_multiplier", "beam_width",
        "angle_weight", "curvature_weight", "trend_weight", "switch_margin",
        "perp_weight", "flip_weight", "ema_alpha",
        "theta_max", "taper_last_frac", "min_gate_multiplier",
        "_heur_cache", "_dtype"
    )

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 max_cands: int = 10,
                 step_candidates: int = 5,
                 gate_multiplier: float = 3.0,
                 beam_width: int = 8,
                 angle_weight: float = 8.0,
                 curvature_weight: float = 200.0,
                 trend_weight: float = 6.0,
                 switch_margin: float = 1.5,
                 perp_weight: float = 12.0,
                 flip_weight: float = 5.0,
                 ema_alpha: float = 0.5,
                 theta_max: float = 0.35,
                 taper_last_frac: float = 0.30,
                 min_gate_multiplier: float = 1.2) -> None:

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands,
                         step_candidates=step_candidates)

        self.layer_surfaces = layer_surfaces
        self.step_candidates = int(step_candidates)
        self.gate_multiplier = float(gate_multiplier)
        self.beam_width = int(beam_width)

        self.angle_weight = float(angle_weight)
        self.curvature_weight = float(curvature_weight)
        self.trend_weight = float(trend_weight)
        self.switch_margin = float(switch_margin)

        self.perp_weight = float(perp_weight)
        self.flip_weight = float(flip_weight)
        self.ema_alpha = float(ema_alpha)

        self.theta_max = float(theta_max)
        self.taper_last_frac = float(taper_last_frac)
        self.min_gate_multiplier = float(min_gate_multiplier)

        self._heur_cache: Dict[Tuple[int, int, int, int], float] = {}

        # pick a consistent dtype from the first layer
        try:
            any_layer = next(iter(trees))
            self._dtype = trees[any_layer][1].dtype
        except Exception:
            self._dtype = np.float64

    @staticmethod
    def _unit_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        r"""
        Normalize rows of a matrix to unit :math:`\ell_2` norm (safe).

        Parameters
        ----------
        M : ndarray, shape (m, d)
            Input matrix whose rows are normalized independently.
        eps : float, optional
            Minimum norm used to avoid division by zero (default ``1e-12``).

        Returns
        -------
        ndarray, shape (m, d)
            Row-normalized matrix :math:`\tilde{M}` with
            :math:`\|\tilde{M}_{i,:}\|_2 \approx 1`.
        """
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, eps)

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        r"""
        Normalize a vector to unit :math:`\ell_2` norm (safe).

        Parameters
        ----------
        v : ndarray, shape (d,)
            Input vector.
        eps : float, optional
            Small threshold below which the vector is returned unchanged.

        Returns
        -------
        ndarray, shape (d,)
            Unit vector :math:`\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|_2`
            or ``v`` if :math:`\|\mathbf{v}\|_2<\text{eps}`.
        """
        n = np.linalg.norm(v)
        return v if n < eps else (v / n)

    def _angle_pen_vec(self, t_new: np.ndarray, diff: np.ndarray) -> np.ndarray:
        r"""
        Angle penalty for multiple candidates (vectorized).

        Uses

        .. math::

            C_{\theta,i} = w_\theta\,\bigl(1-\cos\theta_i\bigr)^2, \qquad
            \cos\theta_i = \frac{\hat{\mathbf{t}}_i^\top \hat{\mathbf{d}}_i}
                                {\|\hat{\mathbf{t}}_i\|_2\,\|\hat{\mathbf{d}}_i\|_2},

        where :math:`\hat{\mathbf{t}}_i` is the smoothed tangent for candidate
        :math:`i` and :math:`\hat{\mathbf{d}}_i` is the normalized residual.

        Parameters
        ----------
        t_new : ndarray, shape (m, 3)
            Candidate tangent directions (before normalization is okay).
        diff : ndarray, shape (m, 3)
            Residuals :math:`\mathbf{r}_i = \mathbf{z}_i - \hat{\mathbf{x}}_{0:3}`.

        Returns
        -------
        ndarray, shape (m,)
            Penalty values. All zeros if ``angle_weight <= 0``.
        """
        if self.angle_weight <= 0.0:
            return np.zeros(diff.shape[0], dtype=self._dtype)
        t_hat = self._unit_rows(t_new.astype(self._dtype, copy=False))
        d_hat = self._unit_rows(diff.astype(self._dtype, copy=False))
        c = np.einsum('ij,ij->i', t_hat, d_hat).clip(-1.0, 1.0)
        return self.angle_weight * (1.0 - c) ** 2

    def _curv_pen_vec(self, k_prev: float, k_new: np.ndarray) -> np.ndarray:
        r"""
        Curvature-change penalty (vectorized).

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
            Penalty values. All zeros if ``curvature_weight <= 0``.
        """
        if self.curvature_weight <= 0.0:
            return np.zeros_like(k_new, dtype=self._dtype)
        dk = (k_new - float(k_prev)).astype(self._dtype, copy=False)
        return self.curvature_weight * dk * dk

    def _perp_pen_vec(self, S: np.ndarray, t_new: np.ndarray, diff: np.ndarray) -> np.ndarray:
        r"""
        Cross-track (perpendicular) penalty (vectorized).

        Decompose residuals into along/orthogonal components relative to the
        smoothed tangent :math:`\hat{\mathbf{t}}`. Penalize the perpendicular
        part scaled by an isotropic variance proxy :math:`\sigma^2`:

        .. math::

            \mathbf{r}_{\perp,i} &= \mathbf{r}_i -
                (\mathbf{r}_i^\top \hat{\mathbf{t}}_i)\,\hat{\mathbf{t}}_i, \\
            C_{\perp,i} &= \frac{w_\perp}{\sigma^2}\,\|\mathbf{r}_{\perp,i}\|_2^2, \qquad
            \sigma^2 = \tfrac{1}{3}\operatorname{tr}(\mathbf{S}).

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance for the layer.
        t_new : ndarray, shape (m, 3)
            Candidate tangents.
        diff : ndarray, shape (m, 3)
            Candidate residuals.

        Returns
        -------
        ndarray, shape (m,)
            Penalty values. All zeros if ``perp_weight <= 0``.
        """
        if self.perp_weight <= 0.0:
            return np.zeros(diff.shape[0], dtype=self._dtype)
        t_hat = self._unit_rows(t_new.astype(self._dtype, copy=False))
        r_along = np.einsum('ij,ij->i', diff, t_hat)            # (m,)
        r_perp = diff - r_along[:, None] * t_hat                # (m,3)
        r2 = np.einsum('ij,ij->i', r_perp, r_perp)
        sigma2 = float(np.trace(S)) / 3.0 + 1e-12
        return (self.perp_weight / sigma2) * r2

    def _trend_pen_vec(self,
                       layer_key: Tuple[int, int],
                       x_pred: np.ndarray,
                       v_tan: np.ndarray,
                       pts: np.ndarray) -> np.ndarray:
        r"""
        Monotonic trend penalty (vectorized).

        Encourage consistent movement:
        - On **disks**, along :math:`z` with sign given by :math:`v_z`.
        - On **cylinders**, in radius :math:`r=\sqrt{x^2+y^2}` with sign given
          by :math:`\dot{r} = (x v_x + y v_y)/\max(r, \varepsilon)`.

        .. math::

            C_{\text{trend},i} = w_{\text{trend}}\,|\Delta_i|\cdot
            \mathbb{1}\{\text{sign}(\Delta_i)\ \text{disagrees with predicted sign}\},

        where :math:`\Delta_i` is either :math:`z_i - z_\text{pred}` (disk)
        or :math:`r_i - r_\text{pred}` (cylinder).

        Parameters
        ----------
        layer_key : tuple[int, int]
            Current layer identifier.
        x_pred : ndarray, shape (7,)
            Predicted state before update.
        v_tan : ndarray, shape (3,)
            Representative tangent direction (e.g., mean of candidate velocities).
        pts : ndarray, shape (m, 3)
            Candidate hit positions.

        Returns
        -------
        ndarray, shape (m,)
            Penalty values. All zeros if ``trend_weight <= 0``.
        """
        if self.trend_weight <= 0.0:
            return np.zeros(pts.shape[0], dtype=self._dtype)
        surf = self.layer_surfaces[layer_key]
        if surf['type'] == 'disk':
            s = np.sign(v_tan[2]) if abs(v_tan[2]) > 1e-9 else 0.0
            if s == 0.0:
                return np.zeros(pts.shape[0], dtype=self._dtype)
            dpz = pts[:, 2] - x_pred[2]
            wrong = (np.sign(dpz) != s)
            return self.trend_weight * np.abs(dpz) * wrong.astype(self._dtype)
        # cylinder
        x0, y0 = float(x_pred[0]), float(x_pred[1])
        r0 = float(np.hypot(x0, y0))
        vr = (x0 * v_tan[0] + y0 * v_tan[1]) / max(r0, 1e-9)
        s = np.sign(vr) if abs(vr) > 1e-9 else 0.0
        if s == 0.0:
            return np.zeros(pts.shape[0], dtype=self._dtype)
        r1 = np.hypot(pts[:, 0], pts[:, 1])
        dr = r1 - r0
        wrong = (np.sign(dr) != s)
        return self.trend_weight * np.abs(dr) * wrong.astype(self._dtype)

    @staticmethod
    def _softmax_weights(w: np.ndarray) -> np.ndarray:
        r"""
        Numerically stable softmax over weights (optional tie-breaking).

        Computes

        .. math::

            p_i = \frac{\exp(w_i - \max_j w_j)}{\sum_k \exp(w_k - \max_j w_j)}.

        Parameters
        ----------
        w : ndarray, shape (m,)
            Unnormalized weights.

        Returns
        -------
        ndarray, shape (m,)
            Probabilities. If the sum underflows to zero, returns a uniform
            distribution.
        """
        m = np.max(w)
        e = np.exp(w - m)
        s = e.sum()
        return e / s if s > 0 else np.full_like(w, 1.0 / len(w))

    def _substep_predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        EKF predict with adaptive sub-stepping when turn angle is large.

        Let :math:`p_T=\sqrt{v_x^2+v_y^2}` and define a local angular rate
        :math:`\omega = B_z\,\kappa\,p_T`. If :math:`|\omega\,\Delta t| >
        \theta_\max`, the step is split into :math:`n=\lceil |\omega\Delta t| /
        \theta_\max \rceil` sub-steps of size :math:`h=\Delta t/n`. For each
        sub-step:

        .. math::

            \mathbf{F}_k &= \texttt{compute\_F}(\mathbf{x}_k, h), \\
            \mathbf{x}_{k+1} &= \texttt{propagate}(\mathbf{x}_k, h), \\
            \mathbf{P}_{k+1} &= \mathbf{F}_k \mathbf{P}_k \mathbf{F}_k^\top + \mathbf{Q}_0\,h.

        Parameters
        ----------
        x : ndarray, shape (7,)
            Current state.
        P : ndarray, shape (7, 7)
            Current covariance.
        dt : float
            Time step to propagate.

        Returns
        -------
        x2 : ndarray, shape (7,)
            Predicted state after (sub-)stepping.
        P2 : ndarray, shape (7, 7)
            Predicted covariance after (sub-)stepping.
        """
        vx, vy = x[3], x[4]
        pT = float(np.hypot(vx, vy))
        omega = float(self.B_z * x[6] * pT)
        turn = abs(omega * dt)
        if turn <= self.theta_max:
            F = self.compute_F(x, dt)
            x2 = self.propagate(x, dt)
            P2 = F @ P @ F.T + self.Q0 * dt
            return x2, P2
        n = int(np.ceil(turn / self.theta_max))
        h = dt / n
        xk, Pk = x, P
        for _ in range(n):
            F = self.compute_F(xk, h)
            xk = self.propagate(xk, h)
            Pk = F @ Pk @ F.T + self.Q0 * h
        return xk, Pk

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: Optional[np.ndarray],
            plot_tree: bool = False,
            **kwargs) -> Tuple[List[Dict[str, Any]], Any]:
        r"""
        Execute A* with EKF propagation and physics-aware scoring.

        The search proceeds layer-by-layer. For the current node:
        (1) solve geometry for :math:`\Delta t` to the layer surface,
        (2) predict with :meth:`_substep_predict`,
        (3) obtain χ²-gated top candidates with
            :meth:`Brancher._layer_topk_candidates`,
        (4) apply vectorized EKF updates using a single layer gain
            :math:`\mathbf{K}`,
        (5) add physics penalties and keep a beam of the lowest totals,
        (6) push successors with :math:`f=g+h` into the priority queue.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed positions used to initialize a helix (two segments).
        layers : list[tuple[int, int]]
            Ordered layer keys defining the traversal.
        t : ndarray or None
            Optional timestamps aligned with ``seed_xyz``. If provided, the
            initial :math:`\Delta t` is ``t[1]-t[0]``; otherwise ``1.0``.
        plot_tree : bool, optional
            If ``True``, also build and return a :class:`networkx.DiGraph` of
            explored edges annotated with penalty components.
        **kwargs
            Optional:
                ``deny_hits`` : iterable of hit IDs to exclude from selection.

        Returns
        -------
        results : list[dict]
            A singleton list containing the best (or best partial) branch with
            keys ``'traj'``, ``'state'``, ``'cov'``, ``'score'``, ``'hit_ids'``.
        G : networkx.DiGraph or None
            If ``plot_tree`` is ``True``, the exploration graph; else ``None``.

        Notes
        -----
        - **Beam & gating.** The per-layer gate is tapered for the last
          ``taper_last_frac`` fraction of layers:

          .. math:: \text{gate\_mul} \in
                    \bigl[\text{min\_gate\_multiplier},\ \text{gate\_multiplier}\bigr],

          decreasing linearly toward the end to reduce branching.
        - **Switch margin.** A candidate that revisits the same (layer, hit)
          is only accepted if it improves :math:`g` by at least
          :attr:`switch_margin`.
        - If the search terminates without reaching the final layer, the best
          partial node by :math:`g` is returned.
        """
        if not layers:
            if plot_tree:
                import networkx as nx
                return [], nx.DiGraph()
            return [], None

        deny_hits = kwargs.get('deny_hits', None)
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz.astype(self._dtype), dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2].astype(self._dtype), v0.astype(self._dtype), np.array([k0], dtype=self._dtype)])
        P0 = np.eye(self.state_dim, dtype=self._dtype) * 0.1

        # heuristic target (best-effort)
        try:
            dt_goal = self._solve_dt_to_surface(x0, self.layer_surfaces[layers[-1]])
            xg, _ = self._substep_predict(x0, P0, dt_goal)
            goal_pos = xg[:3]
        except Exception:
            goal_pos = seed_xyz[2].astype(self._dtype)

        # graph (lazy import)
        G = None
        if plot_tree:
            import networkx as nx
            G = nx.DiGraph()

        # A* structures
        open_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        g_best: Dict[Tuple[int, int], float] = {(0, -1): 0.0}  # (layer_idx, last_hit_id) -> best g
        counter = 0

        tan0 = self._unit(v0.astype(self._dtype))
        start = {
            'layer_idx': 0,
            'state': x0,
            'cov': P0,
            'pos': seed_xyz[2].astype(self._dtype),
            'traj': [seed_xyz[0].astype(self._dtype), seed_xyz[1].astype(self._dtype), seed_xyz[2].astype(self._dtype)],
            'g': 0.0,
            'h': float(self._heur_to_goal(x0, P0, goal_pos, layers[-1], len(layers))),
            'hit_ids': [],
            'tan_ema': tan0,
            'prev_along': None
        }
        heapq.heappush(open_heap, (start['g'] + start['h'], counter, start))
        counter += 1

        N = len(layers)
        expansions = 0
        max_exp = 40000  # safety valve

        H = self.H_jac(None)  # constant 3x7

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            expansions += 1
            if expansions > max_exp:
                break

            i = int(cur['layer_idx'])
            if i >= N:
                branch = {
                    'traj': cur['traj'],
                    'state': cur['state'],
                    'cov': cur['cov'],
                    'score': cur['g'],
                    'hit_ids': cur['hit_ids']
                }
                return [branch], (G if plot_tree else None)

            layer_key = layers[i]
            surf = self.layer_surfaces[layer_key]

            # geometric time step
            try:
                dt = self._solve_dt_to_surface(cur['state'], surf)
            except Exception:
                continue

            # predict with possible sub-steps
            x_pred, P_pred = self._substep_predict(cur['state'], cur['cov'], dt)
            S = H @ P_pred @ H.T + self.R

            # depth-aware gate taper
            progress = i / max(N, 1)
            gate_mul = self.gate_multiplier
            if progress >= (1.0 - self.taper_last_frac):
                frac = (progress - (1.0 - self.taper_last_frac)) / max(self.taper_last_frac, 1e-9)
                gate_mul = max(self.min_gate_multiplier, self.gate_multiplier * (1.0 - 0.5 * frac))

            # get top-k gated candidates (already χ²-sorted)
            depth_frac = (i + 1) / max(N, 1)
            pts, ids, chi2 = self._layer_topk_candidates(
                x_pred, S, layer_key,
                k=max(1, min(self.step_candidates, 32)),
                depth_frac=depth_frac,
                gate_mul=gate_mul,
                gate_tighten=0.15,
                deny_hits=deny_hits
            )
            if ids.size == 0:
                continue

            # vectorized EKF update for these candidates
            diff = (pts - x_pred[:3]).astype(self._dtype, copy=False)      # (m,3)
            K = kalman_gain(P_pred, H, S).astype(self._dtype)              # (7,3), stable (Cholesky)
            x_upds = x_pred + diff @ K.T                                   # (m,7)

            v_news = x_upds[:, 3:6]
            t_news = self._unit_rows((1.0 - self.ema_alpha) * cur['tan_ema'][None, :] + self.ema_alpha * v_news)

            # penalties (vectorized)
            perp_pen = self._perp_pen_vec(S, t_news, diff)
            ang_pen = self._angle_pen_vec(t_news, diff)
            curv_pen = self._curv_pen_vec(float(cur['state'][6]), x_upds[:, 6])
            trend_pen = self._trend_pen_vec(layer_key, x_pred, v_news.mean(axis=0), pts)  # v_tan: any consistent tangent; use mean

            if cur['prev_along'] is None or self.flip_weight <= 0.0:
                flip_pen = np.zeros(ids.size, dtype=self._dtype)
                r_along = np.einsum('ij,ij->i', diff, t_news)
            else:
                r_along = np.einsum('ij,ij->i', diff, t_news)
                flips = (np.sign(r_along) != np.sign(cur['prev_along'])) & (r_along != 0.0)
                flip_pen = self.flip_weight * np.abs(r_along) * flips.astype(self._dtype)

            # total local cost per candidate
            total = chi2.astype(self._dtype, copy=False) + perp_pen + ang_pen + curv_pen + trend_pen + flip_pen

            # keep beam of size <= beam_width
            bw = max(1, min(self.beam_width, total.size))
            if total.size > bw:
                keep = np.argpartition(total, bw - 1)[:bw]
                keep = keep[np.argsort(total[keep])]
            else:
                keep = np.argsort(total)

            for j in keep:
                hid = int(ids[j])
                z = pts[j]
                xj = x_upds[j]
                # Joseph form omitted for speed; K from Cholesky is stable enough here
                KH = K @ H
                Pj = (np.eye(self.state_dim, dtype=self._dtype) - KH) @ P_pred

                g_new = float(cur['g'] + total[j])
                remain = max(0, N - (i + 1))
                h_new = float(self._heur_to_goal(xj, Pj, goal_pos, layers[-1], remain))
                f_new = g_new + h_new

                key = (i + 1, hid)
                prev = g_best.get(key, np.inf)
                if g_new + self.switch_margin < prev:
                    g_best[key] = g_new
                elif g_new >= prev:
                    continue

                nxt = {
                    'layer_idx': i + 1,
                    'state': xj,
                    'cov': Pj,
                    'pos': z,
                    'traj': cur['traj'] + [z],
                    'g': g_new,
                    'h': h_new,
                    'hit_ids': cur['hit_ids'] + [hid],
                    'tan_ema': t_news[j],
                    'prev_along': float(r_along[j]),
                }
                counter += 1
                heapq.heappush(open_heap, (f_new, counter, nxt))

                if plot_tree:
                    G.add_edge((i, tuple(cur['pos'])),
                               (i + 1, tuple(z)),
                               cost=float(chi2[j]),
                               perp_pen=float(perp_pen[j]),
                               flip_pen=float(flip_pen[j]),
                               angle_pen=float(ang_pen[j]),
                               curv_pen=float(curv_pen[j]),
                               trend_pen=float(trend_pen[j]))

        # If nothing reached the end, return the best partial in the heap
        if open_heap:
            _, _, best = min(open_heap, key=lambda t3: t3[2]['g'])
            return [{
                'traj': best['traj'],
                'state': best['state'],
                'cov': best['cov'],
                'score': best['g'],
                'hit_ids': best['hit_ids']
            }], (G if plot_tree else None)

        return [], (G if plot_tree else None)

    def _heur_to_goal(self,
                      state: np.ndarray,
                      cov: np.ndarray,
                      goal_pos: np.ndarray,
                      goal_layer: Tuple[int, int],
                      remaining_layers: int) -> float:
        r"""
        Heuristic to the goal surface: Mahalanobis distance + small layer bias.

        Approximates future cost by predicting one geometric step to the goal
        surface, forming an innovation covariance, and evaluating a Mahalanobis
        distance from the predicted position to a target ``goal_pos``:

        .. math::

            \mathbf{x}_g &= \texttt{propagate}(\mathbf{x}, \Delta t_g), \quad
            \mathbf{P}_g = \mathbf{F}\mathbf{P}\mathbf{F}^\top + \mathbf{Q}_0\,\Delta t_g, \\
            \mathbf{S}_g &= \mathbf{H}\mathbf{P}_g\mathbf{H}^\top + \mathbf{R}, \\
            h_\text{base} &= (\mathbf{x}_{g,0:3} - \mathbf{z}_\text{goal})^\top
                             \mathbf{S}_g^{-1}(\mathbf{x}_{g,0:3} - \mathbf{z}_\text{goal}).

        The final heuristic is

        .. math:: h = h_\text{base} + \lambda \cdot \text{remaining\_layers},

        with a small bias :math:`\lambda = \tfrac{1}{2}\operatorname{tr}(\mathbf{R})`
        to mildly prefer shorter completions.

        To reduce recomputation, :math:`h_\text{base}` is cached using a key
        built from millimeter-quantized position and ``remaining_layers``.

        Parameters
        ----------
        state : ndarray, shape (7,)
            Current state :math:`\mathbf{x}`.
        cov : ndarray, shape (7, 7)
            Current covariance :math:`\mathbf{P}`.
        goal_pos : ndarray, shape (3,)
            Target position used for the Mahalanobis distance.
        goal_layer : tuple[int, int]
            Layer key used to compute :math:`\Delta t_g` (goal surface).
        remaining_layers : int
            Number of layers left to traverse (non-negative).

        Returns
        -------
        float
            Heuristic value :math:`h \ge 0`. Falls back to Euclidean distance
            if geometric solving fails.
        """
        # quantize (mm) and include remaining_layers to reduce collisions
        key = (int(round(state[0] * 1e3)),
               int(round(state[1] * 1e3)),
               int(round(state[2] * 1e3)),
               int(remaining_layers))
        base = self._heur_cache.get(key)
        if base is None:
            H = self.H_jac(None)
            try:
                dtg = self._solve_dt_to_surface(state, self.layer_surfaces[goal_layer])
                Fg = self.compute_F(state, dtg)
                xg = self.propagate(state, dtg)
                Pg = Fg @ cov @ Fg.T + self.Q0 * dtg
                Sg = H @ Pg @ H.T + self.R
                d = (xg[:3] - goal_pos).astype(self._dtype, copy=False)
                # Mahalanobis via solve (no inverse)
                base = float(d @ np.linalg.solve(Sg, d))
            except Exception:
                base = float(np.linalg.norm(state[:3] - goal_pos))
            self._heur_cache[key] = base
        # small bias per remaining layer
        return base + remaining_layers * 0.5 * float(np.trace(self.R))

