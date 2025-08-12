import heapq
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from trackml_reco.branchers.brancher import Brancher


class HelixEKFAStarBrancher(Brancher):
    r"""
    EKF-based A* brancher with cross-track suppression and flip damping.

    This variant reduces diagonal "wiggle/straddling" by augmenting the candidate
    cost with two physics-informed terms:

    1) **Perpendicular (cross-track) penalty** — residual components perpendicular to
    the local tangent are penalized more than along-tangent components.
    2) **Flip penalty** — flipping the sign of the along-tangent innovation between
    consecutive layers is discouraged.

    It retains the usual angle, curvature, and trend penalties, and supports
    time-step subdivision to stabilize EKF predictions.

    Parameters
    ----------
    trees : dict[(int, int) -> (cKDTree, ndarray of shape (N, 3), ndarray of shape (N,))]
        KD-trees and hit banks per layer.
    layer_surfaces : dict[(int, int) -> dict]
        Geometry per layer. Either ``{'type': 'disk', 'n': (3,), 'p': (3,)}`` or
        ``{'type': 'cylinder', 'R': float}``.
    noise_std : float, optional
        Measurement noise :math:`\sigma` in millimeters. Default is ``2.0``.
    B_z : float, optional
        Magnetic field along :math:`z` (Tesla). Default is ``0.002``.
    max_cands : int, optional
        Maximum neighbors to query per layer (KD-tree). Default is ``10``.
    step_candidates : int, optional
        Keep at most this many nearest gated hits per layer. Default is ``5``.
    gate_multiplier : float, optional
        Base gating multiplier. Gate radius
        :math:`r_\text{gate} = \text{gate\_multiplier}\,\sqrt{\lambda_{\max}(\mathbf{S})}`.
        Default is ``3.0``.
    beam_width : int, optional
        Expand at most this many local candidates per node (A* beam). Default is ``8``.
    angle_weight : float, optional
        :math:`\chi^2` weight for tangent misalignment. Default is ``8.0``.
    curvature_weight : float, optional
        :math:`\chi^2` weight for :math:`(\Delta \kappa)^2`. Default is ``200.0``.
    trend_weight : float, optional
        :math:`\chi^2` weight for monotonic :math:`dz/dr` consistency. Default is ``6.0``.
    switch_margin : float, optional
        Required improvement margin when revisiting a (layer, hit). Default is ``1.5``.

    Other Parameters
    ----------------
    perp_weight : float, optional
        :math:`\chi^2` weight for the squared cross-track residual (perpendicular to the
        tangent). Default is ``12.0``.
    flip_weight : float, optional
        :math:`\chi^2` weight added when the along-track residual flips sign versus the
        previous step. Default is ``5.0``.
    ema_alpha : float, optional
        Exponential moving-average smoothing factor for the tangent used in penalties
        (``0..1``; higher = less memory). Default is ``0.5``.
    theta_max : float, optional
        Maximum helix turn per EKF step (radians); larger :math:`\Delta t` is subdivided.
        Default is ``0.35``.
    taper_last_frac : float, optional
        Fraction of the last layers where the gate is tapered smaller. Default is ``0.30``.
    min_gate_multiplier : float, optional
        Minimum gate multiplier in the taper zone. Default is ``1.2``.

    Notes
    -----
    **Perpendicular penalty.** Let :math:`\mathbf{t}` be the current smoothed unit tangent
    (from EMA of the EKF velocity), :math:`\mathbf{z}` a candidate hit, and
    :math:`\hat{\mathbf{x}}` the predicted position with innovation covariance
    :math:`\mathbf{S}`. With :math:`\sigma^2 \approx \tfrac{1}{3}\operatorname{tr}(\mathbf{S})`,

    .. math::

    C_\perp \;=\; \lambda_\perp\,
    \frac{\big\Vert\!\big(\mathbf{I}-\mathbf{t}\mathbf{t}^\top\big)\,(\mathbf{z}-\hat{\mathbf{x}})\big\Vert^2}
            {\sigma^2}\, .

    **Flip penalty.** With along-tangent residual :math:`r_\parallel=\mathbf{t}^\top(\mathbf{z}-\hat{\mathbf{x}})`,

    .. math::

    C_{\mathrm{flip}} \;=\;
    \begin{cases}
        \lambda_{\mathrm{flip}}\,\lvert r_\parallel\rvert, &
        \text{if } \operatorname{sign}(r_\parallel)\neq \operatorname{sign}(r_\parallel^{\text{prev}}),\\[3pt]
        0, & \text{otherwise.}
    \end{cases}

    **Additional penalties.**
    Angle misalignment :math:`C_\angle=\lambda_\angle(1-\cos\theta)^2`,
    curvature change :math:`C_\kappa=\lambda_\kappa(\Delta\kappa)^2`,
    and a trend term that discourages reversals in :math:`dz/dr`.

    **Gating.** The gate radius uses the largest eigenvalue of :math:`\mathbf{S}` and is
    optionally tapered in the final fraction of layers:

    .. math:: r_\text{gate} \propto \sqrt{\lambda_{\max}(\mathbf{S})}.

    **Stability via sub-stepping.** The EKF prediction subdivides the step when
    :math:`|\omega\,\Delta t|>\theta_\text{max}`, with
    :math:`\omega \approx B_z\,\kappa\,p_T`.

    See Also
    --------
    run : Execute the search and return the best branch and expansion graph.
    """

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
                         max_cands=max_cands)

        self.layer_surfaces = layer_surfaces
        self.step_candidates = int(step_candidates)
        self.state_dim = 7

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

        self._heur_cache: Dict[Tuple[int, int, int], float] = {}

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        r"""
        Normalize a vector to unit length.

        Parameters
        ----------
        v : ndarray, shape (N,)
            Input vector.

        Returns
        -------
        ndarray, shape (N,)
            Unit vector :math:`\frac{v}{\|v\|}` if the norm is larger than
            :math:`10^{-12}`, otherwise a zero vector of the same shape.
        """
        n = np.linalg.norm(v)
        return v * 0.0 if n < 1e-12 else (v / n)

    def _angle_pen(self, tangent: np.ndarray, disp: np.ndarray) -> float:
        r"""
        Compute angle misalignment penalty in :math:`\chi^2` units.

        Parameters
        ----------
        tangent : ndarray, shape (3,)
            Estimated tangent direction vector.
        disp : ndarray, shape (3,)
            Displacement vector to the measurement.

        Returns
        -------
        float
            Angle penalty:

            .. math::

                \text{angle\_weight} \cdot (1 - \cos\theta)^2

            where :math:`\theta` is the angle between ``tangent`` and ``disp``.
        """
        t = self._unit(tangent)
        d = self._unit(disp)
        c = float(np.clip(np.dot(t, d), -1.0, 1.0))
        return self.angle_weight * (1.0 - c) ** 2

    def _curv_pen(self, k_prev: float, k_new: float) -> float:
        r"""
        Compute curvature change penalty.

        Parameters
        ----------
        k_prev : float
            Previous curvature estimate.
        k_new : float
            New curvature estimate.

        Returns
        -------
        float
            Curvature penalty:

            .. math::

                \text{curvature\_weight} \cdot (\Delta k)^2
        """
        dk = float(k_new - k_prev)
        return self.curvature_weight * (dk * dk)

    def _trend_pen(self,
                   layer_key: Tuple[int, int],
                   x_pred: np.ndarray,
                   v_tan: np.ndarray,
                   z: np.ndarray) -> float:
        r"""
        Compute monotonic :math:`\Delta z / \Delta r` penalty to discourage
        endcap/barrel reversals.

        Parameters
        ----------
        layer_key : tuple of int
            ``(volume_id, layer_id)`` identifying the tracking layer surface.
        x_pred : ndarray, shape (>=3,)
            Predicted state vector (position in first 3 components).
        v_tan : ndarray, shape (3,)
            Tangent (velocity) vector.
        z : ndarray, shape (3,)
            Measurement (hit) position.

        Returns
        -------
        float
            Trend penalty proportional to :math:`|\Delta z|` (disk) or
            :math:`|\Delta r|` (cylinder) if the displacement direction
            disagrees with the tangent direction.
        """
        surf = self.layer_surfaces[layer_key]
        dp = z - x_pred[:3]
        if surf['type'] == 'disk':
            s = np.sign(v_tan[2]) if abs(v_tan[2]) > 1e-9 else 0.0
            wrong = (np.sign(dp[2]) != s) and (s != 0.0)
            return self.trend_weight * (abs(dp[2]) if wrong else 0.0)
        # cylinder
        r0 = float(np.hypot(x_pred[0], x_pred[1]))
        r1 = float(np.hypot(z[0], z[1]))
        dr = r1 - r0
        vr = (x_pred[0]*v_tan[0] + x_pred[1]*v_tan[1]) / max(r0, 1e-9)
        s = np.sign(vr) if abs(vr) > 1e-9 else 0.0
        wrong = (np.sign(dr) != s) and (s != 0.0)
        return self.trend_weight * (abs(dr) if wrong else 0.0)

    def _perp_pen(self,
                  S: np.ndarray,
                  tangent: np.ndarray,
                  resid: np.ndarray) -> float:
        r"""
        Compute cross-track residual penalty in :math:`\chi^2` units.

        The perpendicular component of the residual is penalized relative
        to the estimated position covariance.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Residual covariance matrix.
        tangent : ndarray, shape (3,)
            Tangent (velocity) vector.
        resid : ndarray, shape (3,)
            Position residual vector.

        Returns
        -------
        float
            Perpendicular penalty:

            .. math::

                \lambda_{\perp} \frac{\| (I - t t^{\mathsf{T}}) \, \text{resid} \|^2}
                                {\sigma^2},

            where :math:`\sigma^2 \approx \frac{\mathrm{trace}(S)}{3}`.
        """
        t = self._unit(tangent)
        P_perp = np.eye(3) - np.outer(t, t)
        r_perp = P_perp @ resid
        sigma2 = float(np.trace(S)) / 3.0 + 1e-12
        return self.perp_weight * float(r_perp @ r_perp) / sigma2

    @staticmethod
    def _sign_flip_pen(prev_along: Optional[float], cur_along: float, weight: float) -> float:
        r"""
        Penalize sign flips in along-track residuals.

        Parameters
        ----------
        prev_along : float or None
            Previous along-track residual; ``None`` if undefined.
        cur_along : float
            Current along-track residual.
        weight : float
            Penalty weight factor.

        Returns
        -------
        float
            Penalty:

            .. math::

                \text{weight} \cdot |\text{cur\_along}|

            if :math:`\text{prev\_along} \cdot \text{cur\_along} < 0`, else 0.
        """
        if prev_along is None:
            return 0.0
        return (weight * abs(cur_along)) if (prev_along * cur_along) < 0.0 else 0.0

    def _substep_predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Predict state and covariance forward in time with angular step subdivision.

        Subdivides the propagation if :math:`|\omega \cdot \Delta t|` exceeds
        ``theta_max`` to improve integration accuracy.

        Parameters
        ----------
        x : ndarray, shape (state_dim,)
            Current state vector.
        P : ndarray, shape (state_dim, state_dim)
            Current state covariance matrix.
        dt : float
            Time increment.

        Returns
        -------
        x_new : ndarray, shape (state_dim,)
            Predicted state vector.
        P_new : ndarray, shape (state_dim, state_dim)
            Predicted state covariance matrix.
        """
        vx, vy = x[3], x[4]
        pT = float(np.hypot(vx, vy))
        omega = float(self.B_z * x[6] * pT)
        turn = abs(omega * dt)

        n = int(np.ceil(turn / self.theta_max)) if turn > self.theta_max else 1
        h = dt / n
        xk, Pk = x.copy(), P.copy()
        for _ in range(n):
            F = self.compute_F(xk, h)
            xk = self.propagate(xk, h)
            Pk = F @ Pk @ F.T + self.Q0 * h
        return xk, Pk

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Execute A* track-building with cross-track suppression and flip damping.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed points [inner, middle, outer] in 3D space.
        layers : list of tuple
            Ordered list of ``(volume_id, layer_id)`` for layers to traverse.
        t : ndarray
            Time points associated with the seed (for EKF initialization).
        plot_tree : bool, optional
            Ignored (kept for API compatibility).

        Returns
        -------
        branches : list of dict
            One-best branch in TrackBuilder format.
        graph : networkx.DiGraph
            Expansion graph containing search states and edges.
        """
        if not layers:
            return [], nx.DiGraph()

        dt0 = float(t[1] - t[0]) if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, k0])
        P0 = np.eye(self.state_dim) * 0.1

        # goal (for heuristic)
        try:
            dt_goal = self._solve_dt_to_surface(x0, self.layer_surfaces[layers[-1]], dt_init=dt0)
            xg, _ = self._substep_predict(x0, P0, dt_goal)
            goal_pos = xg[:3]
        except Exception:
            goal_pos = seed_xyz[2].copy()

        # A* data
        G = nx.DiGraph()
        open_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        g_best: Dict[Tuple[int, int], float] = {(0, -1): 0.0}
        counter = 0

        # keep a smoothed tangent & previous along-residual in the node
        tan0 = self._unit(v0)
        start = {
            'layer_idx': 0,
            'state': x0,
            'cov': P0,
            'pos': seed_xyz[2],
            'traj': [seed_xyz[0], seed_xyz[1], seed_xyz[2]],
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
        max_exp = 40000
        H = self.H_jac(None)

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
                return [branch], G

            layer_key = layers[i]
            surf = self.layer_surfaces[layer_key]

            # Predict to surface (with subdivision)
            try:
                dt = self._solve_dt_to_surface(cur['state'], surf)
            except Exception:
                continue
            x_pred, P_pred = self._substep_predict(cur['state'], cur['cov'], dt)
            S = H @ P_pred @ H.T + self.R

            # tapered gate near the end
            progress = i / max(N, 1)
            taper = 1.0
            if progress >= (1.0 - self.taper_last_frac):
                frac = (progress - (1.0 - self.taper_last_frac)) / max(self.taper_last_frac, 1e-9)
                taper = max(self.min_gate_multiplier / self.gate_multiplier, 1.0 - 0.5 * frac)
            try:
                gate_r = float(self.gate_multiplier * taper * np.sqrt(np.max(np.linalg.eigvalsh(S))))
            except Exception:
                gate_r = float(self.gate_multiplier * taper * 3.0)

            pts, ids = self._get_candidates_in_gate(x_pred[:3], layer_key, gate_r)
            if len(ids) == 0:
                continue

            # local scoring
            tan_ema = cur['tan_ema']
            prev_along = cur['prev_along']
            k_prev = float(cur['state'][6])

            local: List[Tuple[float, Dict[str, Any]]] = []
            sigma2 = float(np.trace(S)) / 3.0 + 1e-12

            for z, hid in zip(pts, ids):
                r = z - x_pred[:3]
                # χ² innovation
                try:
                    chi2 = float(r @ np.linalg.solve(S, r))
                except Exception:
                    continue
                if not np.isfinite(chi2) or chi2 > 1e4:
                    continue

                # EKF update at candidate
                K = P_pred @ H.T @ np.linalg.inv(S)
                x_upd = x_pred + K @ r
                P_upd = (np.eye(self.state_dim) - K @ H) @ P_pred

                # update tangent EMA using updated velocity
                vel_new = x_upd[3:6]
                t_new = self._unit((1.0 - self.ema_alpha) * tan_ema + self.ema_alpha * vel_new)

                # split residual into along / perp wrt smoothed tangent
                r_along = float(np.dot(r, t_new))
                r_perp_cost = self._perp_pen(S, t_new, r)

                # penalties
                ang_pen = self._angle_pen(vel_new, r)
                curv_pen = self._curv_pen(k_prev, float(x_upd[6]))
                trend_pen = self._trend_pen(layer_key, x_pred, vel_new, z)
                flip_pen = self._sign_flip_pen(prev_along, r_along, self.flip_weight)

                total = chi2 + r_perp_cost + ang_pen + curv_pen + trend_pen + flip_pen
                item = {
                    'hid': int(hid),
                    'z': z,
                    'x_upd': x_upd,
                    'P_upd': P_upd,
                    't_new': t_new,
                    'r_along': r_along,
                    'chi2': chi2,
                    'perp_pen': r_perp_cost,
                    'ang_pen': ang_pen,
                    'curv_pen': curv_pen,
                    'trend_pen': trend_pen,
                    'flip_pen': flip_pen
                }
                local.append((total, item))

            if not local:
                continue
            local.sort(key=lambda x: x[0])
            topk = local[:max(1, min(len(local), self.beam_width))]

            for total, it in topk:
                g_new = float(cur['g'] + it['chi2'] + it['perp_pen'] + it['ang_pen'] + it['curv_pen'] + it['trend_pen'] + it['flip_pen'])
                remain = max(0, N - (i + 1))
                h_new = float(self._heur_to_goal(it['x_upd'], it['P_upd'], goal_pos, layers[-1], remain))
                f_new = g_new + h_new

                key = (i + 1, it['hid'])
                prev = g_best.get(key, np.inf)
                if g_new + self.switch_margin < prev:
                    g_best[key] = g_new
                elif g_new >= prev:
                    continue

                nxt = {
                    'layer_idx': i + 1,
                    'state': it['x_upd'],
                    'cov': it['P_upd'],
                    'pos': it['z'],
                    'traj': cur['traj'] + [it['z']],
                    'g': g_new,
                    'h': h_new,
                    'hit_ids': cur['hit_ids'] + [it['hid']],
                    'tan_ema': it['t_new'],
                    'prev_along': it['r_along']
                }
                counter += 1
                heapq.heappush(open_heap, (f_new, counter, nxt))
                G.add_edge((i, tuple(cur['pos'])),
                           (i + 1, tuple(it['z'])),
                           cost=float(it['chi2']),
                           perp_pen=float(it['perp_pen']),
                           flip_pen=float(it['flip_pen']),
                           angle_pen=float(it['ang_pen']),
                           curv_pen=float(it['curv_pen']),
                           trend_pen=float(it['trend_pen']))

        # fallback: best partial
        if open_heap:
            nodes = [n for _, _, n in open_heap]
            best = min(nodes, key=lambda n: n['g'])
            return [{
                'traj': best['traj'],
                'state': best['state'],
                'cov': best['cov'],
                'score': best['g'],
                'hit_ids': best['hit_ids']
            }], G

        return [], G

    def _heur_to_goal(self,
                      state: np.ndarray,
                      cov: np.ndarray,
                      goal_pos: np.ndarray,
                      goal_layer: Tuple[int, int],
                      remaining_layers: int) -> float:
        r"""
        Compute heuristic cost to goal layer.

        Uses Mahalanobis distance between predicted position and goal position,
        plus a small per-layer constant to encourage shorter paths.

        Parameters
        ----------
        state : ndarray, shape (state_dim,)
            Current state vector.
        cov : ndarray, shape (state_dim, state_dim)
            State covariance matrix.
        goal_pos : ndarray, shape (3,)
            Target position for the goal layer.
        goal_layer : tuple of int
            ``(volume_id, layer_id)`` of the goal layer surface.
        remaining_layers : int
            Number of layers left to traverse.

        Returns
        -------
        float
            Heuristic score:

            .. math::

                d_{\text{Mahalanobis}} + 0.5 \cdot \mathrm{trace}(R)
                \cdot \text{remaining\_layers}.
        """
        key = (int(round(state[0]*1e3)), int(round(state[1]*1e3)), int(round(state[2]*1e3)))
        if key in self._heur_cache:
            base = self._heur_cache[key]
        else:
            try:
                dtg = self._solve_dt_to_surface(state, self.layer_surfaces[goal_layer], dt_init=1.0)
                Fg = self.compute_F(state, dtg)
                xg = self.propagate(state, dtg)
                Pg = Fg @ cov @ Fg.T + self.Q0 * dtg
                H = self.H_jac(None)
                Sg = H @ Pg @ H.T + self.R
                d = xg[:3] - goal_pos
                base = float(d @ np.linalg.solve(Sg, d))
            except Exception:
                base = float(np.linalg.norm(state[:3] - goal_pos))
            self._heur_cache[key] = base
        # small constant per layer; we keep it mild to not dominate
        return base + remaining_layers * 0.5 * float(np.trace(self.R))
