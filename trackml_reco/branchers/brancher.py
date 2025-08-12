import abc
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Sequence
from scipy.spatial import cKDTree
from scipy.optimize import newton

class Brancher(abc.ABC):
    r"""
    Abstract base class for branching track finders.

    This class provides shared utilities for Extended Kalman Filter (EKF)
    prediction/update, geometry propagation in a solenoidal field :math:`B_z`,
    candidate lookup and gating using KD-trees, deny-list handling, and
    shape-safe linear algebra helpers. Children implement the search strategy
    in :meth:`run`.

    Parameters
    ----------
    trees : dict
        Mapping ``(volume_id, layer_id) -> (cKDTree, points, ids)``, where
        ``points`` is an ``(N, 3)`` array of hit positions and ``ids`` are
        hit identifiers aligned with ``points``.
    layers : list of tuple of int
        Ordered list of layers to traverse, as ``(volume_id, layer_id)``.
    noise_std : float, optional
        Measurement noise standard deviation (meters). Sets
        :math:`R = \sigma^2 I_3` and scales the process noise ``Q0``.
    B_z : float, optional
        Longitudinal magnetic field [Tesla]. Affects curvature and angular
        frequency :math:`\omega = B_z \,\kappa\, p_T`.
    max_cands : int, optional
        Maximum number of nearest neighbors to query from the KD-tree before
        local filtering.
    step_candidates : int, optional
        Maximum number of candidates to keep per step after local sorting.

    Attributes
    ----------
    state_dim : int
        State dimension (default 7). Layout is
        :math:`x = [x, y, z, v_x, v_y, v_z, \kappa]^{\mathsf{T}}`.
    R : ndarray, shape (3, 3)
        Measurement noise covariance.
    Q0 : ndarray, shape (7, 7)
        Base process noise used as ``Q0 * dt`` during propagation.
    _H : ndarray, shape (3, 7)
        Cached measurement Jacobian that extracts :math:`(x,y,z)`.
    _deny : set of int
        Global deny-list of hit IDs (see :meth:`set_deny_hits`).

    Notes
    -----
    The continuous-time kinematics assume transverse rotation with angular
    frequency :math:`\omega = B_z\,\kappa\,p_T`, with :math:`p_T =
    \sqrt{v_x^2 + v_y^2}`. For small :math:`|\omega|`, linear motion is used
    as a stable limit.
    """

    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layers: List[Tuple[int,int]],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 max_cands: int = 10,
                 step_candidates: int = 5):
        self.trees       = trees
        self.layers      = layers
        self.noise_std   = float(noise_std)
        self.B_z         = float(B_z)
        self.max_cands   = int(max_cands)
        self.step_candidates = int(step_candidates)

        # model sizes (children can override)
        self.state_dim = 7

        # measurement & process noise
        self.R   = (self.noise_std**2) * np.eye(3)
        self.Q0  = np.diag([1e-4]*3 + [1e-5]*3 + [1e-6]) * self.noise_std

        # cached measurement jacobian H (3x7)
        self._H = np.zeros((3, self.state_dim))
        self._H[0,0] = self._H[1,1] = self._H[2,2] = 1.0

        # deny-list (shared across calls unless overridden by run(..., deny_hits=...))
        self._deny: set[int] = set()
        self._deny_mode: str = "hard"       # "hard" or "penalize"
        self._deny_penalty: float = 50.0

    def set_rng(self, seed: Optional[int]) -> None:
        r"""
        Set the internal random number generator.

        Parameters
        ----------
        seed : int or None
            Seed for :class:`numpy.random.Generator`. If ``None``, a
            non-deterministic generator is created.
        """
        self._rng = np.random.default_rng(seed)

    def set_deny_hits(self,
                      hits: Optional[Sequence[int]],
                      mode: str = "hard",
                      penalty: float = 50.0) -> None:
        r"""
        Configure a global deny-list for hit IDs.

        Parameters
        ----------
        hits : sequence of int or None
            Hit IDs to deny. Use ``None`` or empty to clear.
        mode : {"hard", "penalize"}, optional
            * ``"hard"`` drops denied hits.
            * ``"penalize"`` keeps them but adds a cost (see ``penalty``).
        penalty : float, optional
            Cost added when ``mode="penalize"``.

        Notes
        -----
        Children may also pass per-call deny lists to :meth:`run` if they
        want call-scoped behavior. This method sets a persistent global list.
        """
        self._deny = set(int(h) for h in (hits or []))
        self._deny_mode = str(mode)
        self._deny_penalty = float(penalty)

    def H_jac(self, _: np.ndarray) -> np.ndarray:
        r"""
        Return the measurement Jacobian :math:`H`.

        Parameters
        ----------
        _ : ndarray
            Ignored (kept for signature compatibility).

        Returns
        -------
        ndarray, shape (3, 7)
            Jacobian that extracts position: :math:`H x = (x,y,z)`.
        """
        # Return cached 3x7 Jacobian (extracts x,y,z)
        return self._H

    def to_local_frame_jac(self, plane_normal: np.ndarray) -> np.ndarray:
        r"""
        Build a 2D local-frame Jacobian for a plane.

        Given a plane with unit normal :math:`w`, this constructs orthonormal
        in-plane axes :math:`u, v` such that :math:`[u; v]` maps a 3D position
        to 2D local coordinates.

        Parameters
        ----------
        plane_normal : ndarray, shape (3,)
            Plane normal (will be normalized internally).

        Returns
        -------
        ndarray, shape (2, 3)
            Matrix with rows :math:`u^{\mathsf{T}}` and :math:`v^{\mathsf{T}}`.
        """
        w = plane_normal/np.linalg.norm(plane_normal)
        arbitrary = np.array([1,0,0]) if abs(w[0])<0.9 else np.array([0,1,0])
        u = np.cross(arbitrary, w); u /= np.linalg.norm(u)
        v = np.cross(w, u)
        return np.vstack([u, v])

    def to_local_frame(self, pos: np.ndarray, cov: np.ndarray,
                       plane_normal: np.ndarray, plane_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Transform a position and covariance to a plane's local 2D frame.

        Parameters
        ----------
        pos : ndarray, shape (3,)
            Global position.
        cov : ndarray, shape (>=3, >=3)
            Global covariance; the leading :math:`3\times 3` block is used.
        plane_normal : ndarray, shape (3,)
            Plane normal.
        plane_point : ndarray, shape (3,)
            A point lying on the plane.

        Returns
        -------
        meas : ndarray, shape (2,)
            Local 2D coordinates :math:`[u^{\mathsf{T}}(pos-p_0),\ v^{\mathsf{T}}(pos-p_0)]`.
        cov_local : ndarray, shape (2, 2)
            Projected covariance :math:`H\,\mathrm{Cov}_{xyz}\,H^{\mathsf{T}}`,
            with :math:`H=[u;v]`.
        """
        H = self.to_local_frame_jac(plane_normal)
        meas = H @ (pos - plane_point)
        cov_local = H @ cov[:3, :3] @ H.T
        return meas, cov_local

    def _safe_cov2d(self, P: np.ndarray) -> np.ndarray:
        r"""
        Ensure covariance is 2D.

        Parameters
        ----------
        P : ndarray
            Covariance array with shape ``(n, n)`` or broadcastable to it.

        Returns
        -------
        ndarray, shape (n, n)
            2D covariance matrix.
        """
        P = np.asarray(P, dtype=float)
        return P if P.ndim == 2 else np.atleast_2d(P)

    def _solve_dt_to_surface(self, x0: np.ndarray, surf: dict, dt_init: float = 1.0) -> float:
        r"""
        Solve time-of-flight :math:`\Delta t` to intersect a surface.

        Uses a 1D Newton solve on the signed distance to either a disk plane
        or a cylinder radius, with propagation :math:`x(\Delta t)` defined by
        :meth:`propagate`.

        Parameters
        ----------
        x0 : ndarray, shape (7,)
            Initial state :math:`[x,y,z,v_x,v_y,v_z,\kappa]^{\mathsf{T}}`.
        surf : dict
            Surface specification. One of::

                {'type': 'disk', 'n': n, 'p': p}   # plane normal n, point p
                {'type': 'cylinder', 'R': R}       # cylinder radius R

        dt_init : float, optional
            Initial guess for Newton's method.

        Returns
        -------
        float
            Time step :math:`\Delta t` such that the propagated state
            intersects the surface within tolerance.

        Notes
        -----
        For a disk, solve :math:`f(\Delta t)=(x(\Delta t)-p)\cdot n = 0`.
        For a cylinder, solve :math:`f(\Delta t)=\|x_{xy}(\Delta t)\|-R = 0`.
        """
        if surf['type']=='disk':
            n, p = surf['n'], surf['p']
            def f(dt): return (self.propagate(x0,dt)[:3]-p).dot(n)
        else:
            R = surf['R']
            def f(dt):
                xyt = self.propagate(x0,dt)[:2]
                return np.hypot(xyt[0], xyt[1]) - R
        return newton(f, dt_init, maxiter=20, tol=1e-6)

    def compute_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        r"""
        Compute discrete-time state Jacobian :math:`F=\partial x_{k+1}/\partial x_k`.

        The kinematics rotate the transverse velocity by
        :math:`\theta = \omega\,\Delta t`, with
        :math:`\omega = B_z\,\kappa\,p_T` and :math:`p_T=\sqrt{v_x^2+v_y^2}`.
        Small-:math:`\omega` limits are handled explicitly.

        Parameters
        ----------
        x : ndarray, shape (7,)
            State at time :math:`k`.
        dt : float
            Time step :math:`\Delta t`.

        Returns
        -------
        ndarray, shape (7, 7)
            Linearized transition matrix :math:`F`.
        """
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pT = np.hypot(vx, vy)
        ω = self.B_z * κ * pT
        θ = ω * dt
        F = np.eye(self.state_dim)
        if abs(ω) < 1e-6:
            F[0,3] = F[1,4] = F[2,5] = dt
            return F
        c, s = np.cos(θ), np.sin(θ)
        F[0,3] = c * dt - s/ω
        F[1,3] = s * dt + (1 - c)/ω
        F[0,4] = -s * dt - (1 - c)/ω
        F[1,4] = c * dt - s/ω
        F[2,5] = dt
        if abs(θ) < 1e-3:
            F[0,6] = -0.5 * self.B_z * pT * dt**2
            F[1,6] =  0.5 * self.B_z * pT * dt**2
        else:
            F[0,6] = (vy/κ) * (s/ω - c*dt)
            F[1,6] = (vx/κ) * (-(1 - c)/ω + s*dt)
        F[3,3], F[4,4], F[5,5], F[3,4], F[4,3] = c, c, 1, -s, s
        F[3,6] = (-s * vx + c * vy) * self.B_z * dt * pT
        F[4,6] = (-c * vx - s * vy) * self.B_z * dt * pT
        return F

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        r"""
        Propagate the state forward by :math:`\Delta t`.

        Parameters
        ----------
        x : ndarray, shape (7,)
            Current state :math:`[x,y,z,v_x,v_y,v_z,\kappa]^{\mathsf{T}}`.
        dt : float
            Time step :math:`\Delta t`.

        Returns
        -------
        ndarray, shape (7,)
            Propagated state :math:`x'`.

        Notes
        -----
        For :math:`|\omega| \ll 1`, use linear motion:
        :math:`x'_{pos} = x_{pos} + v \Delta t`.

        Otherwise, rotate the transverse velocity by
        :math:`\theta=\omega\,\Delta t`:

        .. math::

            \begin{aligned}
            v_x' &= \cos\theta\, v_x - \sin\theta\, v_y,\\
            v_y' &= \sin\theta\, v_x + \cos\theta\, v_y,\\
            x'_{pos} &= x_{pos} + [v_x', v_y', v_z] \Delta t.
            \end{aligned}
        """
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pT = np.hypot(vx, vy)
        ω = self.B_z * κ * pT
        if abs(ω) < 1e-6:
            dx = np.array([vx, vy, vz]) * dt
            return x + np.hstack([dx, np.zeros(4)])
        θ = ω * dt
        c, s = np.cos(θ), np.sin(θ)
        vx2 = c*vx - s*vy
        vy2 = s*vx + c*vy
        pos2 = x[:3] + np.array([vx2, vy2, vz]) * dt
        return np.hstack([pos2, [vx2, vy2, vz, κ]])

    def _estimate_seed_helix(self, seed_xyz: np.ndarray, dt: float, B_z: float) -> Tuple[np.ndarray, float]:
        r"""
        Estimate initial velocity and curvature from three seed points.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three ordered seed points :math:`(s_0, s_1, s_2)`.
        dt : float
            Nominal time between consecutive seed samples.
        B_z : float
            Longitudinal magnetic field [Tesla] to set curvature sign.

        Returns
        -------
        v0 : ndarray, shape (3,)
            Initial velocity estimate.
        kappa : float
            Initial curvature :math:`\kappa`. Sign is chosen so that
            :math:`\mathrm{sign}(\kappa) = \mathrm{sign}(cr_z) \cdot \mathrm{sign}(B_z)`,
            where :math:`cr = (s_1-s_0) \times (s_2-s_1)`.

        Notes
        -----
        The curvature magnitude is estimated via the circumcircle of the
        triangle:

        .. math::

            \kappa \approx \frac{2\| (s_1-s_0)\times(s_2-s_1) \|}
                               {\|s_1-s_0\|\,\|s_2-s_1\|\,\|s_2-s_0\|}.

        The initial direction uses a normalized sum of segment directions.
        """
        s0, s1, s2 = seed_xyz
        d1, d2, d02 = s1 - s0, s2 - s1, s2 - s0
        n1, n2, n02 = np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(d02)
        if min(n1,n2,n02) < 1e-6:
            return (s2 - s0)/(2*dt), 0.0
        cr = np.cross(d1, d2)
        kappa = 2*np.linalg.norm(cr)/(n1*n2*n02)
        if cr[2]*B_z < 0: kappa = -kappa
        t = (d1/n1 + d2/n2)
        tn = np.linalg.norm(t)
        t = d2/n2 if tn<1e-6 else t/tn
        v0 = t * (n2/dt)
        return v0, kappa

    def ekf_predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        EKF predict step.

        Parameters
        ----------
        x : ndarray, shape (7,)
            Current state.
        P : ndarray, shape (7, 7)
            Current covariance.
        dt : float
            Time step.

        Returns
        -------
        x_pred : ndarray, shape (7,)
            Predicted state.
        P_pred : ndarray, shape (7, 7)
            Predicted covariance :math:`F P F^{\mathsf{T}} + Q_0\,dt`.
        S : ndarray, shape (3, 3)
            Innovation covariance :math:`S = H P_{\text{pred}} H^{\mathsf{T}} + R`.

        Notes
        -----
        Uses :meth:`compute_F` and :meth:`propagate`.
        """
        P = self._safe_cov2d(P)
        F = self.compute_F(x, dt)
        x_pred = self.propagate(x, dt)
        P_pred = F @ P @ F.T + self.Q0 * dt
        H = self._H
        S = H @ P_pred @ H.T + self.R
        return x_pred, P_pred, S

    def ekf_update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray, S: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        EKF measurement update with position-only measurement.

        Parameters
        ----------
        x_pred : ndarray, shape (7,)
            Predicted state.
        P_pred : ndarray, shape (7, 7)
            Predicted covariance.
        z : ndarray, shape (3,)
            Position measurement.
        S : ndarray, shape (3, 3), optional
            Innovation covariance. If ``None``, it is recomputed.

        Returns
        -------
        x_upd : ndarray, shape (7,)
            Updated state.
        P_upd : ndarray, shape (7, 7)
            Updated covariance.

        Notes
        -----
        With :math:`H` extracting position and :math:`K = P H^{\mathsf{T}} S^{-1}`,
        the update is

        .. math::

            \begin{aligned}
            x^+ &= x^- + K (z - H x^-),\\
            P^+ &= (I - K H)\,P^-.
            \end{aligned}
        """
        H = self._H
        if S is None:
            S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ (z - x_pred[:3])
        P_upd = (np.eye(self.state_dim) - K @ H) @ P_pred
        return x_upd, P_upd

    def _get_candidates(self, pred_xyz: np.ndarray, layer: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Get nearest-neighbor candidate hits for a layer.

        Parameters
        ----------
        pred_xyz : ndarray, shape (3,)
            Predicted position.
        layer : tuple of int
            ``(volume_id, layer_id)`` key.

        Returns
        -------
        points_sel : ndarray, shape (M, 3)
            Selected candidate points, distance-sorted (``M <= step_candidates``).
        ids_sel : ndarray, shape (M,)
            Corresponding hit IDs.
        """
        tree, points, ids = self.trees[layer]
        dists, idxs = tree.query(pred_xyz, k=self.max_cands)
        idxs = np.atleast_1d(idxs)
        pts = points[idxs]
        d2 = np.linalg.norm(pts - pred_xyz, axis=1)
        best_local = np.argsort(d2)[:self.step_candidates]
        sel = idxs[best_local]
        return points[sel], ids[sel]

    def _get_candidates_in_gate(self, pred_pos: np.ndarray, layer: Tuple[int,int], radius: float):
        r"""
        Get candidate hits within a radial gate.

        Parameters
        ----------
        pred_pos : ndarray, shape (3,)
            Predicted position.
        layer : tuple of int
            ``(volume_id, layer_id)`` key.
        radius : float
            Radial gate in meters.

        Returns
        -------
        points_sel : ndarray, shape (M, 3)
            Points within ``radius``, distance-sorted and clipped to
            ``step_candidates``.
        ids_sel : ndarray, shape (M,)
            Corresponding hit IDs.

        Notes
        -----
        Deny-list handling is **not** applied here; see :meth:`_apply_deny`.
        """
        tree, points, ids = self.trees[layer]
        idxs = tree.query_ball_point(pred_pos, r=radius)
        if not idxs:
            return np.empty((0,3)), np.array([], dtype=ids.dtype)
        pts = points[idxs]
        d2 = np.linalg.norm(pts - pred_pos, axis=1)
        order = np.argsort(d2)[:self.step_candidates]
        sel = [idxs[i] for i in order]
        return points[sel], ids[sel]

    # fast, shared gate radii (children can pick either)
    @staticmethod
    def gate_radius_trace(S: np.ndarray, mul: float) -> float:
        r"""
        Gate radius from trace heuristic.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance.
        mul : float
            Multiplier (gain).

        Returns
        -------
        float
            Radius :math:`r = \text{mul}\,\sqrt{\max(10^{-12}, \mathrm{trace}(S)/3)}`.
        """
        return float(mul * np.sqrt(max(1e-12, np.trace(S)/3.0)))

    @staticmethod
    def gate_radius_maxeig(S: np.ndarray, mul: float) -> float:
        r"""
        Gate radius from maximum eigenvalue.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance.
        mul : float
            Multiplier (gain).

        Returns
        -------
        float
            Radius :math:`r = \text{mul}\,\sqrt{\lambda_{\max}(S)}`.
            Falls back to ``3*mul`` if eigen-decomposition fails.
        """
        try:
            return float(mul * np.sqrt(np.max(np.linalg.eigvalsh(S))))
        except Exception:
            return float(mul * 3.0)

    # optional deny application (useful if a child wants to centralize it)
    def _apply_deny(self,
                    pts: np.ndarray,
                    ids: np.ndarray,
                    base_cost: np.ndarray,
                    mode: Optional[str] = None,
                    penalty: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Apply the current deny policy to candidates.

        Parameters
        ----------
        pts : ndarray, shape (N, 3)
            Candidate points.
        ids : ndarray, shape (N,)
            Candidate hit IDs.
        base_cost : ndarray, shape (N,)
            Base costs per candidate (will be returned possibly modified).
        mode : {"hard", "penalize"}, optional
            Override the configured mode for this call.
        penalty : float, optional
            Override the configured penalty (only for ``"penalize"`` mode).

        Returns
        -------
        pts_out : ndarray, shape (M, 3)
            Kept points after deny application.
        ids_out : ndarray, shape (M,)
            Kept IDs.
        cost_out : ndarray, shape (M,)
            Updated costs (penalty added if applicable).

        Notes
        -----
        * ``"hard"`` removes denied hits.
        * ``"penalize"`` adds ``penalty`` to their cost but keeps them.
        """
        if ids.size == 0 or not self._deny:
            return pts, ids, base_cost
        mode = self._deny_mode if mode is None else mode
        penalty = self._deny_penalty if penalty is None else penalty
        deny_mask = np.array([int(h) in self._deny for h in ids], dtype=bool)
        if mode.lower().startswith("hard"):
            keep = ~deny_mask
            if not np.any(keep):
                return np.empty((0,3)), np.array([], dtype=ids.dtype), np.empty(0)
            return pts[keep], ids[keep], base_cost[keep]
        # penalize
        base_cost = base_cost + deny_mask.astype(float) * float(penalty)
        return pts, ids, base_cost

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        r"""
        Execute the branching/search algorithm.

        Returns
        -------
        branches : list of dict
            One or more branches in TrackBuilder format (implementation-defined).
        graph : object
            Optional expansion/search graph for debugging/visualization.

        Notes
        -----
        Children should document their return structure precisely,
        including any scores, per-layer details, and auxiliary fields.
        """
