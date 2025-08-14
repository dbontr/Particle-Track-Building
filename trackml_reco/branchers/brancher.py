import abc
from functools import lru_cache
import logging
from typing import Tuple, Dict, List, Optional, Sequence, Callable, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import newton

# default fast kernels (Cholesky-based) from your project
from trackml_reco.ekf_kernels import chi2_batch as _chi2_batch_default, kalman_gain as _kalman_gain_default

class MathBackend:
    r"""
    Pluggable math backend for EKF kernels (χ² and Kalman gain).

    This thin wrapper lets you swap in accelerated implementations (NumPy,
    Numba, CuPy, etc.) without changing call sites. Two callables are required:

    - ``chi2_batch(diff, S) -> chi2``: vectorized Mahalanobis distances
      for residuals ``diff`` against innovation covariance :math:`\mathbf{S}`:

      .. math::

          \chi^2_i \;=\; \mathbf{d}_i^\top \mathbf{S}^{-1} \mathbf{d}_i,
          \qquad \mathbf{d}_i = \text{diff}[i,:].

    - ``kalman_gain(P, H, S) -> K``: Cholesky-based EKF gain
      :math:`\mathbf{K}\in\mathbb{R}^{d\times 3}` for state covariance
      :math:`\mathbf{P}\in\mathbb{R}^{d\times d}`, measurement Jacobian
      :math:`\mathbf{H}\in\mathbb{R}^{3\times d}`, and innovation covariance
      :math:`\mathbf{S}=\mathbf{H}\mathbf{P}\mathbf{H}^\top+\mathbf{R}`:

      .. math::

          \mathbf{K} \;=\; \mathbf{P}\mathbf{H}^\top \mathbf{S}^{-1}
          \quad\text{(computed via Cholesky solves, no explicit inverses).}

    Parameters
    ----------
    chi2_batch : callable, optional
        Function with signature ``(diff:(N,3), S:(3,3)) -> (N,)``.
        Defaults to :data:`trackml_reco.ekf_kernels.chi2_batch`.
    kalman_gain : callable, optional
        Function with signature ``(P:(d,d), H:(3,d), S:(3,3)) -> (d,3)``.
        Defaults to :data:`trackml_reco.ekf_kernels.kalman_gain`.

    Notes
    -----
    The default kernels are Cholesky-based and avoid matrix inversions.
    """
    __slots__ = ("chi2_batch", "kalman_gain")

    def __init__(self,
                 chi2_batch: Callable[[np.ndarray, np.ndarray], np.ndarray] = _chi2_batch_default,
                 kalman_gain: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = _kalman_gain_default):
        self.chi2_batch = chi2_batch
        self.kalman_gain = kalman_gain


class Brancher(abc.ABC):
    r"""
    Abstract base class for branching track finders.

    Provides shared utilities for Extended Kalman Filter (EKF) prediction and
    update, geometric propagation in a solenoidal field :math:`B_z`, KD-tree
    candidate lookup and χ² gating, deny-list handling, and shape-safe linear
    algebra helpers. Concrete search strategies are implemented in :meth:`run`.

    **State/measurement convention.**
    The state is
    :math:`\mathbf{x}=[x,y,z,v_x,v_y,v_z,\kappa]^\top\in\mathbb{R}^7`,
    measurement is 3D position :math:`\mathbf{z}\in\mathbb{R}^3`, and the
    measurement Jacobian :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}` extracts
    position (first three components). With predicted :math:`(\hat{\mathbf{x}},\mathbf{P})`:

    .. math::

        \mathbf{S} &= \mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R}, \\
        \chi^2(\mathbf{z}) &= (\mathbf{z}-\hat{\mathbf{x}}_{0:3})^\top
                              \mathbf{S}^{-1}(\mathbf{z}-\hat{\mathbf{x}}_{0:3}), \\
        \mathbf{K} &= \mathbf{P}\mathbf{H}^\top\mathbf{S}^{-1} \quad
                     \text{(computed via Cholesky solves).}

    Performance features
    --------------------
    - No matrix inverses: EKF uses Cholesky-based gain (``kalman_gain``) and χ²
      (``chi2_batch``).
    - Vectorized candidate scoring + top-:math:`k` selection via ``argpartition``.
    - Memory wins: ``__slots__``, reusable identity, consistent dtype.
    - Smart KD-tree gating with radius query and k-NN fallback for extreme fan-out.
    - Pluggable math backend via :class:`MathBackend`.
    - Deny-list supports hard drop or additive penalty.

    Parameters
    ----------
    trees : dict[tuple[int, int], tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) -> (tree, points, ids)`` where
        ``points`` has shape ``(N,3)`` and ``ids`` is aligned of shape ``(N,)``.
    layers : list[tuple[int, int]]
        Ordered layer keys to traverse.
    noise_std : float, optional
        Measurement noise standard deviation (meters). Sets
        :math:`\mathbf{R}=\sigma^2\mathbf{I}_3` and scales :math:`\mathbf{Q}_0`.
    B_z : float, optional
        Longitudinal magnetic field (Tesla). Affects angular rate
        :math:`\omega = B_z\,\kappa\,p_T`, where :math:`p_T=\sqrt{v_x^2+v_y^2}`.
    max_cands : int, optional
        KD-tree preselect per layer (upper bound) before local filtering.
    step_candidates : int, optional
        Maximum number of candidates kept per step after sorting.
    dtype : numpy dtype, optional
        Computation dtype for internal arrays (default ``float64``).
    gate_multiplier : float, optional
        Base gating multiplier for radius (trace heuristic) used by
        :meth:`_gate_radius_trace`.
    gate_tighten : float, optional
        Linear gate tightening factor as depth increases in the layer list.
    ball_k_fallback : int, optional
        If a radius query returns more than this many hits, fall back to
        k-NN to cap fan-out.
    backend : MathBackend or None, optional
        Pluggable numeric backend for χ² and Kalman gain.

    Attributes
    ----------
    state_dim : int
        State dimension (``7``).
    dtype : numpy dtype
        Internal numeric dtype.
    R : ndarray, shape (3, 3)
        Measurement covariance :math:`\mathbf{R}`.
    Q0 : ndarray, shape (7, 7)
        Base process noise, used as :math:`\mathbf{Q}_0\,\Delta t`.
    _H : ndarray, shape (3, 7)
        Measurement Jacobian extracting position.
    _I : ndarray, shape (7, 7)
        Reusable identity.
    _deny : set[int]
        Global deny-list of hit IDs; see :meth:`set_deny_hits`.

    Notes
    -----
    The state layout is fixed as above; subclasses can specialize the search
    (beam, A*, ACO, etc.) by reusing the provided EKF and gating utilities.
    """

    # keep subclass flexibility but still save memory on base attributes
    __slots__ = (
        "trees", "layers", "noise_std", "B_z", "max_cands", "step_candidates",
        "state_dim", "dtype", "R", "Q0", "_H", "_I", "_deny", "_deny_mode",
        "_deny_penalty", "_rng", "layer_surfaces", "gate_multiplier",
        "gate_tighten", "ball_k_fallback", "_backend", "__dict__", "__weakref__"
    )

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layers: List[Tuple[int, int]],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 max_cands: int = 10,
                 step_candidates: int = 5,
                 dtype: np.dtype = np.float64,
                 gate_multiplier: float = 3.0,
                 gate_tighten: float = 0.15,
                 ball_k_fallback: int = 512,
                 backend: Optional[MathBackend] = None):
        # config
        self.trees = trees
        self.layers = layers
        self.noise_std = float(noise_std)
        self.B_z = float(B_z)
        self.max_cands = int(max_cands)
        self.step_candidates = int(step_candidates)
        self.dtype = np.dtype(dtype)
        self.gate_multiplier = float(gate_multiplier)
        self.gate_tighten = float(gate_tighten)
        self.ball_k_fallback = int(ball_k_fallback)
        self._backend = backend or MathBackend()

        # model sizes
        self.state_dim = 7

        # measurement & process noise
        self.R = (self.noise_std ** 2) * np.eye(3, dtype=self.dtype)
        self.Q0 = np.diag([1e-4] * 3 + [1e-5] * 3 + [1e-6]).astype(self.dtype) * self.noise_std

        # cached measurement jacobian H (3x7)
        self._H = np.zeros((3, self.state_dim), dtype=self.dtype)
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = 1.0

        # reusable identity (avoids np.eye allocations)
        self._I = np.eye(self.state_dim, dtype=self.dtype)

        # deny-list (shared across calls unless overridden by run(..., deny_hits=...))
        self._deny: set[int] = set()
        self._deny_mode: str = "hard"          # "hard" or "penalize"
        self._deny_penalty: float = 50.0

        # RNG + optional geometry surface map
        self._rng = np.random.default_rng(None)
        self.layer_surfaces: Dict[Any, dict] = {}

        # logger (opt-in)
        self.log = logging.getLogger(self.__class__.__name__)

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

    def set_backend(self, backend: MathBackend) -> None:
        r"""
        Swap the math backend at runtime.

        Parameters
        ----------
        backend : MathBackend
            Backend providing ``chi2_batch`` and ``kalman_gain`` kernels.

        Notes
        -----
        Useful for switching to JIT/GPU implementations without touching
        algorithmic code paths.
        """
        self._backend = backend

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

    @lru_cache(maxsize=None)
    def _H_cached(self):
        r"""
        Return and memoize the measurement Jacobian :math:`\mathbf{H}`.

        Notes
        -----
        Hook point if a child overrides :meth:`H_jac`; memoization avoids
        repeated allocations.
        """
        # Hook if a child overrides H_jac; we still memoize.
        return self.H_jac(None)

    def H_jac(self, _: np.ndarray) -> np.ndarray:
        r"""
        Return the measurement Jacobian :math:`\mathbf{H}`.

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
    
    def _ekf_predict(self, state: np.ndarray, cov: np.ndarray, dt: float):
        r"""
        Internal EKF predict; returns ``x_pred``, ``P_pred``, ``S`` and cached ``H``.

        Computes

        .. math::

            \mathbf{x}^- &= f(\mathbf{x}, \Delta t),\\
            \mathbf{F} &= \frac{\partial f}{\partial \mathbf{x}},\\
            \mathbf{P}^- &= \mathbf{F}\mathbf{P}\mathbf{F}^\top + \mathbf{Q}_0\,\Delta t,\\
            \mathbf{S} &= \mathbf{H}\mathbf{P}^-\mathbf{H}^\top + \mathbf{R}.

        Parameters
        ----------
        state : ndarray, shape (7,)
            Current state :math:`\mathbf{x}`.
        cov : ndarray, shape (7, 7)
            Current covariance :math:`\mathbf{P}`.
        dt : float
            Time step :math:`\Delta t`.

        Returns
        -------
        x_pred : ndarray, shape (7,)
        P_pred : ndarray, shape (7, 7)
        S : ndarray, shape (3, 3)
        H : ndarray, shape (3, 7)
        """
        H = self._H_cached()
        F = self.compute_F(state, dt)
        x_pred = self.propagate(state, dt)
        P_pred = F @ cov @ F.T + self.Q0 * dt
        S = H @ P_pred @ H.T + self.R
        return x_pred, P_pred, S, H

    def _ekf_update_meas(self,
                         x_pred: np.ndarray,
                         P_pred: np.ndarray,
                         z: np.ndarray,
                         H: np.ndarray,
                         S: np.ndarray):
        r"""
        Internal EKF update using Cholesky-based gain.

        With residual :math:`\mathbf{r}=\mathbf{z}-\hat{\mathbf{x}}_{0:3}`,
        and :math:`\mathbf{K}=\texttt{kalman\_gain}(\mathbf{P}^-,\mathbf{H},\mathbf{S})`,

        .. math::

            \mathbf{x}^+ = \mathbf{x}^- + \mathbf{K}\mathbf{r}, \qquad
            \mathbf{P}^+ \approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^-.

        Returns
        -------
        x_upd : ndarray, shape (7,)
        P_upd : ndarray, shape (7, 7)
        """
        K = self._backend.kalman_gain(P_pred, H, S)
        x_upd = x_pred + K @ (z - x_pred[:3])
        P_upd = (self._I - K @ H) @ P_pred
        return x_upd, P_upd
    
    
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
            Predicted covariance :math:`\mathbf{F}\mathbf{P}\mathbf{F}^{\mathsf{T}} + \mathbf{Q}_0\,dt`.
        S : ndarray, shape (3, 3)
            Innovation covariance :math:`\mathbf{S} = \mathbf{H}\mathbf{P}_{\text{pred}}\mathbf{H}^{\mathsf{T}} + \mathbf{R}`.

        Notes
        -----
        Uses :meth:`compute_F` and :meth:`propagate`.
        """
        x_pred, P_pred, S, _ = self._ekf_predict(np.asarray(x, self.dtype), self._safe_cov2d(P), float(dt))
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
        With :math:`\mathbf{H}` extracting position and
        :math:`\mathbf{K} = \mathbf{P}\mathbf{H}^{\mathsf{T}}\mathbf{S}^{-1}`,
        the update is

        .. math::

            \begin{aligned}
            \mathbf{x}^+ &= \mathbf{x}^- + \mathbf{K} (\mathbf{z} - \mathbf{H} \mathbf{x}^-),\\
            \mathbf{P}^+ &= (\mathbf{I} - \mathbf{K} \mathbf{H})\,\mathbf{P}^-.
            \end{aligned}
        """
        H = self._H_cached()
        if S is None:
            S = H @ P_pred @ H.T + self.R
        return self._ekf_update_meas(x_pred, P_pred, np.asarray(z, self.dtype), H, S)

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
        P = np.asarray(P, dtype=self.dtype)
        return P if P.ndim == 2 else np.atleast_2d(P)

    def _solve_dt_to_surface(self, x0: np.ndarray, surf: dict, dt_init: float = 1.0) -> float:
        r"""
        Solve time-of-flight :math:`\Delta t` to intersect a surface.

        Uses a 1D Newton solve on the signed distance to either a disk plane
        or a cylinder radius, with propagation :math:`\mathbf{x}(\Delta t)`
        defined by :meth:`propagate`.

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
        x0 = np.asarray(x0, dtype=self.dtype)

        if surf['type'] == 'disk':
            n, p = np.asarray(surf['n'], self.dtype), np.asarray(surf['p'], self.dtype)

            def f(dt): return (self.propagate(x0, dt)[:3] - p).dot(n)
        else:
            R = float(surf['R'])

            def f(dt):
                xyt = self.propagate(x0, dt)[:2]
                return np.hypot(xyt[0], xyt[1]) - R

        return float(newton(f, float(dt_init), maxiter=20, tol=1e-6))

    def compute_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        r"""
        Compute the discrete-time state Jacobian :math:`\mathbf{F} = \partial \mathbf{x}_{k+1} / \partial \mathbf{x}_k`.

        The transverse velocity undergoes a rotation by
        :math:`\theta=\omega\,\Delta t`, with
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
            Linearized transition matrix :math:`\mathbf{F}`.

        Notes
        -----
        This implementation matches the companion :meth:`propagate` model,
        including the small-angle branches to avoid catastrophic cancellation.
        """
        x = np.asarray(x, dtype=self.dtype)
        dt = float(dt)
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pT = np.hypot(vx, vy)
        ω = self.B_z * κ * pT
        θ = ω * dt
        F = self._I.copy()
        if abs(ω) < 1e-6:
            F[0, 3] = F[1, 4] = F[2, 5] = dt
            return F
        c, s = np.cos(θ), np.sin(θ)
        F[0, 3] = c * dt - s / ω
        F[1, 3] = s * dt + (1 - c) / ω
        F[0, 4] = -s * dt - (1 - c) / ω
        F[1, 4] = c * dt - s / ω
        F[2, 5] = dt
        if abs(θ) < 1e-3:
            F[0, 6] = -0.5 * self.B_z * pT * dt * dt
            F[1, 6] = 0.5 * self.B_z * pT * dt * dt
        else:
            F[0, 6] = (vy / κ) * (s / ω - c * dt)
            F[1, 6] = (vx / κ) * (-(1 - c) / ω + s * dt)
        F[3, 3], F[4, 4], F[5, 5], F[3, 4], F[4, 3] = c, c, 1.0, -s, s
        F[3, 6] = (-s * vx + c * vy) * self.B_z * dt * pT
        F[4, 6] = (-c * vx - s * vy) * self.B_z * dt * pT
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
            Propagated state :math:`\mathbf{x}'`.

        Notes
        -----
        For :math:`|\omega| \ll 1`, use linear motion
        :math:`\mathbf{x}'_{\text{pos}} = \mathbf{x}_{\text{pos}} + \mathbf{v}\,\Delta t`.

        Otherwise, rotate the transverse velocity by
        :math:`\theta=\omega\,\Delta t`:

        .. math::

            \begin{aligned}
            v_x' &= \cos\theta\, v_x - \sin\theta\, v_y,\\
            v_y' &= \sin\theta\, v_x + \cos\theta\, v_y,\\
            \mathbf{x}'_{\text{pos}} &= \mathbf{x}_{\text{pos}} + [v_x', v_y', v_z] \,\Delta t.
            \end{aligned}
        """
        x = np.asarray(x, dtype=self.dtype)
        dt = float(dt)
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pT = np.hypot(vx, vy)
        ω = self.B_z * κ * pT
        if abs(ω) < 1e-6:
            dx = np.array([vx, vy, vz], dtype=self.dtype) * dt
            return x + np.hstack([dx, np.zeros(4, dtype=self.dtype)])
        θ = ω * dt
        c, s = np.cos(θ), np.sin(θ)
        vx2 = c * vx - s * vy
        vy2 = s * vx + c * vy
        pos2 = x[:3] + np.array([vx2, vy2, vz], dtype=self.dtype) * dt
        return np.hstack([pos2, np.array([vx2, vy2, vz, κ], dtype=self.dtype)])

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

            \kappa \approx \frac{2\left\| (s_1-s_0)\times(s_2-s_1) \right\|}
                               {\|s_1-s_0\|\,\|s_2-s_1\|\,\|s_2-s_0\|}.

        The initial direction uses a normalized sum of segment directions.
        """
        seed_xyz = np.asarray(seed_xyz, dtype=self.dtype)
        s0, s1, s2 = seed_xyz
        d1, d2, d02 = s1 - s0, s2 - s1, s2 - s0
        n1, n2, n02 = np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(d02)
        if min(n1, n2, n02) < 1e-6:
            return (s2 - s0) / (2 * dt), 0.0
        cr = np.cross(d1, d2)
        kappa = 2 * np.linalg.norm(cr) / (n1 * n2 * n02)
        if cr[2] * B_z < 0:
            kappa = -kappa
        t = (d1 / n1 + d2 / n2)
        tn = np.linalg.norm(t)
        t = d2 / n2 if tn < 1e-6 else t / tn
        v0 = t * (n2 / dt)
        return v0.astype(self.dtype), float(kappa)
    
    def _gate_radius_trace(self, S: np.ndarray, depth_frac: float = 0.0, base_mul: float = None, tighten: float = None):
        r"""
        Trace-based scalar gate with progressive tightening.

        The gate radius is

        .. math::

            r_{\text{gate}} \;=\; \text{base}\cdot
            \max\!\left(0.5,\; 1 - \text{tighten}\cdot \text{depth\_frac}\right),

        where

        .. math::

            \text{base} \;=\; \text{base\_mul}\,\sqrt{\max(10^{-12}, \tfrac{1}{3}\operatorname{tr}(\mathbf{S}))}.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance.
        depth_frac : float, optional
            Fractional depth in the layer list (``0`` start → ``1`` end).
        base_mul : float or None, optional
            Base multiplier; defaults to :attr:`gate_multiplier`.
        tighten : float or None, optional
            Tightening factor; defaults to :attr:`gate_tighten`.

        Returns
        -------
        float
            Gate radius.
        """
        if base_mul is None:
            base_mul = self.gate_multiplier
        if tighten is None:
            tighten = self.gate_tighten
        base = float(base_mul * np.sqrt(max(1e-12, np.trace(S) / 3.0)))
        return base * max(0.5, 1.0 - float(tighten) * float(depth_frac))

    def _get_candidates_in_gate(self, pred_pos: np.ndarray, layer: Tuple[int,int], radius: float):
        r"""
        Get candidate hits within a radial gate.

        Parameters
        ----------
        pred_pos : ndarray, shape (3,)
            Predicted position.
        layer : tuple[int, int]
            ``(volume_id, layer_id)`` key.
        radius : float
            Radial gate in meters.

        Returns
        -------
        points_sel : ndarray, shape (M, 3)
            Points within ``radius``, distance-sorted and clipped to
            ``step_candidates`` (``M`` can be zero).
        ids_sel : ndarray, shape (M,)
            Corresponding hit IDs.

        Notes
        -----
        Deny-list handling is **not** applied here; see :meth:`_apply_deny`.
        A k-NN fallback caps fan-out if the radius query yields more than
        :attr:`ball_k_fallback` hits.
        """
        pred_pos = np.asarray(pred_pos, dtype=self.dtype)
        tree, points, ids = self.trees[layer]

        # 1) try radius query
        idxs = tree.query_ball_point(pred_pos, r=float(radius))
        if idxs:
            # Guard against absurd fan-out by falling back to kNN
            if len(idxs) > self.ball_k_fallback:
                k = min(max(self.step_candidates * 4, self.max_cands), len(points))
                dists, knn_idx = tree.query(pred_pos, k=k)
                knn_idx = np.atleast_1d(knn_idx)
                # keep only those within radius
                if np.ndim(dists) == 0:
                    dists = np.array([dists], dtype=self.dtype)
                mask = np.asarray(dists <= radius, dtype=bool)
                sel = knn_idx[mask]
                return points[sel], ids[sel]
            # Normal (bounded) radius flow
            pts = points[idxs]
            d2 = np.einsum('ij,ij->i', pts - pred_pos, pts - pred_pos)
            order = np.argpartition(d2, min(len(d2) - 1, self.step_candidates - 1))[:self.step_candidates]
            order = order[np.argsort(d2[order])]
            sel = np.asarray(idxs, dtype=np.int64)[order]
            return points[sel], ids[sel]

        # 2) empty ball: fall back to a small k-NN
        k = min(max(self.step_candidates * 2, self.max_cands), len(points))
        dists, knn_idx = tree.query(pred_pos, k=k)
        knn_idx = np.atleast_1d(knn_idx)
        if np.ndim(dists) == 0:
            dists = np.array([dists], dtype=self.dtype)
        order = np.argsort(dists)[:self.step_candidates]
        sel = knn_idx[order]
        return points[sel], ids[sel]
    
    def _layer_topk_candidates(
        self,
        x_pred: np.ndarray,
        S: np.ndarray,
        layer: Tuple[int, int],
        k: int,
        depth_frac: float = 0.0,
        gate_mul: float = None,
        gate_tighten: float = None,
        deny_hits: Optional[Sequence[int]] = None,
        apply_global_deny: bool = True,
    ):
        r"""
        Gate via radius, compute χ² vectorized, keep top-:math:`k` sorted by χ².

        Steps
        -----
        1. Compute gate radius :math:`r_{\text{gate}}` using
           :meth:`_gate_radius_trace` with optional ``gate_mul``/``gate_tighten``.
        2. Query KD-tree for points inside the ball and distance-prune to
           at most :attr:`step_candidates`.
        3. Optionally remove per-call deny hits (``deny_hits``).
        4. Compute vectorized χ² via ``backend.chi2_batch``:

           .. math:: \chi^2_i = (\mathbf{z}_i-\hat{\mathbf{x}}_{0:3})^\top
                                \mathbf{S}^{-1}(\mathbf{z}_i-\hat{\mathbf{x}}_{0:3}).
        5. Optionally apply the persistent global deny policy
           (:meth:`_apply_deny`).
        6. Return the top-:math:`k` by χ² (ascending).

        Parameters
        ----------
        x_pred : ndarray, shape (7,)
            Predicted state (only position is used here).
        S : ndarray, shape (3, 3)
            Innovation covariance.
        layer : tuple[int, int]
            Layer key.
        k : int
            Number of candidates to keep (upper bound).
        depth_frac : float, optional
            Progress through the layer list (``0`` start → ``1`` end) for
            gate tightening.
        gate_mul, gate_tighten : float or None, optional
            Overrides for :attr:`gate_multiplier` and :attr:`gate_tighten`.
        deny_hits : sequence[int] or None, optional
            Per-call deny list to drop before χ² computation.
        apply_global_deny : bool, optional
            Whether to apply the persistent global deny policy.

        Returns
        -------
        pts_sel : ndarray, shape (M, 3)
            Selected candidate points (``M<=k``).
        ids_sel : ndarray, shape (M,)
            Corresponding hit IDs.
        chi2_sel : ndarray, shape (M,)
            χ² values for the selected candidates (ascending).
        """
        r = self._gate_radius_trace(S, depth_frac, gate_mul, gate_tighten)
        pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, r)
        if ids.size == 0:
            return pts, ids, np.empty(0, dtype=self.dtype)

        # Filter per-call deny if provided
        if deny_hits:
            deny = set(int(h) for h in deny_hits)
            mask = np.fromiter((int(h) not in deny for h in ids), dtype=bool, count=len(ids))
            pts, ids = pts[mask], ids[mask]
            if ids.size == 0:
                return pts, ids, np.empty(0, dtype=self.dtype)

        # χ² (fast kernel)
        diff = pts - x_pred[:3]
        chi2 = self._backend.chi2_batch(diff, S).astype(self.dtype, copy=False)

        # Apply persistent global deny if requested
        if apply_global_deny and self._deny:
            pts, ids, chi2 = self._apply_deny(pts, ids, chi2)

        keep = int(min(k, len(chi2)))
        if keep <= 0:
            return pts[:0], ids[:0], chi2[:0]
        idx = np.argpartition(chi2, keep - 1)[:keep]
        idx = idx[np.argsort(chi2[idx])]
        return pts[idx], ids[idx], chi2[idx]

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
        mode = (self._deny_mode if mode is None else mode).lower()
        penalty = self._deny_penalty if penalty is None else float(penalty)
        deny_mask = np.fromiter((int(h) in self._deny for h in ids), dtype=bool, count=len(ids))
        if mode.startswith("hard"):
            keep = ~deny_mask
            if not np.any(keep):
                return np.empty((0, 3), dtype=pts.dtype), np.array([], dtype=ids.dtype), np.empty(0, dtype=base_cost.dtype)
            return pts[keep], ids[keep], base_cost[keep]
        # penalize
        return pts, ids, base_cost + deny_mask.astype(base_cost.dtype) * penalty

    def to_local_frame_jac(self, plane_normal: np.ndarray) -> np.ndarray:
        r"""
        Build a 2D local-frame Jacobian for a plane.

        Given a plane with unit normal :math:`\mathbf{w}`, this constructs
        orthonormal in-plane axes :math:`\mathbf{u},\mathbf{v}` such that
        :math:`[\,\mathbf{u}^\top;\ \mathbf{v}^\top\,]` maps a 3D position to
        2D local coordinates.

        Parameters
        ----------
        plane_normal : ndarray, shape (3,)
            Plane normal (will be normalized internally).

        Returns
        -------
        ndarray, shape (2, 3)
            Matrix with rows :math:`\mathbf{u}^{\mathsf{T}}` and
            :math:`\mathbf{v}^{\mathsf{T}}`.
        """
        w = np.asarray(plane_normal, dtype=self.dtype)
        w = w / np.linalg.norm(w)
        arbitrary = np.array([1, 0, 0], dtype=self.dtype) if abs(w[0]) < 0.9 else np.array([0, 1, 0], dtype=self.dtype)
        u = np.cross(arbitrary, w); u /= np.linalg.norm(u)
        v = np.cross(w, u)
        return np.vstack([u, v]).astype(self.dtype, copy=False)

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
            Local 2D coordinates
            :math:`[\mathbf{u}^{\mathsf{T}}(pos-p_0),\ \mathbf{v}^{\mathsf{T}}(pos-p_0)]`.
        cov_local : ndarray, shape (2, 2)
            Projected covariance
            :math:`\mathbf{H}\,\mathrm{Cov}_{xyz}\,\mathbf{H}^{\mathsf{T}}`,
            with :math:`\mathbf{H}=[\mathbf{u};\mathbf{v}]`.
        """
        H = self.to_local_frame_jac(plane_normal)
        meas = H @ (np.asarray(pos, self.dtype) - np.asarray(plane_point, self.dtype))
        cov_local = H @ np.asarray(cov, self.dtype)[:3, :3] @ H.T
        return meas, cov_local

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