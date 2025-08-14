import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Sequence
from scipy.spatial import cKDTree
from scipy.stats import chi2 as _chi2
import networkx as nx

from trackml_reco.branchers.brancher import Brancher


class HelixEKFPSOBrancher(Brancher):
    r"""
    EKF-based **Particle Swarm Optimization** (PSO) track builder — turbo edition.

    This brancher performs track finding by running a **particle swarm** over
    per-layer EKF-gated candidate sets. Each particle proposes one path
    (one chosen hit per layer), evaluated by a **single** EKF update per
    layer. Swarm **velocities** live sparsely over the discrete space of
    ``(layer, hit_id)`` pairs, biasing the sampling distribution toward
    historically good hits from the particle's personal best (pbest) and the
    global best (gbest).

    **State/measurement model.**
    The EKF state is
    :math:`\mathbf{x}=[x,y,z,v_x,v_y,v_z,\kappa]^\top\in\mathbb{R}^7`,
    the measurement is position :math:`\mathbf{z}\in\mathbb{R}^3`, and
    :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}` extracts position.
    For predicted :math:`(\hat{\mathbf{x}},\mathbf{P}^- )` at a layer:

    .. math::

        \mathbf{S}
        = \mathbf{H}\mathbf{P}^- \mathbf{H}^\top + \mathbf{R},\qquad
        \mathbf{K}
        = \texttt{kalman\_gain}(\mathbf{P}^-,\mathbf{H},\mathbf{S}), \\
        \mathbf{x}^+ = \hat{\mathbf{x}} + \mathbf{K}(\mathbf{z}-\hat{\mathbf{x}}_{0:3}),\qquad
        \mathbf{P}^+ \approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^- .

    **Per-layer gating.**
    Candidates are selected with :meth:`Brancher._layer_topk_candidates`, which
    gates by a trace-based radius and sorts by χ² (Cholesky solves; no inverses).
    A **quantile-scaled gate** is supported:

    .. math::

        r_\text{gate}
        \propto \text{gate\_multiplier}\;\sqrt{\tfrac{1}{3}\operatorname{tr}(\mathbf{S})}\cdot
        \underbrace{\sqrt{F^{-1}_{\chi^2_3}(\text{gate\_quantile})}}_{\text{gate\_qscale}}.

    **Swarm policy.**
    At a layer with gated candidates :math:`\{\mathbf{z}_j\}_{j=1}^m` and
    χ² costs :math:`\chi^2_j`, a particle chooses index :math:`j^\star` via an
    :math:`\varepsilon`-greedy mixture of three distributions:

    - inverse χ²: :math:`p^{(\chi^{-2})}_j \propto 1/(\chi^2_j+\varepsilon)`,
    - Boltzmann: :math:`p^{(\mathrm{boltz})}_j \propto e^{-(\chi^2_j-\min\chi^2)/T}`,
    - velocity prior: :math:`p^{(v)}_j \propto \max(0,v_{(layer,hit_j)})`,

    combined as :math:`0.5\,p^{(\chi^{-2})}+0.4\,p^{(\mathrm{boltz})}+0.1\,p^{(v)}`.
    With probability :math:`\varepsilon`, the argmin-χ² is chosen directly.

    **Velocity update (sparse, per-hit).**
    For a touched key :math:`k=(\text{layer},\text{hit\_id})`, define labels
    :math:`y_p,y_g\in\{0,1\}` indicating whether :math:`k` belongs to the
    particle's pbest or the global gbest path. The velocity is updated as

    .. math::

        v_{t+1}(k) = \omega\,v_t(k)
                     + c_1\,r_1\,(y_p - v_t(k))
                     + c_2\,r_2\,(y_g - v_t(k)),

    with :math:`r_1,r_2\sim\mathcal{U}(0,1)`. We clamp to
    :math:`|v_{t+1}|\le v_\text{clamp}` and drop near-zeros to keep the maps
    sparse.

    What makes this fast
    --------------------
    • **Fused EKF kernels** (no explicit inverses) from the base class:
      :meth:`Brancher._ekf_predict`, :meth:`Brancher._layer_topk_candidates`,
      :meth:`Brancher._ekf_update_meas`.  
    • **Sparse velocities** per ``(layer, hit)`` to minimize memory.  
    • **Single EKF update per layer** for the chosen hit (no per-candidate loop).  
    • **Annealed exploration** (ε-greedy and softmax temperature) for stable
      convergence.  
    • **Minimal allocations** via cached :math:`\mathbf{H}/\mathbf{I}`, dtype
      reuse, and ``__slots__``.  
    • Optional graph built **only** when requested.

    Parameters
    ----------
    trees : dict[tuple[int, int], tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) -> (tree, points, ids)`` where
        ``points`` is ``(N,3)`` and ``ids`` is ``(N,)``.
    layer_surfaces : dict[tuple[int, int], dict]
        Layer geometry: either ``{'type':'disk','n':(3,), 'p':(3,)}`` or
        ``{'type':'cylinder','R': float}``.
    noise_std : float, optional
        Measurement noise std (meters), sets :math:`\mathbf{R}=\sigma^2\mathbf{I}_3`.
        Default ``2.0``.
    B_z : float, optional
        Longitudinal field (Tesla), controls :math:`\omega=B_z\,\kappa\,p_T`.
        Default ``0.002``.
    n_particles : int, optional
        Swarm size. Default ``16``.
    n_iters : int, optional
        Maximum PSO iterations (with early stopping). Default ``8``.
    w, c1, c2 : float, optional
        Inertia and cognitive/social coefficients. Defaults ``0.6, 1.0, 2.0``.
    max_cands : int, optional
        KD-tree preselect limit passed to the base. Default ``10``.
    step_candidates : int, optional
        Per-layer Top-K after gating (also passed to the base). Default ``6``.
    gate_multiplier : float, optional
        Base trace-gate multiplier. Default ``3.0``.
    gate_quantile : float or None, optional
        If set, multiplies the gate by
        :math:`\sqrt{F^{-1}_{\chi^2_3}(\text{gate\_quantile})}`. Default ``0.997``.
    epsilon_greedy, eps_min : float, optional
        Probability of direct exploitation (argmin-χ²), annealed to ``eps_min``.
        Defaults ``0.10, 0.02``.
    temp0, temp_min : float, optional
        Initial and minimum Boltzmann temperatures for near-tie smoothing.
        Defaults ``0.60, 0.05``.
    v_clamp : float, optional
        Velocity magnitude clamp; near-zeros are sparsified. Default ``1.5``.
    patience : int, optional
        Early stop after this many non-improving iterations. Default ``3``.
    rng_seed : int or None, optional
        Seed for the internal RNG.

    Attributes
    ----------
    velocities : list[dict]
        Per-particle sparse velocity maps: ``[{layer: {hit_id: v}}]``.
    pbest : list[dict]
        Personal best summary per particle (score, traj, hit_ids, state, cov).
    gbest : dict
        Global best path summary (same fields as pbest entries).
    _H, _I : ndarray
        Cached measurement Jacobian and identity.
    _dtype : numpy dtype
        Numeric dtype inferred from layer banks.

    Notes
    -----
    - Deny lists may be passed per call (``deny_hits=...``) and are honored
      inside :meth:`Brancher._layer_topk_candidates`. Persistent deny behavior
      can be configured via :meth:`Brancher.set_deny_hits`.
    - Only **one** EKF update is performed per layer per particle (for the
      chosen hit); χ² for non-chosen candidates is used only for sampling.
    """

    __slots__ = (
        # config
        "layer_surfaces", "n_particles", "n_iters",
        "w", "c1", "c2",
        "step_candidates", "gate_multiplier", "gate_qscale",
        "epsilon_greedy", "eps_min", "temp0", "temp_min",
        "v_clamp", "patience",
        # cached
        "state_dim", "_H", "_I", "_dtype",
        # swarm state
        "velocities", "pbest", "gbest",
        # annealed scalars
        "_eps_now", "_temp_now", "_w_now",
    )

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
                 gate_quantile: Optional[float] = 0.997,
                 epsilon_greedy: float = 0.10,
                 eps_min: float = 0.02,
                 temp0: float = 0.60,
                 temp_min: float = 0.05,
                 v_clamp: float = 1.5,
                 patience: int = 3,
                 rng_seed: Optional[int] = None):

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands,
                         step_candidates=step_candidates)

        # config
        self.layer_surfaces = layer_surfaces
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.w, self.c1, self.c2 = float(w), float(c1), float(c2)
        self.step_candidates = int(step_candidates)
        self.gate_multiplier = float(gate_multiplier)
        self.gate_qscale = float(np.sqrt(_chi2.ppf(gate_quantile, df=3))) if gate_quantile else 1.0
        self.epsilon_greedy = float(epsilon_greedy)
        self.eps_min = float(eps_min)
        self.temp0 = float(temp0)
        self.temp_min = float(temp_min)
        self.v_clamp = float(v_clamp)
        self.patience = int(patience)

        # model size and caches
        self.state_dim = 7
        self._H = self.H_jac(None)               # constant 3x7
        self._I = np.eye(self.state_dim)

        # pick a consistent dtype from banks (fall back to float64)
        try:
            any_layer = next(iter(trees))
            self._dtype = trees[any_layer][1].dtype
        except Exception:
            self._dtype = np.float64

        # RNG via base helper so users can reseed with set_rng(...)
        self.set_rng(rng_seed)

        # swarm containers
        self.velocities: List[Dict[Tuple[int, int], Dict[int, float]]] = []
        self.pbest: List[Dict[str, Any]] = []
        self.gbest: Dict[str, Any] = {'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}

        # annealed scalars (set each iter)
        self._eps_now = self.epsilon_greedy
        self._temp_now = self.temp0
        self._w_now = self.w

    def _init_particles(self, layers: List[Tuple[int, int]]) -> None:
        r"""
        Initialize swarm structures (sparse velocities, pbest, gbest).

        Parameters
        ----------
        layers : list[tuple[int, int]]
            Ordered layer keys; used to seed per-particle sparse maps.

        Notes
        -----
        - Initializes ``velocities`` as a list of dicts:
          ``velocities[p][layer][hit_id] -> v``.
        - Resets each particle's personal best (pbest) and the global best (gbest).
        """
        # Sparse velocity maps per particle: {layer: {hit_id: v}}
        self.velocities = [{layer: {} for layer in layers} for _ in range(self.n_particles)]
        # (re)initialize personal/global bests
        self.pbest = [{'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}
                      for _ in range(self.n_particles)]
        self.gbest = {'score': np.inf, 'traj': [], 'hit_ids': [], 'state': None, 'cov': None}

    @staticmethod
    def _lin(a: float, b: float, t: int, T: int) -> float:
        r"""
        Linear interpolation between two scalars.

        Computes :math:`a + (b-a)\,\frac{t}{\max(1,T-1)}`.

        Parameters
        ----------
        a, b : float
            Start and end values.
        t : int
            Current iteration index.
        T : int
            Total number of iterations.

        Returns
        -------
        float
            Interpolated value.
        """
        T = max(1, T - 1)
        return a + (b - a) * (t / T)

    # Build one PSO-guided path (single EKF update per layer using chosen hit)
    def _build_path(self,
                    seed_xyz: np.ndarray,
                    t: Optional[np.ndarray],
                    layers: List[Tuple[int, int]],
                    velocity: Dict[Tuple[int, int], Dict[int, float]],
                    depth_start_idx: int,
                    build_graph: bool,
                    deny_hits: Optional[Sequence[int]]) -> Dict[str, Any]:
        r"""
        Build a single particle path with one EKF update per visited layer.

        For each layer :math:`\ell`:

        1. Solve :math:`\Delta t_\ell` to the surface and compute
           :math:`(\hat{\mathbf{x}}_\ell,\mathbf{P}^-_\ell,\mathbf{S}_\ell)` via
           :meth:`Brancher._ekf_predict`.
        2. Fetch the Top-:math:`K` gated candidates with
           :meth:`Brancher._layer_topk_candidates` using a **quantile-scaled**
           gate (see class Notes).
        3. Choose one candidate index :math:`j^\star` by an
           :math:`\varepsilon`-greedy mixture of inverse-χ², Boltzmann, and
           velocity priors (normalized).
        4. Apply a **single** EKF update with that hit and add its χ² to the
           cumulative score.

        Touched hits are recorded to sparsely update the velocity map.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Seed triplet used to initialize :math:`\mathbf{v}_0` and :math:`\kappa`.
        t : ndarray or None
            Optional timestamps; used for the initial :math:`\Delta t_0`.
        layers : list[tuple[int, int]]
            Ordered layer keys.
        velocity : dict
            Sparse velocity map for this particle: ``velocity[layer][hit_id] -> v``.
        depth_start_idx : int
            Index offset used when computing the depth fraction for gating.
        build_graph : bool
            If ``True``, constructs and returns a small path graph.
        deny_hits : sequence[int] or None
            Optional per-call deny list applied during gating.

        Returns
        -------
        result : dict
            Keys: ``'traj'``, ``'hit_ids'``, ``'state'``, ``'cov'``, ``'score'``,
            ``'graph'`` (``nx.DiGraph`` if requested, else empty), and
            ``'touched'`` mapping layers to lists of hit IDs that were considered.

        Notes
        -----
        The selection distribution at a layer is

        .. math::

            p = 0.5\,p^{(\chi^{-2})} + 0.4\,p^{(\mathrm{boltz})} + 0.1\,p^{(v)},

        followed by :math:`\varepsilon`-greedy exploitation of the minimum χ².
        """
        rng = self._rng
        H = self._H
        I = self._I
        dtype = self._dtype

        # seed state
        seed_xyz = seed_xyz.astype(dtype, copy=False)
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0.astype(dtype), np.array([k0], dtype=dtype)])
        cov = np.eye(self.state_dim, dtype=dtype) * 0.1

        traj: List[np.ndarray] = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        score = 0.0

        # track which hits we interacted with (for sparse velocity update)
        touched: Dict[Tuple[int, int], set] = {layer: set() for layer in layers}

        G = nx.DiGraph() if build_graph else None
        L = len(layers)

        for li, layer in enumerate(layers):
            # predict to the surface
            try:
                dt = self._solve_dt_to_surface(state, self.layer_surfaces[layer], dt_init=dt0)
            except Exception:
                break

            x_pred, P_pred, S, _Hcached = self._ekf_predict(state, cov, float(dt))

            # dynamic gate multiplier (quantile scaling + linear tightening)
            depth_frac = (depth_start_idx + li + 1) / max(1, L)
            gate_mul = self.gate_multiplier * self.gate_qscale

            # fetch per-layer top-K candidates (already deny-aware if provided)
            pts, ids, chi2 = self._layer_topk_candidates(
                x_pred, S, layer,
                k=max(1, min(self.step_candidates, 64)),
                depth_frac=depth_frac,
                gate_mul=gate_mul,
                gate_tighten=0.15,          # kept as sensible default
                deny_hits=deny_hits
            )
            m = ids.size
            if m == 0:
                break  # no viable hits

            # touched set (for sparse velocity update)
            if m <= 32:
                # small-set fast path
                for h in ids:
                    touched[layer].add(int(h))
            else:
                touched[layer].update(map(int, ids.tolist()))

            # ε-greedy exploit or sample from mixed distribution
            if m == 1:
                j = 0
            else:
                if rng.random() < self._eps_now:
                    j = int(np.argmin(chi2))
                else:
                    # inverse-χ² term
                    inv = 1.0 / (chi2 + 1e-6)
                    inv /= inv.sum()

                    # annealed Boltzmann term (near-tie smoother)
                    s = chi2 - chi2.min()
                    probs = np.exp(-s / max(1e-9, self._temp_now))
                    probs /= probs.sum()

                    # velocity bias (sparse map → array aligned to ids)
                    vel_map = velocity[layer]
                    v = np.fromiter((max(0.0, float(vel_map.get(int(h), 0.0))) for h in ids),
                                    dtype=float, count=m)
                    if v.sum() > 0:
                        v /= v.sum()

                    # 50/40/10 mixture (inv χ² / Boltzmann / velocity)
                    mix = 0.5 * inv + 0.4 * probs + 0.1 * v
                    mix_sum = mix.sum()
                    if not np.isfinite(mix_sum) or mix_sum <= 0:
                        j = int(np.argmin(chi2))
                    else:
                        mix /= mix_sum
                        j = int(rng.choice(m, p=mix))

            z = pts[j]
            # one EKF measurement update at chosen hit
            state, cov = self._ekf_update_meas(x_pred, P_pred, z, H, S)

            traj.append(z)
            hid = int(ids[j])
            hit_ids.append(hid)
            score += float(chi2[j])

            if build_graph:
                G.add_edge((li, tuple(traj[-2])), (li + 1, tuple(z)), cost=float(chi2[j]))

        return {
            "traj": traj,
            "hit_ids": hit_ids,
            "state": state,
            "cov": cov,
            "score": float(score),
            "graph": (G if build_graph else nx.DiGraph()),
            "touched": {k: list(v) for k, v in touched.items()}
        }

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: Optional[np.ndarray],
            plot_tree: bool = False,
            *,
            deny_hits: Optional[Sequence[int]] = None
            ) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Execute PSO over per-layer EKF shortlists (one chosen hit per layer).

        Pipeline
        --------
        1. **Init swarm**: sparse velocity maps, clear pbest/gbest.
        2. For iterations :math:`t=0,\dots,\text{n\_iters}-1`:
           a. **Anneal** schedules:
              :math:`\varepsilon_t=\mathrm{lin}(\varepsilon_0,\varepsilon_{\min},t)`,
              :math:`T_t=\mathrm{lin}(T_0,T_{\min},t)`,
              :math:`\omega_t=\mathrm{lin}(w,w_{\min},t)`.  
           b. **Evaluate** each particle with :meth:`_build_path`
              (EKF predict→choose→update across layers).  
           c. **Update** personal/global bests.  
           d. **Early stop** if gbest did not improve for ``patience`` rounds.  
           e. **Velocity update** only for **touched** keys using

              .. math::

                  v_{t+1} = \omega_t v_t + c_1 r_1 (y_p - v_t) + c_2 r_2 (y_g - v_t),

              followed by clamping/sparsification.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hits used to initialize the helix.
        layers : list[tuple[int, int]]
            Ordered layer keys to traverse.
        t : ndarray or None
            Optional timestamps aligned with the seed; if provided, the initial
            step uses :math:`\Delta t_0=t_1-t_0`, else ``1.0``.
        plot_tree : bool, optional
            If ``True``, returns a composed graph of all evaluated paths across
            iterations (sparse). Default ``False``.
        deny_hits : sequence[int] or None, keyword-only
            Per-call deny list; persistent behavior can be configured via
            :meth:`Brancher.set_deny_hits`.

        Returns
        -------
        branches : list[dict]
            Singleton list with the best branch:
            ``{'traj','hit_ids','state','cov','score'}``.
        G : networkx.DiGraph
            Aggregated sparse graph if requested; otherwise empty.

        Notes
        -----
        - The objective is **pure χ²** accumulated along the chosen path (deny
          penalties, if any, are introduced during gating when configured).
        - If ``layers`` is empty, returns ``([], nx.DiGraph())``.
        """
        if not layers:
            return [], nx.DiGraph()

        self._init_particles(layers)
        build_graph = bool(plot_tree)
        global_graph = nx.DiGraph() if build_graph else nx.DiGraph()

        best_prev = np.inf
        stale = 0

        deny_seq = list(map(int, deny_hits)) if deny_hits is not None else None

        for it in range(self.n_iters):
            # anneal schedules
            self._eps_now  = max(self.eps_min,  self._lin(self.epsilon_greedy, self.eps_min, it, self.n_iters))
            self._temp_now = max(self.temp_min, self._lin(self.temp0,          self.temp_min, it, self.n_iters))
            self._w_now    = self._lin(self.w, 0.35, it, self.n_iters)

            # 1) evaluate all particles
            paths = [
                self._build_path(seed_xyz, t, layers, self.velocities[p],
                                 depth_start_idx=0, build_graph=build_graph, deny_hits=deny_seq)
                for p in range(self.n_particles)
            ]
            if build_graph:
                for pth in paths:
                    global_graph = nx.compose(global_graph, pth["graph"])

            # 2) update personal/global bests
            for p, pth in enumerate(paths):
                if pth["score"] < self.pbest[p]["score"]:
                    self.pbest[p] = pth
                if pth["score"] < self.gbest["score"]:
                    self.gbest = pth

            # 3) early stop if stagnant
            if self.gbest["score"] + 1e-9 < best_prev:
                best_prev = self.gbest["score"]
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

            # 4) sparse velocity update on touched hits only
            rng = self._rng
            for p in range(self.n_particles):
                vel_p = self.velocities[p]
                pbest_hits = set(self.pbest[p]["hit_ids"])
                gbest_hits = set(self.gbest["hit_ids"])
                touched = paths[p]["touched"]  # dict[layer] -> list[int]
                for layer, ids in touched.items():
                    if not ids:
                        continue
                    # update velocities only for touched keys (keeps maps compact)
                    lp = vel_p[layer]
                    for hid in set(ids):
                        v_old = lp.get(hid, 0.0)
                        y_p = 1.0 if hid in pbest_hits else 0.0
                        y_g = 1.0 if hid in gbest_hits else 0.0
                        r1 = float(rng.random()); r2 = float(rng.random())
                        v_new = (self._w_now * v_old
                                 + self.c1 * r1 * (y_p - v_old)
                                 + self.c2 * r2 * (y_g - v_old))
                        # clamp and sparsify
                        if v_new >  self.v_clamp: v_new =  self.v_clamp
                        if v_new < -self.v_clamp: v_new = -self.v_clamp
                        if abs(v_new) < 1e-4:
                            if hid in lp: del lp[hid]
                        else:
                            lp[hid] = v_new

        # best result
        best = self.gbest
        out = {k: best[k] for k in ("traj", "hit_ids", "state", "cov", "score")}
        return [out], global_graph
