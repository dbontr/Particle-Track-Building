from __future__ import annotations

import time
import numpy as np
from typing import Tuple, Dict, List, Optional, Sequence, Any
from scipy.spatial import cKDTree
from scipy.stats import chi2 as _chi2
import networkx as nx

from trackml_reco.branchers.brancher import Brancher


class HelixEKFSABrancher(Brancher):
    r"""
    EKF + **Simulated Annealing** (SA) with *incremental tail rebuilds* and fused kernels.

    This brancher performs track finding by running **simulated annealing** over
    per-layer EKF-gated hit candidates. A *prefix cache* of EKF states enables
    **O(tail)** recomputation after a local change: when a single layer choice is
    perturbed, only the suffix (tail) of the path is rebuilt.

    **State & measurement model.**
    The EKF state is
    :math:`\mathbf{x}=[x,y,z,v_x,v_y,v_z,\kappa]^\top\in\mathbb{R}^7`,
    measurement is position :math:`\mathbf{z}\in\mathbb{R}^3`, and
    :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}` extracts position. With predicted
    :math:`(\hat{\mathbf{x}},\mathbf{P}^- )` at a layer:

    .. math::

        \mathbf{S}=\mathbf{H}\mathbf{P}^- \mathbf{H}^\top+\mathbf{R},\qquad
        \mathbf{K}=\texttt{kalman\_gain}(\mathbf{P}^-,\mathbf{H},\mathbf{S}),\\
        \mathbf{x}^+ = \hat{\mathbf{x}}+\mathbf{K}\bigl(\mathbf{z}-\hat{\mathbf{x}}_{0:3}\bigr),\qquad
        \mathbf{P}^+ \approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^- .

    **Annealing policy.**
    Let :math:`E` be the cumulative cost (sum of per-layer :math:`\chi^2` plus
    small regularizers). A candidate with :math:`\Delta = E_\text{new}-E_\text{cur}`
    is accepted with probability

    .. math::

        \Pr[\text{accept}] =
        \begin{cases}
        1, & \Delta < 0,\\
        \exp(-\Delta/T), & \Delta \ge 0,
        \end{cases}

    where :math:`T` is the temperature. Cooling is geometric with adaptive
    tweaks based on the *acceptance ratio* to mildly reheat or accelerate cooling.

    **Gating.**
    Per-layer candidates are obtained via a **trace-based** gate multiplied by a
    :math:`\chi^2_3`-quantile factor:

    .. math::

        r_\mathrm{gate}
        \propto \underbrace{\text{gate\_multiplier}}_{\alpha}\,
                   \sqrt{\tfrac{1}{3}\operatorname{tr}(\mathbf{S})}\;
                   \underbrace{\sqrt{F^{-1}_{\chi^2_3}(\text{gate\_quantile})}}_{\text{gate\_qscale}},
        \qquad 0<\text{gate\_quantile}<1.

    **Incremental tail rebuild.**
    The current best path caches per-layer tuples
    ``(state, cov, z, hit_id)``. When layer :math:`k` is mutated, we reuse the
    prefix :math:`0{:}k-1` and rebuild layers :math:`k{:}L-1` only.

    Key optimizations
    -----------------
    • Fused base-class ops (Cholesky-based; no explicit inverses):

      - :meth:`Brancher._ekf_predict` → :math:`(\hat{\mathbf{x}},\mathbf{P}^-,\mathbf{S},\mathbf{H})`
      - :meth:`Brancher._layer_topk_candidates` → gated Top-K with vectorized :math:`\chi^2`
      - :meth:`Brancher._ekf_update_meas` → stable update with one gain per update

    • Prefix cache for **O(tail)** rebuild after a local change.  
    • Quantile-scaled gates; depth-based linear tightening.  
    • Mutation layer is sampled with weights ∝ local residuals (focus where poor).  
    • Adaptive cooling with mild reheating from acceptance statistics.  
    • Graph assembly sampled every ``graph_stride`` steps when enabled.  
    • Deny-list supported (hard drop via constructor or per-call override).

    Parameters
    ----------
    trees : dict[tuple[int, int], tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) -> (tree, points, ids)`` with
        ``points`` of shape ``(N,3)`` and aligned ``ids`` of shape ``(N,)``.
    layer_surfaces : dict[tuple[int, int], dict]
        Geometry per layer: either
        ``{'type':'disk','n':(3,), 'p':(3,)}`` or ``{'type':'cylinder','R': float}``.
    noise_std : float, optional
        Measurement std in meters; sets :math:`\mathbf{R}=\sigma^2\mathbf{I}_3`. Default ``2.0``.
    B_z : float, optional
        Longitudinal field (Tesla); :math:`\omega=B_z\,\kappa\,p_T`. Default ``0.002``.
    initial_temp : float, optional
        Initial temperature :math:`T_0`. Default ``1.0``.
    cooling_rate : float, optional
        Geometric cooling factor :math:`T\leftarrow \text{cooling\_rate}\cdot T`. Default ``0.95``.
    n_iters : int, optional
        Nominal SA iterations (capped internally to a linear function of number of layers). Default ``1000``.
    max_cands : int, optional
        KD preselect bound passed to the base class. Default ``10``.
    step_candidates : int, optional
        Per-layer Top-K retained after gating. Default ``5``.
    max_no_improve : int, optional
        Stop if no improvement after this many accepted moves. Default ``150``.
    gate_multiplier : float, optional
        Multiplier :math:`\alpha` for trace-based gate. Default ``3.0``.
    gate_tighten : float, optional
        Linear tightening along depth :math:`d\in[0,1]`. Default ``0.15``.
    gate_quantile : float or None, optional
        If set, multiplies the gate by :math:`\sqrt{F^{-1}_{\chi^2_3}(q)}`. Default ``0.997``.
    time_budget_s : float or None, optional
        Optional wall-clock time budget (seconds). Default ``3.0``.
    build_graph : bool, optional
        If ``True``, assemble a sparse debug graph. Default ``False``.
    graph_stride : int, optional
        Add at most one edge every ``graph_stride`` layers when graphing. Default ``25``.
    min_temp : float, optional
        Temperature floor :math:`T_{\min}`. Default ``1e{-3}``.
    deny_hits : sequence[int] or None, optional
        Persistent (hard) deny-list applied in addition to any per-call list.

    Attributes
    ----------
    _H, _I : ndarray
        Cached measurement Jacobian and identity.
    _rng : numpy.random.Generator
        Internal RNG.
    _deny : set[int]
        Persistent hard deny-list.

    Notes
    -----
    - **Cost.** The objective minimized by SA is the sum of per-layer
      :math:`\chi^2`, plus small continuity regularizers (angle/curvature
      tie-breakers) during candidate proposal; the committed path stores plain
      :math:`\chi^2`.
    - **Deny.** You may also pass a *per-call* deny list via ``run(..., deny_hits=...)``.
    - **Complexity.** One mutation triggers a tail rebuild whose expected cost
      is :math:`O(L-k)` EKF steps (with Top-K shortlist at each layer).
    """

    __slots__ = (
        # config
        "layer_surfaces", "initial_temp", "cooling_rate", "n_iters",
        "step_candidates", "max_no_improve", "gate_multiplier", "gate_tighten",
        "gate_qscale", "time_budget_s", "build_graph", "graph_stride", "min_temp",
        # caches
        "state_dim", "_H", "_I", "_rng",
        # deny
        "_deny",
    )

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 initial_temp: float = 1.0,
                 cooling_rate: float = 0.95,
                 n_iters: int = 1000,
                 max_cands: int = 10,
                 step_candidates: int = 5,
                 max_no_improve: int = 150,
                 gate_multiplier: float = 3.0,
                 gate_tighten: float = 0.15,
                 gate_quantile: Optional[float] = 0.997,
                 time_budget_s: Optional[float] = 3.0,
                 build_graph: bool = False,
                 graph_stride: int = 25,
                 min_temp: float = 1e-3,
                 deny_hits: Optional[Sequence[int]] = None) -> None:

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands,
                         step_candidates=step_candidates)

        # config
        self.layer_surfaces = layer_surfaces
        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.n_iters = int(n_iters)
        self.step_candidates = int(step_candidates)
        self.max_no_improve = int(max_no_improve)
        self.gate_multiplier = float(gate_multiplier)
        self.gate_tighten = float(gate_tighten)
        self.gate_qscale = float(np.sqrt(_chi2.ppf(gate_quantile, df=3))) if gate_quantile else 1.0
        self.time_budget_s = None if time_budget_s is None else float(time_budget_s)
        self.build_graph = bool(build_graph)
        self.graph_stride = int(max(1, graph_stride))
        self.min_temp = float(min_temp)

        # model size + small caches
        self.state_dim = 7
        self._H = self.H_jac(None)              # constant 3×7
        self._I = np.eye(self.state_dim)
        self._rng = np.random.default_rng()

        # deny-list (hard)
        self._deny = set(int(h) for h in (deny_hits or []))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        r"""
        Safe unit normalization.

        Parameters
        ----------
        v : ndarray, shape (n,)
            Input vector.

        Returns
        -------
        ndarray, shape (n,)
            :math:`v/\|v\|` if the norm is nonzero; otherwise returns ``v``.
        """
        n = np.linalg.norm(v)
        return v if n < 1e-12 else (v / n)

    def _angle_pen(self, prev_vec: Optional[np.ndarray], cand_vec: np.ndarray, w_ang: float) -> float:
        r"""
        Angle continuity penalty.

        Computes :math:`w_\mathrm{ang}\,(1-\cos\theta)` where
        :math:`\theta` is the angle between ``prev_vec`` and ``cand_vec``.

        Parameters
        ----------
        prev_vec : ndarray or None, shape (3,)
            Previous step direction (can be ``None`` at the start).
        cand_vec : ndarray, shape (3,)
            Candidate step vector at the current layer.
        w_ang : float
            Weight in :math:`\chi^2` units.

        Returns
        -------
        float
            Penalty value (nonnegative).

        Notes
        -----
        Returns ``0.0`` when inputs are degenerate (near-zero norms) or
        ``w_ang <= 0``.
        """
        if prev_vec is None or w_ang <= 0.0:
            return 0.0
        a = np.linalg.norm(prev_vec); b = np.linalg.norm(cand_vec)
        if a < 1e-12 or b < 1e-12:
            return 0.0
        c = float(np.clip(np.dot(prev_vec, cand_vec) / (a * b), -1.0, 1.0))
        return w_ang * (1.0 - c)

    @staticmethod
    def _curv_pen(k_prev: float, k_new: float, w_curv: float) -> float:
        r"""
        Curvature-change penalty.

        Computes :math:`w_\kappa\,(\Delta\kappa)^2` with
        :math:`\Delta\kappa = \kappa_\text{new}-\kappa_\text{prev}`.

        Parameters
        ----------
        k_prev, k_new : float
            Previous and new curvature values.
        w_curv : float
            Weight in :math:`\chi^2` units.

        Returns
        -------
        float
            Penalty value (nonnegative).
        """
        if w_curv <= 0.0:
            return 0.0
        dk = float(k_new - k_prev)
        return w_curv * (dk * dk)

    # Use base fused gate+χ², with quantile-scaled multiplier + linear tightening
    def _layer_topk(self,
                    x_pred: np.ndarray,
                    S: np.ndarray,
                    layer: Tuple[int, int],
                    depth_frac: float,
                    deny_hits: Optional[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Gated Top-K candidates for a layer (wrapper around the base helper).

        Parameters
        ----------
        x_pred : ndarray, shape (7,)
            Predicted state.
        S : ndarray, shape (3, 3)
            Innovation covariance.
        layer : tuple[int, int]
            Layer key.
        depth_frac : float
            Depth fraction :math:`\in(0,1]` used for linear gate tightening.
        deny_hits : sequence[int] or None
            Optional per-call deny list.

        Returns
        -------
        pts : ndarray, shape (K, 3)
            Candidate positions (sorted by :math:`\chi^2` ascending).
        ids : ndarray, shape (K,)
            Candidate hit IDs.
        chi2 : ndarray, shape (K,)
            Corresponding :math:`\chi^2` values.
        """
        return self._layer_topk_candidates(
            x_pred, S, layer,
            k=max(1, min(self.step_candidates, 64)),
            depth_frac=depth_frac,
            gate_mul=(self.gate_multiplier * self.gate_qscale),
            gate_tighten=self.gate_tighten,
            deny_hits=deny_hits
        )

    # Build an initial greedy path and a prefix cache for fast tail rebuild
    def _greedy_with_prefix(self,
                            seed_xyz: np.ndarray,
                            t: Optional[np.ndarray],
                            layers: List[Tuple[int, int]],
                            deny_hits: Optional[Sequence[int]],
                            graph_every: int = 0) -> Dict[str, Any]:
        r"""
        Build a greedy initial path and the prefix cache.

        For each layer, pick the minimum-:math:`\chi^2` candidate within the gate,
        perform a single EKF update, and record the tuple
        ``(state, cov, z, hit_id)`` for prefix reuse.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Seed triplet used to bootstrap :math:`\mathbf{v}_0` and :math:`\kappa`.
        t : ndarray or None
            Optional timestamps; if provided, :math:`\Delta t_0 = t_1-t_0`, else ``1.0``.
        layers : list[tuple[int, int]]
            Ordered layer keys.
        deny_hits : sequence[int] or None
            Per-call deny list for gating.
        graph_every : int, optional
            When graphing is enabled, sample one edge every ``graph_every`` layers.

        Returns
        -------
        result : dict
            Keys: ``'traj','hit_ids','state','cov','score','residuals','prefix','graph'``.

        Notes
        -----
        The **prefix** list has length equal to the number of committed layers;
        entry ``i`` stores the EKF state *after* committing layer ``i``.
        """
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1

        traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        residuals: List[float] = []
        prefix: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        G = nx.DiGraph() if self.build_graph else nx.DiGraph()

        L = len(layers)
        for i, layer in enumerate(layers):
            # predict to surface (fused)
            try:
                dt = self._solve_dt_to_surface(state, self.layer_surfaces[layer], dt_init=dt0)
            except Exception:
                break
            x_pred, P_pred, S, H = self._ekf_predict(state, cov, float(dt))

            depth_frac = (i + 1) / max(1, L)
            pts, ids, chi2 = self._layer_topk(x_pred, S, layer, depth_frac, deny_hits)
            if ids.size == 0:
                break

            j = int(np.argmin(chi2))
            z = pts[j]; hid = int(ids[j]); c = float(chi2[j])

            # one EKF update (fused)
            state, cov = self._ekf_update_meas(x_pred, P_pred, z, H, S)

            traj.append(z)
            hit_ids.append(hid)
            residuals.append(c)
            prefix.append((state.copy(), cov.copy(), z.copy(), hid))

            if self.build_graph and graph_every > 0 and (i % graph_every == 0):
                G.add_edge((i, tuple(traj[-2])), (i + 1, tuple(z)), cost=c)

        return {
            "traj": traj,
            "hit_ids": hit_ids,
            "state": state,
            "cov": cov,
            "score": float(np.sum(residuals)) if residuals else 0.0,
            "residuals": residuals,
            "prefix": prefix,
            "graph": G
        }

    # Recompute from layer k onward, optionally forcing the hit at k
    def _rebuild_from(self,
                      k: int,
                      current: Dict[str, Any],
                      seed_xyz: np.ndarray,
                      t: Optional[np.ndarray],
                      layers: List[Tuple[int, int]],
                      deny_hits: Optional[Sequence[int]],
                      forced_hit: Optional[int] = None,
                      graph_every: int = 0) -> Dict[str, Any]:
        r"""
        Rebuild the path from layer ``k`` onward (tail), optionally forcing a hit at ``k``.

        The prefix up to (but not including) layer ``k`` is reused from
        ``current['prefix']``; layers ``k,k+1,\dots`` are recomputed greedily.

        Parameters
        ----------
        k : int
            Index of the first layer to rebuild (``0 <= k < L``).
        current : dict
            Current solution with keys ``'traj','hit_ids','state','cov','residuals','prefix'``.
        seed_xyz : ndarray, shape (3, 3)
            Seed triplet (only used when ``k==0``).
        t : ndarray or None
            Optional timestamps for initial :math:`\Delta t_0` when ``k==0``.
        layers : list[tuple[int, int]]
            Ordered layer keys.
        deny_hits : sequence[int] or None
            Per-call deny list for gating.
        forced_hit : int or None, optional
            If provided, enforce this hit ID at layer ``k`` (falls back to
            the cheapest in-gate hit if unavailable).
        graph_every : int, optional
            When graphing is enabled, sample one edge every ``graph_every`` layers.

        Returns
        -------
        result : dict
            Same structure as :meth:`_greedy_with_prefix`'s return.

        Notes
        -----
        The rebuild uses the same fused EKF predict/update and gated Top-K
        selection as the greedy initializer.
        """
        L = len(layers)
        assert 0 <= k < L

        if k == 0:
            dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
            v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
            state = np.hstack([seed_xyz[2], v0, k0])
            cov = np.eye(self.state_dim) * 0.1
            traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
            hit_ids: List[int] = []
            residuals: List[float] = []
            prefix: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
            G = nx.DiGraph() if self.build_graph else nx.DiGraph()
            start_i = 0
        else:
            state, cov, _, _ = current["prefix"][k - 1]
            traj = current["traj"][:k + 1]
            hit_ids = current["hit_ids"][:k]
            residuals = current["residuals"][:k]
            prefix = list(current["prefix"][:k])  # reuse prefix up to k-1
            G = nx.DiGraph() if self.build_graph else nx.DiGraph()
            start_i = k

        for i in range(start_i, L):
            layer = layers[i]
            try:
                dt = self._solve_dt_to_surface(state, self.layer_surfaces[layer])
            except Exception:
                break
            x_pred, P_pred, S, H = self._ekf_predict(state, cov, float(dt))

            depth_frac = (i + 1) / max(1, L)
            pts, ids, chi2 = self._layer_topk(x_pred, S, layer, depth_frac, deny_hits)
            if ids.size == 0:
                break

            if (i == k) and (forced_hit is not None):
                pos = np.where(ids == int(forced_hit))[0]
                j = int(pos[0]) if pos.size else int(np.argmin(chi2))
            else:
                j = int(np.argmin(chi2))

            z = pts[j]; hid = int(ids[j]); c = float(chi2[j])

            state, cov = self._ekf_update_meas(x_pred, P_pred, z, H, S)

            traj.append(z)
            hit_ids.append(hid)
            residuals.append(c)
            prefix.append((state.copy(), cov.copy(), z.copy(), hid))

            if self.build_graph and graph_every > 0 and (i % graph_every == 0):
                G.add_edge((i, tuple(traj[-2])), (i + 1, tuple(z)), cost=c)

        return {
            "traj": traj,
            "hit_ids": hit_ids,
            "state": state,
            "cov": cov,
            "score": float(np.sum(residuals)) if residuals else 0.0,
            "residuals": residuals,
            "prefix": prefix,
            "graph": G
        }

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: Optional[np.ndarray],
            plot_tree: bool = False,
            **kwargs) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Simulated Annealing with **incremental tail rebuilds**.

        Pipeline
        --------
        1. **Greedy init** with :meth:`_greedy_with_prefix` builds the initial path,
           residuals, and prefix cache.
        2. **SA loop** over at most ``n_iters`` iterations (also bounded by a
           layer-dependent cap and optional wall-time):
           a. Sample a layer :math:`k` with probability proportional to its
              current residual (softened).  
           b. Propose a small set of alternatives at layer :math:`k` (Top-M
              from the gated shortlist), apply continuity tie-breakers (angle,
              curvature), and draw one via Boltzmann at temperature :math:`T`.  
           c. **Rebuild tail** from :math:`k` with the chosen hit enforced using
              :meth:`_rebuild_from`.  
           d. **Metropolis accept** using :math:`\Pr[\text{accept}]=\min\{1,\exp(-\Delta/T)\}`.  
           e. **Adaptive cooling**: adjust :math:`T` based on acceptance ratio
              in a sliding window; also apply geometric cooling.

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
            If ``True``, returns a sparse debug graph of sampled/accepted edges.
        **kwargs
            - ``deny_hits`` (sequence[int] or None): per-call deny list that
              overrides the persistent set from the constructor.

        Returns
        -------
        branches : list[dict]
            Singleton list with keys:

            - ``'traj'`` : list of 3D points including the seed triplet,
            - ``'hit_ids'`` : committed hit IDs by layer,
            - ``'state'`` : final EKF state :math:`\in\mathbb{R}^7`,
            - ``'cov'`` : final covariance :math:`7\times 7`,
            - ``'score'`` : cumulative :math:`\chi^2`.
        G : networkx.DiGraph
            Sparse graph if enabled/requested; otherwise empty.

        Notes
        -----
        - The loop stops when any of these occur: temperature reaches
          ``min_temp``, time budget is exceeded, ``max_no_improve`` is hit, or
          the iteration cap is reached.
        - If ``layers`` is empty or no viable path exists after the greedy init,
          returns an empty branch and an empty graph.
        """
        if not layers:
            return [], nx.DiGraph()

        # per-call deny-list override
        deny_hits = kwargs.get("deny_hits", None)
        deny_hits = list(map(int, deny_hits)) if deny_hits is not None else (list(self._deny) or None)

        start = time.perf_counter()
        deadline = None if self.time_budget_s is None else (start + float(self.time_budget_s))

        # 1) Greedy init + prefix cache
        cur = self._greedy_with_prefix(seed_xyz, t, layers, deny_hits, graph_every=self.graph_stride)
        best = dict(cur)  # shallow copy ok (we replace wholesale on improvements)

        if not cur["hit_ids"]:
            # no viable path
            return [], (cur["graph"] if (self.build_graph or plot_tree) else nx.DiGraph())

        # annealing schedule (adaptive cap to keep runtime sane)
        L = len(layers)
        max_iters = min(int(self.n_iters), 80 + 12 * L)
        T = float(self.initial_temp)
        Tmin = float(self.min_temp)
        cool = float(self.cooling_rate)

        # modest continuity regularizers (tie-breakers vs χ²)
        w_ang = 6.0
        w_curv = 120.0

        # acceptance stats → adaptive cooling
        acc_cnt = 0
        tri_cnt = 0
        acc_win = 40

        rng = self._rng
        global_graph = cur["graph"] if (self.build_graph or plot_tree) else nx.DiGraph()
        no_imp = 0

        for it in range(max_iters):
            if T <= Tmin:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

            # weight layers by residual to focus where poor
            res = np.asarray(cur["residuals"], dtype=float)
            if res.size == 0:
                break
            # avoid zero-sum; soften weights
            w = res + (0.05 * res.mean() if res.mean() > 0 else 1e-6)
            w = w / w.sum()
            k = int(rng.choice(len(res), p=w))

            # state/cov up to k
            if k == 0:
                dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
                v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
                st = np.hstack([seed_xyz[2], v0, k0])
                cv = np.eye(self.state_dim) * 0.1
                prev_vec = None
                prev_kappa = float(st[6])
            else:
                st, cv, _, _ = cur["prefix"][k - 1]
                prev_vec = (np.asarray(cur["traj"][k]) - np.asarray(cur["traj"][k - 1])) if k >= 1 else None
                prev_kappa = float(st[6])

            # predict to layer k
            layer_k = layers[k]
            try:
                dt = self._solve_dt_to_surface(st, self.layer_surfaces[layer_k])
            except Exception:
                # failed propagation — cool and continue
                T *= cool
                continue

            x_pred, P_pred, S, H = self._ekf_predict(st, cv, float(dt))

            depth_frac = (k + 1) / max(1, L)
            pts, ids, chi2 = self._layer_topk(x_pred, S, layer_k, depth_frac, deny_hits)
            if ids.size == 0:
                T *= cool
                continue

            # exclude current choice at layer k
            cur_hid_k = int(cur["hit_ids"][k]) if k < len(cur["hit_ids"]) else None
            if cur_hid_k is not None:
                mask_alt = (ids != cur_hid_k)
                if not np.any(mask_alt):
                    T *= cool
                    continue
                ids_alt = ids[mask_alt]; pts_alt = pts[mask_alt]; chi2_alt = chi2[mask_alt]
            else:
                ids_alt, pts_alt, chi2_alt = ids, pts, chi2

            # evaluate up to M best alternatives with continuity penalties
            M = int(min(3, ids_alt.size))
            # already sorted by χ² from _layer_topk; take the first M
            cand_scores = []
            cand_hits = []
            for j in range(M):
                z = pts_alt[j]; hid = int(ids_alt[j])
                # predict-only penalties require kappa_new → do one-step update (fused)
                x_upd = x_pred  # avoid extra alloc; _ekf_update_meas returns new arrays
                x_upd, _ = self._ekf_update_meas(x_pred, P_pred, z, H, S)
                kappa_new = float(x_upd[6])

                ang = self._angle_pen(prev_vec, z - x_pred[:3], w_ang)
                curv = self._curv_pen(prev_kappa, kappa_new, w_curv)
                total = float(chi2_alt[j]) + ang + curv

                cand_scores.append(total)
                cand_hits.append((hid, z))

            s = np.asarray(cand_scores, dtype=float)
            m = float(s.min())
            probs = np.exp(-(s - m) / max(1e-9, T))
            probs /= probs.sum()
            pick = int(rng.choice(len(cand_hits), p=probs))
            hid_forced, _ = cand_hits[pick]

            # rebuild tail from k with forced hit
            cand = self._rebuild_from(k, cur, seed_xyz, t, layers, deny_hits,
                                      forced_hit=hid_forced, graph_every=self.graph_stride)

            # Metropolis accept on full Δ score
            delta = float(cand["score"] - cur["score"])
            accept = (delta < 0.0) or (rng.random() < np.exp(-delta / max(1e-9, T)))
            tri_cnt += 1
            if accept:
                cur = cand
                acc_cnt += 1
                if (self.build_graph or plot_tree):
                    global_graph = nx.compose(global_graph, cand["graph"])
                if cur["score"] < best["score"]:
                    best = cur
                    no_imp = 0
                else:
                    no_imp += 1
            else:
                no_imp += 1

            # adaptive cooling every acc_win trials; otherwise standard cooling
            if tri_cnt >= acc_win:
                ar = acc_cnt / max(1, tri_cnt)  # acceptance ratio
                if ar < 0.12:
                    T *= 1.04  # mild reheat
                elif ar > 0.55:
                    T *= 0.92  # cool a tad faster
                else:
                    T *= cool
                acc_cnt = 0
                tri_cnt = 0
            else:
                T *= cool

            if no_imp >= self.max_no_improve:
                break

        out = {
            "traj": best["traj"],
            "hit_ids": best["hit_ids"],
            "state": best["state"],
            "cov": best["cov"],
            "score": float(best["score"]),
        }
        return [out], global_graph
