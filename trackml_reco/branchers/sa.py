import time
import numpy as np
from typing import Tuple, Dict, List, Optional, Sequence
from scipy.spatial import cKDTree
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFSABrancher(Brancher):
    r"""
    EKF-based track finder using **Simulated Annealing (SA)** with incremental recomputation.

    The algorithm maintains a current path (one hit per layer) and proposes
    local mutations at a chosen layer :math:`k`. The tail from :math:`k`
    onward is **recomputed incrementally** using fast EKF prediction/update,
    making each proposal cheap. Acceptance is Metropolis–Hastings–style with a
    temperature schedule.

    Speed-oriented features
    -----------------------
    * **Per-seed wall-clock budget** (optional)
    * **Adaptive iteration cap** based on number of layers
    * **Trace-based gate radius** (avoids eigen-decomp)
    * **Optional graph recording** (off by default)

    Parameters
    ----------
    trees : dict[(int, int), tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) → (tree, points, ids)`` where
        ``points`` is ``(N,3)`` and ``ids`` is ``(N,)``.
    layer_surfaces : dict[(int, int), dict]
        Per-layer surface geometry. Either
        ``{'type': 'disk', 'n': normal_vec, 'p': point_on_plane}`` or
        ``{'type': 'cylinder', 'R': radius}``.
    noise_std : float, optional
        Measurement noise std (meters). Sets :math:`R=\sigma^2 I_3`. Default ``2.0``.
    B_z : float, optional
        Magnetic field (Tesla). Affects :math:`\omega = B_z\,\kappa\,p_T`.
        Default ``0.002``.
    initial_temp : float, optional
        Initial SA temperature :math:`T_0`. Default ``1.0``.
    cooling_rate : float, optional
        Multiplicative cooling factor :math:`T \leftarrow \alpha T`. Default ``0.95``.
    n_iters : int, optional
        Hard cap on iterations (also adaptively limited by problem size). Default ``1000``.
    max_cands : int, optional
        KD-tree neighbor upper bound (pre-gating). Default ``10``.
    step_candidates : int, optional
        Per-layer Top-:math:`K` kept after gating. Default ``5``.
    max_no_improve : int, optional
        Early stop after this many non-improving accepts/rejects. Default ``150``.
    gate_multiplier : float, optional
        Base gate factor in :math:`r = \text{mul}\,\sqrt{\operatorname{trace}(S)/3}`.
        Default ``3.0``.
    gate_tighten : float, optional
        Linear tightening with depth fraction:
        :math:`r \leftarrow r \cdot \max(0.5,\, 1 - \text{tighten}\cdot \text{depth\_frac})`.
        Default ``0.15``.
    time_budget_s : float or None, optional
        Wall-clock seconds budget per seed; ``None`` disables. Default ``3.0``.
    build_graph : bool, optional
        If ``True``, record a sparse debug graph. Default ``False``.
    graph_stride : int, optional
        Record every ``graph_stride``-th edge to limit graph size. Default ``25``.
    min_temp : float, optional
        Minimum temperature threshold to stop the anneal. Default ``1e-3``.
    deny_hits : sequence of int or None, optional
        Global deny-list of hit IDs to exclude.

    Notes
    -----
    The primary objective per candidate hit :math:`z` is the Mahalanobis distance

    .. math::

       \chi^2 = (z - \hat{x})^\mathsf{T} \, S^{-1} \, (z - \hat{x}),

    with :math:`S = H P_{\text{pred}} H^\mathsf{T} + R`. Two light regularizers
    (in :math:`\chi^2` units) may be added as tie-breakers:

    .. math::

       \text{angle} &= w_{\text{ang}} \, (1 - \cos\theta),\\
       \text{curv}  &= w_{\text{curv}} \, (\Delta\kappa)^2.

    The acceptance probability for a proposal with score change :math:`\Delta`
    at temperature :math:`T` is

    .. math::

       \Pr[\text{accept}] = \min\{1,\; e^{-\Delta / T}\}.
    """

    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int,int], dict],
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
                 time_budget_s: Optional[float] = 3.0,
                 build_graph: bool = False,
                 graph_stride: int = 25,
                 min_temp: float = 1e-3,
                 deny_hits: Optional[Sequence[int]] = None):
        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.n_iters = int(n_iters)
        self.step_candidates = int(step_candidates)
        self.max_no_improve = int(max_no_improve)
        self.gate_multiplier = float(gate_multiplier)
        self.gate_tighten = float(gate_tighten)
        self.state_dim = 7
        self._deny = set(int(h) for h in (deny_hits or []))

        # perf
        self.time_budget_s = time_budget_s
        self.build_graph = bool(build_graph)
        self.graph_stride = int(max(1, graph_stride))
        self.min_temp = float(min_temp)

    def _gate_radius_fast(self, S: np.ndarray, depth_frac: float) -> float:
        r"""
        Trace-based gating radius with linear tightening.

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance :math:`S = H P_{\text{pred}} H^\mathsf{T} + R`.
        depth_frac : float
            Progress through the layer list in :math:`[0,1]`.

        Returns
        -------
        float
            Radius

            .. math::

               r = \text{gate\_multiplier}
                   \cdot \sqrt{\tfrac{\operatorname{trace}(S)}{3}}
                   \cdot \max\!\bigl(0.5, 1 - \text{gate\_tighten}\cdot \text{depth\_frac}\bigr).
        """
        # Use sqrt(trace(S)/3) instead of max eigenvalue: ~same scale, much faster
        base = self.gate_multiplier * float(np.sqrt(max(1e-12, np.trace(S) / 3.0)))
        tighten = max(0.5, 1.0 - self.gate_tighten * depth_frac)
        return base * tighten

    def _layer_candidates(self,
                          x_pred: np.ndarray,
                          S: np.ndarray,
                          layer: Tuple[int,int],
                          depth_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute in-gate Top-:math:`K` candidates for a layer, sorted by :math:`\chi^2`.

        Parameters
        ----------
        x_pred : ndarray, shape (7,)
            Predicted state; only position :math:`\hat{x}\in\mathbb{R}^3` is used.
        S : ndarray, shape (3, 3)
            Innovation covariance for the layer.
        layer : tuple(int, int)
            Layer key.
        depth_frac : float
            Progress fraction in :math:`[0,1]` for gate tightening.

        Returns
        -------
        pts : ndarray, shape (K, 3)
            Candidate points (sorted by increasing :math:`\chi^2`).
        ids : ndarray, shape (K,)
            Corresponding hit IDs.
        chi2 : ndarray, shape (K,)
            Mahalanobis distances for the kept candidates.

        Notes
        -----
        Deny-listed hits are removed before ranking.
        """
        r = self._gate_radius_fast(S, depth_frac)
        pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, r)
        if len(ids) == 0:
            return pts, ids, np.empty(0)
        if self._deny:
            mask = np.array([int(h) not in self._deny for h in ids], dtype=bool)
            pts, ids = pts[mask], ids[mask]
            if len(ids) == 0:
                return pts, ids, np.empty(0)
        invS = np.linalg.inv(S)
        diff = pts - x_pred[:3]
        chi2 = np.einsum('ni,ij,nj->n', diff, invS, diff)
        keep = min(self.step_candidates, len(chi2))
        order = np.argpartition(chi2, keep-1)[:keep]
        # sort the kept small set
        order = order[np.argsort(chi2[order])]
        return pts[order], ids[order], chi2[order]

    def _greedy_with_prefix(self, seed_xyz, t, layers, graph_every=0):
        r"""
        Build an initial greedy path and a prefix cache for fast tail rebuilds.

        The greedy pass selects the minimum-:math:`\chi^2` candidate at each layer,
        performing EKF updates. Along the accepted path, a **prefix cache** stores
        state/covariance snapshots to enable :math:`O(\text{tail})` recomputation
        after a local change.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions.
        t : ndarray
            Seed times for initial velocity/curvature estimation.
        layers : list of tuple(int, int)
            Ordered layer keys.
        graph_every : int, optional
            If ``>0`` and graph recording is enabled, record every ``graph_every``-th edge.

        Returns
        -------
        out : dict
            Keys:
            * ``traj`` : list of positions,
            * ``hit_ids`` : list of IDs,
            * ``state`` : final EKF state,
            * ``cov`` : final covariance,
            * ``score`` : total :math:`\chi^2`,
            * ``residuals`` : per-layer :math:`\chi^2`,
            * ``prefix`` : list of tuples ``(state_k, cov_k, meas_k, id_k)`` for fast rebuild,
            * ``graph`` : :class:`networkx.DiGraph`.
        """
        dt0 = float(t[1] - t[0]) if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1

        traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        residuals: List[float] = []
        prefix: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []
        G = nx.DiGraph() if self.build_graph else None

        L = len(layers)
        H = self.H_jac(None)

        for i, layer in enumerate(layers):
            try:
                dt = self._solve_dt_to_surface(state, self.layer_surfaces[layer])
            except Exception:
                break
            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R

            depth_frac = (i + 1) / max(1, L)
            pts, ids, chi2 = self._layer_candidates(x_pred, S, layer, depth_frac)
            if len(ids) == 0:
                break

            j = int(np.argmin(chi2))
            chosen_pt, chosen_id, c = pts[j], int(ids[j]), float(chi2[j])

            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov   = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            hit_ids.append(chosen_id)
            residuals.append(c)
            prefix.append((state.copy(), cov.copy(), chosen_pt.copy(), chosen_id))

            if self.build_graph and (graph_every > 0) and (i % graph_every == 0):
                G.add_edge((i, tuple(traj[-2])), (i+1, tuple(chosen_pt)), cost=c)

        out = {
            'traj': traj,
            'hit_ids': hit_ids,
            'state': state,
            'cov': cov,
            'score': float(np.sum(residuals)) if residuals else 0.0,
            'residuals': residuals,
            'prefix': prefix,
            'graph': (G if G is not None else nx.DiGraph())
        }
        return out

    def _rebuild_from(self, k, current, seed_xyz, t, layers, forced_hit=None, graph_every=0):
        r"""
        Recompute the path **from layer k onward**, optionally forcing the hit at k.

        Uses the prefix cache (if :math:`k>0`) to resume from a saved state and then
        proceeds greedily as in :meth:`_greedy_with_prefix`.

        Parameters
        ----------
        k : int
            Index of the layer to mutate/rebuild from.
        current : dict
            Current solution as returned by :meth:`_greedy_with_prefix` or prior rebuilds.
        seed_xyz : ndarray, shape (3, 3)
            Seed hits.
        t : ndarray
            Seed times.
        layers : list of tuple(int, int)
            Ordered layer keys.
        forced_hit : int or None, optional
            If provided, try to use this hit ID at layer :math:`k` (falls back to best).
        graph_every : int, optional
            Graph sampling stride; see :meth:`_greedy_with_prefix`.

        Returns
        -------
        out : dict
            Same structure as :meth:`_greedy_with_prefix` return value, with an updated
            ``prefix`` reflecting the new accepted path.
        """
        layers_len = len(layers)
        assert 0 <= k < layers_len

        if k == 0:
            dt0 = float(t[1] - t[0]) if t is not None else 1.0
            v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
            state = np.hstack([seed_xyz[2], v0, k0])
            cov = np.eye(self.state_dim) * 0.1
            traj = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
            hit_ids: List[int] = []
            residuals: List[float] = []
            G = nx.DiGraph() if self.build_graph else None
            start_i = 0
        else:
            state, cov, _, _ = current['prefix'][k-1]
            traj = current['traj'][:k+1]
            hit_ids = current['hit_ids'][:k]
            residuals = current['residuals'][:k]
            G = nx.DiGraph() if self.build_graph else None
            start_i = k

        L = len(layers)
        H = self.H_jac(None)

        for i in range(start_i, L):
            layer = layers[i]
            try:
                dt = self._solve_dt_to_surface(state, self.layer_surfaces[layer])
            except Exception:
                break
            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R

            depth_frac = (i + 1) / max(1, L)
            pts, ids, chi2 = self._layer_candidates(x_pred, S, layer, depth_frac)
            if len(ids) == 0:
                break

            if (i == k) and (forced_hit is not None):
                idx = np.where(ids == forced_hit)[0]
                j = int(idx[0]) if idx.size else int(np.argmin(chi2))
            else:
                j = int(np.argmin(chi2))

            chosen_pt, chosen_id, c = pts[j], int(ids[j]), float(chi2[j])

            K = P_pred @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ (chosen_pt - x_pred[:3])
            cov   = (np.eye(self.state_dim) - K @ H) @ P_pred

            traj.append(chosen_pt)
            hit_ids.append(chosen_id)
            residuals.append(c)

            if self.build_graph and (graph_every > 0) and (i % graph_every == 0):
                G.add_edge((i, tuple(traj[-2])), (i+1, tuple(chosen_pt)), cost=c)

        # refresh prefix cache along accepted path (fast pass)
        prefix: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []
        dt0 = float(t[1] - t[0]) if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        st = np.hstack([seed_xyz[2], v0, k0])
        cv = np.eye(self.state_dim) * 0.1
        for i, layer in enumerate(layers[:len(hit_ids)]):
            try:
                dt = self._solve_dt_to_surface(st, self.layer_surfaces[layer])
            except Exception:
                break
            F = self.compute_F(st, dt)
            x_pred = self.propagate(st, dt)
            P_pred = F @ cv @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R
            meas = traj[i+1]
            K = P_pred @ H.T @ np.linalg.inv(S)
            st = x_pred + K @ (meas - x_pred[:3])
            cv = (np.eye(self.state_dim) - K @ H) @ P_pred
            prefix.append((st.copy(), cv.copy(), np.asarray(meas).copy(), hit_ids[i]))

        return {
            'traj': traj,
            'hit_ids': hit_ids,
            'state': st,
            'cov': cv,
            'score': float(np.sum(residuals)) if residuals else 0.0,
            'residuals': residuals,
            'prefix': prefix,
            'graph': (G if G is not None else nx.DiGraph())
        }

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: np.ndarray,
            plot_tree: bool = False,
            **kwargs) -> Tuple[List[Dict], nx.DiGraph]:
        """
        Run Simulated Annealing with incremental recomputation and light regularization.

        Workflow
        --------
        1. Build a greedy initialization and a prefix cache via :meth:`_greedy_with_prefix`.
        2. At each iteration, pick a layer :math:`k` to mutate (biased by local residual).
        3. Propose a new hit at :math:`k` (respecting gating and deny-list), and
           **rebuild the tail** via :meth:`_rebuild_from`.
        4. Accept with probability :math:`\min\{1, e^{-\Delta/T}\}`; cool :math:`T`.
        5. Stop on time budget, :math:`T \le \text{min\_temp}`, max iterations, or
           ``max_no_improve``.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions.
        layers : list of tuple(int, int)
            Ordered layer keys to traverse.
        t : ndarray
            Seed times for initial velocity/curvature estimation.
        plot_tree : bool, optional
            If ``True``, compose/return a debug graph of accepted steps.
        **kwargs
            Optional:
            * ``deny_hits`` : sequence[int], per-call deny-list (overrides constructor).

        Returns
        -------
        branches : list of dict
            Single best branch with:
            * ``traj`` : list of positions,
            * ``hit_ids`` : list[int],
            * ``state`` : final EKF state,
            * ``cov`` : final EKF covariance,
            * ``score`` : total :math:`\chi^2` (including accepted steps only).
        G : :class:`networkx.DiGraph`
            Debug graph (empty if graphing disabled).

        Notes
        -----
        * Uses the class-level time budget (``time_budget_s``) if provided,
          returning the best-so-far on timeout.
        * Regularizers (angle/curvature) are **kept small** so that
          :math:`\chi^2` dominates; they mainly break ties and suppress
          zig–zag behavior.
        * Gate radius uses the trace heuristic to avoid eigen decompositions.
        """
        import time
        start_time = time.perf_counter()
        time_budget = None if (self.time_budget_s is None) else float(self.time_budget_s)
        deadline = None if time_budget is None else (start_time + time_budget)

        # optional per-call denylist
        deny = kwargs.get('deny_hits', None)
        if deny is not None:
            self._deny = set(int(h) for h in deny)

        if not layers:
            return [], nx.DiGraph()

        # Greedy init + prefix cache (fast)
        cur = self._greedy_with_prefix(seed_xyz, t, layers)
        best = cur.copy()

        # If nothing built, bail early
        if len(cur['hit_ids']) == 0:
            return [], (cur['graph'] if self.build_graph else nx.DiGraph())

        # annealing schedule (adaptive)
        L = len(layers)
        # Cap iterations based on problem size to avoid runaway runtimes
        max_iters = min(int(self.n_iters), 80 + 12 * L)
        T = float(self.initial_temp)
        Tmin = float(getattr(self, "min_temp", 1e-3))
        cool = float(self.cooling_rate)

        # small regularizers to reduce wiggle 
        # Keep them modest: they act only as tie-breakers vs chi2
        w_ang = 6.0          # angle penalty weight (in chi2 units)
        w_curv = 120.0       # curvature change weight (in chi2 units)

        def _angle_penalty(prev_vec: Optional[np.ndarray], cand_vec: np.ndarray) -> float:
            if prev_vec is None:
                return 0.0
            a = np.linalg.norm(prev_vec)
            b = np.linalg.norm(cand_vec)
            if a < 1e-12 or b < 1e-12:
                return 0.0
            cos_th = float(np.clip(np.dot(prev_vec, cand_vec) / (a * b), -1.0, 1.0))
            return w_ang * (1.0 - cos_th)

        def _curv_penalty(k_prev: float, k_new: float) -> float:
            dk = float(k_new - k_prev)
            return w_curv * (dk * dk)

        # cached Jacobian
        H = self.H_jac(None)

        # Utility: compute gate + chi2 and return top K candidates quickly
        def _top_candidates(x_pred: np.ndarray, P_pred: np.ndarray, layer: Tuple[int, int], depth_frac: float):
            S = H @ P_pred @ H.T + self.R                         # 3x3
            r = self._gate_radius_fast(S, depth_frac)
            pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, r)
            if len(ids) == 0:
                return None
            if self._deny:
                mask = np.array([int(h) not in self._deny for h in ids], dtype=bool)
                pts, ids = pts[mask], ids[mask]
                if len(ids) == 0:
                    return None
            # chi2 = (p - μ)^T S^{-1} (p - μ) via solve (faster & stable)
            diff = pts - x_pred[:3]
            chi2 = np.einsum('ni,ni->n', diff @ np.linalg.solve(S, np.eye(3)), diff)
            kkeep = min(self.step_candidates, len(chi2))
            idx = np.argpartition(chi2, kkeep - 1)[:kkeep]
            idx = idx[np.argsort(chi2[idx])]
            return pts[idx], ids[idx], chi2[idx], S

        # fast Kalman update pieces
        I7 = np.eye(self.state_dim)

        def _ekf_update(x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray, S: np.ndarray):
            r"""
            One-step EKF measurement update for a position-only measurement.

            Returns
            -------
            x_upd : ndarray, shape (7,)
                Updated state.
            P_upd : ndarray, shape (7, 7)
                Updated covariance.
            """
            # K = P_pred H^T S^{-1}
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_upd = x_pred + K @ (z - x_pred[:3])
            P_upd = (I7 - K @ H) @ P_pred
            return x_upd, P_upd

        # For angle penalty we need a local direction:
        def _prev_dir_from_traj(traj: List[np.ndarray]) -> Optional[np.ndarray]:
            r"""
            Extract last segment direction from a trajectory.

            Returns
            -------
            ndarray or None
                Difference of last two positions, or ``None`` if unavailable.
            """
            if len(traj) < 2:
                return None
            return np.asarray(traj[-1]) - np.asarray(traj[-2])

        # main SA loop
        rng = np.random.default_rng()
        global_graph = (cur['graph'] if self.build_graph else nx.DiGraph())
        no_imp = 0

        for it in range(max_iters):
            # wall-clock check (keeps parallel responsive)
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if T <= Tmin:
                break

            # pick layer index to mutate, weighted by local residual (focus where it's bad)
            res = np.array(cur['residuals'] + [1e-6] * (len(cur['hit_ids']) - len(cur['residuals'])))
            if res.size == 0:
                break
            w = res / (res.sum() if res.sum() > 0 else 1.0)
            k = int(rng.choice(len(cur['hit_ids']), p=w))

            # state/cov up to layer k
            if k == 0:
                dt0 = float(t[1] - t[0]) if t is not None else 1.0
                v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
                st = np.hstack([seed_xyz[2], v0, k0])
                cv = np.eye(self.state_dim) * 0.1
                prev_vec = None
                prev_kappa = float(st[6])
            else:
                st, cv, _, _ = cur['prefix'][k - 1]
                prev_vec = _prev_dir_from_traj(cur['traj'][:k+1])
                prev_kappa = float(st[6])

            # propagate to layer k
            layer_k = layers[k]
            try:
                dt = self._solve_dt_to_surface(st, self.layer_surfaces[layer_k])
            except Exception:
                # failed propagation, just cool and continue
                T *= cool
                continue

            F = self.compute_F(st, dt)
            x_pred = self.propagate(st, dt)
            P_pred = F @ cv @ F.T + self.Q0 * dt

            # candidates at layer k
            depth_frac = (k + 1) / max(1, L)
            tc = _top_candidates(x_pred, P_pred, layer_k, depth_frac)
            if tc is None:
                T *= cool
                continue
            pts, ids, chi2, S = tc

            # must choose an alternative different from current
            cur_hid_k = int(cur['hit_ids'][k])
            mask_alt = np.array([int(h) != cur_hid_k for h in ids], dtype=bool)
            if not np.any(mask_alt):
                T *= cool
                continue

            # Evaluate up to 3 best alternatives with continuity penalties, pick one by softmax
            alt_idx = np.where(mask_alt)[0][:3]
            cand_scores = []
            cand_objs = []
            for j in alt_idx:
                z = pts[j]
                # one-step updated state to get κ_new for curvature penalty
                x_upd, P_upd = _ekf_update(x_pred, P_pred, z, S)
                kappa_new = float(x_upd[6])
                # penalties
                ang_pen = _angle_penalty(prev_vec, z - x_pred[:3])
                curv_pen = _curv_penalty(prev_kappa, kappa_new)
                total_local = float(chi2[j]) + ang_pen + curv_pen
                cand_scores.append(total_local)
                cand_objs.append((z, int(ids[j])))

            # Soft choice: sharper at low T, more exploratory at high T
            s = np.array(cand_scores, dtype=float)
            # numerically stable softmin
            m = s.min()
            probs = np.exp(-(s - m) / max(1e-9, T))
            probs /= probs.sum()
            j_pick = int(rng.choice(len(cand_objs), p=probs))
            z_forced, hid_forced = cand_objs[j_pick]

            # rebuild tail from k with the forced hit
            cand = self._rebuild_from(k, cur, seed_xyz, t, layers, forced_hit=hid_forced)

            # Metropolis accept on FULL score change (already includes continuity via chosen steps)
            delta = cand['score'] - cur['score']
            accept = (delta < 0) or (rng.random() < np.exp(-delta / max(1e-9, T)))
            if accept:
                cur = cand
                if self.build_graph:
                    global_graph = nx.compose(global_graph, cand['graph'])
                if cur['score'] < best['score']:
                    best = cur
                    no_imp = 0
                else:
                    no_imp += 1
            else:
                no_imp += 1

            # cooling & early stop
            T *= cool
            if no_imp >= self.max_no_improve:
                break

        result = {
            'traj': best['traj'],
            'hit_ids': best['hit_ids'],
            'state': best['state'],
            'cov': best['cov'],
            'score': best['score']
        }
        return [result], global_graph
