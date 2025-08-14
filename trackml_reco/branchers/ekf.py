import heapq
from typing import Tuple, Dict, List, Optional, Any, Sequence
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from trackml_reco.branchers.brancher import Brancher
from trackml_reco.ekf_kernels import kalman_gain


class HelixEKFBrancher(Brancher):
    r"""
    High-performance branching EKF track finder (beam search, vectorized updates).

    This brancher performs a **beam search** across ordered detector layers while
    propagating a 7-D helical state with an Extended Kalman Filter (EKF). At each
    layer, candidates are produced by χ² gating of KD-tree neighbors
    (:meth:`Brancher._layer_topk_candidates`). A **single** EKF gain
    :math:`\mathbf{K}` is computed per *(branch, layer)* and reused to update all
    candidates vectorized:

    .. math::

        \mathbf{x}^+_j &= \hat{\mathbf{x}} + \mathbf{K}\,\mathbf{r}_j,
        \quad \mathbf{r}_j = \mathbf{z}_j - \hat{\mathbf{x}}_{0:3}, \\
        \mathbf{P}^+     &\approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^-,
        \quad \forall j \in \text{candidates at layer } \ell,

    where :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}` extracts position and
    :math:`\mathbf{K}` is computed via a Cholesky solve (no inverses). The branch
    score :math:`g` accumulates per-step χ² (plus any deny penalties).

    Key optimizations
    -----------------
    - **One :math:`\mathbf{K}` per branch+layer:** compute EKF gain once; update all
      candidates with a single matrix multiply.
    - **KD-gating + χ²** via :meth:`Brancher._layer_topk_candidates`
      (fast Cholesky χ² + optional deny policy).
    - **Argpartition** for local Top-K and global beam pruning (linear-time select).
    - **Allocation hygiene:** cached :math:`\mathbf{H}`/:math:`\mathbf{I}`, dtype tracking,
      minimal temporaries.
    - **Sparse graph option:** only *surviving* branches record edges.
    - **Deny-list** supported (per call and/or persistent via :meth:`Brancher.set_deny_hits`).

    Parameters
    ----------
    trees : dict[tuple[int, int], tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) -> (tree, points, ids)`` where
        ``points`` has shape ``(N,3)`` and ``ids`` is aligned of shape ``(N,)``.
    layer_surfaces : dict[tuple[int, int], dict]
        Geometry per layer. Each value is either
        ``{'type':'disk','n':(3,), 'p':(3,)}`` or ``{'type':'cylinder','R': float}``.
    noise_std : float, optional
        Isotropic measurement std (meters); sets :math:`\mathbf{R}=\sigma^2\mathbf{I}_3`.
        Default ``2.0``.
    B_z : float, optional
        Longitudinal field (Tesla), influences
        :math:`\omega = B_z\,\kappa\,p_T`, with :math:`p_T=\sqrt{v_x^2+v_y^2}`.
        Default ``0.002``.
    num_branches : int, optional
        Beam width retained after each layer (total survivors). Default ``30``.
    survive_top : int, optional
        Deterministic elites per layer (lowest scores); the remainder are sampled
        uniformly for diversity. Default ``12``.
    max_cands : int, optional
        KD-tree preselect bound (forwarded to base). Default ``10``.
    step_candidates : int, optional
        Local Top-K per branch per layer (post-gate). Default ``5``.
    gate_multiplier : float, optional
        Base gate radius scale, combined with covariance via a trace heuristic and
        tapered by depth (see :meth:`_gate_radius_depth`). Default ``3.0``.
    gate_tighten : float, optional
        Linear taper factor along depth :math:`d\in[0,1]`. Default ``0.15``.
    build_graph : bool, optional
        If ``True``, record edges for *surviving* branches only (sparse graph).
        Default ``False``.

    Attributes
    ----------
    layer_surfaces : dict
        Cached geometry per layer.
    num_branches : int
        Beam width retained after each layer.
    survive_top : int
        Number of elites deterministically kept each layer.
    step_candidates : int
        Max candidates kept per branch per layer after gating.
    gate_multiplier, gate_tighten : float
        Gate radius controls; see :meth:`_gate_radius_depth`.
    build_graph : bool
        Whether to build the sparse survivor graph during :meth:`run`.
    _I, _H : ndarray
        Cached identity and measurement Jacobian.
    _rng : numpy.random.Generator
        RNG for diversity sampling.
    _dtype : numpy dtype
        Numeric dtype inferred from layer points.

    See Also
    --------
    :class:`Brancher` : EKF, gating, deny policy, geometry utils.
    :class:`HelixEKFAStarBrancher` : A* with physics penalties.
    :class:`HelixEKFACOBrancher` : Ant Colony Optimization variant.

    Notes
    -----
    - Gating and χ² are delegated to :meth:`Brancher._layer_topk_candidates`,
      which uses a Cholesky solve under the hood (no inverses).
    - The covariance update uses :math:`(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}^-`
      (Joseph form omitted for speed given Cholesky stability).
    - Deny lists may be supplied per call (``deny_hits=...``) or configured
      persistently via :meth:`Brancher.set_deny_hits`.
    """

    __slots__ = (
        "layer_surfaces", "num_branches", "survive_top", "step_candidates",
        "gate_multiplier", "gate_tighten", "build_graph",
        "_I", "_H", "_rng", "_dtype"
    )

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 num_branches: int = 30,
                 survive_top: int = 12,
                 max_cands: int = 10,
                 step_candidates: int = 5,
                 gate_multiplier: float = 3.0,
                 gate_tighten: float = 0.15,
                 build_graph: bool = False) -> None:

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands,
                         step_candidates=step_candidates)

        self.layer_surfaces = layer_surfaces
        self.num_branches   = int(num_branches)
        self.survive_top    = int(survive_top)
        self.step_candidates = int(step_candidates)
        self.gate_multiplier = float(gate_multiplier)
        self.gate_tighten     = float(gate_tighten)
        self.build_graph      = bool(build_graph)

        self.state_dim = 7
        self._I = np.eye(self.state_dim)
        self._H = self.H_jac(None)               # constant 3x7 extractor
        self._rng = np.random.default_rng()

        # pick a consistent float dtype (from any layer's points if available)
        try:
            any_layer = next(iter(trees))
            self._dtype = trees[any_layer][1].dtype
        except Exception:
            self._dtype = np.float64

    def _gate_radius_depth(self, S: np.ndarray, depth_frac: float) -> float:
        r"""
        Depth-tapered trace gate.

        The gate radius is computed from the innovation covariance trace and
        linearly tightened as we progress deeper into the layer stack:

        .. math::

            r_{\text{gate}}(\text{depth}) \;=\;
            \underbrace{\text{gate\_multiplier}\,
            \sqrt{\max(10^{-12},\tfrac{1}{3}\operatorname{tr}(\mathbf{S}))}}_{\text{trace baseline}}
            \cdot \max\!\bigl(0.5,\; 1 - \text{gate\_tighten}\cdot \text{depth\_frac}\bigr).

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance :math:`\mathbf{S}` at the current layer.
        depth_frac : float
            Progress through layers in ``[0, 1]`` (start → end).

        Returns
        -------
        float
            Gate radius :math:`r_{\text{gate}}`.
        """
        base = self.gate_multiplier * float(np.sqrt(max(1e-12, np.trace(S) / 3.0)))
        return base * max(0.5, 1.0 - self.gate_tighten * depth_frac)

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: Optional[np.ndarray],
            plot_tree: bool = False,
            *,
            deny_hits: Optional[Sequence[int]] = None
            ) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Execute beam search with EKF propagation and χ²-based scoring.

        Algorithm
        ---------
        For each surviving branch at layer :math:`\ell`:

        1. **Geometric step** to the layer surface by solving for
           :math:`\Delta t_\ell` with :meth:`Brancher._solve_dt_to_surface`.
        2. **EKF predict** to obtain :math:`(\hat{\mathbf{x}},\mathbf{P}^-,\mathbf{S})`
           via :meth:`Brancher._ekf_predict`.
        3. **Gate & score** candidates with :meth:`Brancher._layer_topk_candidates`
           to get Top-K by χ² (deny policy applied if configured).
        4. **Single-gain updates**: compute one
           :math:`\mathbf{K}=\texttt{kalman\_gain}(\mathbf{P}^-,\mathbf{H},\mathbf{S})`
           and update all candidates vectorized:

           .. math::

               \mathbf{x}^+_j = \hat{\mathbf{x}} + \mathbf{K}(\mathbf{z}_j-\hat{\mathbf{x}}_{0:3}), \qquad
               \mathbf{P}^+ = (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^-.

        5. **Expand** children with local costs (χ² plus any deny penalty) added
           to the parent's accumulated score :math:`g`.
        6. **Beam prune**: keep ``survive_top`` deterministic elites (lowest
           :math:`g`), then sample the remaining survivors uniformly from the rest
           to reach ``num_branches`` for diversity.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed points used to initialize a helix (two segments). Initial
            velocity and curvature are estimated via
            :meth:`Brancher._estimate_seed_helix`.
        layers : list[tuple[int, int]]
            Ordered layer keys to traverse.
        t : ndarray or None
            Optional timestamps aligned with ``seed_xyz``; if provided, the
            initial step uses :math:`\Delta t_0 = t[1]-t[0]`, else ``1.0``.
        plot_tree : bool, optional
            If ``True``, also return a sparse directed graph containing edges
            only for *survivors* (keeps it compact).
        deny_hits : sequence[int] or None, keyword-only
            Per-call deny list; persistent deny behavior can be set via
            :meth:`Brancher.set_deny_hits`.

        Returns
        -------
        branches : list[dict]
            Surviving branches sorted by total score :math:`g` (ascending). Each
            dict contains:

            - ``'traj'`` : list of 3D points (including the seed triplet).
            - ``'state'`` : final EKF state :math:`\in\mathbb{R}^7`.
            - ``'cov'`` : final covariance :math:`7\times 7`.
            - ``'score'`` : accumulated χ² (+ penalties if any).
            - ``'hit_ids'`` : list of chosen hit IDs along the path.
            - ``'id'``, ``'parent'`` : internal IDs for graph linking.
        G : networkx.DiGraph
            Sparse graph with edges between survivor nodes; edge ``data['cost']``
            stores the child's cumulative score at creation time.

        Notes
        -----
        - Gate multiplier is tapered with depth (see :meth:`_gate_radius_depth`)
          to reduce branching near the end.
        - Covariance :math:`\mathbf{P}^+` is **shared** among a branch's
          candidates at a layer because :math:`\mathbf{K}` and
          :math:`(\mathbf{I}-\mathbf{K}\mathbf{H})` do not depend on
          :math:`\mathbf{z}_j` for the position-only model.
        - If ``layers`` is empty, returns ``([], nx.DiGraph())``.
        """
        if not layers:
            return [], nx.DiGraph()

        # merge call-scoped denies with persistent deny-list (if mode='penalize', base will add cost)
        deny = set(map(int, deny_hits)) if deny_hits is not None else None

        do_graph = self.build_graph or bool(plot_tree)
        G = nx.DiGraph() if do_graph else nx.DiGraph()

        # ----- seed state from 3 seed hits -----
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz.astype(self._dtype), dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2].astype(self._dtype), v0.astype(self._dtype), np.array([k0], dtype=self._dtype)])
        P0 = np.eye(self.state_dim, dtype=self._dtype) * 0.1

        branches: List[Dict[str, Any]] = [{
            'id': 0,
            'parent': None,
            'traj': [seed_xyz[0].astype(self._dtype), seed_xyz[1].astype(self._dtype), seed_xyz[2].astype(self._dtype)],
            'state': x0,
            'cov': P0,
            'score': 0.0,
            'hit_ids': []
        }]

        next_id = 1
        N = len(layers)

        for i, layer in enumerate(layers):
            depth_frac = (i + 1) / max(1, N)
            expanded: List[Dict[str, Any]] = []

            # Tighten gates as we go deeper
            gate_mul = self.gate_multiplier * max(0.5, 1.0 - self.gate_tighten * depth_frac)

            for br in branches:
                st = br['state']
                cv = br['cov']

                # 1) Predict to the layer surface (fast base EKF predict)
                try:
                    dt = self._solve_dt_to_surface(st, self.layer_surfaces[layer])
                except Exception:
                    continue

                # Use base helper with fast kernels & caching
                x_pred, P_pred, S, H = self._ekf_predict(st, cv, dt)

                # 2) Gated candidates + χ² (already vectorized & sorted top-k inside)
                #    We ask for up to `step_candidates` best by χ² after gate.
                pts, ids, chi2 = self._layer_topk_candidates(
                    x_pred, S, layer,
                    k=max(1, min(self.step_candidates, 32)),
                    depth_frac=depth_frac,
                    gate_mul=gate_mul,
                    gate_tighten=self.gate_tighten,
                    deny_hits=deny
                )
                if ids.size == 0:
                    continue

                # 3) EKF update for all candidates with ONE gain
                #    K = kalman_gain(P_pred, H, S)  (stable via Cholesky)
                K = kalman_gain(P_pred, H, S).astype(self._dtype)          # (7,3)
                KH = K @ H
                I_KH = (self._I - KH)
                P_upd_shared = I_KH @ P_pred                                # shared for all candidates

                diff = (pts - x_pred[:3]).astype(self._dtype, copy=False)   # (m,3)
                x_upds = x_pred + diff @ K.T                                # (m,7)

                # 4) Expand children (use χ² as score; optional deny penalties already baked in if used)
                m = ids.size
                # Global add (keep Python loop to avoid large object creation; m ≤ step_candidates)
                for j in range(m):
                    z   = pts[j]
                    hid = int(ids[j])
                    c   = float(chi2[j])

                    node_id = next_id; next_id += 1
                    child = {
                        'id': node_id,
                        'parent': br['id'],
                        'traj': br['traj'] + [z],
                        'state': x_upds[j],
                        'cov': P_upd_shared,                 # same for all j
                        'score': br['score'] + c,
                        'hit_ids': br['hit_ids'] + [hid]
                    }
                    expanded.append(child)

            if not expanded:
                break

            # 5) Beam prune: keep `survive_top` elites; sample remainder uniformly (diversity)
            #    (Work with indices to minimize Python attr access.)
            scores = np.fromiter((b['score'] for b in expanded), dtype=self._dtype, count=len(expanded))
            # elites
            k_elite = min(self.survive_top, len(expanded))
            elite_idx = np.argpartition(scores, k_elite - 1)[:k_elite]
            # order elites
            elite_idx = elite_idx[np.argsort(scores[elite_idx])]

            survivors = [expanded[int(idx)] for idx in elite_idx]

            # fill remaining slots via random pick among the rest (without replacement)
            need = max(0, self.num_branches - len(survivors))
            if need > 0 and len(expanded) > k_elite:
                rest_idx_pool = np.setdiff1d(np.arange(len(expanded)), elite_idx, assume_unique=False)
                if need < rest_idx_pool.size:
                    pick = self._rng.choice(rest_idx_pool, size=need, replace=False)
                else:
                    pick = rest_idx_pool
                survivors.extend(expanded[int(i)] for i in pick)

            branches = survivors

            # 6) Optional: add only *survivor* edges to keep graph sparse
            if do_graph:
                for b in branches:
                    if b['parent'] is not None:
                        # parent node coordinate = last coord of its traj before this z
                        G.add_edge(b['parent'], b['id'], cost=float(b['score']))

        # Final ordering by score
        branches.sort(key=lambda b: b['score'])
        return branches[:self.num_branches], G
