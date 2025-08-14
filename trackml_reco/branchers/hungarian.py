import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Sequence
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import networkx as nx

from trackml_reco.branchers.brancher import Brancher


class HelixEKFHungarianBrancher(Brancher):
    r"""
    EKF track finding with **per-layer global one-to-one assignment** (Hungarian).

    This brancher runs a global assignment per layer to associate multiple
    **predicted tracks** (beam hypotheses) with a **union** of EKF-gated hit
    candidates, ensuring a one-to-one mapping between tracks and hits at each
    layer. Each hypothesis is predicted to the layer surface with an Extended
    Kalman Filter (EKF), candidates are shortlisted via χ² gating, their
    per-track χ² costs are placed into a **rectangular cost matrix**, and a
    Hungarian solve returns the optimal matching. Assigned tracks are then
    EKF-updated; unassigned tracks are dropped for that layer.

    **State/measurement convention.**
    The EKF state is
    :math:`\mathbf{x}=[x,y,z,v_x,v_y,v_z,\kappa]^\top\in\mathbb{R}^7`,
    measurement is position :math:`\mathbf{z}\in\mathbb{R}^3`, and the
    measurement Jacobian :math:`\mathbf{H}\in\mathbb{R}^{3\times 7}` extracts
    position. With predicted :math:`(\hat{\mathbf{x}},\mathbf{P}^- )`:

    .. math::

        \mathbf{S} = \mathbf{H}\mathbf{P}^- \mathbf{H}^\top + \mathbf{R},\qquad
        \mathbf{K} = \texttt{kalman\_gain}(\mathbf{P}^-,\mathbf{H},\mathbf{S}), \\
        \mathbf{x}^+ = \hat{\mathbf{x}} + \mathbf{K}(\mathbf{z} - \hat{\mathbf{x}}_{0:3}),\qquad
        \mathbf{P}^+ \approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^- .

    **Per-layer assignment.**
    Suppose at layer :math:`\ell` we have :math:`T` surviving hypotheses and a
    union of :math:`M` distinct candidate hits from all per-track shortlists.
    We build a cost matrix :math:`\mathbf{C}\in\mathbb{R}^{T\times M}`:

    .. math::

        C_{r,c} = \begin{cases}
        \chi^2(\mathbf{z}_c \mid \text{track } r), & \text{if } \mathbf{z}_c
        \text{ is in the shortlist of } r,\\[4pt]
        \mathrm{BIG}, & \text{otherwise},
        \end{cases}

    where :math:`\chi^2(\mathbf{z}) = (\mathbf{z}-\hat{\mathbf{x}}_{0:3})^\top
    \mathbf{S}^{-1}(\mathbf{z}-\hat{\mathbf{x}}_{0:3})` and ``BIG`` is a large
    sentinel (effectively forbids the pairing). A single call to
    :func:`scipy.optimize.linear_sum_assignment` yields the minimum-cost
    one-to-one pairing between tracks and union hits (rectangular inputs allowed).
    Only pairs with finite/valid costs are committed; others are ignored.

    Key optimizations
    -----------------
    - Uses base-class fast kernels:
      * :meth:`Brancher._ekf_predict` (predict + :math:`\mathbf{S}`; single allocation path)
      * :meth:`Brancher._layer_topk_candidates` (KD gate + χ² via Cholesky; deny-aware)
      * :meth:`Brancher._ekf_update_meas` (gain via Cholesky; numerically stable)
    - No matrix inverses; avoids Python loops where possible.
    - Minimal allocations: cached :math:`\mathbf{H}/\mathbf{I}`, dtype from banks, ``__slots__``.
    - Rectangular cost matrices (no padding) → direct Hungarian solve.
    - Global beam prune by cumulative χ²; graphs built only on demand.

    Parameters
    ----------
    trees : dict[tuple[int, int], tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) -> (tree, points, ids)`` where
        ``points`` has shape ``(N,3)`` and ``ids`` is aligned of shape ``(N,)``.
    layer_surfaces : dict[tuple[int, int], dict]
        Geometry per layer. Each value is either
        ``{'type':'disk','n':(3,), 'p':(3,)}`` or ``{'type':'cylinder','R': float}``.
    noise_std : float, optional
        Measurement noise std (meters); sets :math:`\mathbf{R}=\sigma^2\mathbf{I}_3`.
        Default ``2.0``.
    B_z : float, optional
        Longitudinal field (Tesla), influences
        :math:`\omega = B_z\,\kappa\,p_T`, with :math:`p_T=\sqrt{v_x^2+v_y^2}`.
        Default ``0.002``.
    max_cands : int, optional
        KD-tree preselect upper bound passed down to the base class. Default ``20``.
    beam_size : int, optional
        Maximum number of concurrent hypotheses kept after each layer. Default ``4``.
    step_candidates : int, optional
        Per-track Top-K kept *after gating* (cheap beam at the track level). Default ``6``.
    gate_multiplier : float, optional
        Base multiplier for :meth:`Brancher._layer_topk_candidates` gate. Default ``3.0``.
    gate_tighten : float, optional
        Linear tightening factor along traversal depth :math:`d\in[0,1]`. Default ``0.15``.
    build_graph : bool, optional
        If ``True`` (or when ``plot_tree=True`` in :meth:`run`), assemble a sparse
        debug graph of survivor edges. Default ``False``.

    Attributes
    ----------
    beam_size, step_candidates, gate_multiplier, gate_tighten, build_graph
        Algorithmic controls (see Parameters).
    _H, _I : ndarray
        Cached measurement Jacobian and identity for EKF updates.
    _dtype : numpy dtype
        Numeric dtype inferred from the first available layer bank.

    Notes
    -----
    - Within each layer: build per-track Top-K shortlists → take the **union of
      hits** → solve a single **tracks × union hits** assignment → EKF-update
      assigned tracks. Tracks without a valid assignment at a layer are dropped.
    - Deny lists can be passed per call (``deny_hits=...``) or configured
      persistently via :meth:`Brancher.set_deny_hits`; the gating helper is
      deny-aware and can also apply additive penalties in ``"penalize"`` mode.
    """

    __slots__ = (
        "layer_surfaces", "beam_size", "step_candidates",
        "gate_multiplier", "gate_tighten", "build_graph",
        "state_dim", "_H", "_I", "_dtype"
    )

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 max_cands: int = 20,
                 beam_size: int = 4,
                 step_candidates: int = 6,
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
        self.beam_size = int(beam_size)
        self.step_candidates = int(step_candidates)
        self.gate_multiplier = float(gate_multiplier)
        self.gate_tighten = float(gate_tighten)
        self.build_graph = bool(build_graph)

        self.state_dim = 7
        self._H = self.H_jac(None)                 # constant 3x7 extractor
        self._I = np.eye(self.state_dim)

        # Choose a consistent dtype from any bank; fall back to float64
        try:
            any_layer = next(iter(trees))
            self._dtype = trees[any_layer][1].dtype
        except Exception:
            self._dtype = np.float64

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: Optional[np.ndarray],
            plot_tree: bool = False,
            *,
            deny_hits: Optional[Sequence[int]] = None
            ) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Execute beam search with **per-layer global assignment** and EKF updates.

        Pipeline
        --------
        1. **Seed**: estimate initial velocity/curvature from the seed triplet
           (:meth:`Brancher._estimate_seed_helix`) to form :math:`(\mathbf{x}_0,\mathbf{P}_0)`.
        2. For each layer :math:`\ell` (in order):
           a. For each hypothesis, solve :math:`\Delta t_\ell` to the surface and
              compute :math:`(\hat{\mathbf{x}}_\ell,\mathbf{P}^-_\ell,\mathbf{S}_\ell)`
              via :meth:`Brancher._ekf_predict`.  
           b. **Gate & shortlist** with :meth:`Brancher._layer_topk_candidates`
              (Top-K by χ²; deny-aware); collect *per-track* lists.  
           c. Build the **union** of candidate hits and a rectangular cost matrix
              :math:`\mathbf{C}\in\mathbb{R}^{T\times M}` with entries

              .. math::

                  C_{r,c} = \begin{cases}
                  \chi^2(\mathbf{z}_c \mid r), & \text{if } \mathbf{z}_c \in \text{shortlist}(r),\\
                  \mathrm{BIG}, & \text{otherwise}.
                  \end{cases}

           d. Solve the Hungarian assignment on :math:`\mathbf{C}` and **update**
              assigned tracks with :math:`\mathbf{x}^+_\ell = \hat{\mathbf{x}}_\ell +
              \mathbf{K}_\ell(\mathbf{z}_\ell - \hat{\mathbf{x}}_{\ell,0:3})`,
              :math:`\mathbf{K}_\ell=\texttt{kalman\_gain}(\mathbf{P}^-_\ell,\mathbf{H},\mathbf{S}_\ell)`.  
           e. **Prune** to the best ``beam_size`` hypotheses by cumulative score.

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
            If ``True``, return a sparse edge graph of survivor transitions.
        deny_hits : sequence[int] or None, keyword-only
            Per-call deny list; persistent behavior may be configured via
            :meth:`Brancher.set_deny_hits`.

        Returns
        -------
        branches : list[dict]
            The best branch (singleton list) with keys:

            - ``'traj'`` : list of 3D points (includes the seed triplet).
            - ``'hit_ids'`` : chosen hit IDs per committed layer.
            - ``'state'`` : final EKF state :math:`\in\mathbb{R}^7`.
            - ``'cov'`` : final covariance :math:`7\times 7`.
            - ``'score'`` : accumulated χ².
        G : networkx.DiGraph
            Sparse survivor graph if requested, else an empty graph. Each edge
            stores the committed pair's χ² in ``data['cost']``.

        Notes
        -----
        - The assignment is **rectangular** and does not require padding; only
          valid pairings (finite costs) are committed.
        - Tracks with empty gates or without a valid assignment at a layer are
          dropped; if the beam empties, the search stops early.
        - Deny lists interact with gating; in ``"penalize"`` mode, additional
          costs can be applied via :meth:`Brancher._apply_deny`.
        """
        if not layers:
            return [], nx.DiGraph()

        # --- Seed EKF from the 3 seed hits
        seed_xyz = seed_xyz.astype(self._dtype, copy=False)
        dt0: float = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0.astype(self._dtype), np.array([k0], dtype=self._dtype)])
        P0 = np.eye(self.state_dim, dtype=self._dtype) * 0.1

        def _new_hypo(state: np.ndarray, cov: np.ndarray) -> Dict[str, Any]:
            return {
                "state": state.copy(),
                "cov": cov.copy(),
                "traj": [seed_xyz[0], seed_xyz[1], seed_xyz[2]],
                "hit_ids": [],           # type: List[int]
                "score": 0.0
            }

        beam: List[Dict[str, Any]] = [_new_hypo(x0, P0)]
        do_graph = self.build_graph or bool(plot_tree)
        G = nx.DiGraph() if do_graph else nx.DiGraph()

        H = self._H
        Ltot = len(layers)
        deny_arr = list(map(int, deny_hits)) if deny_hits is not None else None

        for i, layer in enumerate(layers):
            depth_frac = (i + 1) / max(1, Ltot)

            # --- Predict & shortlist per hypothesis
            per_track: List[Dict[str, Any]] = []
            union_ids_list: List[int] = []
            union_pts_list: List[np.ndarray] = []

            for h in beam:
                st = h["state"]; cv = h["cov"]
                try:
                    dt = self._solve_dt_to_surface(st, self.layer_surfaces[layer], dt_init=dt0)
                except Exception:
                    # Drop this hypothesis if it can't intersect this surface
                    continue

                # Fast predict (x_pred, P_pred, S, H)
                x_pred, P_pred, S, _H_cached = self._ekf_predict(st, cv, float(dt))

                # Top-K candidates via base helper (gating + vec χ² internally)
                pts_k, ids_k, chi2_k = self._layer_topk_candidates(
                    x_pred, S, layer,
                    k=max(1, min(self.step_candidates, 64)),
                    depth_frac=depth_frac,
                    gate_mul=self.gate_multiplier,
                    gate_tighten=self.gate_tighten,
                    deny_hits=deny_arr
                )
                if ids_k.size == 0:
                    continue

                # Optional global deny policy (hard/penalize); already deny-aware above,
                # but allow additive penalties if global mode="penalize".
                if getattr(self, "_apply_deny", None) is not None and self._deny:
                    pts_k, ids_k, chi2_k = self._apply_deny(pts_k, ids_k, chi2_k)

                per_track.append({
                    "x_pred": x_pred,
                    "P_pred": P_pred,
                    "S": S,
                    "pts": pts_k.astype(self._dtype, copy=False),
                    "ids": ids_k.astype(int, copy=False),
                    "chi2": chi2_k.astype(self._dtype, copy=False),
                    "hypo": h
                })

                # Accumulate for union set
                union_pts_list.append(pts_k)
                union_ids_list.extend(map(int, ids_k))

            if not per_track:
                break  # no viable tracks/candidates at this layer

            if not union_ids_list:
                break

            # --- Build union candidate set (dedup) and id→col map
            if len(union_pts_list) == 1:
                all_pts = union_pts_list[0]
            else:
                all_pts = np.concatenate(union_pts_list, axis=0)

            union_ids, first_idx = np.unique(np.asarray(union_ids_list, dtype=int), return_index=True)
            union_pts = all_pts[first_idx]                        # representative positions
            M = union_ids.size
            T = len(per_track)

            # id -> column index (fast fill of the cost matrix)
            id2col = {int(h): j for j, h in enumerate(union_ids)}

            # --- Tracks × union hits cost matrix (fill with sentinel)
            BIG = 1e12
            C = np.full((T, M), BIG, dtype=self._dtype)
            for r, row in enumerate(per_track):
                # Map this row's ids into union cols, then place χ²
                if row["ids"].size == 0:
                    continue
                cols = np.fromiter((id2col[int(h)] for h in row["ids"]), dtype=int, count=row["ids"].size)
                C[r, cols] = row["chi2"]

            # --- Hungarian assignment (rectangular OK)
            row_ind, col_ind = linear_sum_assignment(C)

            # --- Build next beam from valid assignments
            new_beam: List[Dict[str, Any]] = []
            for r, c in zip(row_ind, col_ind):
                cost = float(C[r, c])
                if cost >= BIG * 0.5:         # invalid / out-of-gate pairing → drop
                    continue

                row = per_track[r]
                hid = int(union_ids[c])
                z = union_pts[c]              # measurement (3,)

                # EKF update (fast, Cholesky-based gain)
                x_upd, P_upd = self._ekf_update_meas(row["x_pred"], row["P_pred"], z, H, row["S"])

                old = row["hypo"]
                nh = {
                    "state": x_upd,
                    "cov": P_upd,
                    "traj": old["traj"] + [z],
                    "hit_ids": old["hit_ids"] + [hid],
                    "score": old["score"] + cost
                }
                new_beam.append(nh)

                if do_graph:
                    G.add_edge((i, tuple(old["traj"][-1])), (i + 1, tuple(z)), cost=cost)

            if not new_beam:
                break

            # Global beam prune by cumulative score
            new_beam.sort(key=lambda h: h["score"])
            beam = new_beam[:self.beam_size]

        if not beam:
            return [], G

        best = min(beam, key=lambda h: h["score"])
        out = {
            "traj": best["traj"],
            "hit_ids": best["hit_ids"],
            "state": best["state"],
            "cov": best["cov"],
            "score": float(best["score"])
        }
        return [out], G
