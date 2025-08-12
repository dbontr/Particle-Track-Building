import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Sequence
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import networkx as nx
from trackml_reco.branchers.brancher import Brancher


class HelixEKFHungarianBrancher(Brancher):
    r"""
    EKF-based track finder with **global one-to-one hit assignment per layer**
    using the Hungarian (linear sum) algorithm.

    For each detector layer:

    1. Propagate every active track hypothesis to the layer surface with the EKF.
    2. For each track, gate hits and keep a small per-track shortlist by the
       Mahalanobis distance :math:`\chi^2`.
    3. Form the **union** of all shortlisted hits across tracks and build a
       cost matrix (tracks :math:`\times` union hits) using :math:`\chi^2`
       (very large cost if a hit is not in a track's gate).
    4. Solve a minimum-cost **one-to-one** assignment with
       :func:`scipy.optimize.linear_sum_assignment`.
    5. Update assigned tracks; unassigned tracks are dropped.
    6. Keep a fixed-size beam of the best hypotheses by cumulative score and continue.

    This enforces collision-free hit usage **within a layer** while remaining greedy
    across layers, yielding a fast and parallel-friendly search.

    Parameters
    ----------
    trees : dict[(int, int), tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) → (tree, points, ids)`` with
        ``points`` of shape ``(N, 3)`` and ``ids`` of shape ``(N,)``.
    layer_surfaces : dict[(int, int), dict]
        Surface geometry per layer. Either
        ``{'type': 'disk', 'n': n_vec, 'p': point_on_plane}`` or
        ``{'type': 'cylinder', 'R': radius}``.
    noise_std : float, optional
        Isotropic measurement std (meters). Sets :math:`R=\sigma^2 I_3`. Default ``2.0``.
    B_z : float, optional
        Magnetic field along +z (Tesla). Affects
        :math:`\omega = B_z\,\kappa\,p_T`. Default ``0.002``.
    max_cands : int, optional
        Upper bound on KD-tree neighbors before gating. Default ``20``.
    beam_size : int, optional
        Number of concurrent track hypotheses to maintain. Default ``4``.
    step_candidates : int, optional
        Per-track shortlist size after gating. Default ``6``.
    gate_multiplier : float, optional
        Base gate radius
        :math:`r = \text{gate\_multiplier}\,\sqrt{\operatorname{trace}(S)/3}`. Default ``3.0``.
    gate_tighten : float, optional
        Linear tightening along depth:
        :math:`r \leftarrow r \cdot \max(0.5,\,1 - \text{gate\_tighten}\cdot \text{depth\_frac})`.
        Default ``0.15``.
    build_graph : bool, optional
        If ``True``, record a sparse debug graph of accepted steps. Default ``False``.

    Notes
    -----
    * If the base :class:`Brancher` implements ``_apply_deny(pts, ids, cost)``,
      it is used to enforce a global deny-list (hard drop or penalize).
    * This class returns **one** branch (the best hypothesis) at the end of the search.
    * Efficiency comes from tight gating and per-track top-:math:`K` reduction
      before constructing the union set.
    """

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
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.state_dim: int = 7
        self.beam_size: int = int(beam_size)
        self.step_candidates: int = int(step_candidates)
        self.gate_multiplier: float = float(gate_multiplier)
        self.gate_tighten: float = float(gate_tighten)
        self.build_graph: bool = bool(build_graph)

    def _gate_radius_fast(self, S: np.ndarray, depth_frac: float) -> float:
        r"""
        Compute a trace-based gating radius with linear tightening along depth.

        .. math::

            r \;=\; \text{gate\_multiplier}\,
                    \sqrt{\tfrac{\operatorname{trace}(S)}{3}}
                    \times \max\!\bigl(0.5,\, 1 - \text{gate\_tighten}\cdot \text{depth\_frac}\bigr)

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance :math:`S = H P_{\text{pred}} H^{\mathsf{T}} + R`.
        depth_frac : float
            Progress through layers in :math:`[0,1]`.

        Returns
        -------
        float
            Gate radius :math:`r`.
        """
        base = self.gate_multiplier * float(np.sqrt(max(1e-12, np.trace(S) / 3.0)))
        tighten = max(0.5, 1.0 - self.gate_tighten * depth_frac)
        return base * tighten

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: np.ndarray,
            plot_tree: bool = False,
            **kwargs: Any) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Execute the Hungarian assignment–based EKF tracking across the layer sequence.

        For each layer, per-track candidates are gated and reduced to top-:math:`K`,
        a union candidate set is formed, and a tracks :math:`\times` hits cost matrix
        is solved for a minimum-cost one-to-one assignment. Assigned tracks are
        EKF-updated; unassigned tracks are dropped. The beam is then pruned to a
        fixed size by cumulative score.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions :math:`(x,y,z)` for helix initialization.
        layers : list of tuple(int, int)
            Ordered list of layer keys to traverse.
        t : ndarray
            Seed "times" (used only to estimate initial velocity/curvature).
        plot_tree : bool, optional
            If ``True``, return a composed graph of accepted steps.

        Returns
        -------
        branches : list of dict
            Single best branch with:
            * ``traj`` : list of :math:`(3,)` hit positions,
            * ``hit_ids`` : list of ``int``,
            * ``state`` : final EKF state :math:`(7,)`,
            * ``cov`` : final covariance :math:`(7\times 7)`,
            * ``score`` : accumulated :math:`\chi^2`.
        G : :class:`networkx.DiGraph`
            Debug graph (empty when neither ``plot_tree`` nor ``build_graph`` is set).

        Notes
        -----
        * Candidate scoring uses:

          .. math::

             \chi^2 \;=\; (z - \hat{x})^{\mathsf{T}}\,S^{-1}\,(z - \hat{x}),

          where :math:`\hat{x}` is the predicted position and :math:`S`
          the innovation covariance.
        * The cost matrix is filled with :math:`\chi^2` where a track–hit pair
          is valid (within that track's gate), and with a very large sentinel
          otherwise.
        * Assignment is computed by :func:`scipy.optimize.linear_sum_assignment`.
        """
        if not layers:
            return [], nx.DiGraph()

        # initialize EKF from seeds
        dt0: float = float(t[1] - t[0]) if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, k0])  # 7,
        P0 = np.eye(self.state_dim) * 0.1

        # one hypothesis struct
        def _new_hypo(state: np.ndarray, cov: np.ndarray) -> Dict[str, Any]:
            return {
                "state": state.copy(),
                "cov": cov.copy(),
                "traj": [seed_xyz[0], seed_xyz[1], seed_xyz[2]],
                "hit_ids": [],            # type: List[int]
                "score": 0.0
            }

        # beam of hypotheses
        beam: List[Dict[str, Any]] = [_new_hypo(x0, P0)]
        G = nx.DiGraph() if (self.build_graph or plot_tree) else nx.DiGraph()
        H = self.H_jac(None)
        I7 = np.eye(self.state_dim)

        L = len(layers)

        for i, layer in enumerate(layers):
            depth_frac = (i + 1) / max(1, L)

            # ---- propagate each hypothesis & get per-track top-K candidates
            per_track: List[Dict[str, Any]] = []
            all_hits: List[int] = []
            all_pts: List[np.ndarray] = []

            for h in beam:
                st = h["state"]; cv = h["cov"]
                # propagate to surface
                try:
                    dt = self._solve_dt_to_surface(st, self.layer_surfaces[layer])
                except Exception:
                    continue  # drop this hypothesis

                F = self.compute_F(st, dt)
                x_pred = self.propagate(st, dt)
                P_pred = F @ cv @ F.T + self.Q0 * dt
                S = H @ P_pred @ H.T + self.R

                # gate
                r = self._gate_radius_fast(S, depth_frac)
                pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, r)
                if len(ids) == 0:
                    continue

                # χ² (vectorized)
                diff = pts - x_pred[:3]
                chi2 = np.einsum('ni,ij,nj->n', diff, np.linalg.inv(S), diff)

                # (optional) centralized deny-list from parent
                apply_deny = getattr(self, "_apply_deny", None)
                if callable(apply_deny):
                    pts, ids, chi2 = apply_deny(pts, ids, chi2)

                # keep top-K per track
                kkeep = min(self.step_candidates, len(chi2))
                order = np.argpartition(chi2, kkeep - 1)[:kkeep]
                pts_k = pts[order]
                ids_k = ids[order].astype(int)
                chi2_k = chi2[order]

                per_track.append({
                    "x_pred": x_pred, "P_pred": P_pred, "S": S,
                    "pts": pts_k, "ids": ids_k, "chi2": chi2_k,
                    "hypo": h
                })

                # accumulate for union set
                all_pts.append(pts_k)
                all_hits.extend(list(map(int, ids_k)))

            if not per_track:
                break  # beam died

            # ---- build union candidate set (dedup)
            if len(all_hits) == 0:
                break
            uniq_hits, uniq_idx = np.unique(np.array(all_hits, dtype=int), return_index=True)
            # map hit_id -> representative position (first occurrence)
            # concatenate pts in same order as all_hits:
            if len(all_pts) == 1:
                concat_pts = all_pts[0]
            else:
                concat_pts = np.concatenate(all_pts, axis=0)
            union_pts = concat_pts[uniq_idx]
            union_ids = uniq_hits
            M = len(union_ids)

            # ---- cost matrix (tracks x union_hits)
            Trows = len(per_track)
            big = 1e12
            C = np.full((Trows, M), big, dtype=float)

            for r_i, row in enumerate(per_track):
                # For hits that came from this track's gate, we have chi2; others remain big.
                # Build a map hit_id -> chi2 for this row
                row_map = {int(hid): float(c) for hid, c in zip(row["ids"], row["chi2"])}
                for c_j, hid in enumerate(union_ids):
                    if hid in row_map:
                        C[r_i, c_j] = row_map[hid]

            # Pad to square with dummy columns to allow "unassigned" via large cost
            if Trows <= M:
                padded_C = C
            else:
                # more tracks than hits: pad columns
                pad_cols = Trows - M
                padded_C = np.hstack([C, np.full((Trows, pad_cols), big, dtype=float)])

            row_ind, col_ind = linear_sum_assignment(padded_C)

            # ---- build next beam from assignments
            new_beam: List[Dict[str, Any]] = []
            for r_i, c_j in zip(row_ind, col_ind):
                if r_i >= Trows:
                    continue  # shouldn't happen
                row = per_track[r_i]
                old = row["hypo"]

                # if this is a dummy assignment or large cost, drop the track
                if c_j >= M or padded_C[r_i, c_j] >= big * 0.5:
                    continue

                # assigned real hit
                hid = int(union_ids[c_j])

                # get its measurement z for this row (we know it must be in row.ids)
                match_idx = np.where(row["ids"] == hid)[0]
                if match_idx.size == 0:
                    continue  # safety
                j = int(match_idx[0])
                z = row["pts"][j]
                x_pred = row["x_pred"]; P_pred = row["P_pred"]; S = row["S"]

                # EKF update
                K = P_pred @ H.T @ np.linalg.inv(S)
                x_upd = x_pred + K @ (z - x_pred[:3])
                P_upd = (I7 - K @ H) @ P_pred

                # extend hypothesis
                nh = {
                    "state": x_upd,
                    "cov": P_upd,
                    "traj": old["traj"] + [z],
                    "hit_ids": old["hit_ids"] + [hid],
                    "score": old["score"] + float(C[r_i, c_j])
                }
                new_beam.append(nh)

                if self.build_graph or plot_tree:
                    G.add_edge((i, tuple(old["traj"][-1])),
                               (i + 1, tuple(z)),
                               cost=float(C[r_i, c_j]))

            if not new_beam:
                break

            # keep only best beam_size by cumulative score
            new_beam.sort(key=lambda h: h["score"])
            beam = new_beam[:self.beam_size]

        # pick best hypothesis
        if not beam:
            return [], G
        best = min(beam, key=lambda h: h["score"])

        branch = {
            "traj": best["traj"],
            "hit_ids": best["hit_ids"],
            "state": best["state"],
            "cov": best["cov"],
            "score": float(best["score"])
        }
        return [branch], G
