import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Sequence
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from numpy.linalg import solve
from trackml_reco.branchers.brancher import Brancher

class HelixEKFBrancher(Brancher):
    r"""
    Branching EKF track finder with beam pruning.

    At each layer, every active branch is propagated by an EKF to the layer
    surface. Candidates within a gating radius are scored by Mahalanobis
    :math:`\chi^2`, each branch fans out to its top-K local candidates, and
    then global beam pruning keeps only a fixed number of best branches.

    This yields a fast, robust search with a predictable branching width.

    Parameters
    ----------
    trees : dict[(int, int), tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) → (KD-tree, points, hit_ids)`` where
        ``points`` has shape ``(N, 3)`` and ``hit_ids`` has shape ``(N,)``.
    layer_surfaces : dict[(int, int), dict]
        Geometry per layer. One of

        * ``{'type': 'disk', 'n': n, 'p': p}``  (plane normal and point), or
        * ``{'type': 'cylinder', 'R': R}``      (cylindrical radius).
    noise_std : float, optional
        Isotropic measurement std (meters). Sets :math:`R=\sigma^2 I_3`.
        Default is ``2.0``.
    B_z : float, optional
        Magnetic field along +z (Tesla). Affects
        :math:`\omega = B_z\,\kappa\,p_T`. Default ``0.002``.
    num_branches : int, optional
        Beam width after each layer (max branches kept). Default ``30``.
    survive_top : int, optional
        Number of elite branches kept deterministically each layer. Default ``12``.
        The remaining ``num_branches - survive_top`` are sampled from the rest.
    max_cands : int, optional
        Upper bound passed to KD queries (base class). Default ``10``.
    step_candidates : int, optional
        Per-branch top-K candidates expanded per layer. Default ``5``.
    gate_multiplier : float, optional
        Base gate radius
        :math:`r = \text{gate\_multiplier}\,\sqrt{\operatorname{trace}(S)/3}`.
        Default ``3.0``.
    gate_tighten : float, optional
        Linear gate tightening along depth:
        :math:`r \leftarrow r \cdot \max\!\bigl(0.5,\, 1 - \text{gate\_tighten}\cdot \text{depth\_frac}\bigr)`.
        Default ``0.15``.
    build_graph : bool, optional
        If ``True``, record a sparse debug graph of chosen edges. Default ``False``.

    Notes
    -----
    * Each branch's ``traj`` stores the **measured hit positions** it used.
    * Passing ``deny_hits`` to :meth:`run` excludes those hits from consideration.
    """

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
                         max_cands=max_cands)

        self.layer_surfaces = layer_surfaces
        self.num_branches = int(num_branches)
        self.survive_top = int(survive_top)
        self.step_candidates = int(step_candidates)
        self.state_dim = 7

        # gating
        self.gate_multiplier = float(gate_multiplier)
        self.gate_tighten = float(gate_tighten)

        # debug graph
        self.build_graph = bool(build_graph)

        # RNG for diversity sampling
        self._rng = np.random.default_rng()

    def _gate_radius_fast(self, S: np.ndarray, depth_frac: float) -> float:
        r"""
        Compute a fast scalar gating radius from innovation covariance.

        Uses trace-based scale with mild linear tightening along depth:

        .. math::

            r \;=\; \text{gate\_multiplier}\,\sqrt{\tfrac{\operatorname{trace}(S)}{3}}
            \times \max\!\bigl(0.5,\; 1 - \text{gate\_tighten}\cdot \text{depth\_frac}\bigr).

        Parameters
        ----------
        S : ndarray, shape (3, 3)
            Innovation covariance.
        depth_frac : float
            Progress through the layer sequence in :math:`[0,1]`.

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
            *,
            deny_hits: Optional[Sequence[int]] = None) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Execute branching EKF across the provided layer sequence.

        For each active branch and layer:
        1) predict to the layer surface, 2) compute :math:`S` and gate,
        3) score candidates by Mahalanobis :math:`\chi^2`, 4) keep local top-K,
        5) perform EKF update, 6) globally prune to a fixed beam.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions :math:`(x,y,z)` used to initialize the helix.
        layers : list of tuple(int, int)
            Ordered list of ``(volume_id, layer_id)`` to traverse.
        t : ndarray
            Nominal times for seed spacing (used for seed velocity/curvature).
        plot_tree : bool, optional
            If ``True``, returns a composed graph of chosen steps.
        deny_hits : sequence of int, optional
            Hit IDs to exclude (e.g., already claimed by other threads).

        Returns
        -------
        branches : list of dict
            Final set of branches (up to ``num_branches``). Each branch has:
            * ``traj`` : list of :math:`(3,)` measured hit positions,
            * ``hit_ids`` : list of ``int``,
            * ``state`` : :math:`(7,)` EKF state,
            * ``cov`` : :math:`(7\times 7)` covariance,
            * ``score`` : accumulated :math:`\chi^2` (lower is better).
        G : :class:`networkx.DiGraph`
            Debug graph (empty if not recording).

        Notes
        -----
        * Gate radius uses :math:`S = H P_{\text{pred}} H^{\mathsf{T}} + R`.
        * Candidate scoring uses the quadratic form
          :math:`\chi^2 = (z - \hat{x})^{\mathsf{T}} S^{-1} (z - \hat{x})`.
        * Beam pruning keeps ``survive_top`` elites and samples uniformly
          without replacement from the remainder to fill up to ``num_branches``.
        """
        if not layers:
            return [], nx.DiGraph()

        deny: set[int] = set(map(int, deny_hits)) if deny_hits is not None else set()

        # Seed EKF
        dt0 = float(t[1] - t[0]) if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, k0])          # (7,)
        P0 = np.eye(self.state_dim) * 0.1              # (7,7)

        # Branch record
        branches: List[Dict[str, Any]] = [{
            'id': 0,
            'parent': None,
            'traj': [seed_xyz[0], seed_xyz[1], seed_xyz[2]],  # measured hits
            'state': x0,
            'cov': P0,
            'score': 0.0,
            'hit_ids': []  # we haven’t consumed any yet (seeds are not from candidate bank)
        }]

        G = nx.DiGraph() if (self.build_graph or plot_tree) else nx.DiGraph()
        next_id = 1
        H = self.H_jac(None)
        I7 = np.eye(self.state_dim)
        L = len(layers)

        for i, layer in enumerate(layers):
            depth_frac = (i + 1) / max(1, L)
            layer_surf = self.layer_surfaces[layer]
            expanded: List[Dict[str, Any]] = []

            for br in branches:
                # propagate to surface
                try:
                    dt_layer = self._solve_dt_to_surface(br['state'], layer_surf, dt_init=dt0)
                except Exception:
                    continue

                F = self.compute_F(br['state'], dt_layer)
                x_pred = self.propagate(br['state'], dt_layer)
                P_pred = F @ br['cov'] @ F.T + self.Q0 * dt_layer

                # innovation covariance & gating
                S = H @ P_pred @ H.T + self.R
                gate_r = self._gate_radius_fast(S, depth_frac)

                # fetch candidates within gate
                pts, ids = self._get_candidates_in_gate(x_pred[:3], layer, gate_r)
                if len(ids) == 0:
                    continue

                # deny-list filtering
                if deny:
                    mask = np.array([int(h) not in deny for h in ids], dtype=bool)
                    pts, ids = pts[mask], ids[mask]
                    if len(ids) == 0:
                        continue

                # vectorized χ² via solve
                # Solve S a = diff^T  ⇒ chi2 = diff · a
                diff = pts - x_pred[:3]                 # (m,3)
                Sinv_diffT = solve(S, diff.T)           # (3,m)
                chi2 = np.einsum('ni,in->n', diff, Sinv_diffT)

                # keep top-K for this branch
                kkeep = min(self.step_candidates, len(chi2))
                order = np.argpartition(chi2, kkeep - 1)[:kkeep]
                pts_k = pts[order]; ids_k = ids[order]; chi2_k = chi2[order]

                # generate child branches
                K = P_pred @ H.T @ np.linalg.inv(S)     # (7,3)
                for z, hid, c in zip(pts_k, ids_k, chi2_k):
                    x_upd = x_pred + K @ (z - x_pred[:3])
                    P_upd = (I7 - K @ H) @ P_pred

                    node_id = next_id; next_id += 1
                    if self.build_graph or plot_tree:
                        G.add_node(node_id, pos=z)
                        G.add_edge(br['id'], node_id, cost=float(c))

                    expanded.append({
                        'id': node_id,
                        'parent': br['id'],
                        'traj': br['traj'] + [z],              # measured point
                        'state': x_upd,
                        'cov': P_upd,
                        'score': br['score'] + float(c),
                        'hit_ids': br['hit_ids'] + [int(hid)]
                    })

            if not expanded:
                break

            # prune to beam: top `survive_top`, plus random from the rest (diversity)
            expanded.sort(key=lambda b: b['score'])
            elite = expanded[:min(self.survive_top, len(expanded))]
            rest = expanded[len(elite):]
            need = max(0, self.num_branches - len(elite))
            if need > 0 and rest:
                take = min(need, len(rest))
                idx = self._rng.choice(len(rest), size=take, replace=False)
                elite.extend([rest[j] for j in idx])

            branches = elite

        # done — return the final beam (already sorted by score)
        branches.sort(key=lambda b: b['score'])
        return branches, G

    def _plot_tree(self, G: nx.DiGraph) -> None:
        r"""
        Plot the branch tree in an :math:`x\!-\!y` projection.

        Parameters
        ----------
        G : :class:`networkx.DiGraph`
            Directed graph produced during branching. Nodes should have a
            ``'pos'`` attribute with a 3D point; the first two coordinates are used.

        Notes
        -----
        This is a convenience visualization for debugging. It will do nothing
        if the graph has no nodes or nodes lack a ``'pos'`` attribute.
        """
        if G.number_of_nodes() == 0:
            return
        pos = {n: tuple(np.asarray(G.nodes[n]['pos'])[:2]) for n in G.nodes() if 'pos' in G.nodes[n]}
        if not pos:
            return
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=False, node_size=30, arrowsize=8, width=0.6)
        plt.title('Branching tree (XY projection)')
        plt.tight_layout()
        plt.show()
