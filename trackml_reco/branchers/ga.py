import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy.spatial import cKDTree
import networkx as nx
from trackml_reco.branchers.brancher import Brancher

class HelixEKFGABrancher(Brancher):
    r"""
    Fast EKF-based track finder with a Genetic Algorithm (GA) over *per-layer shortlists*.

    The GA operates on **indices into precomputed per-layer shortlists**, rather
    than on the full set of hits. For each layer, a shortlist is built using a
    single seed-initialized EKF prediction, taking the Top-K candidates ranked
    by Mahalanobis :math:`\chi^2`. Each GA individual encodes one hit index per
    layer, selecting a candidate from that layer's shortlist.

    This dramatically reduces the search space, making mutation and crossover
    cheap, and removing most per-individual KD-tree queries — ideal for running
    many seeds in parallel.

    Parameters
    ----------
    trees : dict[(int, int), tuple(cKDTree, ndarray, ndarray)]
        Mapping ``(volume_id, layer_id) → (KD-tree, points, hit_ids)``.
        * ``points`` has shape ``(N, 3)``.
        * ``hit_ids`` has shape ``(N,)``.
    layer_surfaces : dict[(int, int), dict]
        Geometry per layer:

        * Disk: ``{'type': 'disk', 'n': normal_vec, 'p': point_on_plane}``.
        * Cylinder: ``{'type': 'cylinder', 'R': radius}``.
    noise_std : float, optional
        Isotropic measurement standard deviation (meters).
        Sets :math:`R = \sigma^2 I_3`. Default is ``2.0``.
    B_z : float, optional
        Magnetic field along z-axis (Tesla). Default ``0.002``.
    pop_size : int, optional
        GA population size. Default ``24``.
    n_gens : int, optional
        Maximum number of generations. Default ``12``.
    cx_rate : float, optional
        One-point crossover probability. Default ``0.7``.
    mut_rate : float, optional
        Per-gene mutation probability. Default ``0.15``.
    max_cands : int, optional
        Maximum KD-tree neighbors to fetch (base class). Default ``10``.
    step_candidates : int, optional
        Shortlist size per layer. Default ``6``.
    elite_frac : float, optional
        Fraction of elites preserved each generation. Default ``0.1``.
    tournament_k : int, optional
        Tournament size for selection. Default ``3``.
    patience : int, optional
        Stop early if no improvement in best score for this many generations.
        Default ``3``.
    rng_seed : int or None, optional
        Random number generator seed. Default ``None``.

    Notes
    -----
    * NetworkX graphs are **not** assembled unless ``plot_tree=True`` in :meth:`run`.
    * Shortlists are rebuilt for every :meth:`run` call from the seed state,
      so they are specific to the given seed's geometry and timing.
    """

    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layer_surfaces: Dict[Tuple[int, int], dict],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 pop_size: int = 24,
                 n_gens: int = 12,
                 cx_rate: float = 0.7,
                 mut_rate: float = 0.15,
                 max_cands: int = 10,
                 step_candidates: int = 6,
                 elite_frac: float = 0.10,
                 tournament_k: int = 3,
                 patience: int = 3,
                 rng_seed: Optional[int] = None) -> None:

        super().__init__(trees=trees,
                         layers=list(layer_surfaces.keys()),
                         noise_std=noise_std,
                         B_z=B_z,
                         max_cands=max_cands)
        self.layer_surfaces = layer_surfaces
        self.pop_size = int(pop_size)
        self.n_gens = int(n_gens)
        self.cx_rate = float(cx_rate)
        self.mut_rate = float(mut_rate)
        self.step_candidates = int(step_candidates)
        self.elite_frac = float(elite_frac)
        self.tournament_k = int(tournament_k)
        self.patience = int(patience)
        self.state_dim = 7
        self.rng = np.random.default_rng(rng_seed)

    def _build_shortlists(self,
                          seed_xyz: np.ndarray,
                          t: np.ndarray,
                          layers: List[Tuple[int, int]]
                          ) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        r"""
        Build Top-K candidate shortlists per layer using a seed-based EKF prediction.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions :math:`(x,y,z)`.
        t : ndarray
            Time points for seed spacing. Used to estimate seed velocity and curvature.
        layers : list of tuple(int, int)
            Ordered layer keys.

        Returns
        -------
        shortlists : dict
            Maps each ``layer_key`` to a dictionary:

            * ``'pts'`` : ndarray, shape (K, 3), candidate hit positions.
            * ``'ids'`` : ndarray, shape (K,), corresponding hit IDs.

        Notes
        -----
        * Candidates are gated using :math:`r = 3 \sqrt{\lambda_{\max}(S)}`,
          where :math:`S = H P_{\text{pred}} H^{\mathsf{T}} + R`.
        * Sorted by Mahalanobis distance:

          .. math::

             \chi^2 = (z - \hat{x})^{\mathsf{T}} S^{-1} (z - \hat{x}).
        """
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1

        H = self.H_jac(None)
        short: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

        for layer in layers:
            surf = self.layer_surfaces[layer]
            # Predict to layer once from the seed state
            try:
                dt = self._solve_dt_to_surface(state, surf, dt_init=dt0)
            except Exception:
                # No shortlist (empty) if intersection fails
                short[layer] = {'pts': np.empty((0, 3), float),
                                'ids': np.empty((0,), int)}
                continue

            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R
            # gate radius from S
            try:
                gate_r = 3.0 * np.sqrt(np.max(np.linalg.eigvalsh(S)))
            except Exception:
                gate_r = 3.0 * 3.0

            tree, pts_all, ids_all = self.trees[layer]
            idxs = tree.query_ball_point(x_pred[:3], r=float(gate_r))
            if not idxs:
                short[layer] = {'pts': np.empty((0, 3), float),
                                'ids': np.empty((0,), int)}
                continue

            cand_pts = pts_all[idxs]
            cand_ids = ids_all[idxs]

            # Rank by Mahalanobis distance w.r.t. S (vectorized)
            try:
                S_inv = np.linalg.inv(S)
                diff = cand_pts - x_pred[:3]
                chi2 = np.einsum('ij,jk,ik->i', diff, S_inv, diff)
            except Exception:
                # fallback: Euclidean
                diff = cand_pts - x_pred[:3]
                chi2 = np.sum(diff * diff, axis=1)

            k = min(self.step_candidates, cand_pts.shape[0])
            order = np.argpartition(chi2, k - 1)[:k]
            short[layer] = {'pts': cand_pts[order].astype(float),
                            'ids': cand_ids[order].astype(int)}

        return short

    def _init_population(self,
                         shortlists: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                         layers: List[Tuple[int, int]]) -> np.ndarray:
        r"""
        Create the initial GA population.

        Each individual is a vector of integers of length ``len(layers)``,
        where gene ``i`` is the index into that layer's shortlist:

        .. math::

            g_i \in \{0, 1, \dots, K_i - 1\}, \quad\text{or}\quad g_i = -1 \ \text{if empty}.

        Parameters
        ----------
        shortlists : dict
            Output from :meth:`_build_shortlists`.
        layers : list of tuple
            Ordered layer keys.

        Returns
        -------
        pop : ndarray, shape (pop_size, n_layers)
            Integer-encoded GA population.
        """
        genes_per_layer = [shortlists[ly]['ids'].shape[0] for ly in layers]
        pop = -np.ones((self.pop_size, len(layers)), dtype=int)
        for L, K in enumerate(genes_per_layer):
            if K == 0:
                continue
            pop[:, L] = self.rng.integers(0, K, size=self.pop_size)
        return pop

    def _tournament(self, fitness: np.ndarray) -> int:
        r"""
        Select an individual index via tournament selection.

        Parameters
        ----------
        fitness : ndarray, shape (pop_size,)
            Fitness scores (lower is better).

        Returns
        -------
        int
            Index of the selected individual.
        """
        idx = self.rng.integers(0, self.pop_size, size=self.tournament_k)
        best = idx[np.argmin(fitness[idx])]
        return int(best)

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Perform one-point crossover between two parents.

        Parameters
        ----------
        a, b : ndarray
            Parent genomes.

        Returns
        -------
        c1, c2 : ndarray
            Child genomes (copies if no crossover occurs).
        """
        if self.rng.random() > self.cx_rate or a.size < 2:
            return a.copy(), b.copy()
        pt = int(self.rng.integers(1, a.size))
        return np.concatenate([a[:pt], b[pt:]]), np.concatenate([b[:pt], a[pt:]])

    def _mutate(self,
                child: np.ndarray,
                shortlists: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                layers: List[Tuple[int, int]]) -> np.ndarray:
        r"""
        Apply per-gene mutation to a child genome.

        Parameters
        ----------
        child : ndarray
            Child genome (modified in-place).
        shortlists : dict
            Output from :meth:`_build_shortlists`.
        layers : list of tuple
            Ordered layer keys.

        Returns
        -------
        ndarray
            Mutated genome.

        Notes
        -----
        Mutation chooses a different index for the gene's layer when possible.
        """
        for L, layer in enumerate(layers):
            if self.rng.random() > self.mut_rate:
                continue
            K = shortlists[layer]['ids'].shape[0]
            if K <= 1:
                continue
            cur = int(child[L]) if child[L] >= 0 else -1
            # choose an alternative index
            alt = self.rng.integers(0, K - 1)
            if cur >= 0 and alt >= cur:
                alt += 1
            child[L] = int(alt)
        return child

    def _eval_sequence(self,
                       seed_xyz: np.ndarray,
                       t: np.ndarray,
                       layers: List[Tuple[int, int]],
                       shortlists: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                       gene_row: np.ndarray,
                       build_graph: bool) -> Dict[str, Any]:
        r"""
        Evaluate a GA individual by running an EKF through its chosen shortlist hits.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions.
        t : ndarray
            Time points for seed spacing.
        layers : list of tuple
            Ordered layer keys.
        shortlists : dict
            Per-layer shortlists from :meth:`_build_shortlists`.
        gene_row : ndarray, shape (n_layers,)
            Genome: one shortlist index per layer.
        build_graph : bool
            If True, record an expansion graph of chosen edges.

        Returns
        -------
        result : dict
            Contains:
            * ``'traj'`` : list of hit positions (ndarray, shape (3,)),
            * ``'hit_ids'`` : list of int,
            * ``'state'`` : final EKF state vector,
            * ``'cov'`` : final EKF covariance matrix,
            * ``'score'`` : accumulated :math:`\chi^2`,
            * ``'graph'`` : :class:`networkx.DiGraph` (empty if not built).

        Notes
        -----
        At each layer, the chosen hit index ``j`` maps to:

        .. math::

           z_j \in \text{shortlist}_{\text{layer}}

        and the branch is updated with the standard EKF equations:

        .. math::

           K &= P_{\text{pred}} H^{\mathsf{T}} S^{-1},\\
           x^+ &= x^- + K (z_j - H x^-),\\
           P^+ &= (I - K H) P_{\text{pred}}.
        """
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        state = np.hstack([seed_xyz[2], v0, k0])
        cov = np.eye(self.state_dim) * 0.1
        traj: List[np.ndarray] = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        hit_ids: List[int] = []
        score = 0.0

        H = self.H_jac(None)
        G = nx.DiGraph() if build_graph else None

        for i, layer in enumerate(layers):
            surf = self.layer_surfaces[layer]
            # Propagate to this surface
            try:
                dt = self._solve_dt_to_surface(state, surf, dt_init=dt0)
            except Exception:
                break

            F = self.compute_F(state, dt)
            x_pred = self.propagate(state, dt)
            P_pred = F @ cov @ F.T + self.Q0 * dt
            S = H @ P_pred @ H.T + self.R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                break

            pts = shortlists[layer]['pts']
            ids = shortlists[layer]['ids']
            if pts.shape[0] == 0:
                break

            j = int(gene_row[i])
            if j < 0 or j >= pts.shape[0]:
                # invalid gene -> pick cheapest within shortlist (failsafe)
                diff_all = pts - x_pred[:3]
                chi2_all = np.einsum('ij,jk,ik->i', diff_all, S_inv, diff_all)
                j = int(np.argmin(chi2_all))

            z = pts[j]
            diff = z - x_pred[:3]
            chi2 = float(diff @ S_inv @ diff)

            # EKF update
            K_gain = P_pred @ H.T @ S_inv
            state = x_pred + K_gain @ diff
            cov = (np.eye(self.state_dim) - K_gain @ H) @ P_pred

            traj.append(z)
            hit_ids.append(int(ids[j]))
            score += chi2

            if build_graph:
                G.add_edge((i, tuple(traj[-2])), (i + 1, tuple(z)), cost=chi2)

        return {'traj': traj, 'hit_ids': hit_ids, 'state': state, 'cov': cov,
                'score': float(score), 'graph': (G if build_graph else nx.DiGraph())}

    def run(self,
            seed_xyz: np.ndarray,
            layers: List[Tuple[int, int]],
            t: np.ndarray,
            plot_tree: bool = False) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        r"""
        Run the GA to optimize a sequence of shortlist indices.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed hit positions.
        layers : list of tuple(int, int)
            Ordered list of ``(volume_id, layer_id)`` to traverse.
        t : ndarray
            Time points for seed spacing.
        plot_tree : bool, optional
            If True, return a NetworkX graph of the best path.

        Returns
        -------
        branches : list of dict
            One best branch with:
            * ``traj`` : list of hit positions,
            * ``hit_ids`` : list of int,
            * ``state`` : final EKF state,
            * ``cov`` : final EKF covariance,
            * ``score`` : accumulated :math:`\chi^2`.
        G : :class:`networkx.DiGraph`
            Graph of the best path (empty if ``plot_tree=False``).

        Notes
        -----
        * GA loop uses:
          - **elitism**: top ``elite_frac`` fraction carried over,
          - **tournament selection** for parents,
          - **one-point crossover** with prob. ``cx_rate``,
          - **per-gene mutation** with prob. ``mut_rate``.
        * Early stopping occurs if the best score does not improve for
          ``patience`` generations.
        """
        if not layers:
            return [], nx.DiGraph()

        # 1) Build per-layer shortlists once
        shortlists = self._build_shortlists(seed_xyz, t, layers)

        # 2) Initialize population over shortlist indices
        pop = self._init_population(shortlists, layers)
        fitness = np.full(self.pop_size, np.inf, dtype=float)
        phenos: List[Optional[Dict[str, Any]]] = [None] * self.pop_size

        build_graph = bool(plot_tree)
        global_graph = nx.DiGraph() if build_graph else nx.DiGraph()

        # Evaluate initial population
        for i in range(self.pop_size):
            ph = self._eval_sequence(seed_xyz, t, layers, shortlists, pop[i], build_graph=False)
            phenos[i] = ph
            fitness[i] = ph['score']

        best_prev = float(np.min(fitness))
        no_improve = 0
        n_elite = max(1, int(self.elite_frac * self.pop_size))

        for gen in range(self.n_gens):
            # --- Elitism ---
            elite_idx = np.argsort(fitness)[:n_elite]
            elites = pop[elite_idx].copy()
            elite_ph = [phenos[i] for i in elite_idx]
            elite_fit = fitness[elite_idx].copy()

            # --- New offspring via tournaments ---
            children = []
            while len(children) < self.pop_size - n_elite:
                p1 = pop[self._tournament(fitness)]
                p2 = pop[self._tournament(fitness)]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1, shortlists, layers)
                c2 = self._mutate(c2, shortlists, layers)
                children.append(c1)
                if len(children) < self.pop_size - n_elite:
                    children.append(c2)

            new_pop = np.vstack([elites, np.array(children, dtype=int)])

            # Evaluate only children (reuse elites)
            new_fitness = np.full(self.pop_size, np.inf, dtype=float)
            new_phenos: List[Optional[Dict[str, Any]]] = [None] * self.pop_size

            # put elites back
            new_fitness[:n_elite] = elite_fit
            for i, ph in enumerate(elite_ph):
                new_phenos[i] = ph

            # evaluate children
            for i in range(n_elite, self.pop_size):
                ph = self._eval_sequence(seed_xyz, t, layers, shortlists, new_pop[i], build_graph=False)
                new_phenos[i] = ph
                new_fitness[i] = ph['score']

            pop, phenos, fitness = new_pop, new_phenos, new_fitness

            # Early stopping on stagnation
            best_now = float(np.min(fitness))
            if best_now + 1e-9 < best_prev:
                best_prev = best_now
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        # Final best individual; rebuild its graph only if requested
        best_idx = int(np.argmin(fitness))
        best_ph = phenos[best_idx]
        if build_graph:
            # Re-evaluate *once* with graph building to avoid Nx overhead inside GA
            best_ph = self._eval_sequence(seed_xyz, t, layers, shortlists, pop[best_idx], build_graph=True)
            global_graph = best_ph['graph']

        result = {k: best_ph[k] for k in ('traj', 'hit_ids', 'state', 'cov', 'score')}
        return [result], global_graph
