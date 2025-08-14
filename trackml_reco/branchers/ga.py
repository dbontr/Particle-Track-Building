import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Sequence
import networkx as nx
from scipy.spatial import cKDTree

from trackml_reco.branchers.brancher import Brancher
from trackml_reco.ekf_kernels import chi2_batch, kalman_gain


class HelixEKFGABrancher(Brancher):
    r"""
    EKF + Genetic Algorithm over *per-layer EKF-gated shortlists* (optimized).

    This brancher runs a Genetic Algorithm (GA) over **indices into fixed
    per-layer shortlists** of hit candidates. Shortlists are built **once**
    per run using χ² gating from :meth:`Brancher._layer_topk_candidates`
    (Cholesky solves; no inverses). Each GA individual represents a sequence
    of shortlist picks—one per layer—and is evaluated by an EKF that shares
    a single layer gain across all candidates at that layer.

    **State/measurement convention.** The EKF state is
    :math:`\mathbf{x}=[x,y,z,v_x,v_y,v_z,\kappa]^\top\in\mathbb{R}^7`,
    the measurement is position :math:`\mathbf{z}\in\mathbb{R}^3`, and the
    measurement Jacobian :math:`\mathbf{H}\in\mathbb{R}^{3\times7}` extracts
    the first three components. With predicted :math:`(\hat{\mathbf{x}},\mathbf{P}^- )`:

    .. math::

        \mathbf{S} = \mathbf{H}\mathbf{P}^- \mathbf{H}^\top + \mathbf{R},\qquad
        \mathbf{K} = \texttt{kalman\_gain}(\mathbf{P}^-,\mathbf{H},\mathbf{S}), \\
        \mathbf{x}^+ = \hat{\mathbf{x}} + \mathbf{K}(\mathbf{z}-\hat{\mathbf{x}}_{0:3}),\qquad
        \mathbf{P}^+ \approx (\mathbf{I}-\mathbf{K}\mathbf{H})\,\mathbf{P}^- .

    **Fitness (objective).** For an individual that selects hits
    :math:`\mathbf{z}_{1:L}`, the score minimized by GA is the accumulated
    per-layer χ²:

    .. math::

        J = \sum_{\ell=1}^{L} \chi^2_\ell, \qquad
        \chi^2_\ell = (\mathbf{z}_\ell - \hat{\mathbf{x}}_{\ell,0:3})^\top
                      \mathbf{S}_\ell^{-1}(\mathbf{z}_\ell - \hat{\mathbf{x}}_{\ell,0:3}).

    What makes this fast
    --------------------
    • **Shortlists once per run** via :meth:`_build_shortlists` /
      :meth:`Brancher._layer_topk_candidates` (stable χ² via Cholesky).  
    • **Per-individual EKF** uses :meth:`Brancher._ekf_predict` +
      :meth:`Brancher._ekf_update_meas` (one gain per layer).  
    • **Allocation hygiene**: ``__slots__``, cached :math:`\mathbf{H}/\mathbf{I}`,
      consistent dtype from banks.  
    • **Vector-friendly GA**: elitism + tournament selection + one-point
      crossover + per-gene mutation.  
    • **Graph** is built **once** for the best individual only (optional).

    Parameters
    ----------
    trees, layer_surfaces, noise_std, B_z, max_cands, step_candidates
        Forwarded to :class:`Brancher`.
    pop_size : int, optional
        GA population size (default ``24``).
    n_gens : int, optional
        Maximum generations (default ``12``).
    cx_rate : float, optional
        Crossover probability (default ``0.7``).
    mut_rate : float, optional
        Independent per-gene mutation probability (default ``0.15``).
    elite_frac : float, optional
        Fraction of elites carried unchanged to the next generation (default ``0.10``).
    tournament_k : int, optional
        Tournament size for parent selection (default ``3``).
    patience : int, optional
        Early-stop if no improvement in best score for this many generations (default ``3``).
    rng_seed : int or None, optional
        Seed for the internal RNG used by GA operators.

    Attributes
    ----------
    layer_surfaces : dict
        Geometry per layer (disks/cylinders).
    pop_size, n_gens, cx_rate, mut_rate, elite_frac, tournament_k, patience
        GA hyperparameters.
    _H, _I : ndarray
        Cached measurement Jacobian and identity for EKF updates.
    _rng : numpy.random.Generator
        RNG for GA operators.
    _dtype : numpy dtype
        Numeric dtype inferred from the first layer's points.

    Notes
    -----
    - You can pass call-scoped ``deny_hits`` to :meth:`run` or configure a
      persistent deny policy via :meth:`Brancher.set_deny_hits`. Denies are
      honored during shortlist construction.
    """

    __slots__ = (
        "layer_surfaces", "pop_size", "n_gens", "cx_rate", "mut_rate",
        "elite_frac", "tournament_k", "patience",
        "state_dim", "_rng", "_H", "_I", "_dtype"
    )

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
                         max_cands=max_cands,
                         step_candidates=step_candidates)

        self.layer_surfaces = layer_surfaces
        self.pop_size = int(pop_size)
        self.n_gens = int(n_gens)
        self.cx_rate = float(cx_rate)
        self.mut_rate = float(mut_rate)
        self.elite_frac = float(elite_frac)
        self.tournament_k = int(tournament_k)
        self.patience = int(patience)

        self.state_dim = 7
        self._rng = np.random.default_rng(rng_seed)
        self._H = self.H_jac(None)                  # constant 3x7 position extractor
        self._I = np.eye(self.state_dim)

        # Detect a consistent dtype from any layer's point bank
        try:
            any_layer = next(iter(trees))
            self._dtype = trees[any_layer][1].dtype
        except Exception:
            self._dtype = np.float64

    def _build_shortlists(self,
                          seed_xyz: np.ndarray,
                          t: Optional[np.ndarray],
                          layers: List[Tuple[int, int]],
                          deny_hits: Optional[Sequence[int]] = None
                          ) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        r"""
        Build per-layer **shortlists** by predict-only EKF from the seed.

        For each layer, perform a geometric time-of-flight solve to the surface,
        run :meth:`Brancher._ekf_predict` to get :math:`(\hat{\mathbf{x}},\mathbf{P}^-,\mathbf{S})`,
        and then call :meth:`Brancher._layer_topk_candidates` to select the
        Top-:math:`K` gated hits by χ² (ascending). Only positions and IDs are
        stored in the shortlist; χ² is recomputed as needed during evaluation.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Seed triplet used to initialize velocity/curvature.
        t : ndarray or None
            Optional timestamps aligned with the seed; if supplied, the initial
            step uses :math:`\Delta t_0=t_1-t_0`, else ``1.0``.
        layers : list[tuple[int, int]]
            Ordered layer keys to build shortlists for.
        deny_hits : sequence[int] or None, optional
            Per-call deny list applied during gating.

        Returns
        -------
        shortlists : dict[layer_key -> dict]
            For each layer ``L``, a dict with keys:
            ``'pts'`` (``(K,3)`` float array) and ``'ids'`` (``(K,)`` int array).

        Notes
        -----
        The predict-only chain advances :math:`(\hat{\mathbf{x}},\mathbf{P}^- )`
        to the next surface **without** applying updates, ensuring shortlist
        construction is independent of individual genomes.
        """
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz.astype(self._dtype), dt0, self.B_z)
        state = np.hstack([seed_xyz[2].astype(self._dtype), v0.astype(self._dtype), np.array([k0], dtype=self._dtype)])
        cov = np.eye(self.state_dim, dtype=self._dtype) * 0.1

        short: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        deny = list(map(int, deny_hits)) if deny_hits is not None else None

        Ltot = len(layers)
        for i, layer in enumerate(layers):
            depth_frac = (i + 1) / max(1, Ltot)
            try:
                dt = self._solve_dt_to_surface(state, self.layer_surfaces[layer], dt_init=dt0)
            except Exception:
                short[layer] = {'pts': np.empty((0, 3), dtype=self._dtype),
                                'ids': np.empty((0,), dtype=int)}
                continue

            # Fast EKF predict (uses base kernels)
            x_pred, P_pred, S, H = self._ekf_predict(state, cov, float(dt))

            # Use Brancher’s optimized gate + vectorized χ² + Top-K (sorted)
            pts, ids, _chi2 = self._layer_topk_candidates(
                x_pred, S, layer,
                k=max(1, min(self.step_candidates, 32)),
                depth_frac=depth_frac,
                gate_mul=3.0,          # base multiplier; shortlist already tight
                gate_tighten=0.15,
                deny_hits=deny
            )
            short[layer] = {'pts': pts.astype(self._dtype, copy=False),
                            'ids': ids.astype(int, copy=False)}

            # Predict-only chain to the next surface
            state = x_pred
            cov = P_pred

        return short

    def _init_population(self,
                         shortlists: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                         layers: List[Tuple[int, int]]) -> np.ndarray:
        r"""
        Initialize a population of genomes over shortlist indices.

        Each genome is a vector of length ``len(layers)``. Gene ``g_L`` is an
        index in ``[0, K_L-1]`` for the layer's shortlist of size ``K_L``.
        The value ``-1`` denotes "no pick" and triggers a cheapest-in-shortlist
        fallback during evaluation.

        Parameters
        ----------
        shortlists : dict
            Output of :meth:`_build_shortlists`.
        layers : list[tuple[int, int]]
            Layer ordering (defines genome length).

        Returns
        -------
        ndarray, shape (pop_size, n_layers), dtype int32
            Initial population with uniform random valid indices where possible,
            else ``-1``.
        """
        nL = len(layers)
        pop = -np.ones((self.pop_size, nL), dtype=np.int32)
        for L, layer in enumerate(layers):
            K = shortlists[layer]['ids'].shape[0]
            if K > 0:
                pop[:, L] = self._rng.integers(0, K, size=self.pop_size, endpoint=False)
        return pop

    def _tournament_indices(self, fitness: np.ndarray, n_winners: int) -> np.ndarray:
        r"""
        Tournament selection (vectorized) — return indices of winners.

        For each of ``n_winners`` tournaments, sample ``tournament_k`` distinct
        individuals (with replacement across tournaments) and pick the one with
        the smallest fitness:

        .. math::

            i^\star = \arg\min_{i \in \mathcal{T}} J_i .

        Parameters
        ----------
        fitness : ndarray, shape (pop_size,)
            Fitness values :math:`J_i` (lower is better).
        n_winners : int
            Number of tournament winners to select.

        Returns
        -------
        ndarray, shape (n_winners,), dtype int32
            Indices of selected parents.
        """
        # shape: (n_winners, tournament_k)
        choices = self._rng.integers(0, self.pop_size, size=(n_winners, self.tournament_k))
        subfit = fitness[choices]                   # (n_winners, k)
        winners = choices[np.arange(n_winners), np.argmin(subfit, axis=1)]
        return winners.astype(np.int32, copy=False)

    def _crossover_pair(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        One-point crossover for two parents.

        With probability ``cx_rate``, draw a cut-point :math:`p\in\{1,\dots,L-1\}`
        and swap tails:

        .. math::

            \text{child}_1 = [a_{:p},\, b_{p:}],\qquad
            \text{child}_2 = [b_{:p},\, a_{p:}].

        If crossover does not occur or the genome is shorter than 2, parents are
        copied unchanged.

        Parameters
        ----------
        a, b : ndarray, shape (L,), dtype int
            Parent genomes.

        Returns
        -------
        child1, child2 : ndarray
            Offspring genomes (copies if no crossover).
        """
        if (a.size < 2) or (self._rng.random() > self.cx_rate):
            return a.copy(), b.copy()
        pt = int(self._rng.integers(1, a.size))
        return np.concatenate([a[:pt], b[pt:]]), np.concatenate([b[:pt], a[pt:]])

    def _mutate_inplace(self,
                        child: np.ndarray,
                        shortlists: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                        layers: List[Tuple[int, int]]) -> None:
        r"""
        Per-gene mutation in place to a *different* valid index (if possible).

        For gene :math:`g_L` with shortlist size :math:`K_L`:

        - With probability ``mut_rate``, if :math:`K_L\le 1` no change.
        - If ``g_L < 0``, sample uniformly from ``[0, K_L-1]``.
        - Else, sample a uniform alternative in ``[0, K_L-1]\setminus\{g_L\}``.

        Parameters
        ----------
        child : ndarray, shape (L,), dtype int
            Genome to mutate (modified in place).
        shortlists : dict
            Per-layer shortlists.
        layers : list[tuple[int, int]]
            Ordered layers corresponding to genome positions.
        """
        for L, layer in enumerate(layers):
            if self._rng.random() > self.mut_rate:
                continue
            K = shortlists[layer]['ids'].shape[0]
            if K <= 1:
                continue
            cur = int(child[L])
            if cur < 0:
                child[L] = int(self._rng.integers(0, K))
            else:
                # sample from 0..K-2 and shift to avoid 'cur'
                alt = int(self._rng.integers(0, K - 1))
                if alt >= cur:
                    alt += 1
                child[L] = alt

    def _eval_sequence(self,
                       seed_xyz: np.ndarray,
                       t: Optional[np.ndarray],
                       layers: List[Tuple[int, int]],
                       shortlists: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                       gene_row: np.ndarray,
                       build_graph: bool) -> Dict[str, Any]:
        r"""
        Evaluate one genome: EKF predict→select→update across layers.

        For layer :math:`\ell`:

        1. Solve :math:`\Delta t_\ell` to the layer surface and compute
           :math:`(\hat{\mathbf{x}}_\ell,\mathbf{P}^-_\ell,\mathbf{S}_\ell)`
           via :meth:`Brancher._ekf_predict`.
        2. Choose the gene's shortlist hit index. If the gene is ``-1`` or
           out-of-range, fall back to the cheapest (minimum χ²) within the
           shortlist under the **current** :math:`\mathbf{S}_\ell`.
        3. Update with a single Cholesky-based gain
           :math:`\mathbf{K}_\ell=\texttt{kalman\_gain}(\mathbf{P}^-_\ell,\mathbf{H},\mathbf{S}_\ell)`:

           .. math::

               \mathbf{x}^+_\ell = \hat{\mathbf{x}}_\ell + \mathbf{K}_\ell(\mathbf{z}_\ell-\hat{\mathbf{x}}_{\ell,0:3}), \qquad
               \mathbf{P}^+_\ell \approx (\mathbf{I}-\mathbf{K}_\ell\mathbf{H})\,\mathbf{P}^-_\ell.

        The fitness increment is :math:`\chi^2_\ell` for the chosen hit, as
        computed by :func:`trackml_reco.ekf_kernels.chi2_batch`.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Seed triplet for initializing the helix.
        t : ndarray or None
            Optional timestamps for determining the initial :math:`\Delta t_0`.
        layers : list[tuple[int, int]]
            Layer sequence.
        shortlists : dict
            Output of :meth:`_build_shortlists`.
        gene_row : ndarray, shape (L,), dtype int
            Genome to evaluate (one index per layer, or ``-1`` for fallback).
        build_graph : bool
            If ``True``, build and return a graph of the chosen edges.

        Returns
        -------
        result : dict
            Keys: ``'traj'``, ``'hit_ids'``, ``'state'``, ``'cov'``, ``'score'``,
            and ``'graph'`` (``nx.DiGraph`` if requested, else empty graph).
        """
        dt0 = float((t[1] - t[0]) if (t is not None and len(t) >= 2) else 1.0)
        v0, k0 = self._estimate_seed_helix(seed_xyz.astype(self._dtype), dt0, self.B_z)
        state = np.hstack([seed_xyz[2].astype(self._dtype), v0.astype(self._dtype), np.array([k0], dtype=self._dtype)])
        cov = np.eye(self.state_dim, dtype=self._dtype) * 0.1

        traj: List[np.ndarray] = [seed_xyz[0].astype(self._dtype),
                                  seed_xyz[1].astype(self._dtype),
                                  seed_xyz[2].astype(self._dtype)]
        hit_ids: List[int] = []
        score = 0.0

        H = self._H
        G = nx.DiGraph() if build_graph else None
        N = len(layers)

        for i, layer in enumerate(layers):
            surf = self.layer_surfaces[layer]
            try:
                dt = self._solve_dt_to_surface(state, surf, dt_init=dt0)
            except Exception:
                break

            # Predict and build S
            x_pred, P_pred, S, _H_cached = self._ekf_predict(state, cov, float(dt))
            # Vector of candidates for this layer (from shortlist)
            pts = shortlists[layer]['pts']
            ids = shortlists[layer]['ids']
            if pts.shape[0] == 0:
                break

            j = int(gene_row[i])
            if j < 0 or j >= pts.shape[0]:
                # Fallback: cheapest within shortlist under current S
                diff_all = pts - x_pred[:3]
                chi2_all = chi2_batch(diff_all, S)        # stable Cholesky inside
                j = int(np.argmin(chi2_all))
                chi2_j = float(chi2_all[j])
            else:
                # Compute just chosen residual's χ²
                diff_j = pts[j] - x_pred[:3]
                chi2_j = float(chi2_batch(diff_j[None, :], S)[0])

            z = pts[j]
            # EKF update with fast gain
            state, cov = self._ekf_update_meas(x_pred, P_pred, z, H, S)

            traj.append(z)
            hit_ids.append(int(ids[j]))
            score += chi2_j

            if build_graph:
                G.add_edge((i, tuple(traj[-2])), (i + 1, tuple(z)), cost=chi2_j)

        return {
            'traj': traj,
            'hit_ids': hit_ids,
            'state': state,
            'cov': cov,
            'score': float(score),
            'graph': (G if build_graph else nx.DiGraph())
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
        Execute GA over shortlist indices and return the best branch (+ optional graph).

        Pipeline
        --------
        1. **Shortlists** (once): predict-only EKF chain from the seed and per-layer
           χ² gating (honoring optional deny lists) via
           :meth:`Brancher._layer_topk_candidates`.
        2. **Initialize** a population over shortlist indices; ``-1`` means
           "fallback to cheapest".
        3. **Evaluate** population with :meth:`_eval_sequence`.
        4. **Generations** (up to ``n_gens`` or early stop):
           - **Elitism**: keep the top ``ceil(elite_frac*pop_size)`` genomes unchanged.
           - **Parent selection**: vectorized tournaments of size ``tournament_k``.
           - **Crossover**: one-point with probability ``cx_rate``.
           - **Mutation**: per-gene with probability ``mut_rate`` to a *different* index.
           - **Evaluation**: reuse elite phenotypes; evaluate children.
           - **Early stopping** after ``patience`` stagnant generations.
        5. **Result**: take the best genome; if ``plot_tree=True``, re-evaluate it
           with ``build_graph=True`` to return its sparse path graph.

        Parameters
        ----------
        seed_xyz : ndarray, shape (3, 3)
            Three seed points to initialize the helix (two segments).
        layers : list[tuple[int, int]]
            Ordered traversal of layer keys.
        t : ndarray or None
            Optional timestamps aligned with the seed.
        plot_tree : bool, optional
            If ``True``, return the best individual's path graph (``nx.DiGraph``).
        deny_hits : sequence[int] or None, keyword-only
            Per-call deny list; persistent behavior can be configured via
            :meth:`Brancher.set_deny_hits`.

        Returns
        -------
        results : list[dict]
            A singleton list with the best branch:
            ``{'traj','hit_ids','state','cov','score'}``.
        G : networkx.DiGraph
            Best individual's sparse path graph if requested, else empty graph.

        Notes
        -----
        - The fitness is **pure χ²** accumulated along the genome path; if a
          persistent deny policy is configured in the base, it is applied during
          shortlist construction (affecting which hits appear there).
        - If ``layers`` is empty, returns ``([], nx.DiGraph())``.
        """
        if not layers:
            return [], nx.DiGraph()

        # 1) Shortlists once, honoring deny list
        shortlists = self._build_shortlists(seed_xyz, t, layers, deny_hits=deny_hits)

        # 2) Init population
        pop = self._init_population(shortlists, layers)
        fitness = np.full(self.pop_size, np.inf, dtype=self._dtype)
        phenos: List[Optional[Dict[str, Any]]] = [None] * self.pop_size

        # Evaluate initial population
        for i in range(self.pop_size):
            ph = self._eval_sequence(seed_xyz, t, layers, shortlists, pop[i], build_graph=False)
            phenos[i] = ph
            fitness[i] = ph['score']

        best_prev = float(np.min(fitness))
        no_improve = 0
        n_elite = max(1, int(self.elite_frac * self.pop_size))

        # 3) GA generations
        for _gen in range(self.n_gens):
            # Elites (copy genomes & re-use phenotypes/fitness)
            elite_idx = np.argsort(fitness)[:n_elite]
            elites = pop[elite_idx].copy()
            elite_ph = [phenos[idx] for idx in elite_idx]
            elite_fit = fitness[elite_idx].copy()

            # Parents via tournaments (vectorized) to produce remaining children
            n_children = self.pop_size - n_elite
            parent_idx = self._tournament_indices(fitness, max(2 * n_children, 2))
            parents = pop[parent_idx]

            # Crossover + mutation
            children = []
            it = iter(parents)
            while len(children) < n_children:
                try:
                    p1 = next(it); p2 = next(it)
                except StopIteration:
                    # In case of odd count, resample a parent
                    p1 = pop[int(self._rng.integers(0, self.pop_size))]
                    p2 = pop[int(self._rng.integers(0, self.pop_size))]
                c1, c2 = self._crossover_pair(p1, p2)
                self._mutate_inplace(c1, shortlists, layers)
                if len(children) < n_children:
                    children.append(c1)
                if len(children) < n_children:
                    self._mutate_inplace(c2, shortlists, layers)
                    children.append(c2)

            new_pop = np.vstack([elites, np.asarray(children, dtype=np.int32)])

            # Evaluate: elites reused, children computed
            new_fitness = np.full(self.pop_size, np.inf, dtype=self._dtype)
            new_phenos: List[Optional[Dict[str, Any]]] = [None] * self.pop_size

            # Put elites back
            new_fitness[:n_elite] = elite_fit
            for i, ph in enumerate(elite_ph):
                new_phenos[i] = ph

            # Evaluate children
            for i in range(n_elite, self.pop_size):
                ph = self._eval_sequence(seed_xyz, t, layers, shortlists, new_pop[i], build_graph=False)
                new_phenos[i] = ph
                new_fitness[i] = ph['score']

            pop, phenos, fitness = new_pop, new_phenos, new_fitness

            # Early stopping
            best_now = float(np.min(fitness))
            if best_now + 1e-9 < best_prev:
                best_prev = best_now
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        # Final best, build its graph exactly once if requested
        best_idx = int(np.argmin(fitness))
        best_ph = phenos[best_idx]
        if plot_tree:
            best_ph = self._eval_sequence(seed_xyz, t, layers, shortlists, pop[best_idx], build_graph=True)
            G = best_ph['graph']
        else:
            G = nx.DiGraph()

        result = {k: best_ph[k] for k in ('traj', 'hit_ids', 'state', 'cov', 'score')}
        return [result], G
