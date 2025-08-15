# Meta-Heuristic Track Building with a Helical Extended Kalman Filter

Fast, headless-safe TrackML reconstruction with a helical EKF core and a family of meta-heuristics (A*, ACO, GA, PSO, SA, per-layer Hungarian). The pipeline is vectorized/NumPy-first, exploits robust Cholesky factorizations (no explicit matrix inverses), and scales via collaborative multithreading with conflict-free shared hit ownership. It ships an optimizer to tune brancher hyper-parameters on your event(s).

## Mathematical model

### State, kinematics, and propagation

We adopt the helical state
$$
\mathbf{x}=(x,\,y,\,z,\,\phi,\,\tan\lambda,\,q/p_T)^\top,\qquad
\kappa \equiv \frac{qB}{p_T} ,
$$
where $(x,y,z)$ are Cartesian positions, $\phi$ is the azimuth, $\tan\lambda=p_z/p_T$ is the dip (with $p_T$ the transverse momentum), and $q/p_T$ is the signed inverse transverse momentum. The curvature is $\kappa = qB/p_T = 1/R$ in a uniform solenoidal field $B\hat z$. In the transverse plane the motion is circular with radius $R$ and uniform angular rate $d\phi/ds=\kappa$ ($s$: path length). Over a step $s$,
$$
\begin{aligned}
\phi' &= \phi + \kappa s,\\
x' &= x + R(\sin\phi' - \sin\phi),\\
y' &= y - R(\cos\phi' - \cos\phi),\\
z' &= z + s\,\tan\lambda .
\end{aligned}
$$

Linearizing $f:\mathbf{x}\mapsto\mathbf{x}'$ gives the Jacobian $F=\partial f/\partial\mathbf{x}$. One explicit form is
$$
F\approx
\begin{pmatrix}
1&0&0&R(\cos\phi'-\cos\phi)&0&\frac{\partial x'}{\partial(q/p_T)}\\
0&1&0&R(\sin\phi'-\sin\phi)&0&\frac{\partial y'}{\partial(q/p_T)}\\
0&0&1&0&s&0\\
0&0&0&1&0&Bs\\
0&0&0&0&1&0\\
0&0&0&0&0&1
\end{pmatrix},
$$
where $\partial x'/\partial(q/p_T)$ and $\partial y'/\partial(q/p_T)$ follow from $R=1/\kappa$ and are $\mathcal{O}(sB/p_T)$. Sensors measure Cartesian positions
$$
h(\mathbf{x})=(x,y,z)^\top,\qquad
H=\begin{pmatrix}1&0&0&0&0&0\\0&1&0&0&0&0\\0&0&1&0&0&0\end{pmatrix}.
$$

### Process noise and scattering

For a thin scatterer with material budget $x/X_0$, the RMS multiple-scattering angle is
$$
\theta_0=\frac{13.6\,\mathrm{MeV}}{p\beta}\sqrt{\frac{x}{X_0}}\left[1+0.038\ln\!\left(\frac{x}{X_0}\right)\right] .
$$
We approximate $Q$ by injecting independent kicks in $(\phi,\lambda)$ with variance $\theta_0^2$ and projecting to the full state through
$$
G=
\begin{pmatrix}
0&0\\0&0\\0&0\\1&0\\0&1\\0&0
\end{pmatrix},\qquad
Q \approx G\,\mathrm{diag}(\theta_0^2,\theta_0^2)\,G^\top .
$$
This thin-scatterer model is adequate for modest layer thicknesses; see Frühwirth [19] and the Particle Data Group [26] for derivations and refinements.

### EKF equations (fused, factorized)

Prediction:
$$
\hat{\mathbf{x}}_k=f(\mathbf{x}_{k-1}),\qquad
\hat P_k = F P_{k-1} F^\top + Q .
$$

Innovation and gate (3-D Cartesian):
$$
r_k=\mathbf{z}_k-h(\hat{\mathbf{x}}_k),\quad
S_k=H \hat P_k H^\top + R,\quad
R=\mathrm{diag}(\sigma_x^2,\sigma_y^2,\sigma_z^2),\quad
\chi^2_k=r_k^\top S_k^{-1} r_k .
$$

Update via Cholesky $S_k=LL^\top$ (two triangular solves, no inverse):
$$
K_k = \hat P_k H^\top S_k^{-1} \equiv \textsf{solve}(L^\top,\ \textsf{solve}(L,\ H\hat P_k^\top))^\top,
$$
$$
\mathbf{x}_k=\hat{\mathbf{x}}_k+K_k r_k,\qquad
P_k=(I-K_k H)\hat P_k .
$$

Gate with global p-value $\alpha$: accept if
$$
\chi^2_k < \chi^2_{3;\,\alpha}\qquad(\alpha\approx0.997\Rightarrow\chi^2_{3;\,\alpha}\simeq14.2) .
$$
We additionally **tighten** the effective radius along depth $d\in[0,1]$ by a linear factor, and scale the nominal gate by $\sqrt{\chi^2_{3;\,\alpha}}$ to avoid eigendecompositions.

## Algorithms at a glance

### EKF baseline

Greedy branching: per layer, fetch Top-K nearest hits passing the χ² gate; expand branches by lowest incremental $\chi^2$.

### A\*

Best-first with $f(n)=g(n)+h(n)$, where $g$ is accumulated $\chi^2$ and $h$ penalizes cross-track residuals
$$
C_\perp=\lambda_\perp\frac{\|(I-\mathbf{t}\mathbf{t}^\top)(\mathbf{z}-\hat{\mathbf{x}})\|^2}{\sigma^2}.
$$

### ACO

Stochastic construction with pheromones $\tau$, heuristic $\eta$, evaporation $\rho$, and $p_i\propto \tau_i^\alpha \eta_i^\beta$.

### GA

Layer-wise gene encoding; tournament selection; one-point crossover; mutation rate $\mu$. Fitness = EKF score.

### PSO

Sparse velocity over layer choices:
$$
v\leftarrow wv+c_1 r_1 (p_{\mathrm{best}}-x)+c_2 r_2 (g_{\mathrm{best}}-x),\quad x\leftarrow x+v .
$$

### Simulated Annealing (SA) — **with incremental tail rebuilds**

Choose a layer $k$ with probability $\propto$ local residuals, propose up to $M$ alternatives, accept with
$$
P=\min\{1,\exp(-\Delta/T)\}\qquad(\Delta=\text{score}_\text{new}-\text{score}_\text{cur}).
$$
We rebuild only the tail $k\to$end using a **prefix cache** of $(x,P,z,\text{id})$. Continuity tie-breakers:
$$
\text{angle penalty}=w_\text{ang}(1-\cos\theta),\qquad
\text{curvature penalty}=w_\text{curv}(\kappa-\kappa_\text{prev})^2 .
$$
Cooling is adaptive with mild reheats driven by the recent acceptance ratio.

### Per-layer Hungarian

Solve $\min\sum_{ij} C_{ij}x_{ij}$ s.t. row/col sums = 1 to enforce collision-free hit usage in a layer.

## Robust linear algebra kernels

All hot-path matrix ops are **factorization first**:

- **Robust Cholesky**: if $S\not\succ 0$, add diagonal jitter $\varepsilon I$ with geometric escalation; final resort: eigenvalue flooring.  
- **Mahalanobis**: for residuals $d_i\in\mathbb{R}^3$,
  $$
  \chi^2_i = d_i^\top S^{-1} d_i
  \quad\Rightarrow\quad
  \texttt{solve}(L^\top,\ \texttt{solve}(L, d_i)).
  $$
- **Kalman gain**: solve $S X^\top=(\hat P H^\top)^\top$ once; reuse for all rows.

Numba implementations are provided (parallel batched χ²; per-row gain), with drop-in NumPy fallbacks.

## Data pipeline and geometry

- **Loading & preprocessing**: units (mm→m) with dtype-safe assignment; derived $r=\sqrt{x^2+y^2}$ and momentum features; **weighting** with CoW disabled inside `trackml.weights` to silence chained-assignment warnings.
- **Layer surfaces**: infer simple disk/cylinder surfaces per `(volume_id, layer_id)` by comparing $z$-span vs $r$-span; used by analytic `solve_dt_to_surface(...)` during propagation.

## HitPool (indexing & reservations)

- Build per-layer KD-trees via a **single** `lexsort((layer, volume))`; contiguous segments form layers; no temporary DataFrame columns.
- Store `(KDTree, points[N,3], ids[N])` plus an immutable `frozenset(ids)` (for O(1) membership).
- **Assignment API**: bulk reserve/release with set semantics; vectorized `np.isin` paths when large; fast kNN/radius candidate queries that **exclude** assigned hits.

_Time & memory_: columns are materialized once to compact `float64`/`int64` arrays; trees are built with `balanced_tree=True, compact_nodes=True`.

## Gating & candidates

For layer $\ell$ with prediction $(\hat x,S)$:

1. Query Top-K nearest via `cKDTree` (small $K$ keeps cache hot).
2. Compute vectorized χ² using the **same** $S$ (Cholesky once).
3. Accept if $\chi^2<(\gamma\cdot q_\alpha)^2$, where $q_\alpha=\sqrt{\chi^2_{3;\,\alpha}}$, and $\gamma=1-\text{tighten}\cdot d$ decays linearly with depth $d\in[0,1]$.

## Collaborative parallel builder

A shared `SharedHitBook` guards hit ownership. For a branch with score $s_b$ to claim hit $h$, we require
$$
s_b < s_h - \text{margin},
$$
where $s_h$ is the current best score recorded for $h$. Claims are atomic; denied hits are added to per-call **deny-lists** to avoid live conflicts. The builder supports:

- Per-seed **time budgets**.
- Optional graph composition (`networkx`) for debugging.
- Headless-safe plotting guard (`Agg` backend; `plt.ioff()`).

## Metrics & evaluation

Given predictions $P=\{p_i\}$ and truth $T=\{t_j\}$:

- **Regression**:
  $$
  \text{MSE}=\frac{1}{|P|}\sum_i \min_j \|p_i-t_j\|^2,\qquad
  \text{RMSE}=\sqrt{\text{MSE}} .
  $$
- **Efficiency** (with tolerance $\tau$):
  $$
  \text{recall}=\frac{\#\{t\in T:\min_{p\in P}\|p-t\|\le\tau\}}{|T|}\cdot 100\%,\quad
  \text{precision}=\frac{\#\{p\in P:\min_{t\in T}\|p-t\|\le\tau\}}{|P|}\cdot 100\%,
  $$
  $$
  \text{F1}= \frac{2\,\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}} .
  $$

A small-problem broadcast path avoids KD-tree overhead when $\min(|P|,|T|)\le 64$.

## Optimizer

Tune any brancher’s hyperparameters:

```python
from pathlib import Path
from trackml_reco.optimize import optimize_brancher

result = optimize_brancher(
    hit_pool=hit_pool,
    base_config=config,
    brancher_key="sa",             # ekf, astar, aco, pso, sa, ga, hungarian
    space_path=Path("param_space.json"),
    metric="f1",                   # mse, rmse, recall, precision, f1, score, ...
    iterations=250,
    n_init=15,
    use_parallel=True,
    parallel_time_budget=0.03,     # s per seed
    max_seeds=200,
    aggregator="mean",
    n_best_tracks=5,
    tol=0.005,
)
print(result.best_value, result.best_params)
```

- Supports **skopt** backends (`gp`, `forest`, `gbrt`, `random`) or a robust random+local fallback.
- Metric aliases: `pct_hits`, `%hits`, `percent_matched` → `recall`.
- For **higher-is-better** metrics we minimize the complement (e.g., `100-recall`).

## Command-line usage

```bash
# Basic run (EKF baseline)
python -m trackml_reco.main --file path/to/train_1.zip --brancher ekf --pt 2.0

# Parallel builder
python -m trackml_reco.main --file train_1.zip --brancher sa --parallel

# Optimize the SA brancher
python -m trackml_reco.main --file train_1.zip --brancher sa --optimize   --opt-metric f1 --opt-iterations 250 --opt-n-init 20   --opt-space param_space.json --opt-parallel --opt-max-seeds 200   --opt-out best_config.json --opt-history trials.csv
```

Useful flags:

- `--plot/--no-plot` and `--extra-plots`
- `--profile [--profile-out FILE]` (context-managed `cProfile`)
- `--config CONFIG.json` (attach inferred layer surfaces automatically)

## Configuration snippets

**Brancher config** (excerpt):

```json
{
  "ekfsa_config": {
    "noise_std": 2.0,
    "B_z": 0.002,
    "initial_temp": 1.0,
    "cooling_rate": 0.95,
    "n_iters": 1000,
    "step_candidates": 5,
    "max_no_improve": 150,
    "gate_multiplier": 3.0,
    "gate_tighten": 0.15,
    "gate_quantile": 0.997,
    "time_budget_s": 3.0,
    "build_graph": false,
    "graph_stride": 25,
    "min_temp": 1e-3
  }
}
```

**Optimizer search space** (excerpt):

```json
{
  "sa": [
    {"name":"initial_temp","type":"float","min":0.2,"max":3.0,"log":false},
    {"name":"cooling_rate","type":"float","min":0.90,"max":0.995},
    {"name":"gate_multiplier","type":"float","min":2.0,"max":5.0},
    {"name":"step_candidates","type":"int","min":2,"max":10}
  ]
}
```

## Performance tips

- Prefer **parallel** builder for dense events; tune `claim_margin` (default 1.0).
- Keep `step_candidates` small (≤8) to preserve cache locality.
- Use **quantile gates** (`gate_quantile≈0.997`) to stabilize false positives early; increase `gate_tighten` to reduce late branching.
- Enable Numba (if available) for batched χ² and gain; otherwise NumPy BLAS is already optimized.

## Visualization

- `plotting.plot_extras(...)` shows detector layers, truth paths (R–Z and 3D).
- `plot_branches(...)` overlays per-layer rectangles, seeds, branches, and truth hits.
- Headless environments are respected (`Agg` backend; `show()` is no-op when disabled).

## Code structure

```text
trackml_reco/
├── branchers/                # EKF, A*, ACO, GA, PSO, SA, Hungarian
├── data.py                   # Load/preprocess, weights, HitPool creation
├── hit_pool.py               # Per-layer cKDTree + reservations API
├── track_builder.py          # Sequential builder (brancher-agnostic)
├── parallel_track_builder.py # Multithreaded collaborative builder
├── metrics.py                # MSE/recall/precision/F1; KD-tree fast paths
├── kernels.py                # Robust Cholesky, χ², gain (Numba/NumPy)
├── optimize.py               # skopt/random search wrapper
├── profiling.py              # `prof(...)` context manager
├── plotting.py               # Optional 2D/3D visuals
└── utils.py                  # Submission helpers, perturbations
```

## Troubleshooting

- **No seeds built**: ensure ≥3 distinct `(volume, layer)` hits per particle after the $p_T$ cut; try lowering `--pt`.
- **Too few branches**: relax gates (`gate_multiplier↑`, `gate_quantile↑`), reduce `gate_tighten`, increase `step_candidates`.
- **Conflicts in parallel mode**: lower `claim_margin` or increase per-seed `--opt-parallel-time-budget`.
- **Numba missing**: the code transparently falls back to NumPy; install Numba for extra speed.

## References

1. J. Yeo, et al. (2024). “Analytic helix propagation for Kalman filters”. Journal of Instrumentation [Online]. vol 15, article P08012.
2. M. Battisti, et al. (2024). “Kalman-filter-based tracking in high-energy physics”. Computer Physics Communications [Online]. vol 324, 108601.
3. A. Strandlie, R. Frühwirth. (2007). “Track fitting with combinatorial Kalman filter”. Nuclear Instruments and Methods in Physics Research Section A [Online]. vol 559, p. 305.
4. A. Schöning. (2021). “Advanced seeding techniques for track reconstruction”. CERN Thesis, CERN-THESIS-2021-042.
5. J. D. Jackson. (1998). Classical Electrodynamics, 3rd ed. Wiley.
6. M. Amrouche, et al. (2018). “The Tracking Machine Learning Challenge: Accuracy Phase”. Advances in Neural Information Processing Systems. arXiv:1904.06778 [cs.LG].
7. J. H. Friedman, J. L. Bentley, R. A. Finkel. (1977). “An algorithm for finding best matches in logarithmic expected time”. ACM Transactions on Mathematical Software [Online]. vol 3, p. 209.
8. SciPy Project. (n.d.). “cKDTree documentation”. SciPy v1.10.1 Documentation [Online]. Available: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
9. G. Welch, G. Bishop. (1995). “An introduction to the Kalman filter”. University of North Carolina at Chapel Hill, Technical Report TR 95-041.
10. MathWorks. (2024). “Tuning Kalman filter to improve state estimation”. MATLAB & Simulink Documentation [Online]. Available: [https://www.mathworks.com/help/control/ug/tuning-kalman-filter.html](https://www.mathworks.com/help/control/ug/tuning-kalman-filter.html)
11. ACTS Collaboration. (n.d.). “Track seeding”. ACTS Documentation v4.0.0 [Online]. Available: [https://acts.readthedocs.io/en/v4.0.0/core/seeding.html](https://acts.readthedocs.io/en/v4.0.0/core/seeding.html)
12. T. Golling, et al. (2019). “TrackML: A tracking machine learning challenge”. EPJ Web of Conferences [Online]. vol 214, 06037.
13. M. Dorigo, T. Stützle. (2004). Ant Colony Optimization. MIT Press.
14. P. E. Hart, N. J. Nilsson, B. Raphael. (1968). “A formal basis for the heuristic determination of minimum cost paths”. IEEE Transactions on Systems Science and Cybernetics [Online]. vol 4, p. 100.
15. J. H. Holland. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press; D. E. Goldberg. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
16. J. Kennedy, R. Eberhart. (1995). “Particle swarm optimization”. Proceedings of the IEEE International Conference on Neural Networks [Online]. vol 4, p. 1942.
17. S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi. (1983). “Optimization by simulated annealing”. Science [Online]. vol 220, p. 671.
18. H. W. Kuhn. (1955). “The Hungarian method for the assignment problem”. Naval Research Logistics Quarterly [Online]. vol 2, p. 83.
19. R. Frühwirth. (1987). “Application of Kalman filtering to track and vertex fitting”. Nuclear Instruments and Methods in Physics Research Section A [Online]. vol 262, p. 444.
20. W. Blum, W. Riegler, L. Rolandi. (2008). Particle Detection with Drift Chambers, 2nd ed. Springer.
21. Y. Bar-Shalom, X. R. Li, T. Kirubarajan. (2001). Estimation with Applications to Tracking and Navigation, 2nd ed. Wiley.
22. A. E. Eiben, J. E. Smith. (2015). Introduction to Evolutionary Computing, 2nd ed. Springer.
23. G. Cerati, et al. (2015). “Parallelized Kalman filter tracking on many-core processors and GPUs”. Journal of Physics: Conference Series [Online]. vol 608, 012057.
24. M. Klijnsma, et al. (2019). “Multi-threaded and vectorized Kalman filter tracking for the CMS experiment”. Computing and Software for Big Science [Online]. vol 3, article 11.
25. V. L. Highland. (1975). “Some practical remarks on multiple scattering”. Nuclear Instruments and Methods [Online]. vol 129, p. 497.
26. Particle Data Group; P. A. Zyla, et al. (2024). “Review of particle physics”. Progress of Theoretical and Experimental Physics [Online]. vol 2024, 083C01.
27. S. Caillou, et al. (2024). “Novel fully-heterogeneous GNN designs for track reconstruction at the HL-LHC”. EPJ Web of Conferences [Online]. vol 295. Available: [https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09028/epjconf_chep2024_09028.html](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09028/epjconf_chep2024_09028.html)
28. G. Cerati, et al. (2019). “Parallelized and vectorized tracking using Kalman filters with CMS detector geometry and events”. EPJ Web of Conferences [Online]. vol 214. Available: [https://www.epj-conferences.org/articles/epjconf/abs/2019/19/epjconf_chep2018_02002/epjconf_chep2018_02002.html](https://www.epj-conferences.org/articles/epjconf/abs/2019/19/epjconf_chep2018_02002/epjconf_chep2018_02002.html)
29. M. Dorigo, V. Maniezzo, A. Colorni. (1996). “Ant system: optimization by a colony of cooperating agents”. IEEE Transactions on Systems, Man, and Cybernetics, Part B [Online]. vol 26, p. 29.
30. J. Kennedy. (2011). “Particle swarm optimization”. Encyclopedia of Machine Learning. Springer [Online]. Available: [https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_630](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_630)
31. D. J. Webb, W. M. Alobaidi, E. Sandgren. (2017). “Maze navigation via genetic optimization”. Intelligent Information Management [Online]. vol 10. Available: [https://www.scirp.org/reference/referencespapers?referenceid=2173538]([https://www.scirp.org/reference/referencespapers?referenceid=2173538)
32. H. M. Gray. (2021). “Quantum pattern recognition algorithms for charged particle tracking”. Philosophical Transactions of the Royal Society A [Online]. vol 380, 20210103. Available: [https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0103](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0103)
33. R. E. Kalman. (1960). “A new approach to linear filtering and prediction problems”. Journal of Basic Engineering [Online]. vol 82, p. 35.

## Getting Started

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Download Dataset**:

     Obtain the TrackML Particle Identification dataset from the
     [Kaggle competition page](https://www.kaggle.com/competitions/trackml-particle-identification/rules)
     and place the desired event ``.zip`` files (e.g. ``train_1.zip``) in a known location.

3. **Run Reconstruction**:

     ```bash
     python -m trackml_reco.main --file path/to/train_1.zip --brancher ekf --pt 2.0
     ```

     Replace ``ekf`` with any brancher key (``astar``, ``aco``, ``pso``, ``sa``,``ga``, ``hungarian``). Add ``--parallel`` to enable the collaborative track builder and ``--config`` to use a custom JSON configuration. Use ``-h`` for the full list of options.

4. **Visualize**: plotting is enabled by default; add ``--no-plot`` to disable or ``--extra-plots`` for additional views of hits, seeds, and the branching tree.

_Tip:_ Tune `noise_std`, `num_branches`, and gating thresholds (`gate_*`) in `config.json` to your detector geometry and occupancy.
