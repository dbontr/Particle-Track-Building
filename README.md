# Meta-Heuristic Track Building with a Helical Extended Kalman Filter

This repository implements meta-heuristic track building atop a helical Extended Kalman Filter (EKF) for the TrackML challenge.
Each candidate track is represented by the state

$$
\mathbf{x} = (x, y, z, \phi, \tan\lambda, q/p_T)^{\top},
$$

where $q$ is the particle charge and $p_T$ its transverse momentum. In a solenoidal field $B\hat{z}$ the trajectory follows a
helix with curvature $\kappa = qB/p_T$ and radius $R = 1/\kappa$ [1,20]. Propagation over an arc length $s$ updates the
parameters as

$$
\begin{aligned}
\phi' &= \phi + \kappa s,\\
x' &= x + R(\sin\phi' - \sin\phi),\\
y' &= y - R(\cos\phi' - \cos\phi),\\
z' &= z + s\tan\lambda ,
\end{aligned}
$$

while the EKF prediction–update cycle transports the state and covariance.
Linearising the motion and measurement models introduces the Jacobians
$F=\partial f/\partial\mathbf{x}$ and $H=\partial h/\partial\mathbf{x}$.
For the helical propagation above one obtains, to first order in $s$,

$$
F =
\begin{pmatrix}
1 & 0 & 0 & R(\cos\phi' - \cos\phi) & 0 & -Bs\frac{q}{p_T^2} \\
0 & 1 & 0 & R(\sin\phi' - \sin\phi) & 0 & 0 \\
0 & 0 & 1 & 0 & s & 0 \\
0 & 0 & 0 & 1 & 0 & -Bs\frac{q}{p_T^2} \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix},
$$

while planar sensor modules measuring Cartesian positions employ

$$
h(\mathbf{x}) =
\begin{pmatrix}x \\ y \\ z\end{pmatrix}, \qquad
H =
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0
\end{pmatrix}.
$$

Stochastic effects from multiple scattering and energy loss are encoded in
the process noise $Q$. Using the Highland expression [25] for the RMS
scattering angle $\theta_0 = 13.6\,\text{MeV}/(p\beta)\,\sqrt{x/X_0}\,[1+0.038\ln(x/X_0)]$, the angular covariances are
approximated as $Q_{\phi\phi} = Q_{\lambda\lambda} = \theta_0^2$ with
remaining elements filled via kinematic relations [3,20]. The prediction and
update equations become

$$
\begin{aligned}
\mathbf{x}_{k|k-1} &= f(\mathbf{x}_{k-1}), & P_{k|k-1} &= F P_{k-1} F^\top + Q,\\
r_k &= \mathbf{z}_k - h(\mathbf{x}_{k|k-1}), & S_k &= H P_{k|k-1} H^\top + R,\\
K_k &= P_{k|k-1} H^\top S_k^{-1}, & \mathbf{x}_k &= \mathbf{x}_{k|k-1} + K_k r_k,\\
&& P_k &= (I - K_k H) P_{k|k-1}.
\end{aligned}
$$

Branches are retained only if the gating test
$\chi^2 = r_k^\top S_k^{-1} r_k < \chi^2_{\max}$ passes [9,19,21]. For
Cartesian sensors the innovation has three degrees of freedom, so
$\chi^2_{\max}$ corresponds to an ellipsoidal gate enclosing the desired
global $p$-value. This statistical pruning reduces combinatorics while
preserving hits consistent with the predicted trajectory.
Starting from three-hit seeds, the framework explores candidate assignments
with algorithms ranging from the baseline EKF to A*, Ant Colony Optimization,
Genetic Algorithms, Particle Swarm Optimization, Simulated Annealing, and
layer-wise Hungarian matching [22].

## Brancher Algorithms

### Extended Kalman Filter (EKF)

Baseline combinatorial tracking where each branch follows the EKF predict–update cycle:

$$
\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1}), \quad P_{k|k-1} = F P_{k-1} F^T + Q,
$$
$$
K = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}, \quad \mathbf{x}_k = \mathbf{x}_{k|k-1} + K(\mathbf{z}_k - H\mathbf{x}_{k|k-1}).
$$

The residual $r_k = \mathbf{z}_k - H\mathbf{x}_{k|k-1}$ yields a score $\chi^2 = r_k^T (H P_{k|k-1} H^T + R)^{-1} r_k$.

### A\* Search (A\*)

Best-first expansion uses a cost function $f(n) = g(n) + h(n)$ where $g$ is accumulated $\chi^2$ and $h$ is a heuristic estimate of remaining cost. Cross-track residuals are damped via

$$
C_{\perp} = \lambda_{\perp}\frac{\big\Vert (I - \mathbf{t}\mathbf{t}^\top)(\mathbf{z} - \hat{\mathbf{x}}) \big\Vert^2}{\sigma^2},
$$

Algorithmic guidance follows the classic formulation of Hart, Nilsson & Raphael [14].

### Ant Colony Optimization (ACO)

Ants probabilistically construct paths using pheromones $\tau$ and heuristic desirability $\eta$. After evaporation, pheromones are updated by

$$
\tau \leftarrow (1 - \rho)\,\tau + \Delta\tau,
$$

with selection probabilities $p_i \propto \tau_i^{\alpha} \eta_i^{\beta}$ [13].

### Genetic Algorithm (GA)

Individuals encode layer-wise hit indices. Selection uses tournaments, one-point crossover recombines parents, and genes mutate with probability $\mu$ to explore new candidates [15]. Fitness is the EKF track score.

### Particle Swarm Optimization (PSO)

Swarm particles carry sparse velocities that bias sampling from each layer's gated hits. Velocities follow

$$
v \leftarrow w v + c_1 r_1 (p_{\mathrm{best}} - x) + c_2 r_2 (g_{\mathrm{best}} - x), \qquad x \leftarrow x + v
$$

to balance exploration and exploitation [16].

### Simulated Annealing (SA)

Local mutations to a current path are accepted with probability

$$
P_{\mathrm{accept}} = \min\{1, \exp(-\Delta/T)\},
$$

while the temperature cools as $T \leftarrow \alpha T$ [17].

### Hungarian Assignment

For each layer, a cost matrix built from $\chi^2$ values is solved for a one-to-one assignment:

$$
\min_{x_{ij}} \sum_{i,j} C_{ij} x_{ij} \quad \text{s.t.}\; \sum_j x_{ij}=1,\; \sum_i x_{ij}=1,
$$

enforcing collision-free hit usage within the layer [18].

## Parallel Track Building

High hit densities require exploring many candidate branches in tandem. The
`CollaborativeParallelTrackBuilder` distributes seed groups across a pool of
worker threads and coordinates hit ownership through a shared score book. For a
hit $h$ with best recorded score $s_h$, a branch with score $s_{\text{branch}}$
claims the hit only if

$$
s_{\text{branch}} < s_h - \text{margin},
$$

ensuring a strictly better track before ownership changes. Denied hits are
passed back to other threads so exploration continues without conflicts. The
builder merges per-thread debug graphs, supports an optional per-seed time
budget, and targets efficient multicore execution [23,24].

Enable the collaborative mode from the CLI:

```bash
python -m trackml_reco.main --file path/to/train_1.zip --brancher ekf --parallel
```

By default eight worker threads and a claim margin of 1.0 are used; these
parameters can be tuned by instantiating `CollaborativeParallelTrackBuilder`
directly in your own scripts.

## Code Structure

```text
trackml_reco/
├── branchers/                # EKF, A*, ACO, GA, PSO, SA, Hungarian
├── data.py                   # Load TrackML events and form seeds
├── hit_pool.py               # Gating and shared hit claims
├── track_builder.py          # Sequential track construction logic
├── parallel_track_builder.py # Multithreaded builder with shared hit pools
├── metrics.py                # Evaluation utilities
├── plotting.py               # Optional 2D/3D visualisation
└── utils.py                  # Submission helpers and random baselines
```

Top-level files:

* `config.json` — hyperparameters controlling branchers and gating.
* `main.py` — CLI entry point for running reconstruction.
* `requirements.txt` / `setup.py` — installation helpers.

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

## References

1. Yeo _et al._, “Analytic Helix Propagation for Kalman Filters,” JINST **15** (2024).
2. Battisti _et al._, “Kalman-Filter-Based Tracking in High-Energy Physics,” Comput. Phys. Commun. **324**, 108601 (2024).
3. Strandlie & Frühwirth, “Track Fitting with Combinatorial Kalman Filter,” Nucl. Instrum. Methods A **559**, 305–308 (2007).
4. Schöning, “Advanced Seeding Techniques for Track Reconstruction,” CERN-THESIS-2021-042 (2021).
5. Jackson, J. D., “Classical Electrodynamics,” 3rd ed., Wiley (1998).
6. Amrouche _et al._, “The Tracking Machine Learning challenge: Accuracy phase,” NeurIPS (2018) / arXiv:1904.06778 (2019).
7. Friedman, Bentley & Finkel, “An Algorithm for Finding Best Matches in Logarithmic Expected Time,” ACM Trans. Math. Softw. **3**, 209–226 (1977).
8. SciPy `cKDTree` documentation, SciPy v1.10.1, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
9. Welch & Bishop, “An Introduction to the Kalman Filter,” University of North Carolina at Chapel Hill (1995).
10. MathWorks, “Tuning Kalman Filter to Improve State Estimation,” MATLAB & Simulink documentation (2024).
11. ACTS Collaboration, “Track Seeding,” ACTS documentation v4.0.0, [https://acts.readthedocs.io/en/v4.0.0/core/seeding.html](https://acts.readthedocs.io/en/v4.0.0/core/seeding.html)
12. Golling _et al._, “TrackML: a tracking Machine Learning challenge,” EPJ Web of Conferences **214**, 06037 (2019).
13. Dorigo & Stützle, _Ant Colony Optimization_, MIT Press (2004).
14. Hart, Nilsson & Raphael, “A Formal Basis for the Heuristic Determination of Minimum Cost Paths,” IEEE Trans. Syst. Sci. Cybern. **4**, 100–107 (1968).
15. Holland, _Adaptation in Natural and Artificial Systems_, Univ. of Michigan Press (1975); Goldberg, _Genetic Algorithms in Search, Optimization, and Machine Learning_, Addison‑Wesley (1989).
16. Kennedy & Eberhart, “Particle Swarm Optimization,” Proc. IEEE Int. Conf. Neural Networks, 1942–1948 (1995).
17. Kirkpatrick, Gelatt & Vecchi, “Optimization by Simulated Annealing,” Science **220**, 671–680 (1983).
18. Kuhn, “The Hungarian Method for the Assignment Problem,” Nav. Res. Logist. Q. **2**, 83–97 (1955).
19. Frühwirth, “Application of Kalman filtering to track and vertex fitting,” Nucl. Instrum. Methods A **262**, 444–450 (1987).
20. Blum, Riegler & Rolandi, _Particle Detection with Drift Chambers_, 2nd ed., Springer (2008).
21. Bar-Shalom, Li & Kirubarajan, _Estimation with Applications to Tracking and Navigation_, 2nd ed., Wiley (2001).
22. Eiben & Smith, _Introduction to Evolutionary Computing_, 2nd ed., Springer (2015).
23. Cerati _et al._, "Parallelized Kalman Filter Tracking on Many-Core Processors and GPUs," J. Phys.: Conf. Ser. **608**, 012057 (2015).
24. Klijnsma _et al._, "Multi-threaded and vectorized Kalman Filter tracking for the CMS experiment," Comput. Softw. Big Sci. **3**, 11 (2019).
25. Highland, "Some Practical Remarks on Multiple Scattering," Nucl. Instrum. Methods **129**, 497–499 (1975).

_Feel free to adapt parameters (`noise_std`, `num_branches`, gating thresholds) in `config.json` to your dataset and detector geometry._
