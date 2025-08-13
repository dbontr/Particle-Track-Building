# Meta-Heuristic Track Building with a Helical Extended Kalman Filter

This repository implements meta-heuristic track building atop a helical Extended Kalman Filter (EKF) for the TrackML challenge.
Each candidate track is represented by the state

$$
\mathbf{x} = (x, y, z, \phi, \tan\lambda, q/p_T)^{\top},
$$

where $q$ is the particle charge and $p_T$ its transverse momentum. The polar angle $\lambda$ relates longitudinal and transverse momenta via $\tan\lambda = p_z/p_T$, so $p = p_T \sqrt{1 + \tan^2\lambda}$. Under the Lorentz force
$$
\dot{\mathbf{p}} = q\,\mathbf{v}\times B\hat{z} = qB\begin{pmatrix} v_y\\-v_x\\0 \end{pmatrix},
$$
the transverse momentum components obey $\dot p_x = qB v_y$ and $\dot p_y = -qB v_x$, yielding uniform circular motion with constant $p_T$ while $\dot p_z = 0$ so $p_z$ and $\lambda$ remain unchanged [5].  The azimuthal angle therefore advances at
$
\frac{d\phi}{dt} = \frac{qB}{p} = \omega,
$
and, using arc length $s = vt$, one has $d\phi/ds = qB/p_T \equiv \kappa$.  In a solenoidal field $B\hat{z}$ the trajectory thus traces a helix of curvature $\kappa$ and radius $R = 1/\kappa = p_T/(qB)$ [1,20]. Integrating the angular increment over an arc length $s$ updates the parameters as

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

The $\phi$– and $x$–rows depend explicitly on the curvature $\kappa = B(q/p_T)$.  Differentiating $\phi' = \phi + \kappa s$ gives $\partial\phi'/\partial(q/p_T) = B s$, while the derivative $\partial x'/\partial(q/p_T)$ follows from $R = 1/\kappa$ and the trigonometric shifts, leading to the small-$s$ term $-B s\, q/p_T^2$ shown above.

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

Stochastic effects from multiple scattering and energy loss are encoded in the process noise $Q$. Using the Highland expression [25] or the Particle Data Group recommendation [26] for the RMS scattering angle $\theta_0 = 13.6\,\text{MeV}/(p\beta)\,\sqrt{x/X_0}\,[1+0.038\ln(x/X_0)]$, the angular covariances are approximated as $Q_{\phi\phi} = Q_{\lambda\lambda} = \theta_0^2$.  For a thin scatterer this can be cast as $Q = G\,\text{diag}(\theta_0^2,\theta_0^2)G^\top$, where $G$ projects kicks in $(\phi,\lambda)$ into the full state basis and remaining elements are filled via kinematic relations [3,20]. The prediction and update equations become

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

1. J. Yeo *et al.*, "Analytic Helix Propagation for Kalman Filters," JINST **15**, P08012 (2024).  
2. M. Battisti *et al.*, "Kalman-Filter-Based Tracking in High-Energy Physics," Comput. Phys. Commun. **324**, 108601 (2024).  
3. A. Strandlie and R. Frühwirth, "Track Fitting with Combinatorial Kalman Filter," Nucl. Instrum. Methods Phys. Res., Sect. A **559**, 305 (2007).  
4. A. Schöning, "Advanced Seeding Techniques for Track Reconstruction," CERN-THESIS-2021-042 (2021).  
5. J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1998).  
6. M. Amrouche *et al.*, "The Tracking Machine Learning Challenge: Accuracy Phase," Adv. Neural Inf. Process. Syst. (2018), arXiv:1904.06778 [cs.LG].  
7. J. H. Friedman, J. L. Bentley and R. A. Finkel, "An Algorithm for Finding Best Matches in Logarithmic Expected Time," ACM Trans. Math. Softw. **3**, 209 (1977).  
8. SciPy `cKDTree` documentation, SciPy v1.10.1, Available at: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html).  
9. G. Welch and G. Bishop, "An Introduction to the Kalman Filter," Univ. of North Carolina at Chapel Hill, TR 95-041 (1995).  
10. MathWorks, "Tuning Kalman Filter to Improve State Estimation," MATLAB & Simulink documentation (2024), Available at: [https://www.mathworks.com/help/control/ug/tuning-kalman-filter.html](https://www.mathworks.com/help/control/ug/tuning-kalman-filter.html).  
11. ACTS Collaboration, "Track Seeding," ACTS documentation v4.0.0, Available at: [https://acts.readthedocs.io/en/v4.0.0/core/seeding.html](https://acts.readthedocs.io/en/v4.0.0/core/seeding.html).  
12. T. Golling *et al.*, "TrackML: A Tracking Machine Learning Challenge," EPJ Web Conf. **214**, 06037 (2019).  
13. M. Dorigo and T. Stützle, *Ant Colony Optimization*, MIT Press (2004).  
14. P. E. Hart, N. J. Nilsson and B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," IEEE Trans. Syst. Sci. Cybern. **4**, 100 (1968).  
15. J. H. Holland, *Adaptation in Natural and Artificial Systems*, Univ. of Michigan Press (1975); D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley (1989).  
16. J. Kennedy and R. Eberhart, "Particle Swarm Optimization," Proc. IEEE Int. Conf. Neural Netw. **4**, 1942 (1995).  
17. S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi, "Optimization by Simulated Annealing," Science **220**, 671 (1983).  
18. H. W. Kuhn, "The Hungarian Method for the Assignment Problem," Nav. Res. Logist. Q. **2**, 83 (1955).  
19. R. Frühwirth, "Application of Kalman Filtering to Track and Vertex Fitting," Nucl. Instrum. Methods Phys. Res., Sect. A **262**, 444 (1987).  
20. W. Blum, W. Riegler and L. Rolandi, *Particle Detection with Drift Chambers*, 2nd ed., Springer (2008).  
21. Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, 2nd ed., Wiley (2001).  
22. A. E. Eiben and J. E. Smith, *Introduction to Evolutionary Computing*, 2nd ed., Springer (2015).  
23. G. Cerati *et al.*, "Parallelized Kalman Filter Tracking on Many-Core Processors and GPUs," J. Phys. Conf. Ser. **608**, 012057 (2015).  
24. M. Klijnsma *et al.*, "Multi-threaded and Vectorized Kalman Filter Tracking for the CMS Experiment," Comput. Softw. Big Sci. **3**, 11 (2019).  
25. V. L. Highland, "Some Practical Remarks on Multiple Scattering," Nucl. Instrum. Methods **129**, 497 (1975).  
26. Particle Data Group, P. A. Zyla *et al.*, "Review of Particle Physics," Prog. Theor. Exp. Phys. **2024**, 083C01 (2024).  
27. S. Caillou *et al.*, "Novel Fully-Heterogeneous GNN Designs for Track Reconstruction at the HL-LHC," EPJ Web Conf. **295**, 09028 (2024), Available at: [https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09028/epjconf_chep2024_09028.html](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09028/epjconf_chep2024_09028.html).  
28. G. Cerati *et al.*, "Parallelized and Vectorized Tracking Using Kalman Filters with CMS Detector Geometry and Events," EPJ Web Conf. **214**, 02002 (2019), Available at: [https://www.epj-conferences.org/articles/epjconf/abs/2019/19/epjconf_chep2018_02002/epjconf_chep2018_02002.html](https://www.epj-conferences.org/articles/epjconf/abs/2019/19/epjconf_chep2018_02002/epjconf_chep2018_02002.html).  
29. M. Dorigo, V. Maniezzo and A. Colorni, "Ant System: Optimization by a Colony of Cooperating Agents," IEEE Trans. Syst. Man Cybern. B **26**, 29 (1996).  
30. J. Kennedy, "Particle Swarm Optimization," in *Encyclopedia of Machine Learning*, Springer (2011), Available at: [https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_630](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_630).  
31. D. J. Webb, W. M. Alobaidi and E. Sandgren, "Maze Navigation via Genetic Optimization," Intell. Inf. Manag. **10**, 215 (2017), Available at: [https://www.scirp.org/reference/referencespapers?referenceid=2173538](https://www.scirp.org/reference/referencespapers?referenceid=2173538).  
32. H. M. Gray, "Quantum Pattern Recognition Algorithms for Charged Particle Tracking," Philos. Trans. R. Soc. A **380**, 20210103 (2021), Available at: [https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0103](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0103).  

_Feel free to adapt parameters (`noise_std`, `num_branches`, gating thresholds) in `config.json` to your dataset and detector geometry._
