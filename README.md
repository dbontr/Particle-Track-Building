# Helical Extended Kalman Filter Track Reconstruction for TrackML

This repository implements a suite of meta-heuristic branchers built atop a helical Extended Kalman Filter (EKF) for charged-particle track reconstruction in a uniform magnetic field. Starting from three-hit seeds, the framework propagates state and covariance through detector layers and explores candidate assignments with algorithms ranging from the baseline EKF to A*, Ant Colony Optimization, Genetic Algorithms, Particle Swarm Optimization, Simulated Annealing, and layer-wise Hungarian matching.

## Brancher Algorithms

### Extended Kalman Filter (EKF)

Baseline combinatorial tracking where each branch follows the EKF predict–update cycle:

$$
\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1}), \quad P_{k|k-1} = F P_{k-1} F^T + Q,
$$
$$
K = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}, \quad \mathbf{x}_k = \mathbf{x}_{k|k-1}  K(\mathbf{z}_k - H\mathbf{x}_{k|k-1}).
$$

### A\* Search (A\*)

Best-first expansion uses a cost function $f(n) = g(n) + h(n)$ where $g$ is accumulated $\chi^2$ and $h$ is a heuristic estimate of remaining cost. Cross-track residuals are damped via

$$
C_{\perp} = \lambda_{\perp}\frac{\big\Vert (I - \mathbf{t}\mathbf{t}^\top)(\mathbf{z} - \hat{\mathbf{x}}) \big\Vert^2}{\sigma^2}.
$$

Algorithmic guidance follows the classic formulation of Hart, Nilsson & Raphael \[14\].

### Ant Colony Optimization (ACO)

Ants probabilistically construct paths using pheromones $\tau$ and heuristic desirability $\eta$. After evaporation, pheromones are updated by

$$
\tau \leftarrow (1 - \rho)\,\tau + \Delta\tau,
$$

with selection probabilities $p_i \propto \tau_i^{\alpha} \eta_i^{\beta}$ \[13\].

### Genetic Algorithm (GA)

Individuals encode layer-wise hit indices. Selection uses tournaments, one-point crossover recombines parents, and genes mutate with probability $\mu$ to explore new candidates \[15\]. Fitness is the EKF track score.

### Particle Swarm Optimization (PSO)

Swarm particles carry sparse velocities that bias sampling from each layer's gated hits. Velocities follow

$$
v \leftarrow w v + c_1 r_1 (p_{\mathrm{best}} - x) + c_2 r_2 (g_{\mathrm{best}} - x), \qquad x \leftarrow x + v
$$

to balance exploration and exploitation \[16\].

### Simulated Annealing (SA)

Local mutations to a current path are accepted with probability

$$
P_{\mathrm{accept}} = \min\{1, \exp(-\Delta/T)\},
$$

while the temperature cools as $T \leftarrow \alpha T$ \[17\].

### Hungarian Assignment

For each layer, a cost matrix built from $\chi^2$ values is solved for a one-to-one assignment:

$$
\min_{x_{ij}} \sum_{i,j} C_{ij} x_{ij} \quad \text{s.t.}\; \sum_j x_{ij}=1,\; \sum_i x_{ij}=1,
$$

enforcing collision-free hit usage within the layer \[18\].

## Supporting Modules

* `track_builder.py` orchestrates seed extraction, brancher execution, hit assignment, and pruning of track hypotheses.
* `parallel_track_builder.py` runs branchers in parallel threads with shared hit-claiming to avoid conflicts.
* `metrics.py` provides evaluation utilities such as mean squared error and hit-recovery percentage.
* `utils.py` supplies submission helpers and baseline generators like random solutions and hit shuffling.

## Getting Started

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Download Dataset**:

     You will also need the TrackML Particle Identification data, available from the [Kaggle competition page](https://www.kaggle.com/competitions/trackml-particle-identification/rules).

3. **Run the Main**:

     ```bash
     python main.py --file train_1.zip --pt 2.0
     ```

4. **Visualize**: Set ```PLOT=True``` to generate 2D/3D plots of hits, seeds, and branching tree.

## Code Structure

## References

1. Yeo _et al._, “Analytic Helix Propagation for Kalman Filters,” JINST **15** (2024).

2. Battisti _et al._, “Kalman-Filter-Based Tracking in High-Energy Physics,” Comput. Phys. Commun. **324**, 108601 (2024).

3. Strandlie & Frühwirth, “Track Fitting with Combinatorial Kalman Filter,” Nucl. Instrum. Methods A **559**, 305–308 (2007).

4. Schöning, “Advanced Seeding Techniques for Track Reconstruction,” CERN-THESIS-2021-042 (2021).

5. Jackson, J. D., “Classical Electrodynamics,” 3rd ed., Wiley (1998).

6. Amrouche _et al._, “The Tracking Machine Learning challenge: Accuracy phase,” NeurIPS (2018) / arXiv:1904.06778 (2019).

7. Friedman, Bentley & Finkel, “An Algorithm for Finding Best Matches in Logarithmic Expected Time,” ACM Trans. Math. Softw. **3**, 209–226 (1977).

8. SciPy ```cKDTree``` documentation, SciPy v1.10.1, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)

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

_Feel free to adapt parameters (```noise_std```, ```num_branches```, gating thresholds) in ```config.json``` to your dataset and detector geometry._
