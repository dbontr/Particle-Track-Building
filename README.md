**Helical Extended Kalman Filter Track Reconstruction for TrackML**

This repository implements a multi-hypothesis Extended Kalman Filter (EKF) for charged-particle track reconstruction in a uniform magnetic field, inspired by the TrackML challenge at the LHC. It seeds tracks from 3 hits, propagates a helical state & covariance through detector layers, and explores multiple track candidates using combinatorial branching and gating.

Features
--------

*   **Helix Motion Model**: Propagate particle state $[x, y, z, v_x, v_y, v_z, \kappa]$ by solving the 3D helix equations in a constant $B_z$ field (Yeo _et al._, 2024) \[1\].
    
*   **Analytic Jacobians**: Compute the state transition Jacobian for EKF propagation, including small-angle expansions for numerical stability (Yeo _et al._, 2024) \[1\].
    
*   **Extended Kalman Filter**: Standard EKF predict/update cycle to incorporate hit measurements with noise covariance (Battisti _et al._, 2024) \[2\]; measurement-model Jacobian discussed in Welch & Bishop (1995) \[9\].
    
*   **Seeding**: Initialize track parameters (velocity & curvature) from 3-point circle fits requiring ≥3 hits (Schöning, 2021) \[4\]; seed jittering simulates detector resolution (TrackML challenge perturbations) \[12\].
    
*   **KD-Tree Spatial Lookup**: Fast nearest-neighbor search per detector layer via ```scipy.spatial.cKDTree``` (Friedman _et al._, 1977) \[7\]; see SciPy docs \[8\].
    
*   **Combinatorial Branching (CKF)**: Create new hypotheses for each compatible hit, prune by $\chi^2$ gating, and keep top-scoring branches (Strandlie & Frühwirth, 2007) \[3\].
    
*   **Metrics & Scoring**: Compute mean squared error (MSE), hit recall, and overall event score against ground truth using TrackML scoring (Amrouche _et al._, 2019) \[6\].
    
*   **Baseline Strategies**: Random assignment, hit dropping, and shuffling baselines matching TrackML random solution scoring and challenge guidelines (TrackML challenge docs) \[3\]\[12\].
    

Theory & Implementation Details
-------------------------------

### 1\. Helical Trajectories in a Magnetic Field

Charged particles in a uniform magnetic field follow helices with

$$r = \frac{p_T}{qB}, \quad \kappa = \frac{1}{r} = \frac{qB}{p_T}.$$

Analytic propagation involves sine/cosine of $\omega \Delta t$, where $\omega = B_z \kappa p_T$. For small $\omega \Delta t$, the motion reduces to straight-line (Yeo _et al._, 2024) \[1\]; Jackson (1998) discusses classical helix motion \[5\].

### 2\. Extended Kalman Filtering (EKF)

The EKF is a recursive Bayesian estimator. In each layer:

1.  **Predict:** $\mathbf{x}_{\text{pred}} = f(\mathbf{x}_{\text{prev}}), \quad P_{\text{pred}} = F P_{\text{prev}} F^T + Q.$
    
2.  **Update:** $K = P_{\text{pred}} H^T (H P_{\text{pred}} H^T + R)^{-1}, \quad \mathbf{x}_{\text{new}} = \mathbf{x}_{\text{pred}} + K(\mathbf{z} - H\mathbf{x}_{\text{pred}}).$
    

*   **Process Noise Tuning**: Covariance $Q_0$ scaling accounts for model uncertainty; process noise is tuned per variable—for curvature, velocity, position—reflecting confidence levels (MathWorks tuning example) \[10\]; NLKF implementations in ACTS explore advanced process noise models \[7\].
    
*   **Measurement-Model Jacobian**: The function ```H_jac``` extracts $(x, y, z)$ from the state; projecting to detector plane via ```to_local_frame_jac``` yields the 2×7 Jacobian (Welch & Bishop, 1995) \[9\].
    

### 3\. Combinatorial Kalman Filter (CKF) & Gating

Compatible hits spawn new branches. $\chi^2$ gating:

$$\chi^2 = (\mathbf{z} - H\mathbf{x}_{\text{pred}})^T S^{-1} (\mathbf{z} - H\mathbf{x}_{\text{pred}}), \quad S = H P_{\text{pred}} H^T + R.$$


Branches with $\chi^2$ above thresholds incur penalties or are pruned; top-scoring branches retained per layer (Strandlie & Frühwirth, 2007) \[3\].

### 4\. Seeding with Three-Point Fits

Three hits define a circle; curvature:

$$
\kappa = \frac{2\| (\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_2 - \mathbf{p}_1) \|}{|d_1|\,|d_2|\,|d_{02}|},
$$


initializes $\kappa$. Jitter of seeds ($\sigma = 0.001$) simulates layer measurement noise (TrackML perturbations) \[12\]; see ACTS seeding notes for similar resolution modeling \[11\].

### 5\. KD-Tree Spatial Indexing

Use ```cKDTree``` for $O(\log N)$
 neighbor searches per layer; ```max_cands``` and ```step_candidates``` parameters control candidate selection (Friedman _et al._, 1977) \[7\]; SciPy ```cKDTree``` docs \[8\].

### 6\. Baseline Submission Strategies

Helper functions:

*   **Random Solution**: randomly assign track IDs; scores ~0 (TrackML random baseline) \[3\]\[12\].
    
*   **Drop Hits**: randomly drop true hits by fake IDs.
    
*   **Shuffle Hits**: randomly reassign hits to incorrect tracks.
    

These illustrate degenerative baselines and help calibrate scoring (TrackML challenge docs) \[3\]\[12\].

### 7\. Metrics & Scoring

*   **Mean Squared Error (MSE)**: avg. squared distance between predicted & true hits.
    
*   **Hit Recall (%)**: % of true hits within tolerance.
    
*   **Event Score**: TrackML weighted accuracy (Amrouche _et al._, 2019) \[6\].
    

Getting Started
---------------

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Download Dataset**:

    You will also need the TrackML Particle Identification data, available from the [Kaggle competition page](https://www.kaggle.com/competitions/trackml-particle-identification/rules).

    
3.  **Run the Main**:
    ```bash
    python main.py --file train_1.zip --pt 2.0
    ```
    
4.  **Visualize**: Set ```PLOT=True``` to generate 2D/3D plots of hits, seeds, and branching tree.
    

Code Structure
--------------

*   ```load_and_preprocess()```: Load TrackML event, filter by $p_T$, compute positions & weights.
    
*   ```build_layer_trees()```: Build KD-trees for fast hit lookup per (volume,layer).
    
*   ```HelixEKFBrancher```: Main class implementing CKF with EKF:
    
    *   ```compute_F()```, ```propagate()```: Helix propagation & Jacobian (Yeo _et al._, 2024) \[1\].
        
    *   ```run()```: Branching logic, gating, updates, and tree construction (Strandlie & Frühwirth, 2007) \[3\].
        
*   Baseline helpers: ```_make_submission```, ```random_solution```, ```drop_hits```, ```shuffle_hits```.
    
*   Utilities for metrics (```compute_metrics```, ```branch_mse```, ```branch_hit_stats```) and submission creation.
    

References
----------

1.  Yeo _et al._, “Analytic Helix Propagation for Kalman Filters,” JINST **15** (2024).
    
2.  Battisti _et al._, “Kalman-Filter-Based Tracking in High-Energy Physics,” Comput. Phys. Commun. **324**, 108601 (2024).
    
3.  Strandlie & Frühwirth, “Track Fitting with Combinatorial Kalman Filter,” Nucl. Instrum. Methods A **559**, 305–308 (2007).
    
4.  Schöning, “Advanced Seeding Techniques for Track Reconstruction,” CERN-THESIS-2021-042 (2021).
    
5.  Jackson, J. D., “Classical Electrodynamics,” 3rd ed., Wiley (1998).
    
6.  Amrouche _et al._, “The Tracking Machine Learning challenge: Accuracy phase,” NeurIPS (2018) / arXiv:1904.06778 (2019).
    
7.  Friedman, Bentley & Finkel, “An Algorithm for Finding Best Matches in Logarithmic Expected Time,” ACM Trans. Math. Softw. **3**, 209–226 (1977).
    
8.  SciPy ```cKDTree``` documentation, SciPy v1.10.1, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
    
9.  Welch & Bishop, “An Introduction to the Kalman Filter,” University of North Carolina at Chapel Hill (1995).
    
10.  MathWorks, “Tuning Kalman Filter to Improve State Estimation,” MATLAB & Simulink documentation (2024).
    
11.  ACTS Collaboration, “Track Seeding,” ACTS documentation v4.0.0, [https://acts.readthedocs.io/en/v4.0.0/core/seeding.html](https://acts.readthedocs.io/en/v4.0.0/core/seeding.html)
    
12.  Golling _et al._, “TrackML: a tracking Machine Learning challenge,” EPJ Web of Conferences **214**, 06037 (2019).
    

_Feel free to adapt parameters (```noise_std```, ```num_branches```, gating thresholds) to your dataset and detector geometry._