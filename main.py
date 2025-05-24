import time
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from trackml.dataset import load_dataset
from kalman import CombinatorialKalmanFilter


# === Constants & Configuration ===
DATASET_PATH: str = 'train_1.zip'
N_EVENTS: int = 1
PARTS: List[str] = ['hits', 'cells', 'truth', 'particles']
B_FIELD: float = 2.0  # Tesla, magnetic field along z-axis
ENERGY_THRESHOLD_PT: float = 2.0  # GeV, minimum transverse momentum


def load_event(
    path: str,
    nevents: int,
    parts: List[str]
) -> Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a single event from TrackML dataset.

    Parameters
    ----------
    path : str
        Path to the dataset .zip file.
    nevents : int
        Number of events to load.
    parts : List[str]
        List of data parts to retrieve (e.g., ['hits', 'truth', ...]).

    Returns
    -------
    event_id : int
        Identifier of the loaded event.
    hits : pd.DataFrame
        Detector hit coordinates and metadata.
    cells : pd.DataFrame
        Detector cell information.
    truth : pd.DataFrame
        Ground truth associations between hits and particles.
    particles : pd.DataFrame
        Particle-level information (momentum, charge, etc.).
    """
    return next(load_dataset(path, nevents=nevents, parts=parts))


def preprocess_data(
    hits: pd.DataFrame,
    truth: pd.DataFrame,
    particles: pd.DataFrame,
    pt_threshold: float = ENERGY_THRESHOLD_PT
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean and filter raw TrackML data.

    Steps:
    1. Cast IDs to consistent integer type.
    2. Filter out low-energy particles (pt < threshold).
    3. Merge hits with truth for valid tracks.
    4. Compute cylindrical radius for hits.

    Parameters
    ----------
    hits : pd.DataFrame
        Raw hit-level data.
    truth : pd.DataFrame
        Raw truth associations.
    particles : pd.DataFrame
        Raw particle-level data.
    pt_threshold : float, optional
        Minimum transverse momentum (GeV) to keep a particle.

    Returns
    -------
    hits : pd.DataFrame
        Filtered hit data with computed radius 'r'.
    truth_hits : pd.DataFrame
        Merged truth and hit data for selected particles.
    high_energy_particles : pd.DataFrame
        Particle DataFrame after energy filtering.
    """
    # Ensure consistent ID types
    truth['particle_id'] = truth['particle_id'].astype(np.int64)
    particles['particle_id'] = particles['particle_id'].astype(np.int64)

    # Filter by transverse momentum
    particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)
    high_energy_particles = particles[particles['pt'] >= pt_threshold]

    # Filter truth to high-energy particles
    truth = truth[truth['particle_id'].isin(high_energy_particles['particle_id'])]

    # Compute radial coordinate
    hits['r'] = np.sqrt(hits['x']**2 + hits['y']**2)

    # Merge truth and hits
    truth_hits = truth.merge(hits, on='hit_id', how='inner')
    truth_hits = truth_hits.sort_values(['particle_id', 'r'])

    return hits, truth_hits, high_energy_particles


def get_layer_mappings(hits: pd.DataFrame) -> Tuple[List[Tuple[int, int]], dict]:
    """
    Generate sorted unique layer tuples and numeric labels.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit DataFrame containing 'volume_id' and 'layer_id'.

    Returns
    -------
    layer_tuples : List[Tuple[int, int]]
        Sorted unique (volume_id, layer_id) pairs.
    layer_to_label : dict
        Mapping from (volume_id, layer_id) to integer label.
    """
    layer_tuples = sorted(
        set(zip(hits['volume_id'], hits['layer_id'])),
        key=lambda x: (x[0], x[1])
    )
    layer_to_label = {layer: idx for idx, layer in enumerate(layer_tuples)}
    return layer_tuples, layer_to_label


# Setup data and mappings
event_id, hits, cells, truth, particles = load_event(
    DATASET_PATH, N_EVENTS, PARTS
)
hits, truth_hits, high_energy_particles = preprocess_data(
    hits, truth, particles
)
LAYER_TUPLES, LAYER_TO_LABEL = get_layer_mappings(hits)
HIT_LABELS = np.array([
    LAYER_TO_LABEL[(v, l)] for v, l in zip(hits['volume_id'], hits['layer_id'])
])


# === Plotting Utilities ===

def plot_hits_colored_by_layer(hits_df: pd.DataFrame) -> None:
    """
    Generate 2D and 3D scatter plots of detector hits colored by layer.

    Parameters
    ----------
    hits_df : pd.DataFrame
        Hit DataFrame containing 'x','y','z','r', 'volume_id', 'layer_id'.

    Returns
    -------
    None
    """
    # 2D scatter (z vs r)
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        hits_df['z'], hits_df['r'], c=HIT_LABELS,
        cmap='hsv', s=10, alpha=0.8
    )
    plt.xlabel('z (mm)')
    plt.ylabel('r (mm)')
    plt.title('Hits by layer (z vs r)')
    cbar = plt.colorbar(sc, ticks=range(len(LAYER_TUPLES)))
    cbar.set_ticklabels([f"{v}_{l}" for v, l in LAYER_TUPLES])
    cbar.set_label('volume_layer')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3D scatter
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc3 = ax.scatter(
        hits_df['x'], hits_df['y'], hits_df['z'],
        c=HIT_LABELS, cmap='hsv', s=10, alpha=0.8
    )
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D Hits by layer')
    cbar3 = fig.colorbar(sc3, ax=ax, pad=0.1, fraction=0.03)
    cbar3.set_ticks(list(range(len(LAYER_TUPLES))))
    cbar3.set_ticklabels([f"{v}_{l}" for v, l in LAYER_TUPLES])
    plt.tight_layout()
    plt.show()


def plot_layer_boundaries(hits_df: pd.DataFrame) -> None:
    """
    Draw rectangular boundaries of each detector layer in (z, r) space.

    Parameters
    ----------
    hits_df : pd.DataFrame
        Hit DataFrame with 'z', 'r', 'volume_id', 'layer_id'.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    all_z, all_r = [], []

    for vol_id, lay_id in LAYER_TUPLES:
        layer_hits = hits_df[
            (hits_df['volume_id'] == vol_id) &
            (hits_df['layer_id'] == lay_id)
        ]
        if layer_hits.empty:
            continue

        z_vals, r_vals = layer_hits['z'], layer_hits['r']
        zmin, zmax = z_vals.min(), z_vals.max()
        rmin, rmax = r_vals.min(), r_vals.max()
        all_z += [zmin, zmax]
        all_r += [rmin, rmax]

        rect = patches.Rectangle(
            (zmin, rmin), zmax - zmin, rmax - rmin,
            linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5
        )
        ax.add_patch(rect)
        ax.text(
            (zmin + zmax) / 2,
            (rmin + rmax) / 2,
            f"{vol_id}_{lay_id}",
            fontsize=6, ha='center', va='center'
        )

    ax.set_xlim(min(all_z) - 100, max(all_z) + 100)
    ax.set_ylim(min(all_r) - 50, max(all_r) + 50)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('r (mm)')
    ax.set_title('Layer Boundaries in r vs z')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_truth_paths_rz(
    truth_hits_df: pd.DataFrame,
    max_tracks: Optional[int] = None
) -> None:
    """
    Plot 2D R-Z trajectories for ground truth particle tracks.

    Parameters
    ----------
    truth_hits_df : pd.DataFrame
        Merged truth/hit DataFrame sorted by 'particle_id' and 'r'.
    max_tracks : Optional[int]
        Maximum number of tracks to plot (None for all).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (_, group) in enumerate(truth_hits_df.groupby('particle_id')):
        if max_tracks and i >= max_tracks:
            break
        r = np.sqrt(group['x']**2 + group['y']**2)
        ax.plot(group['z'], r, alpha=0.7)

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('r (mm)')
    ax.set_title('2D R-Z Truth Particle Paths')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_truth_paths_3d(
    truth_hits_df: pd.DataFrame,
    max_tracks: Optional[int] = None
) -> None:
    """
    Plot 3D trajectories for ground truth particle tracks.

    Parameters
    ----------
    truth_hits_df : pd.DataFrame
        Merged truth/hit DataFrame.
    max_tracks : Optional[int]
        Maximum number of tracks to plot (None for all).

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for i, (_, group) in enumerate(truth_hits_df.groupby('particle_id')):
        if max_tracks and i >= max_tracks:
            break
        ax.plot(group['x'], group['y'], group['z'], alpha=0.7)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D Truth Particle Paths')
    plt.tight_layout()
    plt.show()


def plot_seed_paths_rz(
    seeds_df: pd.DataFrame,
    max_seeds: Optional[int] = None
) -> None:
    """
    Plot 2D R–Z trajectories for seed hits.

    Parameters
    ----------
    seeds_df : pd.DataFrame
        DataFrame of seed hits with columns ['x','y','z','r','hit_id','particle_id','charge'].
    max_seeds : Optional[int]
        Maximum number of seeds to plot (None for all).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    grouped = seeds_df.groupby('particle_id')
    for i, (_, group) in enumerate(grouped):
        if max_seeds and i >= max_seeds:
            break
        # ensure 'r' exists or compute it:
        if 'r' not in group.columns:
            group = group.copy()
            group['r'] = np.sqrt(group['x']**2 + group['y']**2)
        ax.plot(group['z'], group['r'], marker='o', linestyle='-', alpha=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('r (mm)')
    ax.set_title('2D R–Z Seed Trajectories')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_seed_paths_3d(
    seeds_df: pd.DataFrame,
    max_seeds: Optional[int] = None
) -> None:
    """
    Plot 3D X–Y–Z trajectories for seed hits.

    Parameters
    ----------
    seeds_df : pd.DataFrame
        DataFrame of seed hits with columns ['x','y','z','hit_id','particle_id','charge'].
    max_seeds : Optional[int]
        Maximum number of seeds to plot (None for all).
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    grouped = seeds_df.groupby('particle_id')
    for i, (_, group) in enumerate(grouped):
        if max_seeds and i >= max_seeds:
            break
        ax.plot(
            group['x'], group['y'], group['z'],
            marker='o', linestyle='-', alpha=0.8
        )
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D XYZ Seed Trajectories')
    plt.tight_layout()
    plt.show()


def extract_seed_hits(
    truth_hits_df: pd.DataFrame,
    high_energy_particles: pd.DataFrame,
    n_seeds: int = 100,
    seed_length: int = 3,
    start_layer: Tuple[int,int] = (8, 2)
) -> Optional[pd.DataFrame]:
    """
    Extract fixed-length seeds starting from the innermost hit in startup layer (8,2),
    going outwards in radius, for BOTH positive and negative z.

    Parameters
    ----------
    truth_hits_df : pd.DataFrame
        DataFrame of hits with ['x','y','z','r','volume_id','layer_id','particle_id'].
    high_energy_particles : pd.DataFrame
        Particle DataFrame with ['particle_id','q', ...].
    n_seeds : int
        Max number of seeds to extract.
    seed_length : int
        Number of hits per seed (including the start-layer hit).
    start_layer : tuple
        (volume_id, layer_id) identifying the seeding layer.

    Returns
    -------
    pd.DataFrame or None
        Concatenated seed hits ['x','y','z','r','hit_id','particle_id','charge'].
    """
    seeds = []
    vol0, lay0 = start_layer

    # Precompute radius if missing
    if 'r' not in truth_hits_df.columns:
        truth_hits_df = truth_hits_df.copy()
        truth_hits_df['r'] = np.sqrt(truth_hits_df.x**2 + truth_hits_df.y**2)

    # Group by particle
    for pid, group in truth_hits_df.groupby('particle_id'):
        if len(seeds) >= n_seeds:
            break

        # 1) find all hits in the start layer
        layer_hits = group[
            (group.volume_id == vol0) & (group.layer_id == lay0)
        ]
        if layer_hits.empty:
            continue

        # 2) pick the innermost one (smallest r)
        first_hit = layer_hits.loc[layer_hits.r.idxmin()]

        # 3) now find the next (seed_length-1) hits outwards by increasing r
        outward_hits = group[group.r > first_hit.r].sort_values('r')
        if len(outward_hits) < (seed_length - 1):
            continue

        seed_hits = pd.concat([
            first_hit.to_frame().T,
            outward_hits.iloc[: seed_length - 1]
        ], ignore_index=True)

        # attach charge
        charge = high_energy_particles.loc[
            high_energy_particles.particle_id == pid, 'q'
        ].iloc[0]
        seed_hits = seed_hits[['x','y','z','r','hit_id','particle_id']].copy()
        seed_hits['charge'] = charge

        seeds.append(seed_hits)

    if not seeds:
        return None
    return pd.concat(seeds, ignore_index=True)


def plot_ckf_branches_xy(seed, candidates, branches, best_idx, true_xy, pid):
    """
    Plot all CKF branches in XY + seed + true track.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(candidates[:,0], candidates[:,1], c='lightgray', s=5, label='all hits')
    ax.plot(seed[:,0], seed[:,1], 'bo-', linewidth=2, markersize=6, label='seed')
    for i, br in enumerate(branches):
        pts = np.array(br)
        if i == best_idx:
            ax.plot(pts[:,0], pts[:,1], 'r-', linewidth=2.5, label='best branch')
        else:
            ax.plot(pts[:,0], pts[:,1], 'r-', alpha=0.3, linewidth=1)
    ax.plot(true_xy[:,0], true_xy[:,1], 'k--', linewidth=1.5, label='true XY')
    ax.set_title(f'PID {pid}: CKF branches')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.axis('equal')
    ax.legend(loc='upper left')
    plt.show()

def debug_gate_evolution(
    seed, candidates, branch, ckf, true_xy, pid
):
    kf = ckf.KalmanFilter(ckf.Q, ckf.R)
    state = np.array([seed[2,0], seed[2,1],
                      seed[2,0]-seed[1,0],
                      seed[2,1]-seed[1,1]])
    cov = np.eye(4)
    tau = ckf.threshold  # Mahalanobis cutoff
    max_ang = np.radians(ckf.max_angle)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(candidates[:,0], candidates[:,1],
               c='lightgray', s=5, label='all hits')
    ax.plot(true_xy[:,0], true_xy[:,1], 'k--', label='true XY')
    ax.plot(seed[:,0], seed[:,1], 'bo-', label='seed')

    for depth, pt in enumerate(branch[3:], start=1):
        # predict
        x_pred, P_pred = kf.predict(state, cov)
        S = P_pred[:2,:2] + ckf.R

        # 1) plot Mahalanobis gate ellipse
        vals, vecs = np.linalg.eigh(S)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        width, height = 2*np.sqrt(vals)*tau
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        ell = patches.Ellipse(
            (x_pred[0], x_pred[1]), width, height, angle=angle,
            edgecolor='orange', facecolor='none', linestyle='--',
            linewidth=1, label='Mahal gate' if depth==1 else None
        )
        ax.add_patch(ell)

        # 2) highlight the hits that lie inside the ellipse
        diff = candidates - x_pred[:2]
        m_dists = np.sum(diff @ np.linalg.inv(S) * diff, axis=1)
        maha_pts = candidates[m_dists <= tau**2]
        ax.scatter(maha_pts[:,0], maha_pts[:,1],
                   c='green', s=30,
                   label='in Mahal' if depth==1 else None)

        # 3) plot angular cone around velocity vector
        v = state[2:]
        norm = np.linalg.norm(v)
        if norm>1e-6:
            v_u = v/norm
            for sign in (+1,-1):
                rot = transforms.Affine2D().rotate_deg_around(
                    x_pred[0], x_pred[1], sign*ckf.max_angle
                ) + ax.transData
                ax.plot(
                    [x_pred[0], x_pred[0]+v_u[0]*width/2],
                    [x_pred[1], x_pred[1]+v_u[1]*height/2],
                    '-', transform=rot,
                    color='blue', linewidth=1,
                    label='angular cone' if depth==1 else None
                )

        # 4) mark the chosen point
        ax.scatter(pt[0], pt[1], c='red', s=70, marker='x',
                   label='picked pt' if depth==1 else None)

        # update state
        state, cov = kf.update(x_pred, P_pred, np.array(pt))

    ax.set_title(f'PID {pid}: Mahalanobis & angular gating')
    ax.axis('equal')
    ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_no_branch(
    pid: float,
    seed: np.ndarray,
    full_candidates: np.ndarray,
    true_xy: np.ndarray,
    ckf: CombinatorialKalmanFilter
):
    """
    Plot seed, true track, and initial gate when no CKF branches were formed.
    """
    # 1) initial gate prediction
    kf0 = ckf.KalmanFilter(ckf.Q, ckf.R)
    init_dir   = seed[2] - seed[1]
    init_state = np.array([seed[2,0], seed[2,1], init_dir[0], init_dir[1]])
    init_cov   = np.eye(4)
    x_pred, _  = kf0.predict(init_state, init_cov)
    wx, wy     = ckf.spatial_window

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(full_candidates[:,0], full_candidates[:,1],
               c='lightgray', s=5, label='all hits')
    ax.plot(seed[:,0], seed[:,1], 'bo-', linewidth=2, markersize=6, label='seed')
    ax.plot(true_xy[:,0], true_xy[:,1], 'k--', linewidth=1.5, label='true XY')

    # draw the initial gate
    rect = patches.Rectangle(
        (x_pred[0]-wx, x_pred[1]-wy),
        2*wx, 2*wy,
        linewidth=1, edgecolor='orange',
        facecolor='none', linestyle='--',
        label='initial gate'
    )
    ax.add_patch(rect)

    ax.set_title(f'PID {pid}: NO CKF branches — see initial gate')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.axis('equal')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    event_id, hits, cells, truth, particles = load_event(DATASET_PATH, N_EVENTS, PARTS)
    hits, truth_hits, high_energy_particles = preprocess_data(hits, truth, particles)
    plot_hits_colored_by_layer(hits)
    plot_layer_boundaries(hits)
    plot_truth_paths_3d(truth_hits)
    plot_truth_paths_rz(truth_hits)

    # Extract seeds
    seeds_df = extract_seed_hits(
        truth_hits,
        high_energy_particles,
        n_seeds=50,
        seed_length=3,
        start_layer=(8, 2)
    )
    if seeds_df is None or seeds_df.empty:
        raise RuntimeError("No seeds extracted!")
    print(f"Extracted {len(seeds_df.groupby('particle_id'))} seeds.")

    ckf = CombinatorialKalmanFilter(
        Q=None, R=None,
        spatial_window=(10.0, 10.0),
        threshold=100,
        max_branches=100,
        max_depth=22,
        max_angle=50.0
    )
    ckf.Q *= 10.0

    full_candidates = hits[['x','y']].values

    for pid, seed_group in seeds_df.groupby('particle_id'):
        seed    = seed_group[['x','y']].values   # shape (3,2)
        true_xy = truth_hits[truth_hits.particle_id == pid][['x','y']].values

        # run CKF on the full pool
        branches = ckf.fit(seed, full_candidates, true_track=true_xy)
        if not branches:
            # instead of skipping, show us the seed/true/initial-gate
            plot_no_branch(pid, seed, full_candidates, true_xy, ckf)
            continue

        best = branches[ckf.best_idx or 0]

        # Re‐simulate the KF along the best branch to extract per‐depth gates:
        kf = ckf.KalmanFilter(ckf.Q, ckf.R)
        state = np.array([seed[2,0], seed[2,1],
                          seed[2,0]-seed[1,0],
                          seed[2,1]-seed[1,1]])
        cov   = np.eye(4)
        wx, wy = ckf.spatial_window

        fig, ax = plt.subplots(figsize=(6,6))
        # 1) Show all hits in light gray
        ax.scatter(full_candidates[:,0], full_candidates[:,1],
                   c='lightgray', s=5, label='all hits')
        # 2) Show your 3‐point seed
        ax.plot(seed[:,0], seed[:,1], 'bo-', label='seed')

        # Step through each chosen update in the best branch
        for depth, pt in enumerate(best[3:], start=1):
            # Predict next
            state_pred, cov_pred = kf.predict(state, cov)

            # 3) Draw the spatial window
            rect = patches.Rectangle(
                (state_pred[0]-wx, state_pred[1]-wy),
                2*wx, 2*wy,
                linewidth=1, edgecolor='orange',
                facecolor='none', linestyle='--',
                label='gate window' if depth==1 else None
            )
            ax.add_patch(rect)

            # 4) Highlight exactly those hits inside that window
            dx = np.abs(full_candidates[:,0] - state_pred[0])
            dy = np.abs(full_candidates[:,1] - state_pred[1])
            in_gate = full_candidates[(dx<=wx)&(dy<=wy)]
            ax.scatter(in_gate[:,0], in_gate[:,1],
                       c='orange', s=20,
                       label='in gate' if depth==1 else None)

            # 5) Mark the one the filter picked
            ax.scatter(pt[0], pt[1],
                       c='red', s=70, marker='x',
                       label='picked pt' if depth==1 else None)

            # Update for the next depth
            state, cov = kf.update(state_pred, cov_pred, np.array(pt))

        ax.set_title(f"PID {pid}: spatial‐window gating & picks")
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        ax.axis('equal')
        ax.legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.show()

        debug_gate_evolution(
            seed=seed,
            candidates=full_candidates,
            branch=best,
            ckf=ckf,
            true_xy=true_xy,
            pid=pid
        )