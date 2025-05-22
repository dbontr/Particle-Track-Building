# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from trackml.dataset import load_dataset
from filterpy.kalman import KalmanFilter
import time

# === Constants & Configuration ===
DATASET_PATH = 'train_1.zip'
N_EVENTS = 1
PARTS = ['hits', 'cells', 'truth', 'particles']
B_FIELD = 2.0  # Tesla, magnetic field along z-axis

# === Load Dataset ===
event_id, hits, cells, truth, particles = next(load_dataset(DATASET_PATH, nevents=N_EVENTS, parts=PARTS))

# Convert particle_id in truth and particles to consistent type
truth['particle_id'] = truth['particle_id'].astype(np.int64)
particles['particle_id'] = particles['particle_id'].astype(np.int64)

# Remove low energy particles (momentum < 2 GeV)
particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)
high_energy_particles = particles[particles['pt'] >= 2.0]

# Filter truth table by valid particle_ids
truth = truth[truth['particle_id'].isin(high_energy_particles['particle_id'])]

# Precompute cylindrical radius r for hits
hits['r'] = np.sqrt(hits['x']**2 + hits['y']**2)

# Merge hits and truth to get true tracks
truth_hits = truth.merge(hits, on='hit_id', how='inner')
truth_hits = truth_hits.sort_values(['particle_id', 'r'])

# Identify unique (volume_id, layer_id) tuples sorted numerically
LAYER_TUPLES = sorted(
    set(zip(hits['volume_id'], hits['layer_id'])),
    key=lambda x: (x[0], x[1])
)

# Map each layer to a numeric label for coloring
LAYER_TO_LABEL = {layer: idx for idx, layer in enumerate(LAYER_TUPLES)}

# Prepare labels array for hits
HIT_LABELS = np.array([LAYER_TO_LABEL[(v, l)] for v, l in zip(hits['volume_id'], hits['layer_id'])])

# === Plotting Functions ===
def plot_hits_colored_by_layer(hits_df):
    """2D and 3D scatter plots of hits colored by unique layer."""
    # 2D scatter (z vs r)
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(hits_df['z'], hits_df['r'], c=HIT_LABELS, cmap='hsv', s=10, alpha=0.8)
    plt.xlabel('z (mm)'); plt.ylabel('r (mm)')
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
    sc3 = ax.scatter(hits_df['x'], hits_df['y'], hits_df['z'], c=HIT_LABELS, cmap='hsv', s=10, alpha=0.8)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
    ax.set_title('3D Hits by layer')
    cbar3 = fig.colorbar(sc3, ax=ax, pad=0.1, fraction=0.03)
    tick_locs = list(range(len(LAYER_TUPLES)))
    tick_labels = [f"{v}_{l}" for v, l in LAYER_TUPLES]
    cbar3.set_ticks(tick_locs)
    cbar3.set_ticklabels(tick_labels)
    plt.show()

def plot_layer_boundaries(hits_df):
    """Draw layer boundaries as rectangles in (z, r) space."""
    fig, ax = plt.subplots(figsize=(14, 10))
    all_z, all_r = [], []

    for vol_id, lay_id in LAYER_TUPLES:
        layer_hits = hits_df[(hits_df['volume_id'] == vol_id) & (hits_df['layer_id'] == lay_id)]
        if layer_hits.empty:
            continue
        r_vals = layer_hits['r']; z_vals = layer_hits['z']
        zmin, zmax = z_vals.min(), z_vals.max()
        rmin, rmax = r_vals.min(), r_vals.max()
        all_z += [zmin, zmax]; all_r += [rmin, rmax]

        rect = patches.Rectangle((zmin, rmin), zmax - zmin, rmax - rmin,
                                 linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        ax.text((zmin + zmax)/2, (rmin + rmax)/2, f"{vol_id}_{lay_id}",
                fontsize=6, ha='center', va='center')

    ax.set_xlim(min(all_z) - 100, max(all_z) + 100)
    ax.set_ylim(min(all_r) - 50, max(all_r) + 50)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('r (mm)')
    ax.set_title('Layer Boundaries in r vs z')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# === Plotting Functions ===

def plot_truth_paths_rz(truth_hits_df, max_tracks=None):
    """Plot 2D rz trajectories of truth particle paths."""
    fig, ax = plt.subplots(figsize=(12, 8))
    grouped = truth_hits_df.groupby('particle_id')
    for i, (pid, group) in enumerate(grouped):
        if max_tracks and i >= max_tracks:
            break
        r = np.sqrt(group['x']**2 + group['y']**2)
        ax.plot(group['z'], r, alpha=0.7)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('r (mm)')
    ax.set_title('2D R-Z Truth Particle Paths')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_truth_paths_3d(truth_hits_df, max_tracks=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    grouped = truth_hits_df.groupby('particle_id')
    for i, (pid, group) in enumerate(grouped):
        if max_tracks and i >= max_tracks:
            break
        ax.plot(group['x'], group['y'], group['z'], alpha=0.7)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
    ax.set_title('3D Truth Particle Paths')
    plt.tight_layout()
    plt.show()

def plot_kalman_vs_truth_curves(kf_results, truth_hits_df):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for pid, kf_traj in kf_results:
        true_group = truth_hits_df[truth_hits_df['particle_id'] == pid].sort_values('z')
        if len(true_group) >= len(kf_traj):
            ax.plot(true_group['x'].values[:len(kf_traj)],
                    true_group['y'].values[:len(kf_traj)],
                    true_group['z'].values[:len(kf_traj)],
                    color='gray', linestyle='--', alpha=0.5)
        ax.plot(kf_traj[:,0], kf_traj[:,1], kf_traj[:,2], label=f'PID {pid}', alpha=0.8)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('Kalman Filter vs Truth Particle Curves')
    plt.tight_layout()
    plt.show()

def extract_seed_hits(truth_hits_df, n_seeds=100, seed_length=3):
    seeds = []
    grouped = truth_hits_df.groupby('particle_id')
    for i, (pid, group) in enumerate(grouped):
        if len(seeds) >= n_seeds:
            break
        group_sorted = group.sort_values('z').reset_index()
        if len(group_sorted) >= seed_length:
            seed_hits = group_sorted.loc[:seed_length-1, ['x','y','z','r','hit_id']]
            seed_hits['particle_id'] = pid
            charge = high_energy_particles[high_energy_particles['particle_id'] == pid]['q'].values[0]
            seed_hits['charge'] = charge
            seeds.append(seed_hits)
    if seeds:
        return pd.concat(seeds, ignore_index=True)
    else:
        return None

def run_kalman_filter_on_seeds(seed_df):
    results = []
    grouped = seed_df.groupby('particle_id')
    for pid, group in grouped:
        group = group.sort_values('z')
        charge = group['charge'].values[0]
        kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = 1.0
        omega = 0.3 * B_FIELD * charge
        kf.F = np.array([[1, 0, 0, dt, 0, 0],
                         [0, 1, 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, omega, 0],
                         [0, 0, 0, -omega, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.R *= 0.5
        kf.P *= 1000.0
        kf.Q *= 0.01
        x0, y0, z0 = group.iloc[0][['x', 'y', 'z']]
        kf.x = np.array([x0, y0, z0, 0, 0, 0]).reshape(6, 1)

        trajectory = []
        for _, row in group.iterrows():
            z = row[['x', 'y', 'z']].values.reshape(3, 1)
            kf.predict()
            kf.update(z)
            trajectory.append(kf.x[:3].flatten())

        for _ in range(10):  # Extend beyond seed length
            kf.predict()
            trajectory.append(kf.x[:3].flatten())

        results.append((pid, np.array(trajectory)))
    return results

def evaluate_accuracy(kf_results, truth_hits_df):
    errors = []
    for pid, traj in kf_results:
        true_hits = truth_hits_df[truth_hits_df['particle_id'] == pid].sort_values('z')
        truth_pos = true_hits[['x', 'y', 'z']].values[:len(traj)]
        if len(truth_pos) < len(traj):
            continue
        mse = np.mean((traj - truth_pos)**2)
        errors.append(mse)
    return np.mean(errors) if errors else None


# === Execute ===
plot_hits_colored_by_layer(hits)
plot_layer_boundaries(hits)
plot_truth_paths_3d(truth_hits)
plot_truth_paths_rz(truth_hits)
seed_df = extract_seed_hits(truth_hits, n_seeds=50, seed_length=3)
print("Extracted seed hits shape:", seed_df.shape if seed_df is not None else None)

if seed_df is not None:
    start = time.time()
    kf_results = run_kalman_filter_on_seeds(seed_df)
    end = time.time()

    print(f"Kalman Filter runtime: {end - start:.3f} seconds")

    # Plot Kalman vs Truth curved paths
    plot_kalman_vs_truth_curves(kf_results, truth_hits)

    # Accuracy evaluation
    mse = evaluate_accuracy(kf_results, truth_hits)
    print(f"Mean squared error of Kalman trajectories: {mse:.4f}")

    # Throughput
    print(f"Processed {len(kf_results)} tracks in {end - start:.3f} seconds (~{len(kf_results)/(end-start):.2f} tracks/sec)")
