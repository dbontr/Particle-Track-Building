import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import *

def check_seed_and_plot(model: object, seed_points: List[np.ndarray]) -> None:
    """
    Estimates a seed state from 3 seed points, compares it with geometric expectations,
    and visualizes the position and velocity vs. true circle tangent in 2D.

    Parameters
    ----------
    model : object
        An object with an `estimate_seed` method that takes a list of 3D points and returns a tuple of initial state and covariance.
    seed_points : List[np.ndarray]
        A list of 3 seed points (each a 3-element array-like object representing x, y, z).

    Returns
    -------
    None
    """
    # generate seed
    x0, P0 = model.estimate_seed(seed_points)[0]

    # 1. Position check
    print("Seed position:", np.round(x0[:3],5))
    print("Last pt      :", np.round(seed_points[-1],5))

    # 2. Tangent check
    p1,p2,p3 = seed_points
    # compute true tangent
    A = np.array([[p2[0]-p1[0], p2[1]-p1[1]],
                  [p3[0]-p1[0], p3[1]-p1[1]]])
    B = np.array([[(p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2)/2],
                  [(p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2)/2]])
    cx, cy = (np.linalg.solve(A, B).flatten() + p1[:2])
    radial = p3[:2] - np.array([cx, cy])
    true_tan = np.array([-radial[1], radial[0]])
    true_tan /= np.linalg.norm(true_tan)
    if np.dot(true_tan, (p3[:2]-p2[:2])/np.linalg.norm(p3[:2]-p2[:2])) < 0:
        true_tan *= -1

    v_hat = x0[3:5]/np.linalg.norm(x0[3:5])
    angle = np.degrees(np.arccos(np.clip(v_hat.dot(true_tan), -1, 1)))
    print(f"Angle seed-v to true-tangent: {angle:.3f}°")

    # 3. Plot
    plt.figure(figsize=(4,4))
    plt.scatter(*zip(*[p[:2] for p in seed_points]), label="seed points")
    plt.scatter(cx, cy, marker='x', color='k', label="circle center")
    plt.arrow(p3[0], p3[1],
              true_tan[0]*0.1, true_tan[1]*0.1,
              head_width=0.003, color='r', label="true tangent")
    plt.arrow(p3[0], p3[1],
              v_hat[0]*0.1, v_hat[1]*0.1,
              head_width=0.003, color='b', label="seed velocity")
    plt.legend()
    plt.gca().set_aspect('equal', 'box')
    plt.show()


def plot_hits_colored_by_layer(hits_df: pd.DataFrame, layer_tuples: List[Tuple[int, int]]) -> None:
    """
    Plots a 2D z–r and 3D xyz scatter plot of hits, colored by (volume_id, layer_id).

    Parameters
    ----------
    hits_df : pd.DataFrame
        DataFrame with columns ['x', 'y', 'z', 'volume_id', 'layer_id'].
    layer_tuples : List[Tuple[int, int]]
        Ordered list of (volume_id, layer_id) combinations to assign color labels.

    Returns
    -------
    None
    """
    # 1) Build integer labels 0..N-1 for each hit
    labels = np.array([
        layer_tuples.index((vol, lay))
        for vol, lay in zip(hits_df.volume_id, hits_df.layer_id)
    ])

    # 2) Create a Normalize + Colormap once
    norm = Normalize(vmin=0, vmax=len(layer_tuples)-1)
    cmap = cm.get_cmap('hsv', len(layer_tuples))

    # 3) 2D z vs r plot
    hits_df['r'] = np.sqrt(hits_df.x**2 + hits_df.y**2)
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        hits_df['z'], hits_df['r'],
        c=labels, cmap=cmap, norm=norm,
        s=10, alpha=0.8
    )
    plt.xlabel('z (mm)')
    plt.ylabel('r (mm)')
    plt.title('Hits by layer (z vs r)')
    cbar = plt.colorbar(sc, ticks=range(len(layer_tuples)))
    cbar.set_ticklabels([f"{v}_{l}" for v, l in layer_tuples])
    cbar.set_label('volume_layer')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) 3D scatter with the same colors
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc3 = ax.scatter(
        hits_df['x'], hits_df['y'], hits_df['z'],
        c=labels, cmap=cmap, norm=norm,
        s=10, alpha=0.8
    )
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D Hits by layer')
    cbar3 = fig.colorbar(sc3, ax=ax, pad=0.1, fraction=0.03, ticks=range(len(layer_tuples)))
    cbar3.set_ticklabels([f"{v}_{l}" for v, l in layer_tuples])
    plt.tight_layout()
    plt.show()



def plot_truth_paths_rz(truth_hits_df: pd.DataFrame, max_tracks: Optional[int] = None) -> None:
    """
    Plots the ground truth particle paths in 2D (z vs r).

    Parameters
    ----------
    truth_hits_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z', and 'particle_id' columns.
    max_tracks : Optional[int], default=None
        Maximum number of tracks to plot. If None, plots all.

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


def plot_truth_paths_3d(truth_hits_df: pd.DataFrame, max_tracks: Optional[int] = None) -> None:
    """
    Plots the ground truth particle paths in 3D (x, y, z).

    Parameters
    ----------
    truth_hits_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z', and 'particle_id' columns.
    max_tracks : Optional[int], default=None
        Maximum number of tracks to plot. If None, plots all.

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

def plot_seed_paths_rz(seeds_df: pd.DataFrame, max_seeds: Optional[int] = None) -> None:
    """
    Plots seed trajectories in 2D (z vs r) for visual inspection.

    Parameters
    ----------
    seeds_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z', and 'particle_id' columns.
    max_seeds : Optional[int], default=None
        Maximum number of seed tracks to plot. If None, plots all.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    grouped = seeds_df.groupby('particle_id')
    for i, (_, group) in enumerate(grouped):
        if max_seeds and i >= max_seeds:
            break
        if 'r' not in group.columns:
            group = group.copy()
            group['r'] = np.sqrt(group['x']**2 + group['y']**2)
        ax.plot(group['z'], group['r'], 'o-', alpha=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('r (mm)')
    ax.set_title('2D R–Z Seed Trajectories')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_seed_paths_3d(seeds_df: pd.DataFrame, max_seeds: Optional[int] = None) -> None:
    """
    Plots seed trajectories in 3D (x, y, z) space.

    Parameters
    ----------
    seeds_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z', and 'particle_id' columns.
    max_seeds : Optional[int], default=None
        Maximum number of seed tracks to plot. If None, plots all.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    grouped = seeds_df.groupby('particle_id')
    for i, (_, group) in enumerate(grouped):
        if max_seeds and i >= max_seeds:
            break
        ax.plot(group['x'], group['y'], group['z'], 'o-', alpha=0.8)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D XYZ Seed Trajectories')
    plt.tight_layout()
    plt.show()

def plot_layer_boundaries(hits_df: pd.DataFrame, layer_tuples: List[Tuple[int, int]], pad_frac: float = 0.05) -> None:
    """
    Draws rectangular boundaries for each detector layer in (z, r) space.

    Parameters
    ----------
    hits_df : pd.DataFrame
        DataFrame with columns ['x', 'y', 'z', 'volume_id', 'layer_id'].
    layer_tuples : List[Tuple[int, int]]
        List of (volume_id, layer_id) tuples representing layers to draw.
    pad_frac : float, default=0.05
        Fractional padding around the global z and r ranges for better visualization.

    Returns
    -------
    None
    """
    # compute global min/max
    z_all = hits_df['z']
    r_all = np.sqrt(hits_df['x']**2 + hits_df['y']**2)  # ensure r exists
    zmin_glob, zmax_glob = z_all.min(), z_all.max()
    rmin_glob, rmax_glob = r_all.min(), r_all.max()

    # dynamic padding
    z_pad = (zmax_glob - zmin_glob) * pad_frac
    r_pad = (rmax_glob - rmin_glob) * pad_frac

    fig, ax = plt.subplots(figsize=(16, 12))  # larger canvas
    all_z, all_r = [], []

    for vol_id, lay_id in layer_tuples:
        layer_hits = hits_df[
            (hits_df.volume_id == vol_id) &
            (hits_df.layer_id == lay_id)
        ]
        if layer_hits.empty:
            continue

        z_vals = layer_hits['z']
        r_vals = np.sqrt(layer_hits.x**2 + layer_hits.y**2)  # recompute r
        zmin, zmax = z_vals.min(), z_vals.max()
        rmin, rmax = r_vals.min(), r_vals.max()

        rect = patches.Rectangle(
            (zmin, rmin),
            zmax - zmin,
            rmax - rmin,
            linewidth=1.5, edgecolor='black', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)

        # larger, bold text
        ax.text(
            (zmin + zmax) / 2,
            (rmin + rmax) / 2,
            f"{vol_id}_{lay_id}",
            fontsize=10, fontweight='bold',
            ha='center', va='center'
        )

    # apply padded limits
    ax.set_xlim(zmin_glob - z_pad, zmax_glob + z_pad)
    ax.set_ylim(rmin_glob - r_pad, rmax_glob + r_pad)

    ax.set_xlabel('z (mm)', fontsize=14)
    ax.set_ylabel('r (mm)', fontsize=14)
    ax.set_title('Layer Boundaries in r vs z', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()