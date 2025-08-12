import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Tuple, Dict, List, Optional
import logging

def plot_extras(hits: pd.DataFrame, pt_cut_hits: pd.DataFrame, enabled: bool) -> None:
    r"""
    Optionally generate presentation‐style plots of detector geometry and truth trajectories.

    This function can plot:

    1. All hits colored by layer in :math:`(z, r)` and 3D.
    2. Detector layer boundaries in :math:`(z, r)` space.
    3. Ground‐truth particle paths in :math:`(z, r)` and 3D.

    Parameters
    ----------
    hits : pandas.DataFrame
        All reconstructed hits with columns ``['x', 'y', 'z', 'volume_id', 'layer_id']``.
    pt_cut_hits : pandas.DataFrame
        Ground‐truth hits passing :math:`p_T` selection cuts.
    enabled : bool
        If ``False``, the function returns immediately without plotting.

    Returns
    -------
    None
    """
    if not enabled:
        return
    logging.info("Plotting detector and truth (extra plots)...")
    layer_tuples = sorted(set(zip(hits.volume_id, hits.layer_id)), key=lambda x: (x[0], x[1]))
    plot_hits_colored_by_layer(hits, layer_tuples)
    plot_layer_boundaries(hits, layer_tuples)
    plot_truth_paths_rz(pt_cut_hits, max_tracks=None)
    plot_truth_paths_3d(pt_cut_hits, max_tracks=None)

def plot_seeds(track_builder, show: bool, max_seeds: Optional[int]) -> None:
    r"""
    Plot reconstructed seed trajectories in both :math:`(z, r)` and 3D space.

    The cylindrical radius is computed as:

    .. math::

        r = \sqrt{x^2 + y^2}

    Only seeds with at least three points are plotted.

    Parameters
    ----------
    track_builder : object
        Track builder instance with a ``get_seeds_dataframe()`` method.
    show : bool
        If ``False``, plotting is skipped.
    max_seeds : int, optional
        Maximum number of seeds to plot. If ``None``, all seeds are shown.

    Returns
    -------
    None
    """
    if not show:
        return

    seeds_df = track_builder.get_seeds_dataframe()
    if seeds_df.empty:
        logging.info("No seeds to plot.")
        return

    # build a tidy DataFrame of (x,y,z,r) sequences per particle with >=3 points
    plot_rows: List[dict] = []
    for _, group in seeds_df.groupby("particle_id"):
        if len(group) < 3:
            continue
        g = group.sort_values("seed_point_index")
        xs, ys, zs = g["x"].to_numpy(), g["y"].to_numpy(), g["z"].to_numpy()
        rs = np.sqrt(xs**2 + ys**2)
        for x, y, z, r in zip(xs, ys, zs, rs):
            plot_rows.append({"particle_id": int(g["particle_id"].iloc[0]), "x": x, "y": y, "z": z, "r": r})

    if not plot_rows:
        logging.info("No seed groups with >=3 points to plot.")
        return

    plot_df = pd.DataFrame(plot_rows)
    plot_seed_paths_rz(plot_df, max_seeds=max_seeds)
    plot_seed_paths_3d(plot_df, max_seeds=max_seeds)

def plot_best_track_3d(track_builder, track, truth_particle: pd.DataFrame, idx: int) -> None:
    r"""
    Visualize a single reconstructed track compared to its ground truth in 3D.

    Parameters
    ----------
    track_builder : object
        Object with a ``hit_pool`` attribute containing all hits.
    track : object
        Track object with a ``trajectory`` attribute (array of shape :math:`(N, 3)`).
    truth_particle : pandas.DataFrame
        Truth hits for the particle, with columns ``['x', 'y', 'z', 'r']``.
    idx : int
        Index or label for the reconstructed track.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    hits = track_builder.hit_pool.hits
    ax.scatter(hits.x, hits.y, hits.z, s=1, alpha=0.08)

    tp = truth_particle.sort_values("r")
    ax.plot(tp.x, tp.y, tp.z, "--", label="truth")

    traj = np.asarray(track.trajectory)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", label="reconstructed")

    ax.set_title(f"Track {idx} (PID {track.particle_id})")
    ax.legend()
    plt.show()

def check_seed_and_plot(model: object, seed_points: List[np.ndarray]) -> None:
    r"""
    Estimate and validate a seed state from three points, then visualize its velocity vs. the true circle tangent.

    Given three seed points :math:`P_1, P_2, P_3` in the transverse plane,
    the true circle center :math:`(c_x, c_y)` is computed via perpendicular bisector equations.
    The true tangent vector is orthogonal to the radial vector from center to :math:`P_3`.

    The seed's velocity vector :math:`\hat{v}` from the model estimate is compared to this
    true tangent via:

    .. math::

        \theta = \cos^{-1}(\hat{v} \cdot \hat{t})

    where :math:`\theta` is the angle difference in degrees.

    Parameters
    ----------
    model : object
        Must have a ``estimate_seed(points)`` method returning ``(x0, P0)``.
    seed_points : list of ndarray
        Three seed points, each a length‐3 array :math:`(x, y, z)`.

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
    r"""
    Plot detector hits colored by their :math:`(\text{volume\_id}, \text{layer\_id})` in both :math:`(z, r)` and 3D.

    Parameters
    ----------
    hits_df : pandas.DataFrame
        Must contain columns ``['x', 'y', 'z', 'volume_id', 'layer_id']``.
    layer_tuples : list of tuple
        Unique :math:`(\text{volume\_id}, \text{layer\_id})` pairs in desired color order.

    Notes
    -----
    The cylindrical radius is computed as :math:`r = \sqrt{x^2 + y^2}`.

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
    r"""
    Plot ground‐truth particle trajectories in the :math:`(z, r)` plane.

    Parameters
    ----------
    truth_hits_df : pandas.DataFrame
        Columns: ``['x', 'y', 'z', 'particle_id']``.
    max_tracks : int, optional
        Limit number of tracks plotted. If ``None``, all are shown.

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
    r"""
    Plot ground‐truth particle trajectories in 3D Cartesian space.

    Parameters
    ----------
    truth_hits_df : pandas.DataFrame
        Columns: ``['x', 'y', 'z', 'particle_id']``.
    max_tracks : int, optional
        Limit number of tracks plotted. If ``None``, all are shown.

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
    r"""
    Plot seed trajectories in :math:`(z, r)` for visual inspection.

    Parameters
    ----------
    seeds_df : pandas.DataFrame
        Columns: ``['x', 'y', 'z', 'particle_id']``. Will compute :math:`r` if missing.
    max_seeds : int, optional
        Limit number of seeds plotted. If ``None``, all are shown.

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
    r"""
    Plot seed trajectories in 3D Cartesian space.

    Parameters
    ----------
    seeds_df : pandas.DataFrame
        Columns: ``['x', 'y', 'z', 'particle_id']``.
    max_seeds : int, optional
        Limit number of seeds plotted. If ``None``, all are shown.

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
    r"""
    Draw rectangular boundaries in :math:`(z, r)` for each detector layer.

    Parameters
    ----------
    hits_df : pandas.DataFrame
        Columns: ``['x', 'y', 'z', 'volume_id', 'layer_id']``.
    layer_tuples : list of tuple
        :math:`(\text{volume\_id}, \text{layer\_id})` pairs to draw.
    pad_frac : float, default=0.05
        Fraction of global :math:`z` and :math:`r` ranges to add as padding.

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

    
def plot_branches(branches: List[Dict], seed_points: List[np.ndarray], future_layers: List[Tuple[int, int]], hits_df: pd.DataFrame, 
                  truth_hits: Optional[pd.DataFrame] = None, particle_id: Optional[int] = None, pad_frac: float = 0.05) -> None:
    r"""
    Overlay track‐building branches, seed points, and truth hits on detector layer boundaries.

    The :math:`(z, r)` view is used, with rectangular outlines for each layer and
    trajectories plotted as connected points.

    Parameters
    ----------
    branches : list of dict
        Each must have a ``trajectory`` attribute or key with :math:`(N, 3)` coordinates.
    seed_points : list of ndarray
        List of 3D :math:`[x, y, z]` arrays for the seed.
    future_layers : list of tuple
        Layer identifiers reserved for API compatibility.
    hits_df : pandas.DataFrame
        Hit coordinates and layer IDs.
    truth_hits : pandas.DataFrame, optional
        If provided, must include ``particle_id`` to plot truth hits.
    particle_id : int, optional
        Filter truth hits by this particle ID. Defaults to first present.
    pad_frac : float
        Padding fraction for :math:`z` and :math:`r` axes.

    Returns
    -------
    None
    """
    layer_tuples = sorted(set(zip(hits_df.volume_id, hits_df.layer_id)), key=lambda x:(x[0],x[1]))
    # Compute global z/r extents for padding
    z_all = hits_df['z']
    r_all = np.sqrt(hits_df['x']**2 + hits_df['y']**2)
    zmin_glob, zmax_glob = z_all.min(), z_all.max()
    rmin_glob, rmax_glob = r_all.min(), r_all.max()
    z_pad = (zmax_glob - zmin_glob) * pad_frac
    r_pad = (rmax_glob - rmin_glob) * pad_frac

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1) Draw each layer boundary rectangle
    for vol_id, lay_id in layer_tuples:
        layer_hits = hits_df[
            (hits_df.volume_id == vol_id) &
            (hits_df.layer_id == lay_id)
        ]
        if layer_hits.empty:
            continue

        z_vals = layer_hits['z']
        r_vals = np.sqrt(layer_hits.x**2 + layer_hits.y**2)
        zmin, zmax = z_vals.min(), z_vals.max()
        rmin, rmax = r_vals.min(), r_vals.max()

        rect = patches.Rectangle(
            (zmin, rmin),
            zmax - zmin,
            rmax - rmin,
            linewidth=1.2,
            edgecolor='gray',
            facecolor='none',
            alpha=0.5,
            linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(
            (zmin + zmax) / 2,
            (rmin + rmax) / 2,
            f"{vol_id}_{lay_id}",
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            alpha=0.7
        )

    # 2) Plot truth hits (if any)
    if truth_hits is not None and not truth_hits.empty:
        if particle_id is None:
            # default to first PID
            particle_id = truth_hits['particle_id'].iat[0]
        truth = truth_hits[truth_hits['particle_id'] == particle_id]
        r_truth = np.sqrt(truth['x']**2 + truth['y']**2)
        ax.scatter(
            truth['z'], r_truth,
            c='black', s=30, marker='s',
            alpha=0.7, label=f"Truth hits (PID {particle_id})"
        )

    # 3) Plot all branch trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(branches)))
    for i, branch in enumerate(branches):
        traj = np.array(branch.trajectory)
        r_traj = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        ax.plot(
            traj[:, 2], r_traj,
            marker='o', linestyle='-',
            color=colors[i % len(colors)],
            linewidth=2, alpha=0.8,
            label=f"Branch {i+1} ({len(traj)} hits)"
        )

    # 4) Plot seed points
    sp = np.vstack(seed_points)
    r_sp = np.sqrt(sp[:, 0]**2 + sp[:, 1]**2)
    ax.scatter(
        sp[:, 2], r_sp,
        c='red', marker='*', s=120,
        edgecolors='k', linewidth=1,
        label='Seed points', zorder=10
    )

    # Final styling
    ax.set_xlim(zmin_glob - z_pad, zmax_glob + z_pad)
    ax.set_ylim(rmin_glob - r_pad, rmax_glob + r_pad)
    ax.set_xlabel('z (mm)', fontsize=12)
    ax.set_ylabel('r (mm)', fontsize=12)
    ax.set_title('Track Building Branches over Layer Boundaries (R–Z view)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()

def plot_track_building_debug(track_builder, seed_row: pd.Series, branches: List[Dict], 
                            max_branches_to_plot: int = 10) -> None:
    r"""
    Plot debug view of track‐building results for a single seed.

    Steps
    -----
    1. Extract the first 3 hits as the seed points.
    2. Limit plotted branches to ``max_branches_to_plot``.
    3. Call :func:`plot_branches` with truth hits overlay.

    Parameters
    ----------
    track_builder : object
        Must have ``hit_pool`` with ``pt_cut_hits`` and ``hits`` attributes.
    seed_row : pandas.Series
        Row from the seeds DataFrame.
    branches : list of dict
        Branch trajectories from track building.
    max_branches_to_plot : int, default=10
        Limit number of branches plotted.

    Returns
    -------
    None
    """
    # Extract seed points from the seed row
    particle_id = seed_row['particle_id']
    
    # Get the 3 seed points for this particle
    particle_hits = track_builder.hit_pool.pt_cut_hits[
        track_builder.hit_pool.pt_cut_hits['particle_id'] == particle_id
    ].sort_values('r')
    
    if len(particle_hits) < 3:
        print(f"Warning: Particle {particle_id} has fewer than 3 hits")
        return
    
    # Take first 3 hits as seed points
    seed_points = particle_hits.head(3)[['x', 'y', 'z']].values.tolist()
    
    # Limit branches for plotting
    branches_to_plot = branches[:max_branches_to_plot]
    
    # Plot with truth hits
    plot_branches(
        branches=branches_to_plot,
        seed_points=seed_points,
        future_layers=seed_row['future_layers'],
        hits_df=track_builder.hits,
        truth_hits=track_builder.hit_pool.pt_cut_hits,
        particle_id=particle_id
    )
    
    # Print some debug info
    print(f"Particle {particle_id}: {len(branches)} branches, plotting {len(branches_to_plot)}")
    print(f"Seed points: {len(seed_points)}")
    print(f"Future layers: {len(seed_row['future_layers'])}")
    
    if branches:
        branch_lengths = [len(branch['traj']) for branch in branches]
        print(f"Branch lengths: min={min(branch_lengths)}, max={max(branch_lengths)}, mean={np.mean(branch_lengths):.1f}")