import logging
from typing import Tuple, Dict, List, Optional, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize


def _show_and_close(fig, *, do_show: bool = True) -> None:
    r"""
    Show a Matplotlib figure (optionally) and always close it.

    This helper wraps ``plt.show()`` and ``plt.close(fig)`` to prevent figure
    accumulation and memory growth. It is safe in headless mode where
    ``plt.show()`` may be patched to a no-op.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to display and close.
    do_show : bool, optional
        If ``True`` (default) call ``plt.show()`` before closing.

    Notes
    -----
    The function attempts ``fig.tight_layout()`` and suppresses any exceptions,
    so it can be used uniformly in debug code and batch pipelines.
    """
    try:
        fig.tight_layout()
    except Exception:
        pass
    if do_show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_extras(hits: pd.DataFrame, pt_cut_hits: pd.DataFrame, enabled: bool) -> None:
    r"""
    Optionally generate presentation plots of geometry and truth trajectories.

    When ``enabled`` is ``True``, this function renders:
    1) hits colored by (volume, layer) in :math:`(z,r)` and 3D;
    2) rectangular layer boundaries in :math:`(z,r)`;
    3) truth particle paths in :math:`(z,r)` and 3D.

    Parameters
    ----------
    hits : pandas.DataFrame
        All event hits. Must contain at least columns ``x, y, z, volume_id, layer_id``.
    pt_cut_hits : pandas.DataFrame
        Truth-matched, high-:math:`p_T` hits (subset of ``hits``) with
        columns ``x, y, z, particle_id``.
    enabled : bool
        If ``False``, the function returns immediately without plotting.

    Notes
    -----
    - Axes units follow the input data (typically **millimeters** in TrackML raw
    inputs). Radial coordinate is computed as

    .. math:: r \;=\; \sqrt{x^2 + y^2}.
    """
    if not enabled:
        return
    logging.info("Plotting detector and truth (extra plots)...")

    layer_tuples = _sorted_layer_keys(hits)
    plot_hits_colored_by_layer(hits, layer_tuples)
    plot_layer_boundaries(hits, layer_tuples)
    plot_truth_paths_rz(pt_cut_hits, max_tracks=None)
    plot_truth_paths_3d(pt_cut_hits, max_tracks=None)


def plot_seeds(track_builder, show: bool, max_seeds: Optional[int]) -> None:
    r"""
    Plot reconstructed seed trajectories in both :math:`(z,r)` and 3D.

    Seeds are grouped by ``particle_id``; groups with at least three points are
    drawn as ordered polylines.

    Parameters
    ----------
    track_builder : object
        Must expose ``get_seeds_dataframe()`` returning a DataFrame with columns
        ``particle_id, seed_point_index, x, y, z`` (and optionally extras).
    show : bool
        If ``False``, nothing is drawn.
    max_seeds : int or None, optional
        If provided, cap the number of seed groups to plot (by group order).

    Notes
    -----
    The radial coordinate is :math:`r=\sqrt{x^2+y^2}` and plotted against :math:`z`.
    """
    if not show:
        return

    seeds_df = track_builder.get_seeds_dataframe()
    if seeds_df.empty:
        logging.info("No seeds to plot.")
        return

    rows: List[dict] = []
    for pid, g in seeds_df.groupby("particle_id", sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("seed_point_index", kind="stable")
        x = g["x"].to_numpy(dtype=np.float64, copy=False)
        y = g["y"].to_numpy(dtype=np.float64, copy=False)
        z = g["z"].to_numpy(dtype=np.float64, copy=False)
        r = np.sqrt(x * x + y * y)
        for xi, yi, zi, ri in zip(x, y, z, r):
            rows.append({"particle_id": int(pid), "x": xi, "y": yi, "z": zi, "r": ri})

    if not rows:
        logging.info("No seed groups with >=3 points to plot.")
        return

    plot_df = pd.DataFrame.from_records(rows)
    plot_seed_paths_rz(plot_df, max_seeds=max_seeds)
    plot_seed_paths_3d(plot_df, max_seeds=max_seeds)


def plot_best_track_3d(
    track_builder,
    track,
    truth_particle: pd.DataFrame,
    idx: int,
    *,
    max_hits_scatter: int = 30000,
) -> None:
    r"""
    Visualize one reconstructed track against its ground truth in 3D.

    Parameters
    ----------
    track_builder : object
        Must have ``hit_pool.hits`` (DataFrame with ``x,y,z``).
    track : object
        A track candidate with attributes ``particle_id`` and
        ``trajectory`` (sequence of ``(x,y,z)``).
    truth_particle : pandas.DataFrame
        Truth hits for the same particle id. Columns: ``x, y, z``.
    idx : int
        Human-readable index used in the figure title.
    max_hits_scatter : int, optional
        If the full event has more points than this, a random subset is scattered.

    Notes
    -----
    - The reconstructed polyline is drawn on top of a low-opacity point cloud of
    event hits to provide spatial context.
    - Ground truth is sorted by :math:`z` before plotting to produce a continuous line.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    hits = track_builder.hit_pool.hits
    if len(hits) > max_hits_scatter:
        rng = np.random.default_rng()
        sample_idx = rng.choice(len(hits), size=max_hits_scatter, replace=False)
        sx = hits["x"].to_numpy()[sample_idx]
        sy = hits["y"].to_numpy()[sample_idx]
        sz = hits["z"].to_numpy()[sample_idx]
        ax.scatter(sx, sy, sz, s=1, alpha=0.08)
    else:
        ax.scatter(hits["x"], hits["y"], hits["z"], s=1, alpha=0.08)

    tp = truth_particle.sort_values("z", kind="stable")
    ax.plot(tp["x"], tp["y"], tp["z"], "--", label="truth")

    traj = _branch_to_traj(track)
    if traj.size:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", label="reconstructed")

    ax.set_title(f"Track {idx} (PID {track.particle_id})")
    ax.legend()
    _show_and_close(fig)


def check_seed_and_plot(model: object, seed_points: List[np.ndarray]) -> None:
    r"""
    Estimate and sanity-check a seed's initial state from three points.

    This helper compares the model's estimated transverse velocity direction
    to the *true* tangent of the circle through the last two seed points and
    the inferred circle center in the :math:`xy` plane.

    Computation
    -----------
    Given three seed points :math:`p_1, p_2, p_3 \in \mathbb{R}^3`, we solve in
    the plane for the circle center :math:`(c_x,c_y)` via

    .. math::

    \begin{bmatrix}
    p_{2x}-p_{1x} & p_{2y}-p_{1y}\\
    p_{3x}-p_{1x} & p_{3y}-p_{1y}
    \end{bmatrix}
    \begin{bmatrix}c_x-p_{1x}\\c_y-p_{1y}\end{bmatrix}
    =
    \tfrac{1}{2}\begin{bmatrix}
    p_{2x}^2-p_{1x}^2 + p_{2y}^2-p_{1y}^2\\
    p_{3x}^2-p_{1x}^2 + p_{3y}^2-p_{1y}^2
    \end{bmatrix}.

    The "true" tangent at :math:`p_3` is perpendicular to the radial vector
    :math:`r = (p_{3x}-c_x,\,p_{3y}-c_y)`:

    .. math:: t_{\rm true} \propto (-r_y,\, r_x).

    We compare it to the model's seed transverse velocity :math:`\hat v` via

    .. math:: \theta = \arccos\!\big(\mathrm{clip}(\hat v\cdot \hat t_{\rm true},-1,1)\big).

    Parameters
    ----------
    model : object
        Must implement ``estimate_seed(seed_points)`` returning
        ``(state, covariance)`` where ``state[:5]`` includes ``x,y,z,vx,vy``.
    seed_points : list of ndarray, shape (3,), length == 3
        Three 3D points used to initialize the seed.

    Returns
    -------
    None
        Prints angle diagnostics and renders a 2D arrow plot in the :math:`xy` plane.

    Notes
    -----
    The arrows for the estimated velocity and the analytic tangent are scaled
    to a short length for visualization only.
    """
    x0, _P0 = model.estimate_seed(seed_points)[0]  # type: ignore[attr-defined]

    print("Seed position:", np.round(x0[:3], 5))
    print("Last pt      :", np.round(seed_points[-1], 5))

    p1, p2, p3 = seed_points
    A = np.array(
        [[p2[0] - p1[0], p2[1] - p1[1]], [p3[0] - p1[0], p3[1] - p1[1]]],
        dtype=np.float64,
    )
    B = np.array(
        [
            [(p2[0] ** 2 - p1[0] ** 2 + p2[1] ** 2 - p1[1] ** 2) / 2.0],
            [(p3[0] ** 2 - p1[0] ** 2 + p3[1] ** 2 - p1[1] ** 2) / 2.0],
        ],
        dtype=np.float64,
    )
    cx, cy = (np.linalg.solve(A, B).flatten() + p1[:2])
    radial = p3[:2] - np.array([cx, cy])
    true_tan = np.array([-radial[1], radial[0]], dtype=np.float64)
    nrm = np.linalg.norm(true_tan)
    if nrm > 0:
        true_tan /= nrm

    seg = p3[:2] - p2[:2]
    seg_nrm = np.linalg.norm(seg)
    if seg_nrm > 0 and float(np.dot(true_tan, seg / seg_nrm)) < 0:
        true_tan *= -1

    v_hat = x0[3:5]
    v_nrm = np.linalg.norm(v_hat)
    v_hat = v_hat / v_nrm if v_nrm > 0 else v_hat

    dot = float(np.clip(np.dot(v_hat, true_tan), -1.0, 1.0))
    angle = np.degrees(np.arccos(dot))
    print(f"Angle seed-v to true-tangent: {angle:.3f}°")

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()
    pts2 = np.array([p[:2] for p in seed_points])
    ax.scatter(pts2[:, 0], pts2[:, 1], label="seed points")
    ax.scatter(cx, cy, marker="x", color="k", label="circle center")
    ax.arrow(
        p3[0], p3[1], true_tan[0] * 0.1, true_tan[1] * 0.1, head_width=0.003, color="r", label="true tangent"
    )
    ax.arrow(
        p3[0], p3[1], v_hat[0] * 0.1, v_hat[1] * 0.1, head_width=0.003, color="b", label="seed velocity"
    )
    ax.legend()
    ax.set_aspect("equal", "box")
    _show_and_close(fig)


def plot_hits_colored_by_layer(hits_df: pd.DataFrame, layer_tuples: List[Tuple[int, int]]) -> None:
    r"""
    Plot detector hits colored by (volume, layer) in :math:`(z,r)` and in 3D.

    A categorical colormap is built over the sorted unique layer keys
    ``(volume_id, layer_id)`` passed in ``layer_tuples``, and each hit is colored
    by the corresponding index.

    Parameters
    ----------
    hits_df : pandas.DataFrame
        Input hits with columns ``x, y, z, volume_id, layer_id`` (units: mm).
    layer_tuples : list[tuple[int,int]]
        Sorted list of available layers, e.g. ``[(8,2), (8,4), ...]``.
        The legend labels are rendered as ``"vol_layer"`` strings.

    Notes
    -----
    - Radial coordinate is :math:`r=\sqrt{x^2+y^2}`.
    - The function uses a fast integer encoding
    :math:`\mathrm{key} = (\mathrm{vol}\ll 32)\,|\,(\mathrm{layer}\,\&\,2^{32}\!-\!1)`
    and ``np.searchsorted`` to map each hit to its color index, falling back
    to a hash map if any keys are out of order.
    """
    if hits_df.empty or not layer_tuples:
        return

    vol = hits_df["volume_id"].to_numpy(dtype=np.int64, copy=False)
    lay = hits_df["layer_id"].to_numpy(dtype=np.int64, copy=False)
    keys_enc = (vol << 32) | (lay & np.int64(0xFFFFFFFF))

    ref_v = np.fromiter((v for v, _ in layer_tuples), dtype=np.int64, count=len(layer_tuples))
    ref_l = np.fromiter((l for _, l in layer_tuples), dtype=np.int64, count=len(layer_tuples))
    ref_enc = (ref_v << 32) | (ref_l & np.int64(0xFFFFFFFF))

    idx = np.searchsorted(ref_enc, keys_enc)
    bad = (idx < 0) | (idx >= ref_enc.size) | (ref_enc[idx] != keys_enc)
    if np.any(bad):
        back = {((int(v) << 32) | (int(l) & 0xFFFFFFFF)): i for i, (v, l) in enumerate(layer_tuples)}
        idx[bad] = np.fromiter((back.get(int(k), 0) for k in keys_enc[bad]), dtype=np.int64, count=int(bad.sum()))

    x = hits_df["x"].to_numpy(dtype=np.float64, copy=False)
    y = hits_df["y"].to_numpy(dtype=np.float64, copy=False)
    z = hits_df["z"].to_numpy(dtype=np.float64, copy=False)
    r = np.sqrt(x * x + y * y)

    n_layers = len(layer_tuples)
    labels = [f"{v}_{l}" for v, l in layer_tuples]
    norm = Normalize(vmin=0, vmax=n_layers - 1)
    cmap = cm.get_cmap("hsv", n_layers)
    boundaries = np.arange(-0.5, n_layers + 0.5, 1)
    ticks = np.arange(n_layers)

    # ---- 2D (z, r)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sc = ax2.scatter(z, r, c=idx, cmap=cmap, norm=norm, s=8, alpha=0.8)
    ax2.set_xlabel("z (mm)")
    ax2.set_ylabel("r (mm)")
    ax2.set_title("Hits by layer (z vs r)")
    cbar = fig2.colorbar(sc, ax=ax2, ticks=ticks, boundaries=boundaries)
    cbar.set_ticklabels(labels)
    cbar.set_label("volume_layer")
    ax2.grid(True, alpha=0.3)
    _show_and_close(fig2)

    # ---- 3D
    fig3 = plt.figure(figsize=(12, 9))
    ax3 = fig3.add_subplot(111, projection="3d")
    sc3 = ax3.scatter(x, y, z, c=idx, cmap=cmap, norm=norm, s=6, alpha=0.75)
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("y (mm)")
    ax3.set_zlabel("z (mm)")
    ax3.set_title("3D Hits by layer")
    cbar3 = fig3.colorbar(sc3, ax=ax3, pad=0.08, fraction=0.04, ticks=ticks, boundaries=boundaries)
    cbar3.set_ticklabels(labels)
    cbar3.ax.set_ylabel("volume_layer", rotation=270, labelpad=14)
    _show_and_close(fig3)


def plot_truth_paths_rz(truth_hits_df: pd.DataFrame, max_tracks: Optional[int] = None) -> None:
    r"""
    Plot truth particle trajectories as polylines in :math:`(z,r)`.

    Parameters
    ----------
    truth_hits_df : pandas.DataFrame
        Truth-matched hits with columns ``x, y, z, particle_id`` (units: mm).
    max_tracks : int or None, optional
        If provided, plot at most this many particle groups.

    Notes
    -----
    The radial coordinate is :math:`r=\sqrt{x^2+y^2}`. Each particle id is drawn
    as a separate polyline in ascending group order.
    """
    if truth_hits_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    count = 0
    for _pid, g in truth_hits_df.groupby("particle_id", sort=False):
        if max_tracks is not None and count >= max_tracks:
            break
        x = g["x"].to_numpy(dtype=np.float64, copy=False)
        y = g["y"].to_numpy(dtype=np.float64, copy=False)
        z = g["z"].to_numpy(dtype=np.float64, copy=False)
        r = np.sqrt(x * x + y * y)
        ax.plot(z, r, alpha=0.7, linewidth=1.1)
        count += 1

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("2D R–Z Truth Particle Paths")
    ax.grid(True, alpha=0.3)
    _show_and_close(fig)


def plot_truth_paths_3d(truth_hits_df: pd.DataFrame, max_tracks: Optional[int] = None) -> None:
    r"""
    Plot truth particle trajectories as 3D polylines.

    Parameters
    ----------
    truth_hits_df : pandas.DataFrame
        Truth hits with columns ``x, y, z, particle_id`` (units: mm).
    max_tracks : int or None, optional
        If provided, plot at most this many particle groups.

    Notes
    -----
    The function sorts within each particle group by DataFrame order (no implicit
    re-sorting) to preserve any pre-arranged sequence.
    """
    if truth_hits_df.empty:
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    count = 0
    for _pid, g in truth_hits_df.groupby("particle_id", sort=False):
        if max_tracks is not None and count >= max_tracks:
            break
        ax.plot(g["x"], g["y"], g["z"], alpha=0.7, linewidth=1.0)
        count += 1

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D Truth Particle Paths")
    _show_and_close(fig)


def plot_seed_paths_rz(seeds_df: pd.DataFrame, max_seeds: Optional[int] = None) -> None:
    r"""
    Plot seed trajectories in :math:`(z,r)` as small polylines.

    Parameters
    ----------
    seeds_df : pandas.DataFrame
        Seed points with columns ``x, y, z, particle_id`` (units: mm).
    max_seeds : int or None, optional
        If provided, plot at most this many particle groups.

    Notes
    -----
    We draw a point-marker polyline ``"o-"`` to emphasize sparsity of seed data.
    """
    if seeds_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    count = 0
    for _pid, g in seeds_df.groupby("particle_id", sort=False):
        if max_seeds is not None and count >= max_seeds:
            break
        z = g["z"].to_numpy(dtype=np.float64, copy=False)
        x = g["x"].to_numpy(dtype=np.float64, copy=False)
        y = g["y"].to_numpy(dtype=np.float64, copy=False)
        r = np.sqrt(x * x + y * y)
        ax.plot(z, r, "o-", alpha=0.85, linewidth=1.1, markersize=3)
        count += 1

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("2D R–Z Seed Trajectories")
    ax.grid(True, alpha=0.3)
    _show_and_close(fig)


def plot_seed_paths_3d(seeds_df: pd.DataFrame, max_seeds: Optional[int] = None) -> None:
    r"""
    Plot seed trajectories as 3D polylines.

    Parameters
    ----------
    seeds_df : pandas.DataFrame
        Seed points with columns ``x, y, z, particle_id`` (units: mm).
    max_seeds : int or None, optional
        If provided, plot at most this many particle groups.

    Notes
    -----
    A small marker is used to signal the discrete nature of seed triplets.
    """
    if seeds_df.empty:
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    count = 0
    for _pid, g in seeds_df.groupby("particle_id", sort=False):
        if max_seeds is not None and count >= max_seeds:
            break
        ax.plot(g["x"], g["y"], g["z"], "o-", alpha=0.85, linewidth=1.0, markersize=3)
        count += 1

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D XYZ Seed Trajectories")
    _show_and_close(fig)


def plot_layer_boundaries(
    hits_df: pd.DataFrame,
    layer_tuples: List[Tuple[int, int]],
    pad_frac: float = 0.05,
) -> None:
    r"""
    Draw rectangular per-layer **bounds** in the :math:`(z,r)` plane.

    For each layer key ``(volume_id, layer_id)``, compute

    .. math::
    z_{\min},\, z_{\max},\, r_{\min},\, r_{\max},\qquad
    r \,=\, \sqrt{x^2+y^2},

    and render a rectangle spanning :math:`[z_{\min},z_{\max}]\times[r_{\min},r_{\max}]`.

    Parameters
    ----------
    hits_df : pandas.DataFrame
        Hits with ``x, y, z, volume_id, layer_id`` (units: mm).
    layer_tuples : list[tuple[int,int]]
        Present for API symmetry; plotting relies on the groupby over ``hits_df``.
    pad_frac : float, optional
        Fractional padding applied to global axes limits (default 0.05).

    Notes
    -----
    The label inside each rectangle is ``"vol_layer"`` for quick identification.
    """
    if hits_df.empty or not layer_tuples:
        return

    z = hits_df["z"].to_numpy(dtype=np.float64, copy=False)
    x = hits_df["x"].to_numpy(dtype=np.float64, copy=False)
    y = hits_df["y"].to_numpy(dtype=np.float64, copy=False)
    r = np.sqrt(x * x + y * y)

    zmin_glob, zmax_glob = float(np.min(z)), float(np.max(z))
    rmin_glob, rmax_glob = float(np.min(r)), float(np.max(r))
    z_pad = (zmax_glob - zmin_glob) * pad_frac if zmax_glob > zmin_glob else 1.0
    r_pad = (rmax_glob - rmin_glob) * pad_frac if rmax_glob > rmin_glob else 1.0

    df = hits_df[["volume_id", "layer_id", "z", "x", "y"]].copy()
    df["r"] = np.sqrt(df["x"].to_numpy() ** 2 + df["y"].to_numpy() ** 2)
    gb = df.groupby(["volume_id", "layer_id"], sort=False).agg(
        zmin=("z", "min"), zmax=("z", "max"), rmin=("r", "min"), rmax=("r", "max")
    )

    fig, ax = plt.subplots(figsize=(16, 12))
    for (vol, lay), row in gb.iterrows():
        rect = patches.Rectangle(
            (row.zmin, row.rmin),
            row.zmax - row.zmin,
            row.rmax - row.rmin,
            linewidth=1.3,
            edgecolor="black",
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            (row.zmin + row.zmax) * 0.5,
            (row.rmin + row.rmax) * 0.5,
            f"{vol}_{lay}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )

    ax.set_xlim(zmin_glob - z_pad, zmax_glob + z_pad)
    ax.set_ylim(rmin_glob - r_pad, rmax_glob + r_pad)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("Layer Boundaries in r vs z")
    ax.grid(True, alpha=0.3)
    _show_and_close(fig)


def plot_branches(
    branches: List[Dict],
    seed_points: List[np.ndarray],
    future_layers: List[Tuple[int, int]],
    hits_df: pd.DataFrame,
    truth_hits: Optional[pd.DataFrame] = None,
    particle_id: Optional[int] = None,
    pad_frac: float = 0.05,
) -> None:
    r"""
    Overlay candidate branches, seed points, and (optional) truth hits on
    per-layer rectangles in :math:`(z,r)`.

    Each branch is interpreted as a polyline in Cartesian coordinates
    ``(x,y,z)`` and shown in :math:`(z,r)`, where :math:`r=\sqrt{x^2+y^2}`.

    Parameters
    ----------
    branches : list[dict]
        Each dict must contain key ``"traj"`` with sequence of ``(x,y,z)`` points.
    seed_points : list[numpy.ndarray]
        Three seed points used to initialize the track; plotted as star markers.
    future_layers : list[tuple[int,int]]
        Unused in the plot itself; kept for caller symmetry and annotations.
    hits_df : pandas.DataFrame
        Full event hits (``x, y, z, volume_id, layer_id``) used to compute
        layer rectangles.
    truth_hits : pandas.DataFrame or None, optional
        If provided, points for the ``particle_id`` are overlaid as black squares.
    particle_id : int or None, optional
        Particle id to highlight from ``truth_hits``; if ``None``, the first id
        in ``truth_hits`` is chosen.
    pad_frac : float, optional
        Axes padding fraction on both dimensions.

    Notes
    -----
    - Layer rectangles are computed from per-layer min/max ranges of :math:`z`
    and :math:`r`. They are shaded lightly and labeled.
    - Branch colors cycle through ``tab10`` for visual separation.
    """
    if hits_df.empty:
        return

    z = hits_df["z"].to_numpy(dtype=np.float64, copy=False)
    x = hits_df["x"].to_numpy(dtype=np.float64, copy=False)
    y = hits_df["y"].to_numpy(dtype=np.float64, copy=False)
    r = np.sqrt(x * x + y * y)

    zmin_glob, zmax_glob = float(np.min(z)), float(np.max(z))
    rmin_glob, rmax_glob = float(np.min(r)), float(np.max(r))
    z_pad = (zmax_glob - zmin_glob) * pad_frac if zmax_glob > zmin_glob else 1.0
    r_pad = (rmax_glob - rmin_glob) * pad_frac if rmax_glob > rmin_glob else 1.0

    df_bounds = hits_df[["volume_id", "layer_id", "z", "x", "y"]].copy()
    df_bounds["r"] = np.sqrt(df_bounds["x"].to_numpy() ** 2 + df_bounds["y"].to_numpy() ** 2)
    gb = df_bounds.groupby(["volume_id", "layer_id"], sort=False).agg(
        zmin=("z", "min"), zmax=("z", "max"), rmin=("r", "min"), rmax=("r", "max")
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    for (vol, lay), row in gb.iterrows():
        rect = patches.Rectangle(
            (row.zmin, row.rmin),
            row.zmax - row.zmin,
            row.rmax - row.rmin,
            linewidth=1.0,
            edgecolor="gray",
            facecolor="none",
            alpha=0.5,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            (row.zmin + row.zmax) * 0.5,
            (row.rmin + row.rmax) * 0.5,
            f"{vol}_{lay}",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.7,
        )

    # Truth overlay (optional)
    if truth_hits is not None and not truth_hits.empty:
        if particle_id is None:
            particle_id = int(truth_hits["particle_id"].iloc[0])
        truth = truth_hits.loc[truth_hits["particle_id"] == particle_id, ["x", "y", "z"]]
        if not truth.empty:
            tx = truth["x"].to_numpy(dtype=np.float64, copy=False)
            ty = truth["y"].to_numpy(dtype=np.float64, copy=False)
            tz = truth["z"].to_numpy(dtype=np.float64, copy=False)
            tr = np.sqrt(tx * tx + ty * ty)
            ax.scatter(
                tz, tr, c="black", s=28, marker="s", alpha=0.75, label=f"Truth hits (PID {particle_id})"
            )

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(branches))))
    for i, branch in enumerate(branches):
        traj = _branch_to_traj(branch)
        if traj.size == 0:
            continue
        rt = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
        ax.plot(
            traj[:, 2],
            rt,
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            linewidth=2.0,
            alpha=0.85,
            markersize=3,
            label=f"Branch {i+1} ({len(traj)} pts)",
        )

    sp = np.vstack(seed_points).astype(np.float64, copy=False)
    rsp = np.sqrt(sp[:, 0] ** 2 + sp[:, 1] ** 2)
    ax.scatter(
        sp[:, 2], rsp, c="red", marker="*", s=120, edgecolors="k", linewidth=0.8, label="Seed points", zorder=10
    )

    ax.set_xlim(zmin_glob - z_pad, zmax_glob + z_pad)
    ax.set_ylim(rmin_glob - r_pad, rmax_glob + r_pad)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("Track Building Branches over Layer Boundaries (R–Z)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _show_and_close(fig)


def plot_track_building_debug(
    track_builder,
    seed_row: pd.Series,
    branches: List[Dict],
    max_branches_to_plot: int = 10,
) -> None:
    r"""
    One-seed debug view: layer rectangles, seed points, selected branches,
    and corresponding truth hits in :math:`(z,r)`.

    Parameters
    ----------
    track_builder : object
        Must expose ``hit_pool.hits`` and ``hit_pool.pt_cut_hits`` DataFrames.
    seed_row : pandas.Series
        A row describing a seed; must contain ``particle_id`` and ``future_layers``.
    branches : list[dict]
        Candidate branches from the brancher; only up to ``max_branches_to_plot``
        are drawn.
    max_branches_to_plot : int, optional
        Cap on the number of candidate branches to visualize (default 10).

    Notes
    -----
    The function logs basic branch statistics (min/max/mean lengths) to aid
    qualitative assessment of branching behavior.
    """
    pid = int(seed_row["particle_id"])
    truth_df = track_builder.hit_pool.pt_cut_hits
    hits_df = track_builder.hit_pool.hits

    truth_pid = truth_df.loc[truth_df["particle_id"] == pid, ["x", "y", "z"]]
    if len(truth_pid) < 3:
        logging.warning("Particle %s has fewer than 3 truth hits", pid)
        return

    sp = truth_pid.sort_values("z", kind="stable").head(3).to_numpy(dtype=np.float64, copy=False)
    seed_points = [sp[0], sp[1], sp[2]]

    branches_to_plot = branches[: max_branches_to_plot]
    plot_branches(
        branches=branches_to_plot,
        seed_points=seed_points,
        future_layers=seed_row["future_layers"],
        hits_df=hits_df,
        truth_hits=truth_df,
        particle_id=pid,
    )

    if branches:
        lengths = [len(_branch_to_traj(b)) for b in branches]
        logging.info(
            "Particle %s: %d branches (plotting %d). Lengths: min=%d, max=%d, mean=%.1f",
            pid,
            len(branches),
            len(branches_to_plot),
            int(np.min(lengths)),
            int(np.max(lengths)),
            float(np.mean(lengths)),
        )
    else:
        logging.info("Particle %s: no branches to plot.", pid)


def _sorted_layer_keys(hits: pd.DataFrame) -> List[Tuple[int, int]]:
    r"""
    Return sorted unique layer keys as ``(volume_id, layer_id)`` pairs.

    Parameters
    ----------
    hits : pandas.DataFrame
        Input hits with integer columns ``volume_id`` and ``layer_id``.

    Returns
    -------
    list[tuple[int,int]]
        Lexicographically sorted unique pairs.

    Notes
    -----
    Sorting is performed via ``np.lexsort((layer, volume))`` for stability and speed.
    """
    vols = hits["volume_id"].to_numpy(copy=False)
    lays = hits["layer_id"].to_numpy(copy=False)
    keys = np.unique(np.stack([vols, lays], axis=1), axis=0)
    keys = keys[np.lexsort((keys[:, 1], keys[:, 0]))]
    return [(int(v), int(l)) for v, l in keys]


def _branch_to_traj(branch) -> np.ndarray:
    """
    Accept either:
      - dict with key 'traj' (list/array of (x,y,z))
      - object with attribute .trajectory
    Returns an (N,3) float64 array (possibly empty).
    """
    if isinstance(branch, dict):
        arr = branch.get("traj", None)
    else:
        arr = getattr(branch, "trajectory", None)
    if arr is None:
        return np.empty((0, 3), dtype=np.float64)
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return out[:, :3]

def plot_timing_summary(
    timing: "Mapping[str, float] | Sequence[Tuple[str, float]] | pd.Series",
    *,
    title: str = "Pipeline timing",
    annotate: bool = True,
) -> None:
    r'''
    Plot a compact pipeline timing summary as a horizontal bar chart.

    Given a mapping from pipeline stage names to wall-clock durations (in seconds),
    this function renders a single horizontal bar chart with one bar per stage.
    It is intended for quick, visual comparisons of where time is spent across the
    major phases of the TrackML pipeline (e.g., load+preprocess, surface inference,
    build, evaluation).

    Parameters
    ----------
    timing : Mapping[str, float]
        Dictionary-like object mapping stage names to durations in **seconds**.
        Non-finite or negative values are coerced to zero for display purposes.
    title : str or None, optional
        Figure title. If ``None`` (default), a generic title is used.
    show : bool, optional
        If ``True`` (default), call :func:`matplotlib.pyplot.show` after drawing.
        Set to ``False`` when you are saving the figure or composing into another
        layout.
    save_path : str or pathlib.Path or None, optional
        If provided, the figure is saved to this path (e.g., ``"timing.png"``) via
        :func:`matplotlib.figure.Figure.savefig`. The directory must exist.

    Returns
    -------
    None
        The figure is created for its side effects (optionally shown/saved).
        If you need programmatic access to the figure/axes, adapt the function to
        return them.

    Notes
    -----
    - Bars are ordered from **longest to shortest** duration to emphasize the
    dominant stages.
    - Durations are annotated at the end of each bar in seconds with two decimals.
    - The function relies on the **current Matplotlib backend**. In headless
    contexts, ensure an Agg-like backend is active (the main script already
    enforces this when plotting is disabled globally).
    '''
    # Normalize input into a list of (label, seconds)
    if isinstance(timing, pd.Series):
        items = [(str(k), float(v)) for k, v in timing.items()]
    elif isinstance(timing, dict):
        items = [(str(k), float(v)) for k, v in timing.items()]
    else:
        items = [(str(k), float(v)) for k, v in timing]

    items = [(k, v) for k, v in items if np.isfinite(v) and v > 0.0]
    if not items:
        return

    labels, secs = zip(*items)
    secs = np.asarray(secs, dtype=float)
    total = float(secs.sum())
    perc = (secs / total) * 100.0 if total > 0 else np.zeros_like(secs)

    # Figure size scales with number of stages
    fig_h = max(3.0, 0.55 * len(labels) + 1.25)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))

    bars = ax.barh(labels, secs, alpha=0.9)
    ax.invert_yaxis()  # top item first
    ax.set_xlabel("seconds")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)

    if annotate:
        for b, s, p in zip(bars, secs, perc):
            x = b.get_width()
            y = b.get_y() + b.get_height() / 2
            label = f"{s:.2f}s • {p:.1f}%"
            # place label inside the bar if there is space, else just outside
            if x >= 0.20 * secs.max():
                ax.text(x - 0.01 * secs.max(), y, label,
                        va="center", ha="right", color="white", fontsize=9, fontweight="bold")
            else:
                ax.text(x + 0.01 * secs.max(), y, label,
                        va="center", ha="left", color="black", fontsize=9)

    plt.tight_layout()
    _show_and_close(fig)
