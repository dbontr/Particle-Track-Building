#!/usr/bin/env python3
"""
TrackML: Refactored track building runner.

This script loads a TrackML event, constructs geometric layer surfaces,
configures a chosen branching strategy, builds tracks (optionally in
parallel/collaborative mode), visualizes results, and evaluates against truth.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trackml.score import score_event

# Branchers
from trackml_reco.branchers.ekf import HelixEKFBrancher
from trackml_reco.branchers.astar import HelixEKFAStarBrancher
from trackml_reco.branchers.aco import HelixEKFACOBrancher
from trackml_reco.branchers.pso import HelixEKFPSOBrancher
from trackml_reco.branchers.sa import HelixEKFSABrancher
from trackml_reco.branchers.ga import HelixEKFGABrancher
from trackml_reco.branchers.hungarian import HelixEKFHungarianBrancher

# Builders
from trackml_reco.track_builder import TrackBuilder
from trackml_reco.parallel_track_builder import CollaborativeParallelTrackBuilder

# Local modules
import trackml_reco.data as trk_data
import trackml_reco.plotting as trk_plot
import trackml_reco.metrics as trk_metrics

BRANCHER_KEYS: Tuple[str, ...] = ("ekf", "astar", "aco", "pso", "sa", "ga", "hungarian")
BRANCHER_MAP = {
    "ekf": HelixEKFBrancher,
    "astar": HelixEKFAStarBrancher,
    "aco": HelixEKFACOBrancher,
    "pso": HelixEKFPSOBrancher,
    "sa": HelixEKFSABrancher,
    "ga": HelixEKFGABrancher,
    "hungarian": HelixEKFHungarianBrancher,
}

def build_parser() -> argparse.ArgumentParser:
    r"""
    Create the CLI argument parser for the TrackML track-building runner.

    The parser exposes controls for input files, transverse-momentum threshold
    :math:`p_T`, brancher selection, plotting, parallel mode, and config paths.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with the following options (abridged):

        - ``-f/--file``: input TrackML ``.zip`` event (default: ``train_1.zip``)
        - ``-p/--pt``: minimum \(p_T\) in GeV (default: ``2.0``)
        - ``-d/--debug-n``: limit number of seeds (default: ``None``)
        - ``--plot / --no-plot``: enable/disable plotting (default: enabled)
        - ``--extra-plots``: show extra presentation plots (default: disabled)
        - ``--parallel``: collaborative parallel builder (default: disabled)
        - ``-b/--brancher``: one of ``{'ekf','astar','aco','pso','sa','ga','hungarian'}``
        - ``--config``: JSON config path (default: ``config.json``)
        - ``-v/--verbose``: verbose logging
    """
    p = argparse.ArgumentParser(
        description="Run refactored track building on a TrackML event."
    )
    p.add_argument(
        "-f",
        "--file",
        type=str,
        default="train_1.zip",
        help="Input TrackML .zip event (default: train_1.zip).",
    )
    p.add_argument(
        "-p",
        "--pt",
        type=float,
        default=2.0,
        help="Minimum pT threshold in GeV (default: 2.0).",
    )
    p.add_argument(
        "-d",
        "--debug-n",
        type=int,
        default=None,
        help="If set, only process this many seeds (default: None).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Show seed/track plots (default: True).",
    )
    p.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable plotting.",
    )
    p.add_argument(
        "--extra-plots",
        action="store_true",
        default=False,
        help="Display extra presentation plots (default: False).",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable collaborative parallel track building (default: False).",
    )
    p.add_argument(
        "-b",
        "--brancher",
        type=str,
        choices=BRANCHER_KEYS,
        default="ekf",
        metavar="BRANCHER",
        help=(
            "Branching strategy:\n"
            "  ekf        - Extended Kalman Filter branching\n"
            "  astar      - A* search-based branching\n"
            "  aco        - Ant Colony Optimization-based branching\n"
            "  pso        - Particle Swarm Optimization-based branching\n"
            "  sa         - Simulated Annealing-based branching\n"
            "  ga         - Genetic Algorithm-based branching\n"
            "  hungarian  - Hungarian assignment for hit-to-track\n"
            "(default: ekf)"
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config with per-brancher settings (default: config.json).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    return p


def setup_logging(verbose: bool = False) -> None:
    r"""
    Configure Python logging for the runner.

    Parameters
    ----------
    verbose : bool, optional
        If ``True``, set level to ``DEBUG``; otherwise ``INFO``. Default ``False``.

    Returns
    -------
    None
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

def compute_layer_surfaces(hits: pd.DataFrame) -> Dict[Tuple[int, int], Mapping[str, float | np.ndarray]]:
    r"""
    Infer simple **disk** or **cylinder** surfaces per layer ``(volume_id, layer_id)``.

    For each layer, we compare the axial spread :math:`\Delta z` versus the radial
    spread :math:`\Delta r`. If :math:`\Delta z < 0.1\,\Delta r`, we classify as a
    disk at :math:`z \approx \bar{z}`; otherwise as a cylinder at radius
    :math:`R \approx \bar{r}` where :math:`r=\sqrt{x^2+y^2}`.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table with at least columns
        ``['x', 'y', 'z', 'volume_id', 'layer_id']``.

    Returns
    -------
    dict
        Mapping ``(volume_id, layer_id) → surface``:
        - Disk: ``{'type': 'disk', 'n': [0,0,1], 'p': [0,0, z_mean]}``
        - Cylinder: ``{'type': 'cylinder', 'R': mean_radius}``

    Notes
    -----
    This heuristic is intentionally lightweight and robust for building
    first-order geometric constraints for propagation.
    """
    layer_keys = sorted(set(zip(hits.volume_id, hits.layer_id)), key=lambda x: (x[0], x[1]))
    surfaces: Dict[Tuple[int, int], Mapping[str, float | np.ndarray]] = {}

    for vol, lay in layer_keys:
        df = hits[(hits.volume_id == vol) & (hits.layer_id == lay)]
        if df.empty:
            continue

        z_span = float(df.z.max() - df.z.min())
        r_vals = np.sqrt(df.x.values ** 2 + df.y.values ** 2)
        r_span = float(r_vals.max() - r_vals.min())

        if z_span < 0.1 * r_span:
            # Disk-like
            surfaces[(vol, lay)] = {
                "type": "disk",
                "n": np.array([0.0, 0.0, 1.0]),
                "p": np.array([0.0, 0.0, float(df.z.mean())]),
            }
        else:
            # Cylinder-like
            surfaces[(vol, lay)] = {
                "type": "cylinder",
                "R": float(r_vals.mean()),
            }

    return surfaces


def load_config(config_path: Path) -> MutableMapping[str, dict]:
    r"""
    Load a JSON configuration file for brancher settings.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to a JSON file, e.g., containing keys like
        ``'ekf_config'``, ``'ekfastar_config'``, etc.

    Returns
    -------
    dict
        Parsed JSON object (dict-like).

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the file cannot be parsed as valid JSON.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {config_path}: {e}") from e
    return cfg


def inject_layer_surfaces_into_configs(
    cfg: MutableMapping[str, dict],
    surfaces: Mapping[Tuple[int, int], Mapping[str, float | np.ndarray]],
    ) -> None:
    r"""
    Attach ``layer_surfaces`` into any known brancher config blocks that exist.

    Parameters
    ----------
    cfg : mutable mapping
        Parsed configuration (dict-like) that may include
        ``'ekf_config'``, ``'ekfastar_config'``, ``'ekfaco_config'``,
        ``'ekfpso_config'``, ``'ekfsa_config'``, ``'ekfga_config'``,
        ``'ekfhungarian_config'``.
    surfaces : mapping
        Output of :func:`compute_layer_surfaces`, mapping
        ``(volume_id, layer_id)`` → surface dict.

    Returns
    -------
    None

    Notes
    -----
    The function is idempotent: missing keys are skipped with a warning.
    """
    candidate_keys = [
        "ekf_config",
        "ekfastar_config",
        "ekfaco_config",
        "ekfpso_config",
        "ekfsa_config",
        "ekfga_config",
        "ekfhungarian_config",
    ]
    attached = []
    for k in candidate_keys:
        if k in cfg and isinstance(cfg[k], dict):
            cfg[k]["layer_surfaces"] = surfaces
            attached.append(k)

    if not attached:
        logging.warning("No known brancher config keys found to attach layer surfaces.")


def resolve_brancher_config_key(brancher: str) -> str:
    r"""
    Map a brancher short name to its JSON configuration key.

    Parameters
    ----------
    brancher : str
        One of ``{'ekf','astar','aco','pso','sa','ga','hungarian'}``.

    Returns
    -------
    str
        Configuration key, e.g.:
        ``'ekf' → 'ekf_config'``, ``'ga' → 'ekfga_config'``.

    Examples
    --------
    >>> resolve_brancher_config_key('ekf')
    'ekf_config'
    >>> resolve_brancher_config_key('hungarian')
    'ekfhungarian_config'
    """
    return "ekf_config" if brancher == "ekf" else f"ekf{brancher}_config"


def main() -> None:
    r"""
    CLI entry point for TrackML track building.

    Workflow
    --------
    1. Parse CLI arguments and configure logging.
    2. Load and preprocess the event; obtain :math:`p_T`-filtered hits.
    3. Infer layer surfaces (disk/cylinder) from hit geometry.
    4. Load JSON config and inject ``layer_surfaces`` where relevant.
    5. Instantiate the chosen brancher (EKF/A*/ACO/PSO/SA/GA/Hungarian).
    6. Build tracks (optionally collaborative/parallel).
    7. Plot seeds and (optionally) top tracks.
    8. Compute statistics and score against truth with :func:`trackml.score.score_event`.

    Returns
    -------
    None

    Notes
    -----
    * The score is computed from the deduplicated submission mapping
      ``{'hit_id' → 'track_id'}`` built from the best tracks.
    * When ``--parallel`` is set, the
      :class:`~trackml_reco.parallel_track_builder.CollaborativeParallelTrackBuilder`
      is used instead of the default :class:`~trackml_reco.track_builder.TrackBuilder`.
    """
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    logging.info("Loading & preprocessing data...")
    hit_pool = trk_data.load_and_preprocess(args.file, pt_threshold=args.pt)
    hits, pt_cut_hits = hit_pool.hits, hit_pool.pt_cut_hits

    logging.info("Inferring layer surfaces...")
    layer_surfaces = compute_layer_surfaces(hits)

    trk_plot.plot_extras(hits, pt_cut_hits, enabled=args.extra_plots)

    # Load config
    cfg_path = Path(args.config)
    logging.info("Reading config from %s", cfg_path)
    config = load_config(cfg_path)

    # Attach geometry to any present brancher configs
    inject_layer_surfaces_into_configs(config, layer_surfaces)

    # Select brancher & config section
    brancher_key = args.brancher
    brancher_cls = BRANCHER_MAP[brancher_key]
    brancher_config_key = resolve_brancher_config_key(brancher_key)

    if brancher_config_key not in config:
        raise KeyError(
            f"Missing '{brancher_config_key}' in {cfg_path}. "
            f"Available keys: {', '.join(config.keys())}"
        )

    builder_cls = CollaborativeParallelTrackBuilder if args.parallel else TrackBuilder
    logging.info(
        "Building with brancher=%s (%s), parallel=%s",
        brancher_key,
        brancher_cls.__name__,
        args.parallel,
    )
    track_builder = builder_cls(
        hit_pool=hit_pool,
        brancher_cls=brancher_cls,
        brancher_config=config[brancher_config_key],
    )

    # Build tracks
    logging.info("Building seeds & tracks from truth hits...")
    completed_tracks = track_builder.build_tracks_from_truth(
        max_seeds=args.debug_n,
        max_tracks_per_seed=config["ekf_config"]["num_branches"],  # matches original behavior
        max_branches=config["ekf_config"]["survive_top"],
    )

    # Seed plots
    trk_plot.plot_seeds(track_builder, show=args.plot, max_seeds=args.debug_n)

    # Stats
    stats = track_builder.get_track_statistics()
    logging.info("Track building statistics:")
    for k, v in stats.items():
        logging.info("  %s: %s", k, v)

    # Evaluate best tracks
    n_best = min(10, len(completed_tracks))
    best_tracks = track_builder.get_best_tracks(n=n_best)
    logging.info("Evaluating best %d tracks...", n_best)

    submission_rows: List[dict] = []
    mses: List[float] = []
    pct_hits_list: List[float] = []

    for i, track in enumerate(best_tracks, start=1):
        truth_particle = pt_cut_hits[pt_cut_hits.particle_id == track.particle_id]
        if truth_particle.empty:
            continue

        traj = np.asarray(track.trajectory)
        truth_xyz = truth_particle[["x", "y", "z"]].values

        mse = trk_metrics.branch_mse({"traj": traj}, truth_xyz)
        pct_hits, _ = trk_metrics.branch_hit_stats({"traj": traj}, truth_xyz)
        mses.append(float(mse))
        pct_hits_list.append(float(pct_hits))

        logging.info("Track %d (PID=%s): MSE=%.3f, %%hits=%.1f%%",
                     i, track.particle_id, mse, pct_hits)

        for hid in track.hit_ids:
            submission_rows.append({"hit_id": int(hid), "track_id": int(track.particle_id)})

        if args.plot and i <= 3:
            trk_plot.plot_best_track_3d(track_builder, track, truth_particle, i)

    if not submission_rows:
        logging.warning("No tracks were successfully built.")
        return

    submission_df = pd.DataFrame(submission_rows).drop_duplicates("hit_id")

    score = score_event(
        pt_cut_hits[["hit_id", "particle_id", "weight"]],
        submission_df,
    )

    if mses:
        logging.info("Average MSE over %d tracks: %.3f", len(mses), float(np.mean(mses)))
    if pct_hits_list:
        logging.info("Average %%hits over %d tracks: %.1f%%", len(pct_hits_list), float(np.mean(pct_hits_list)))
    logging.info("Event score: %s", score)


if __name__ == "__main__":
    main()
