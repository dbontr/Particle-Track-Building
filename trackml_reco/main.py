#!/usr/bin/env python3
r"""
TrackML track building runner (headless-safe, fast I/O, optimization-aware).

This script loads a TrackML event, constructs simple geometric **layer surfaces**
(disks/cylinders), configures one of several EKF-based branchers, builds tracks
(optionally in collaborative parallel mode), visualizes results (when enabled),
and evaluates against truth with the official TrackML score.

Mathematical conventions
------------------------
Hit positions are in meters, with coordinates :math:`(x,y,z)`. For a layer's hit
cloud :math:`\{q_j\}`, we define

- mean axial location :math:`\bar z = \frac{1}{N}\sum_j z_j`,
- transverse radii :math:`r_j=\sqrt{x_j^2+y_j^2}`, :math:`\bar r=\frac{1}{N}\sum_j r_j`,
- spans :math:`\Delta z = \max z_j - \min z_j`, :math:`\Delta r = \max r_j - \min r_j`.

A layer is treated as a **disk** (plane) if :math:`\Delta z < 0.1\,\Delta r`,
with plane normal :math:`n=(0,0,1)` and a point on the plane :math:`p=(0,0,\bar z)`.
Otherwise it is modeled as a **cylinder** of radius :math:`R=\bar r`.

The event score is the standard TrackML metric; per-track auxiliary metrics
(MSE, recall/precision/F1) are optionally reported for the top tracks.

CLI overview
------------
See :func:`build_parser` for all options. Typical usage:

.. code-block:: bash

   python run.py -f train_1.zip -b ekf --parallel --plot
   python run.py -f train_1.zip -b astar --optimize --opt-space param_space.json

The optimization mode performs parameter search for the selected brancher and
can optionally use the collaborative parallel builder to speed up evaluations.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
from trackml.score import score_event

# Branchers (pure compute — safe to import early)
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

# Local modules (compute-only; plotting imported lazily)
import trackml_reco.data as trk_data
import trackml_reco.metrics as trk_metrics
from trackml_reco.profiling import prof

# Optimizer (imported here to keep main self-contained)
from trackml_reco.optimize import optimize_brancher

try:
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None


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
    Construct the command-line interface for the TrackML runner.

    Returns
    -------
    argparse.ArgumentParser
        Parser with options for input data, plotting, branching strategy,
        profiling, and optional parameter optimization.

    Notes
    -----
    Key options:

    - ``--file``: input TrackML ``.zip`` event.
    - ``--pt``: :math:`p_T` (GeV) threshold used during preprocessing.
    - ``--brancher``: choice among ``{"ekf","astar","aco","pso","sa","ga","hungarian"}``.
    - ``--parallel``: enable collaborative parallel builder.
    - ``--plot``/``--no-plot``/``--extra-plots``: headless-safe plotting control.
    - Optimization block (``--opt-*``) enables parameter search via
      :func:`trackml_reco.optimize.optimize_brancher`. See that function for the
      precise meaning of ``--opt-metric``, search backend selection, and history output.
    """
    p = argparse.ArgumentParser(description="Run refactored track building on TrackML event(s).")
    p.add_argument(
        "-f", "--file", type=str, default="train_1.zip",
        help=(
            "Input TrackML .zip event, a directory containing *.zip, or a glob "
            "(e.g. data/train_*.zip). Default: train_1.zip"
        ),
    )
    p.add_argument(
        "-n", "--n-events", type=int, default=1,
        help=(
            "Number of events to run. When --file is a directory or glob, take the "
            "first N matches (natural order). When --file is a single zip, start at "
            "that file and continue through its siblings. Default: 1."
        ),
    )
    p.add_argument("-p", "--pt", type=float, default=2.0,
                   help="Minimum pT threshold in GeV (default: 2.0).")
    p.add_argument("-d", "--debug-n", type=int, default=None,
                   help="If set, only process this many seeds (default: None).")
    p.add_argument("--plot", action="store_true", default=False,
                   help="Show seed/track plots (default: False).")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="Disable plotting.")
    p.add_argument("--extra-plots", action="store_true", default=False,
                   help="Display extra presentation plots (default: False).")
    p.add_argument("--parallel", action="store_true", default=False,
                   help="Enable collaborative parallel track building (default: False).")
    p.add_argument("-b", "--brancher", type=str, choices=BRANCHER_KEYS, default="ekf",
                   metavar="BRANCHER",
                   help="Branching strategy: ekf | astar | aco | pso | sa | ga | hungarian (default: ekf)")
    p.add_argument("--config", type=str, default="config.json",
                   help="Path to JSON config with per-brancher settings (default: config.json).")
    p.add_argument("--profile", action="store_true", default=False,
                   help="Enable cProfile around the build+score phase.")
    p.add_argument("--profile-out", type=str, default=None, 
                   help="If set, write pstats text to this file.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable verbose logging.")

    # Optimization block
    p.add_argument("--optimize", action="store_true", default=False,
                   help="Run parameter optimization for the selected brancher instead of a single build.")
    p.add_argument("--opt-metric", type=str, default="mse",
                   help="Objective metric to minimize (e.g., mse, rmse, recall, precision, f1, pct_hits, score).")
    p.add_argument("--opt-iterations", type=int, default=40,
                   help="Number of optimization iterations (default: 40).")
    p.add_argument("--opt-n-init", type=int, default=10,
                   help="Number of initial random points for Bayesian search (default: 10).")
    p.add_argument("--opt-space", type=str, default="param_space.json",
                   help="Path to parameter space JSON.")
    p.add_argument("--opt-max-seeds", type=int, default=200,
                   help="Limit seeds per evaluation to speed up optimization (default: 200).")
    p.add_argument("--opt-parallel", action="store_true", default=False,
                   help="Use CollaborativeParallelTrackBuilder during optimization.")
    p.add_argument("--opt-parallel-time-budget", type=float, default=None,
                   help="Optional per-seed time budget (seconds) if --opt-parallel.")
    p.add_argument("--opt-seed", type=int, default=None,
                   help="Random seed for the optimizer.")
    p.add_argument("--opt-out", type=str, default=None,
                   help="If set, write best-tuned full config JSON to this path.")
    p.add_argument("--opt-history", type=str, default=None,
                   help="If set, write a CSV of trial history to this path.")
    p.add_argument("--opt-skopt-kind", type=str, default="auto",
                   choices=("auto", "gp", "forest", "gbrt", "random"),
                   help="Which skopt backend to use if available (default: auto).")
    p.add_argument("--opt-plot", type=str, default=None,
                   help="Path to save the optimization history plot (e.g. opt_progress.png).")
    p.add_argument("--opt-aggregator", type=str, default="mean",
                   choices=("mean", "median", "min", "max"),
                   help="Aggregator for per-track metrics during optimization (default: mean).")
    p.add_argument("--opt-n-best-tracks", type=int, default=0,
                   help="If >0, aggregate metric over the best N tracks; 0 uses ALL tracks (default: 0).")
    p.add_argument("--opt-tol", type=float, default=0.005,
                   help="Hit-matching tolerance for efficiency metrics (default: 0.005).")
    return p


def setup_logging(verbose: bool = False) -> None:
    r"""
    Configure process-wide logging.

    Parameters
    ----------
    verbose : bool, optional
        If ``True``, set level to ``DEBUG``; otherwise ``INFO``.

    Notes
    -----
    Format is ``'%(asctime)s | %(levelname)-8s | %(message)s'`` with ``%H:%M:%S`` timestamps.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def apply_plotting_guard(enable_plots: bool) -> None:
    r"""
    Enforce a **headless-safe** Matplotlib configuration when plotting is disabled.

    Must be called **before** importing any module that might import
    :mod:`matplotlib.pyplot`.

    Parameters
    ----------
    enable_plots : bool
        If ``False``, set backend to ``'Agg'`` (non-interactive), turn off
        interactive mode, and neutralize ``plt.show()``.
    """
    if enable_plots:
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt  # noqa: WPS433
    _plt.ioff()
    _plt.show = lambda *a, **k: None  # type: ignore[assignment]


def compute_layer_surfaces(hits: pd.DataFrame) -> Dict[Tuple[int, int], Mapping[str, float | np.ndarray]]:
    r"""
    Infer simple **disk/cylinder** surfaces for each ``(volume_id, layer_id)``.

    For each layer, compute spans :math:`\Delta z` and :math:`\Delta r` and apply

    .. math::

        \text{disk if}\quad \Delta z < 0.1\,\Delta r,\qquad
        \text{else cylinder}.

    Disk parameters are :math:`n=(0,0,1)` and :math:`p=(0,0,\bar z)`. Cylinder
    radius is :math:`R=\bar r`.

    Parameters
    ----------
    hits : pandas.DataFrame
        Must contain columns ``x,y,z,volume_id,layer_id``.

    Returns
    -------
    dict[(int,int) -> dict]
        Each value is either ``{'type':'disk','n':(3,), 'p':(3,)}`` or
        ``{'type':'cylinder','R': float}``.

    Notes
    -----
    This **coarse** geometric model is adequate for gating and surface-to-surface
    propagation in the branchers; it does not attempt to model detector thickness
    or segmentation.
    """
    surfaces: Dict[Tuple[int, int], Mapping[str, float | np.ndarray]] = {}
    for (vol, lay), df in hits.groupby(["volume_id", "layer_id"], sort=False):
        if df.empty:
            continue
        z_vals = df["z"].to_numpy()
        z_span = float(z_vals.max() - z_vals.min())
        r = np.hypot(df["x"].to_numpy(), df["y"].to_numpy())
        r_span = float(r.max() - r.min())
        if z_span < 0.1 * r_span:
            surfaces[(vol, lay)] = {
                "type": "disk",
                "n": np.array([0.0, 0.0, 1.0]),
                "p": np.array([0.0, 0.0, float(z_vals.mean())]),
            }
        else:
            surfaces[(vol, lay)] = {"type": "cylinder", "R": float(r.mean())}
    return surfaces


def load_config(config_path: Path) -> MutableMapping[str, dict]:
    r"""
    Load a JSON configuration with optional :mod:`orjson` acceleration.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    ValueError
        If the file cannot be parsed.
    """
    try:
        if _orjson is not None:
            return _orjson.loads(config_path.read_bytes())
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to parse {config_path}: {e}") from e


def inject_layer_surfaces_into_configs(
    cfg: MutableMapping[str, dict],
    surfaces: Mapping[Tuple[int, int], Mapping[str, float | np.ndarray]],
) -> None:
    r"""
    Attach ``layer_surfaces`` into any present brancher configuration blocks.

    Parameters
    ----------
    cfg : dict
        Global configuration dictionary (possibly containing multiple brancher blocks).
    surfaces : mapping
        Output of :func:`compute_layer_surfaces`.

    Notes
    -----
    The following keys are recognized and updated in place if present:

    ``"ekf_config"``, ``"ekfastar_config"``, ``"ekfaco_config"``,
    ``"ekfpso_config"``, ``"ekfsa_config"``, ``"ekfga_config"``, ``"ekfhungarian_config"``.
    """
    for key in (
        "ekf_config",
        "ekfastar_config",
        "ekfaco_config",
        "ekfpso_config",
        "ekfsa_config",
        "ekfga_config",
        "ekfhungarian_config",
    ):
        block = cfg.get(key)
        if isinstance(block, dict):
            block["layer_surfaces"] = surfaces


def resolve_brancher_config_key(brancher: str) -> str:
    r"""
    Map a brancher name to its configuration block key.

    Parameters
    ----------
    brancher : {"ekf","astar","aco","pso","sa","ga","hungarian"}

    Returns
    -------
    str
        e.g. ``"ekf_config"`` for ``"ekf"``, or ``"ekf{brancher}_config"`` otherwise.
    """
    return "ekf_config" if brancher == "ekf" else f"ekf{brancher}_config"


def _deep_update(d: dict, u: dict) -> dict:
    r"""
    Recursively merge dictionaries (without side effects).

    Parameters
    ----------
    d : dict
        Base dictionary.
    u : dict
        Overrides (recursively merged).

    Returns
    -------
    dict
        New dictionary where nested dicts are merged and scalars/containers from
        ``u`` replace those in ``d``.
    """
    out = dict(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _natural_key(path: Path):
    """Natural sort key (split digits) so train_2 comes before train_10."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def _resolve_event_paths(file_arg: str, n_events: int) -> List[Path]:
    """
    Turn --file into a list of up to n_events .zip paths.

    Supports:
      - single zip path
      - directory (we take *.zip inside)
      - glob pattern with * ? [ ]

    For a single zip with n_events>1, we continue through siblings in the same
    directory (natural order) starting at the provided file.
    """
    n = max(1, int(n_events))
    s = file_arg
    p = Path(s)

    # glob pattern?
    if any(ch in s for ch in "*?[]"):
        cands = sorted((Path(x) for x in glob(s)), key=_natural_key)
        return cands[:n]

    # directory?
    if p.is_dir():
        cands = sorted(p.glob("*.zip"), key=_natural_key)
        return cands[:n]

    # single file
    if p.is_file():
        sibs = sorted(p.parent.glob("*.zip"), key=_natural_key)
        if p in sibs:
            i = sibs.index(p)
            return sibs[i:i+n]
        return [p]  # fall back
    # otherwise just return as provided (will error later if invalid)
    return [p]


def main() -> None:
    r"""
    End-to-end TrackML pipeline: **load → preprocess → geometry → branch → build → score**.

    Pipeline
    --------
    1. Parse CLI (:func:`build_parser`) and set up logging (:func:`setup_logging`).
    2. Enforce headless plotting guard (:func:`apply_plotting_guard`).
    3. Load & preprocess data (:func:`trackml_reco.data.load_and_preprocess`).
    4. Build per-layer surfaces (:func:`compute_layer_surfaces`) and optionally plot.
    5. Load config (:func:`load_config`), inject surfaces into brancher blocks.
    6. **Two operating modes**:

       - **Optimization mode** (``--optimize``): run
         :func:`trackml_reco.optimize.optimize_brancher` over a parameter space;
         optionally write best config and trial history.
       - **Build mode**: instantiate the requested brancher and builder
         (parallel or single-process) and construct tracks.

    7. Optionally plot seeds / top tracks, then evaluate top tracks with
       :func:`trackml.score.score_event` and auxiliary metrics.

    Notes
    -----
    - The **collaborative parallel builder** shares a hit pool across workers so
      that reservations can be coordinated; this improves wall-clock without
      changing model semantics.
    - The event score is reported on a deduplicated submission built from the
      best tracks; auxiliary metrics (MSE, recall/precision/F1) are logged for
      the top :math:`N` tracks for quick quality feedback.

    See Also
    --------
    trackml_reco.branchers : EKF branchers and search strategies.
    trackml_reco.optimize.optimize_brancher : Parameter search driver.
    """
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Resolve events
    event_paths = _resolve_event_paths(args.file, args.n_events)
    if not event_paths:
        raise FileNotFoundError(f"No events found for --file={args.file}")
    if len(event_paths) > 1:
        logging.info("Running on %d events. First: %s", len(event_paths), event_paths[0].name)
    else:
        logging.info("Running on event: %s", event_paths[0].name)

    # If optimizing across many events, keep API unchanged for now: use first only.
    if args.optimize and len(event_paths) > 1:
        logging.warning("Optimize mode uses only the first event (%s); ignoring %d others.",
                        event_paths[0].name, len(event_paths) - 1)
        event_paths = event_paths[:1]

    apply_plotting_guard(args.plot or args.extra_plots)

    # Load config once
    cfg_path = Path(args.config)
    logging.info("Reading config from %s", cfg_path)
    config = load_config(cfg_path)

    # --------- OPTIMIZE MODE ---------
    if args.optimize:
        # Load first event for optimization pipeline
        logging.info("Loading & preprocessing data for optimization: %s", event_paths[0].name)
        hit_pool = trk_data.load_and_preprocess(str(event_paths[0]), pt_threshold=args.pt)

        logging.info("Inferring layer surfaces...")
        layer_surfaces = compute_layer_surfaces(hit_pool.hits)
        inject_layer_surfaces_into_configs(config, layer_surfaces)

        brancher_key = args.brancher
        brancher_config_key = resolve_brancher_config_key(brancher_key)
        if brancher_config_key not in config:
            raise KeyError(f"Missing '{brancher_config_key}' in {cfg_path}. "
                           f"Available keys: {', '.join(config.keys())}")

        # hard-disable any plotting during optimization
        plots_prev = (args.plot, args.extra_plots)
        args.plot, args.extra_plots = False, False
        apply_plotting_guard(False)

        logging.info("==== Optimization mode ON ====")
        logging.info("Brancher: %s  |  Metric: %s", brancher_key, args.opt_metric)

        result = optimize_brancher(
            hit_pool=hit_pool,
            base_config=config,
            brancher_key=brancher_key,
            space_path=Path(args.opt_space),
            metric=args.opt_metric,
            iterations=args.opt_iterations,
            n_init=args.opt_n_init,
            use_parallel=args.opt_parallel,
            parallel_time_budget=args.opt_parallel_time_budget,
            max_seeds=args.opt_max_seeds,
            rng_seed=args.opt_seed,
            skopt_kind=args.opt_skopt_kind,
            aggregator=args.opt_aggregator,
            n_best_tracks=args.opt_n_best_tracks,
            tol=args.opt_tol,
            plot_path=Path(args.opt_plot) if args.opt_plot else None,
        )

        logging.info("Best objective (lower is better): %.6f", result.best_value)
        logging.info("Best params: %s", result.best_params)

        # write best full-config if requested
        brancher_config_key = resolve_brancher_config_key(args.brancher)
        if args.opt_out:
            out_cfg = copy.deepcopy(config)
            out_cfg[brancher_config_key] = _deep_update(
                out_cfg.get(brancher_config_key, {}),
                result.best_params
            )
            Path(args.opt_out).write_text(json.dumps(out_cfg, indent=2))
            logging.info("Wrote best config to %s", args.opt_out)

        # write history if requested
        if args.opt_history:
            rows = []
            for h in result.history:
                row = {"value": h["value"], "time_s": h["time_s"]}
                for k, v in h["params"].items():
                    row[f"p::{k}"] = v
                rows.append(row)
            pd.DataFrame(rows).to_csv(args.opt_history, index=False)
            logging.info("Wrote optimization history to %s", args.opt_history)

        # restore flags and exit (skip normal build)
        args.plot, args.extra_plots = plots_prev
        return

    # --------- BUILD MODE (possibly multiple events) ---------
    brancher_key = args.brancher
    brancher_cls = BRANCHER_MAP[brancher_key]
    brancher_config_key = resolve_brancher_config_key(brancher_key)
    if brancher_config_key not in config:
        raise KeyError(f"Missing '{brancher_config_key}' in {cfg_path}. "
                       f"Available keys: {', '.join(config.keys())}")

    # Aggregates across events
    scores_all: List[float] = []
    mses_all: List[float] = []
    pct_hits_all: List[float] = []

    for idx, ev_path in enumerate(event_paths, start=1):
        logging.info("=== Event %d/%d: %s ===", idx, len(event_paths), ev_path.name)

        # ---- timing: start of event
        t_load0 = time.time()

        logging.info("Loading & preprocessing data...")
        hit_pool = trk_data.load_and_preprocess(str(ev_path), pt_threshold=args.pt)
        hits, pt_cut_hits = hit_pool.hits, hit_pool.pt_cut_hits
        t_load1 = time.time()

        logging.info("Inferring layer surfaces...")
        layer_surfaces = compute_layer_surfaces(hits)
        t_surf1 = time.time()

        # Lazy import plotting only on the FIRST event to avoid a flood of windows
        if idx == 1 and (args.plot or args.extra_plots):
            import trackml_reco.plotting as trk_plot  # noqa: WPS433
            trk_plot.plot_extras(hits, pt_cut_hits, enabled=args.extra_plots)

        # Attach geometry to any present brancher configs (copy brancher block)
        cfg_local = copy.deepcopy(config)
        inject_layer_surfaces_into_configs(cfg_local, layer_surfaces)

        # Choose builder
        builder_cls = CollaborativeParallelTrackBuilder if args.parallel else TrackBuilder
        logging.info(
            "Building with brancher=%s (%s), parallel=%s",
            brancher_key, brancher_cls.__name__, args.parallel,
        )
        brancher_cfg = dict(cfg_local[brancher_config_key])  # copy; we may read defaults
        ekf_defaults = cfg_local.get("ekf_config", {})
        num_branches = int(brancher_cfg.get("num_branches", ekf_defaults.get("num_branches", 30)))
        survive_top = int(brancher_cfg.get("survive_top", ekf_defaults.get("survive_top", 12)))

        track_builder = builder_cls(
            hit_pool=hit_pool,
            brancher_cls=brancher_cls,
            brancher_config=brancher_cfg,
        )

        # Build tracks
        t_build0 = time.time()
        with prof(args.profile, out_path=args.profile_out):
            logging.info("Building seeds & tracks from truth hits...")
            completed_tracks = track_builder.build_tracks_from_truth(
                max_seeds=args.debug_n,
                max_tracks_per_seed=num_branches,
                max_branches=survive_top,
            )
        t_build1 = time.time()

        # Seed plots (only for first event)
        if idx == 1 and args.plot:
            import trackml_reco.plotting as trk_plot  # noqa: WPS433
            trk_plot.plot_seeds(track_builder, show=True, max_seeds=args.debug_n)

        # Stats
        stats = track_builder.get_track_statistics()
        logging.info("Track building statistics:")
        for k, v in stats.items():
            logging.info("  %s: %s", k, v)

        # Evaluate best tracks
        t_eval0 = time.time()
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

            traj = np.asarray(track.trajectory, dtype=float)
            truth_xyz = truth_particle[["x", "y", "z"]].to_numpy(dtype=float, copy=False)

            m = trk_metrics.compute_metrics(
                traj, truth_xyz, tol=0.005,
                names=("mse", "recall", "precision", "f1")
            )
            mse, recall, precision, f1 = trk_metrics.unpack(m, "mse", "recall", "precision", "f1")

            mses.append(mse)
            pct_hits_list.append(recall)  # recall (%) == your “% hits”

            logging.info(
                "Track %d (PID=%s): MSE=%.3f | recall=%.1f%% | precision=%.1f%% | F1=%.1f%%",
                i, track.particle_id, mse, recall, precision, f1
            )

            for hid in track.hit_ids:
                submission_rows.append({"hit_id": int(hid), "track_id": int(track.particle_id)})

            if idx == 1 and args.plot and i <= 3:
                import trackml_reco.plotting as trk_plot  # noqa: WPS433
                trk_plot.plot_best_track_3d(track_builder, track, truth_particle, i)

        # If nothing built, still finalize timing and (optionally) show timing plot
        if not submission_rows:
            t_eval1 = time.time()
            # Timing summary (first event only to avoid many windows)
            if idx == 1 and args.plot:
                import trackml_reco.plotting as trk_plot  # noqa: WPS433
                timing = {
                    "load+preprocess": t_load1 - t_load0,
                    "layer surfaces":   t_surf1 - t_load1,
                    "build":            t_build1 - t_build0,
                    "evaluation":       t_eval1 - t_eval0,
                }
                trk_plot.plot_timing_summary(timing, title=f"Pipeline timing ({ev_path.name})")
            logging.warning("No tracks were successfully built for event %s.", ev_path.name)
            continue

        submission_df = pd.DataFrame(submission_rows).drop_duplicates("hit_id")

        score = score_event(
            pt_cut_hits[["hit_id", "particle_id", "weight"]],
            submission_df,
        )
        t_eval1 = time.time()

        if mses:
            logging.info("Event %s: Average MSE over %d tracks: %.3f", ev_path.name, len(mses), float(np.mean(mses)))
            mses_all.append(float(np.mean(mses)))
        if pct_hits_list:
            logging.info("Event %s: Average %%hits over %d tracks: %.1f%%",
                         ev_path.name, len(pct_hits_list), float(np.mean(pct_hits_list)))
            pct_hits_all.append(float(np.mean(pct_hits_list)))
        logging.info("Event %s: Score: %s", ev_path.name, score)
        scores_all.append(float(score))

        # Timing summary (first event only to avoid many windows)
        if idx == 1 and args.plot:
            import trackml_reco.plotting as trk_plot  # noqa: WPS433
            timing = {
                "load+preprocess": t_load1 - t_load0,
                "layer surfaces":   t_surf1 - t_load1,
                "build":            t_build1 - t_build0,
                "evaluation":       t_eval1 - t_eval0,
            }
            trk_plot.plot_timing_summary(timing, title=f"Pipeline timing ({ev_path.name})")

    # Aggregate across events
    if len(scores_all) > 1:
        logging.info("=== Aggregate over %d events ===", len(scores_all))
        logging.info("Mean score: %.6f", float(np.mean(scores_all)))
        if mses_all:
            logging.info("Mean of per-event avg MSE: %.6f", float(np.mean(mses_all)))
        if pct_hits_all:
            logging.info("Mean of per-event avg %%hits: %.3f%%", float(np.mean(pct_hits_all)))