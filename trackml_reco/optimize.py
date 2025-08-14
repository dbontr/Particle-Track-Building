from __future__ import annotations

import copy
import json
import math
import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from trackml_reco.track_builder import TrackBuilder
from trackml_reco.parallel_track_builder import CollaborativeParallelTrackBuilder
from trackml_reco.branchers.ekf import HelixEKFBrancher
from trackml_reco.branchers.astar import HelixEKFAStarBrancher
from trackml_reco.branchers.aco import HelixEKFACOBrancher
from trackml_reco.branchers.pso import HelixEKFPSOBrancher
from trackml_reco.branchers.sa import HelixEKFSABrancher
from trackml_reco.branchers.ga import HelixEKFGABrancher
from trackml_reco.branchers.hungarian import HelixEKFHungarianBrancher
import trackml_reco.metrics as trk_metrics

logger = logging.getLogger(__name__)

BRANCHER_MAP = {
    "ekf": HelixEKFBrancher,
    "astar": HelixEKFAStarBrancher,
    "aco": HelixEKFACOBrancher,
    "pso": HelixEKFPSOBrancher,
    "sa": HelixEKFSABrancher,
    "ga": HelixEKFGABrancher,
    "hungarian": HelixEKFHungarianBrancher,
}

def _resolve_cfg_key(brancher_key: str) -> str:
    r"""
    Resolve a human-facing brancher key to its configuration block name.

    Parameters
    ----------
    brancher_key : {"ekf","astar","aco","pso","sa","ga","hungarian"}
        Short identifier selected on the CLI or in configs.

    Returns
    -------
    str
        Name of the JSON config block that should contain parameters for the
        chosen brancher. By convention the base EKF uses ``"ekf_config"``,
        while other branchers use ``"ekf{brancher}_config"``.

    Examples
    --------
    >>> _resolve_cfg_key("ekf")
    'ekf_config'
    >>> _resolve_cfg_key("pso")
    'ekfpso_config'
    """
    return "ekf_config" if brancher_key == "ekf" else f"ekf{brancher_key}_config"

def _deep_update(d: dict, u: dict) -> dict:
    r"""
    Recursively merge two (possibly nested) dictionaries without side effects.

    For keys where both values are dict-like, the merge recurses; otherwise the
    value from ``u`` overwrites that in ``d``.

    Parameters
    ----------
    d : dict
        Base dictionary (not modified).
    u : dict
        Update dictionary whose values take precedence.

    Returns
    -------
    dict
        New merged dictionary with the same nesting as the inputs.

    Notes
    -----
    This function is used to **overlay** per-trial parameter proposals on top of
    a base configuration without mutating either object.
    """
    out = dict(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _param_preview(pdict: Mapping[str, Any], limit: int = 6) -> str:
    r"""
    Format a compact string preview of a parameter dictionary.

    Parameters
    ----------
    pdict : mapping
        Parameter dictionary to preview.
    limit : int, optional
        Maximum number of key–value pairs to include (default 6); an ellipsis
        is appended if more keys exist.

    Returns
    -------
    str
        Human-readable preview like ``"a=1, b=2, c=3, …"``.
    """
    keys = sorted(pdict.keys())
    shown = ", ".join(f"{k}={pdict[k]}" for k in keys[:limit])
    if len(keys) > limit:
        shown += ", …"
    return shown

# For per-track metrics, which ones are "higher is better"?
_HIGHER_IS_BETTER = {
    "recall", "precision", "f1", "f1score", "accuracy",
    "pct_hits", "percent_hits", "percent_matched", "efficiency",
}
# Simple aliases/synonyms -> canonical metric key produced by compute_metrics
_METRIC_ALIASES = {
    "f1score": "f1",
    "percent_hits": "recall",
    "percent_matched": "recall",
    "pct_hits": "recall",     # keep old behavior
    "%hits": "recall",
}

def _normalize_metric_name(metric: str) -> str:
    r"""
    Canonicalize a metric name by lowercasing and applying known aliases.

    Parameters
    ----------
    metric : str
        Metric name such as ``"MSE"`` or ``"percent_hits"``.

    Returns
    -------
    str
        Canonical key used internally, e.g. ``"mse"`` or ``"recall"``.
    """
    m = metric.strip().lower()
    return _METRIC_ALIASES.get(m, m)

def _loss_from_value(metric: str, value: float) -> float:
    r"""
    Convert a metric value into a **loss** suitable for *minimization*.

    For metrics where **lower is better** (e.g., MSE/RMSE), the loss is the
    value itself:

    .. math::
        L(v;\,\text{mse}) \;=\; v.

    For metrics where **higher is better** (recall, precision, F1, etc.),
    the loss is the *complement*:

    .. math::
        L(v) \;=\;
        \begin{cases}
            1 - v, & \text{if } 0 \le v \le 1 \text{ (fraction)}\\[3pt]
            100 - v, & \text{otherwise (percentage)}
        \end{cases}

    Parameters
    ----------
    metric : str
        Metric name (canonicalized by :func:`_normalize_metric_name`).
    value : float
        Observed metric value.

    Returns
    -------
    float
        Loss to minimize.
    """
    m = _normalize_metric_name(metric)
    if m in ("mse", "rmse"):
        return float(value)
    if m in _HIGHER_IS_BETTER:
        v = float(value)
        # Guess scale: if looks like a fraction, use 1-v; otherwise assume percent.
        if 0.0 <= v <= 1.0:
            return 1.0 - v
        return 100.0 - v
    # Default: assume lower-is-better
    return float(value)

@dataclass(frozen=True)
class ParamDef:
    r"""
    Definition of a tunable parameter in the search space.

    Parameters
    ----------
    name : str
        Parameter name (key into the brancher config).
    kind : {"int","float","categorical","bool"}
        Parameter type and sampling family.
    low, high : float, optional
        Numeric bounds (inclusive for ``"int"``). Required for numeric kinds.
    log : bool, optional
        If ``True``, sample the parameter **log-uniformly** between ``low`` and
        ``high``. Otherwise sample uniformly.
    choices : list, optional
        Allowed values for categorical parameters.

    Notes
    -----
    - Integer parameters are rounded to the nearest integer after sampling.
    - For log sampling, we draw :math:`u\sim\mathcal{U}(\log a,\log b)` and set
      :math:`x=\exp(u)` before casting (if integer).
    """
    name: str
    kind: str       # "int" | "float" | "categorical" | "bool"
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None

def _load_space(path: Path, brancher_key: str) -> List[ParamDef]:
    r"""
    Load a parameter search space from JSON and extract the block for a brancher.

    The JSON file is expected to contain a list of parameter objects under either
    ``"<brancher_key>"`` or the resolved config key (e.g. ``"ekfpso_config"``).
    Each object should have fields compatible with :class:`ParamDef`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the JSON file describing the parameter space.
    brancher_key : str
        One of the supported branchers (``"ekf"``, ``"pso"``, etc.).

    Returns
    -------
    list[ParamDef]
        Parsed and validated parameter definitions.

    Raises
    ------
    ValueError
        If the space block is missing or empty.
    """
    spec = json.loads(path.read_text())
    block = spec.get(brancher_key) or spec.get(_resolve_cfg_key(brancher_key))
    if not isinstance(block, list) or not block:
        raise ValueError(f"No parameter space for '{brancher_key}' in {path}")
    out: List[ParamDef] = []
    for p in block:
        out.append(ParamDef(
            name=p["name"],
            kind=p["type"],
            low=p.get("min"),
            high=p.get("max"),
            log=bool(p.get("log", False)),
            choices=p.get("choices"),
        ))
    return out

def _random_point(space: List[ParamDef], rng: np.random.Generator) -> Dict[str, Any]:
    r"""
    Sample a single random configuration from the given search space.

    Sampling rules
    --------------
    - ``categorical``: uniform over ``choices``.
    - ``bool``: Bernoulli(0.5).
    - ``int``:
      - if ``log``: :math:`x=\left\lfloor \exp\big(U(\log a,\log b)\big) \right\rceil`
      - else: integer uniform in ``[a,b]`` (inclusive)
    - ``float``:
      - if ``log``: :math:`x=\exp\big(U(\log a,\log b)\big)`
      - else: :math:`x\sim U(a,b)`

    Parameters
    ----------
    space : list[ParamDef]
        Parameter definitions.
    rng : numpy.random.Generator
        Random generator used for reproducibility.

    Returns
    -------
    dict
        Mapping ``name -> sampled value``.
    """
    x: Dict[str, Any] = {}
    for p in space:
        if p.kind == "categorical":
            x[p.name] = rng.choice(p.choices)
        elif p.kind == "bool":
            x[p.name] = bool(rng.integers(0, 2))
        elif p.kind == "int":
            if p.log:
                lo, hi = math.log(max(p.low, 1e-12)), math.log(p.high)
                x[p.name] = int(round(math.exp(rng.uniform(lo, hi))))
            else:
                x[p.name] = int(rng.integers(int(p.low), int(p.high) + 1))
        elif p.kind == "float":
            if p.log:
                lo, hi = math.log(max(p.low, 1e-12)), math.log(p.high)
                x[p.name] = float(math.exp(rng.uniform(lo, hi)))
            else:
                x[p.name] = float(rng.uniform(p.low, p.high))
        else:
            raise ValueError(f"Unsupported param type: {p.kind}")
    return x

def _aggregate(values: List[float], agg: str = "mean") -> float:
    r"""
    Aggregate a list of per-track losses into a single scalar.

    Parameters
    ----------
    values : list of float
        Losses (already *lower is better*).
    agg : {"mean","median","min","max"}, optional
        Aggregation operator (default ``"mean"``).

    Returns
    -------
    float
        Aggregated value. If ``values`` is empty, returns a large penalty
        (``1e6``) to discourage the configuration.
    """
    if not values:
        return 1e6
    a = agg.lower()
    if a in ("mean", "avg", "average"):
        return float(np.mean(values))
    if a == "median":
        return float(np.median(values))
    if a == "min":
        return float(np.min(values))
    if a == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown aggregator '{agg}'")

def _objective_from_builder(
    builder: TrackBuilder,
    metric: str,
    *,
    aggregator: str = "mean",
    n_best: Optional[int] = None,   # None/<=0 => use ALL completed tracks
    tol: float = 0.005,
) -> float:
    r"""
    Compute a scalar **objective** from a built set of tracks (lower is better).

    Two objective families are supported:

    1. **Event score** (``metric == "score"``): compute the official TrackML
       score :math:`s` against all built tracks and return :math:`1/s`.
    2. **Per-track metrics**: for each track :math:`i`, compute the requested
       metric :math:`m_i` versus its ground truth, transform it to a loss
       :math:`\ell_i` via :func:`_loss_from_value`, then aggregate
       :math:`\phi(\{\ell_i\})` (``mean`` by default). Optionally restrict to
       the ``n_best`` tracks according to the builder’s internal ranking.

    Parameters
    ----------
    builder : TrackBuilder
        Builder containing completed tracks and hit pool with truth.
    metric : str
        Metric name (see :mod:`trackml_reco.metrics`) or ``"score"``.
    aggregator : {"mean","median","min","max"}, optional
        Aggregator :math:`\phi` for per-track losses.
    n_best : int or None, optional
        If positive, use only the best ``n_best`` tracks; otherwise use all.
    tol : float, optional
        Tolerance (meters) forwarded to metric computations.

    Returns
    -------
    float
        Scalar objective to minimize.
    """
    m = _normalize_metric_name(metric)
    if m == "score":
        from trackml.score import score_event
        tracks = builder.completed_tracks
        if not tracks:
            return 1e6
        truth = builder.hit_pool.pt_cut_hits[["hit_id", "particle_id", "weight"]]
        rows = []
        for t in tracks:  # use all tracks for "score" too
            rows.extend({"hit_id": int(h), "track_id": int(t.particle_id)} for h in t.hit_ids)
        if not rows:
            return 1e6
        sub = pd.DataFrame(rows).drop_duplicates("hit_id")
        s = float(score_event(truth, sub))
        return 1.0 / max(s, 1e-9)

    # Per-track metrics (overall = all completed tracks unless n_best > 0)
    tracks = builder.completed_tracks
    if not tracks:
        return 1e6
    if n_best is not None and n_best > 0 and n_best < len(tracks):
        tracks = builder.get_best_tracks(n=n_best)

    truth_df = builder.hit_pool.pt_cut_hits
    vals: List[float] = []
    for t in tracks:
        truth = truth_df[truth_df.particle_id == t.particle_id]
        if truth.empty or not t.trajectory:
            continue
        traj = np.asarray(t.trajectory, dtype=np.float64)
        if traj.size == 0:
            continue
        truth_xyz = truth[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)

        met = trk_metrics.compute_metrics(traj, truth_xyz, tol=tol, names=(m,))
        if m not in met:
            return 1e6
        vals.append(_loss_from_value(m, float(met[m])))

    return _aggregate(vals, aggregator)

def _make_builder(
    hit_pool,
    brancher_key: str,
    branch_cfg: Dict[str, Any],
    use_parallel: bool,
    parallel_time_budget: Optional[float],
    max_workers: Optional[int] = None,
):
    r"""
    Construct a track builder (parallel or single-threaded) for a given brancher.

    Parameters
    ----------
    hit_pool : HitPool
        Input hits and per-layer trees.
    brancher_key : {"ekf","astar","aco","pso","sa","ga","hungarian"}
        Which branching strategy to instantiate.
    branch_cfg : dict
        Configuration block passed to the brancher constructor.
    use_parallel : bool
        If ``True``, return :class:`CollaborativeParallelTrackBuilder`.
    parallel_time_budget : float or None
        Optional per-seed time budget (seconds) for the parallel builder.
    max_workers : int or None, optional
        Worker count for the parallel builder (defaults to ``os.cpu_count()``).

    Returns
    -------
    TrackBuilder or CollaborativeParallelTrackBuilder
        Configured builder instance.

    Notes
    -----
    The parallel builder additionally accepts:

    - ``claim_margin`` in ``branch_cfg``: margin factor used for collaborative
      claiming of seeds.
    - ``per_seed_time_budget_s``: forwarded from ``parallel_time_budget``.
    """
    brancher_cls = BRANCHER_MAP[brancher_key]
    branch_cfg = dict(branch_cfg)  # defensive copy

    if use_parallel:
        if max_workers is None:
            max_workers = os.cpu_count() or 8
        return CollaborativeParallelTrackBuilder(
            hit_pool=hit_pool,
            brancher_cls=brancher_cls,
            brancher_config=branch_cfg,
            max_workers=int(max_workers),
            claim_margin=float(branch_cfg.get("claim_margin", 1.0)),
            per_seed_time_budget_s=(
                float(parallel_time_budget) if parallel_time_budget is not None else None
            ),
        )
    else:
        return TrackBuilder(
            hit_pool=hit_pool,
            brancher_cls=brancher_cls,
            brancher_config=branch_cfg,
        )

def evaluate_once(
    *,
    hit_pool,
    base_config: Mapping[str, Any],
    brancher_key: str,
    params: Mapping[str, Any],
    metric: str = "mse",
    use_parallel: bool = False,
    parallel_time_budget: Optional[float] = None,
    max_seeds: Optional[int] = None,
    aggregator: str = "mean",
    n_best_tracks: Optional[int] = None,  # None/<=0 => ALL
    tol: float = 0.005,
) -> float:
    r"""
    Run a **single evaluation** (build tracks and score) for a parameter set.

    Steps
    -----
    1. Resolve the config block for ``brancher_key`` and overlay ``params`` via
       a deep (recursive) update.
    2. Build tracks using a parallel or single-threaded builder.
    3. Convert the chosen metric into a scalar **loss** via
       :func:`_objective_from_builder` and return it.

    Parameters
    ----------
    hit_pool : HitPool
        Data source used by the builder.
    base_config : mapping
        Full experiment configuration dictionary (contains brancher blocks).
    brancher_key : str
        Which brancher to run (see :data:`BRANCHER_MAP`).
    params : mapping
        Trial parameters to overlay on the base brancher config.
    metric : str, optional
        Objective metric (see :func:`_objective_from_builder` for options).
    use_parallel : bool, optional
        Use the collaborative parallel builder.
    parallel_time_budget : float or None, optional
        Per-seed time budget (seconds) in parallel mode.
    max_seeds : int or None, optional
        If provided, limit the number of seeds processed (for speed).
    aggregator : {"mean","median","min","max"}, optional
        Aggregator for per-track objectives.
    n_best_tracks : int or None, optional
        If positive, aggregate only over that many best tracks.
    tol : float, optional
        Tolerance for geometric matching (meters).

    Returns
    -------
    float
        Objective value to minimize. A large penalty (``1e6``) is returned if
        building fails or produces no useful tracks.
    """
    cfg_key = _resolve_cfg_key(brancher_key)
    if cfg_key not in base_config:
        raise KeyError(f"Config missing '{cfg_key}'")

    branch_cfg = _deep_update(copy.deepcopy(base_config[cfg_key]), dict(params))

    builder = _make_builder(
        hit_pool=hit_pool,
        brancher_key=brancher_key,
        branch_cfg=branch_cfg,
        use_parallel=use_parallel,
        parallel_time_budget=parallel_time_budget,
    )

    ekf_cfg = base_config.get("ekf_config", {})
    max_tracks_per_seed = int(ekf_cfg.get("num_branches", 30))
    survive_top = int(ekf_cfg.get("survive_top", 12))

    try:
        builder.build_tracks_from_truth(
            max_seeds=max_seeds,
            max_tracks_per_seed=max_tracks_per_seed,
            max_branches=survive_top,
            jitter_sigma=1e-4,
        )
    except Exception:
        logger.exception("Evaluation failed; penalty assigned. Params=%s", params)
        return 1e6

    return _objective_from_builder(
        builder, metric, aggregator=aggregator, n_best=n_best_tracks, tol=tol
    )

@dataclass
class OptResult:
    r"""
    Result of a parameter optimization run.

    Attributes
    ----------
    best_params : dict
        Best-found parameter dictionary (minimizes the chosen objective).
    best_value : float
        Objective value achieved by ``best_params`` (lower is better).
    history : list of dict
        Trial-by-trial records with keys ``{"value","params","time_s"}``.
    """
    best_params: Dict[str, Any]
    best_value: float
    history: List[Dict[str, Any]]  # {"value": float, "params": {...}, "time_s": float}

def optimize_brancher(
    *,
    hit_pool,
    base_config: Mapping[str, Any],
    brancher_key: str,
    space_path: Path,
    metric: str = "mse",
    iterations: int = 40,
    n_init: int = 10,
    use_parallel: bool = False,
    parallel_time_budget: Optional[float] = None,
    max_seeds: Optional[int] = 200,
    rng_seed: Optional[int] = None,
    skopt_kind: str = "auto",   # "auto" | "gp" | "forest" | "gbrt" | "random"
    aggregator: str = "mean",   # per-track aggregator ("mean", "median", "min", "max")
    n_best_tracks: int = 5,     # how many best tracks to aggregate
    tol: float = 0.005,         # tolerance passed to compute_metrics
) -> OptResult:
    r"""
    Optimize a brancher’s hyperparameters against a chosen metric.

    The routine minimizes a scalar objective derived from either the **TrackML
    event score** (using :math:`1/\text{score}` so lower is better) or **per-track
    metrics** aggregated across tracks (see :func:`_objective_from_builder`).

    Search strategy
    ---------------
    - If :mod:`scikit-optimize` is available, use one of its minimizers:
      Gaussian-process (``"gp"``), random-forest (``"forest"``), gradient-boosted
      trees (``"gbrt"``), or random (``"random"``). The choice ``"auto"`` uses
      GP by default.
    - Otherwise, fall back to **random search** with a warmup of ``n_init``
      samples followed by local perturbations around the current best.

    Objective definition
    --------------------
    For a per-track metric :math:`m` and tracks :math:`i=1,\dots,N`, the
    objective is

    .. math::
        \text{obj} \;=\; \phi\!\left(\{\,L(m_i)\,\}_{i=1}^{N^\*}\right),

    where :math:`L(\cdot)` converts a metric to a *loss* (see
    :func:`_loss_from_value`), :math:`\phi` is the requested aggregator
    (``"mean"`` by default), and :math:`N^\*` equals either all tracks or the
    best ``n_best_tracks`` according to the builder’s ranking.

    Parameters
    ----------
    hit_pool : HitPool
        Input data for building tracks.
    base_config : mapping
        Experiment configuration containing brancher blocks.
    brancher_key : str
        Which brancher to tune (see :data:`BRANCHER_MAP`).
    space_path : pathlib.Path
        Path to a JSON file describing the search space (see :class:`ParamDef`).
    metric : str, optional
        Objective metric (``"mse"``, ``"rmse"``, ``"recall"``, ``"precision"``, ``"f1"``, or ``"score"``).
    iterations : int, optional
        Total number of trials (calls to the objective).
    n_init : int, optional
        Number of initial random evaluations (used by both skopt and fallback).
    use_parallel : bool, optional
        Use :class:`CollaborativeParallelTrackBuilder` instead of single-threaded.
    parallel_time_budget : float or None, optional
        Per-seed time budget (seconds) in parallel mode.
    max_seeds : int or None, optional
        Limit the number of seeds per trial to control runtime.
    rng_seed : int or None, optional
        Random seed for sampling and skopt (when used).
    skopt_kind : {"auto","gp","forest","gbrt","random"}, optional
        Which :mod:`skopt` backend to use when available.
    aggregator : {"mean","median","min","max"}, optional
        Aggregation function :math:`\phi` for per-track losses.
    n_best_tracks : int, optional
        If > 0, aggregate over the best ``n_best_tracks``; otherwise all tracks.
    tol : float, optional
        Distance tolerance (meters) forwarded to per-track metrics.

    Returns
    -------
    OptResult
        Best parameters, best (lowest) objective value, and a trial history.

    Notes
    -----
    - Failures during building assign a large penalty (``1e6``) but do not stop
      the search.
    - When optimizing by event **score**, all completed tracks are used to build
      the submission for evaluation irrespective of ``n_best_tracks``.
    """
    rng = np.random.default_rng(rng_seed)
    space = _load_space(space_path, brancher_key)

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_params: Dict[str, Any] = {}

    n_trials_total = int(iterations)
    trial_idx = 0

    def run_params(pdict: Dict[str, Any]) -> Tuple[float, float]:
        nonlocal best_val, best_params, trial_idx
        trial_idx += 1
        t0 = time.time()

        val = evaluate_once(
            hit_pool=hit_pool,
            base_config=base_config,
            brancher_key=brancher_key,
            params=pdict,
            metric=metric,
            use_parallel=use_parallel,
            parallel_time_budget=parallel_time_budget,
            max_seeds=max_seeds,
            aggregator=aggregator,
            n_best_tracks=n_best_tracks,
            tol=tol,
        )
        dur = time.time() - t0

        scope = "all" if not n_best_tracks or n_best_tracks <= 0 else f"best {n_best_tracks}"
        logger.info(
            "[opt/%s] trial %d/%d | %s(%s over %s)=%.6g | %.2fs | parallel=%s | %s",
            brancher_key, trial_idx, n_trials_total,
            metric, aggregator, scope, val, dur, use_parallel, _param_preview(pdict)
        )

        if val < best_val:
            best_val, best_params = float(val), dict(pdict)
            logger.info(
                "[opt/%s] new best at trial %d/%d | %s=%.6g",
                brancher_key, trial_idx, n_trials_total, metric, best_val
            )
        return float(val), float(dur)

    # Try scikit-optimize when available
    used_skopt = False
    try:
        from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
        from skopt.space import Real, Integer, Categorical

        def to_dim(p: ParamDef):
            if p.kind == "categorical":
                return Categorical(p.choices, name=p.name)
            if p.kind == "bool":
                return Categorical([False, True], name=p.name)
            if p.kind == "int":
                prior = "log-uniform" if p.log else "uniform"
                return Integer(int(p.low), int(p.high), name=p.name, prior=prior)
            if p.kind == "float":
                prior = "log-uniform" if p.log else "uniform"
                return Real(float(p.low), float(p.high), name=p.name, prior=prior)
            raise ValueError(f"Bad kind: {p.kind}")

        dims = [to_dim(p) for p in space]
        names = [p.name for p in space]

        def objective(vec: List[Any]) -> float:
            pdict = {k: v for k, v in zip(names, vec)}
            val, dur = run_params(pdict)
            history.append({"value": float(val), "params": pdict, "time_s": float(dur)})
            return float(val)

        algo = {
            "auto": gp_minimize,
            "gp": gp_minimize,
            "forest": forest_minimize,
            "gbrt": gbrt_minimize,
            "random": dummy_minimize,
        }[skopt_kind]

        algo(
            objective,
            dims,
            n_calls=int(iterations),
            n_initial_points=max(1, int(n_init)),
            random_state=int(rng.integers(0, 2**31 - 1)),
            verbose=False,
        )
        used_skopt = True
    except Exception as e:
        logger.info("skopt not used (or failed): %s; fallback to random search.", e)

    if not used_skopt:
        # Warmup
        n_init_eff = max(1, int(n_init))
        for _ in range(n_init_eff):
            pdict = _random_point(space, rng)
            val, dur = run_params(pdict)
            history.append({"value": float(val), "params": pdict, "time_s": float(dur)})

        # Local perturbations around current best
        remain = max(0, int(iterations) - n_init_eff)
        for _ in range(remain):
            candidate: Dict[str, Any] = dict(best_params) if best_params else _random_point(space, rng)
            for p in space:
                if rng.random() < 0.35:
                    if p.kind in ("float", "int"):
                        lo, hi = float(p.low), float(p.high)
                        if p.log:
                            span = math.log(hi) - math.log(max(lo, 1e-12))
                            step = math.exp(math.log(max(lo, 1e-12)) + rng.uniform(0.0, span) * 0.35)
                            valf = step
                        else:
                            width = (hi - lo) * 0.35
                            base = float(candidate.get(p.name, (lo + hi) * 0.5))
                            valf = np.clip(base + rng.normal(0.0, width * 0.25), lo, hi)
                        candidate[p.name] = int(round(valf)) if p.kind == "int" else float(valf)
                    elif p.kind == "categorical":
                        candidate[p.name] = rng.choice(p.choices)
                    elif p.kind == "bool":
                        candidate[p.name] = not bool(candidate.get(p.name, False))
            val, dur = run_params(candidate)
            history.append({"value": float(val), "params": candidate, "time_s": float(dur)})

    scope = "all" if not n_best_tracks or n_best_tracks <= 0 else f"best {n_best_tracks}"
    logger.info(
        "[opt/%s] done | best %s(%s over %s)=%.6g | params: %s | parallel=%s",
        brancher_key, metric, aggregator, scope, best_val,
        _param_preview(best_params, limit=8), use_parallel
    )
    return OptResult(best_params=dict(best_params), best_value=float(best_val), history=history)