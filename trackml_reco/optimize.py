from __future__ import annotations

import copy
import json
import math
import os
import io
import sys
import time
import logging
from collections import deque
from contextlib import contextmanager, nullcontext
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

@contextmanager
def _suppress_stdout(enabled: bool = True):
    r'''
    Temporarily silence :func:`print` by redirecting ``sys.stdout`` to an in-memory
    buffer **without touching logging**.

    Within the context, Python-level writes to standard output are diverted to a
    fresh :class:`io.StringIO` instance, so nothing appears on the console. On exit,
    the original ``sys.stdout`` is restored.

    Parameters
    ----------
    enabled : bool, optional
        If ``False``, the context is a no-op. If ``True`` (default), redirect
        ``sys.stdout`` for the duration of the ``with`` block.

    Yields
    ------
    None
        This is a context manager; it yields control to the body of the
        ``with`` statement.

    Notes
    -----
    - Only affects **``sys.stdout``**. It does **not** affect ``sys.stderr``,
    the :mod:`logging` framework (whose handlers typically write to ``stderr``),
    or native extensions that bypass Python I/O and write directly to file
    descriptors.
    - The redirection is **process-global**, not thread-local; concurrent threads
    printing to ``stdout`` during the context are also silenced.
    - The captured buffer is discarded on exit. For capturing instead of suppressing,
    consider :class:`contextlib.redirect_stdout`.

    Examples
    --------
    Silence noisy third-party prints:

    >>> with _suppress_stdout(True):
    ...     print("hidden")
    >>> print("visible")
    visible

    Enable/disable conditionally:

    >>> verbose = False
    >>> with _suppress_stdout(not verbose):
    ...     print("only shown when verbose=True")
    '''
    if not enabled:
        yield
        return
    saved = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = saved

@contextmanager
def _temp_logger_level(name: str, level: int | None):
    r'''
    Temporarily set the level of a single named logger.

    Let ``L`` denote the logger with name ``name``. During the context this function
    sets ``L``'s level to ``level`` (e.g., :data:`logging.WARNING`) and then restores
    its original level on exit. If ``level`` is ``None``, the context is a no-op.

    This only alters the **logger’s own level**; it does not mutate handlers, the
    root logger, or unrelated loggers, and it does not disable propagation.

    Parameters
    ----------
    name : str
        Dotted logger name (e.g., ``"trackml_reco.track_builder"``).
    level : int or None
        Target level for the named logger (e.g., :data:`logging.DEBUG`,
        :data:`logging.INFO`, :data:`logging.WARNING`). If ``None``, nothing is
        changed.

    Yields
    ------
    None
        This is a context manager; it yields control to the body of the
        ``with`` statement.

    Notes
    -----
    - **Effective level.** The level used to decide whether a record is emitted is
    the first non-``NOTSET`` level encountered along the logger hierarchy. Writing
    this as a recursion with parent operator :math:`\pi(\cdot)`,

    .. math::

        \mathrm{level}_{\mathrm{eff}}(L) =
        \begin{cases}
            \mathrm{level}(L), & \mathrm{level}(L) \ne \text{NOTSET},\\[3pt]
            \mathrm{level}_{\mathrm{eff}}\bigl(\pi(L)\bigr), & \text{otherwise}.
        \end{cases}

    Therefore setting a child’s level may or may not change behavior depending on
    its current value and the ancestors’ levels.
    - The change is **process-global** for the duration of the context (logging is
    global state) and not thread-local.
    - Handlers attached to the logger still apply their own level thresholds.

    Examples
    --------
    Silence a chatty module below WARNING for a block:

    >>> import logging
    >>> logging.getLogger("libX").info("before")   # may appear
    >>> with _temp_logger_level("libX", logging.WARNING):
    ...     logging.getLogger("libX").info("hidden")
    ...     logging.getLogger("libX").warning("shown")
    shown
    >>> logging.getLogger("libX").info("after")    # back to original behavior

    No-op when ``level=None``:

    >>> with _temp_logger_level("libX", None):
    ...     logging.getLogger("libX").debug("unchanged")

    See Also
    --------
    logging.getLogger : Retrieve a logger by name.
    logging.Logger.setLevel : Permanently change a logger’s level.
    logging.disable : Process-wide ceiling for all logging.
    logging.Logger.propagate : Control propagation to ancestor loggers.
    '''
    if level is None:
        yield
        return
    lg = logging.getLogger(name)
    old = lg.level
    try:
        lg.setLevel(level)
        yield
    finally:
        lg.setLevel(old)

def _isatty() -> bool:
    r'''
    Return ``True`` if stdout appears to be an interactive ANSI-capable terminal.

    The check combines :meth:`sys.stdout.isatty` with a small Windows heuristic:
    on ``os.name == "nt"`` we additionally require either ``WT_SESSION`` or
    ``TERM`` in the environment, which is a pragmatic proxy for terminals that
    understand ANSI escape codes.

    Returns
    -------
    bool
        ``True`` if interactive; ``False`` otherwise.

    Notes
    -----
    This is a conservative predicate intended for **cosmetic** output decisions
    (e.g. coloring). It is not a guarantee that every control sequence will be
    interpreted by the terminal.

    See Also
    --------
    _ansi : Return an ANSI escape sequence for a named color (or empty when disabled).
    '''
    return sys.stdout.isatty() and (os.name != "nt" or "WT_SESSION" in os.environ or "TERM" in os.environ)

def _ansi(color: str) -> str:
    r'''
    Return an ANSI color escape sequence when coloring is enabled.

    Coloring is **disabled** when either (i) stdout is not an interactive TTY
    (see :func:`_isatty`) or (ii) the environment variable ``NO_COLOR`` is set.
    In those cases this function returns the empty string.

    Parameters
    ----------
    color : {"reset", "green", "dim"}
        Logical color/request key. Unknown keys yield ``""``.

    Returns
    -------
    str
        ANSI escape sequence (e.g. ``"\x1b[32m"``) or ``""`` when disabled.

    Notes
    -----
    Typical usage is to wrap colored segments as
    ``f"{_ansi('green')}text{_ansi('reset')}"``. When coloring is disabled the
    wrapping reduces to just ``"text"``.

    See Also
    --------
    _isatty : Detect interactive terminals.
    '''
    if not _isatty() or os.environ.get("NO_COLOR"):
        return ""
    table = {
        "reset": "\x1b[0m",
        "green": "\x1b[32m",
        "dim": "\x1b[2m",
    }
    return table.get(color, "")

def _fmt_time(seconds: float) -> str:
    r'''
    Format seconds as ``H:MM:SS`` (if :math:`\ge 1\ \mathrm{hour}`) or ``MM:SS``.

    Let :math:`T \ge 0` be the number of seconds. We compute

    .. math::
    H = \left\lfloor \tfrac{T}{3600} \right\rfloor,\quad
    M = \left\lfloor \tfrac{T - 3600H}{60} \right\rfloor,\quad
    S = \left\lfloor T - 3600H - 60M \right\rfloor,

    and return ``f"{H}:{M:02d}:{S:02d}"`` if :math:`H>0`, otherwise
    ``f"{M:02d}:{S:02d}"``.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds. Negative inputs are clamped to ``0``.

    Returns
    -------
    str
        ``"H:MM:SS"`` or ``"MM:SS"``.

    Examples
    --------
    >>> _fmt_time(65.2)
    '01:05'
    >>> _fmt_time(3661.0)
    '1:01:01'
    '''
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _render_bar(cur: int, total: int, width: int = 24) -> str:
    r'''
    Render a short progress bar string, optionally colored for TTYs.

    Given the current step :math:`c`, total steps :math:`T>0`, and bar width
    :math:`W`, we compute the filled cell count

    .. math::
    f \;=\; \operatorname{round}\!\left(W \cdot \frac{\max(0,\min(c,T))}{T}\right),

    and render a bar with :math:`f` ``'█'`` characters and :math:`W-f` ``'-'``.
    On ANSI-capable TTYs (see :func:`_isatty`), the bar is emitted in green.

    Parameters
    ----------
    cur : int
        Current step (clamped to ``[0, total]``).
    total : int
        Total number of steps (clamped to at least ``1``).
    width : int, optional
        Number of character cells in the bar (default ``24``).

    Returns
    -------
    str
        The progress bar string, e.g. ``"█████-----"`` for ``width=10``.

    Examples
    --------
    >>> _render_bar(5, 10, width=10)[:10]
    '█████-----'
    '''
    total = max(1, int(total))
    cur = min(max(0, int(cur)), total)
    frac = cur / total
    fill = int(round(frac * width))
    bar_raw = "█" * fill + "-" * (width - fill)
    if _isatty() and not os.environ.get("NO_COLOR"):
        return f"{_ansi('green')}{bar_raw}{_ansi('reset')}"
    return bar_raw

def _print_progress_line(line: str) -> None:
    r'''
    Rewrite the current console line with a single in-place progress message.

    The function emits the control sequence ``ESC[2K`` (erase entire line),
    carriage return ``\\r``, the provided text, and then flushes stdout. This
    style is suitable for compact, continuously updating progress indicators.

    Parameters
    ----------
    line : str
        The text to display on the current line.

    Notes
    -----
    - Intended for single-threaded console output; mixing with other writers may
    yield interleaved lines.
    - No trailing newline is printed; call :func:`_finish_progress_line` to end
    the progress block.
    '''
    sys.stdout.write("\x1b[2K\r" + line)
    sys.stdout.flush()

def _finish_progress_line() -> None:
    r'''
    Terminate the single-line progress block by printing a newline.

    Writes ``"\\n"`` to stdout and flushes, ensuring subsequent output starts on
    the next line.

    Returns
    -------
    None
    '''
    sys.stdout.write("\n")
    sys.stdout.flush()

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
    suppress_builder_logs: bool = True,
    suppress_builder_prints: bool = True,
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
    suppress_builder_logs : bool, optional
        For suppressing the extra logs
    suppress_builder_prints : bool, optional
        For suppressing the extra prints

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

    # Silence only the builder's own noise during optimization:
    # - mute INFO logs from trackml_reco.track_builder (keep WARNING+)
    # - suppress print() called inside builder
    try:
        with _temp_logger_level("trackml_reco.track_builder", logging.WARNING), \
             _suppress_stdout(True):
            builder.build_tracks_from_truth(
                max_seeds=max_seeds,
                max_tracks_per_seed=max_tracks_per_seed,
                max_branches=survive_top,
                jitter_sigma=1e-4,
            )
    except KeyboardInterrupt:
        raise
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

def _plot_history(history: List[Dict[str, Any]], metric: str, out_path: Path) -> None:
    r'''
    Save a compact two-panel optimization history figure.

    The top panel shows the per-trial objective values :math:`v_t` and the
    best-so-far envelope

    .. math::
    b_t \;=\; \min_{1 \le i \le t} v_i,

    while the bottom panel shows the wall-clock duration per trial.

    Parameters
    ----------
    history : list of dict
        Trial records, each with keys ``"value"`` (float) and ``"time_s"`` (float).
    metric : str
        Label for the objective being plotted (e.g. ``"mse"``, ``"1/score"``).
    out_path : pathlib.Path
        Destination path for the PNG file.

    Notes
    -----
    - The function forces a non-interactive backend (``Agg``) to be safe in
    headless environments.
    - The best-so-far curve is computed via :func:`numpy.minimum.accumulate`.
    - This function **does not** raise if saving fails; it closes the figure
    unconditionally to avoid leaking GUI resources.

    Examples
    --------
    >>> hist = [{"value": 1.2, "time_s": 0.8}, {"value": 0.9, "time_s": 0.7}]
    >>> from pathlib import Path
    >>> _plot_history(hist, "mse", Path("opt_progress.png"))  # doctest: +SKIP
    '''
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    vals = np.array([h["value"] for h in history], dtype=float)
    durs = np.array([h["time_s"] for h in history], dtype=float)
    best_so_far = np.minimum.accumulate(vals)

    fig = plt.figure(figsize=(8.5, 5.5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(vals, label=f"{metric} per trial", linewidth=1.6)
    ax1.plot(best_so_far, label="best so far", linewidth=1.6)
    ax1.set_ylabel(metric)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(durs, label="duration (s)", linewidth=1.6)
    ax2.set_xlabel("trial")
    ax2.set_ylabel("seconds")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

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
    plot_path: Optional[Path] = None,  # where to save history plot
) -> OptResult:
    r'''
    Optimize a brancher’s hyperparameters against a chosen metric.

    This routine searches a user-defined hyperparameter space and **minimizes**
    a scalar objective derived either from the TrackML **event score** (by
    minimizing :math:`1/\mathrm{score}`) or from **per-track metrics** aggregated
    over completed tracks (see :func:`_objective_from_builder`).

    Mathematical objective
    ----------------------
    Let :math:`m` denote the selected metric (e.g., ``"mse"``, ``"rmse"``,
    ``"recall"``, ``"precision"``, ``"f1"``, or ``"score"``). For per-track
    metrics the builder returns a set of completed tracks
    :math:`\{\mathcal{T}_i\}_{i=1}^{N}` and we evaluate :math:`m_i` on each track.
    A loss transform :math:`L(\cdot)` converts the metric to “lower is better”:

    .. math::

    L(v) \;=\;
    \begin{cases}
        v, & \text{if } m \in \{\mathrm{MSE}, \mathrm{RMSE}\}, \\
        1 - v, & \text{if } 0 \le v \le 1 \text{ (fractional metric)}, \\
        100 - v, & \text{otherwise (percentage metric)}.
    \end{cases}

    The scalar objective aggregated over either **all** completed tracks or the
    **best** :math:`N^\*` tracks is

    .. math::

    \mathrm{obj}
    \;=\;
    \phi\!\left(\,\{\,L(m_i)\,\}_{i=1}^{N^\*}\right),

    where :math:`\phi\in\{\text{mean},\text{median},\min,\max\}` is the requested
    aggregator and :math:`N^\*` is either :math:`N` (all tracks) or the user-chosen
    ``n_best_tracks``.  When optimizing the event **score**, the objective is

    .. math::

    \mathrm{obj}_\text{score}
    \;=\;
    \frac{1}{\max(\mathrm{score},\;10^{-9})},

    computed from a submission built from **all** completed tracks in the trial.

    Search strategy
    ---------------
    - If :mod:`scikit-optimize` is available, one of its minimizers is used:
    Gaussian process (``"gp"``), random forest (``"forest"``),
    gradient-boosted trees (``"gbrt"``), or random (``"random"``).
    The choice ``"auto"`` defaults to GP.
    - Otherwise, a robust fallback of **random search** with ``n_init`` warmup
    trials is followed by **local perturbations** around the incumbent best.

    During the search, per-trial progress (running mean, wall time, ETA) is printed
    as a single updating line and a PNG history plot is written at the end.

    Parameters
    ----------
    hit_pool : HitPool
        Input data structure used by the builder (KD-trees, reservations, etc.).
    base_config : Mapping[str, Any]
        Configuration dictionary containing per-brancher parameter blocks.
    brancher_key : str
        Which brancher to tune (e.g., ``"ekf"``, ``"astar"``, ``"aco"``,
        ``"pso"``, ``"sa"``, ``"ga"``, ``"hungarian"``).
    space_path : pathlib.Path
        Path to a JSON file describing the search space.  Each entry specifies a
        parameter with fields ``{"name","type","min","max","log","choices"}``.
    metric : str, optional
        Objective metric. One of ``"mse"``, ``"rmse"``, ``"recall"``,
        ``"precision"``, ``"f1"``, or ``"score"``.  Common aliases are accepted
        (e.g., ``"pct_hits"`` :math:`\to` ``"recall"``).
    iterations : int, optional
        Total number of trials (objective evaluations). Default is ``40``.
    n_init : int, optional
        Number of initial random evaluations. Used by both skopt and the fallback.
    use_parallel : bool, optional
        If ``True``, use :class:`CollaborativeParallelTrackBuilder`; otherwise use
        the single-threaded :class:`TrackBuilder`. Default ``False``.
    parallel_time_budget : float or None, optional
        Per-seed time budget (seconds) in parallel mode; ``None`` disables it.
    max_seeds : int or None, optional
        Optional cap on seeds per trial to control runtime. Default ``200``.
    rng_seed : int or None, optional
        Random seed used for sampling (and for skopt’s RNG, when available).
    skopt_kind : {"auto","gp","forest","gbrt","random"}, optional
        Choice of :mod:`skopt` backend. Default ``"auto"`` (GP).
    aggregator : {"mean","median","min","max"}, optional
        Aggregation :math:`\phi` used to combine per-track losses. Default ``"mean"``.
    n_best_tracks : int, optional
        If ``> 0``, aggregate over the best *N* tracks (builder’s own ranking);
        if ``0`` or ``None``, use **all** completed tracks. Default ``5``.
    tol : float, optional
        Euclidean tolerance (meters) forwarded to per-track metrics such as
        recall/precision. Default ``0.005``.
    plot_path : pathlib.Path or None, optional
        Where to save the PNG history plot of trial values. Defaults to
        ``"opt_progress.png"`` in the current working directory.

    Returns
    -------
    OptResult
        Dataclass with fields
        ``best_params`` (``Dict[str, Any]``),
        ``best_value`` (``float``),
        ``history`` (``List[{"value": float, "params": dict, "time_s": float}]``).

    Notes
    -----
    - **Higher-is-better metrics.** For metrics like recall, precision and F1 the
    routine minimizes the *complement* as in :math:`L(v)=1-v` (fractions) or
    :math:`100-v` (percentages).  For MSE/RMSE the raw value is minimized.
    - **Score objective.** When ``metric="score"``, a full submission is built from
    all completed tracks and the loss :math:`1/\mathrm{score}` is minimized.
    - **Failure handling.** Any exception during a trial returns a penalty of
    ``1e6`` (trial logged and the search continues).
    - **Reproducibility.** Set ``rng_seed`` for deterministic sampling; parallel
    scheduling and internal BLAS/threads can still introduce run-to-run jitter.
    - **Side effects.** Prints a live progress line, logs summary messages, and
    writes a PNG plot to ``plot_path``.

    Warnings
    --------
    - The “best-N tracks” selection influences the objective landscape; consider
    using ``n_best_tracks=0`` to aggregate over **all** completed tracks when the
    number of tracks per trial varies substantially.
    - Tight gating or small seed caps (``max_seeds``) can bias the optimizer
    toward overly conservative configurations.

    See Also
    --------
    evaluate_once : Single evaluation pipeline producing one scalar objective.
    ParamDef : Search-space parameter specification (name, type, bounds, log, choices).
    CollaborativeParallelTrackBuilder : Multithreaded builder with shared hit ownership.

    Examples
    --------
    Minimal run (single-threaded EKF, MSE):

    >>> from pathlib import Path
    >>> res = optimize_brancher(
    ...     hit_pool=hit_pool,
    ...     base_config=config,
    ...     brancher_key="ekf",
    ...     space_path=Path("param_space.json"),
    ...     metric="mse",
    ...     iterations=60,
    ...     n_init=12,
    ...     max_seeds=200,
    ...     rng_seed=42,
    ... )
    >>> res.best_value, sorted(res.best_params.keys())[:3]  # doctest: +SKIP

    Parallel SA optimizing F1 over the best 5 tracks, with history plot:

    >>> res = optimize_brancher(
    ...     hit_pool=hit_pool,
    ...     base_config=config,
    ...     brancher_key="sa",
    ...     space_path=Path("param_space.json"),
    ...     metric="f1",
    ...     iterations=250,
    ...     n_init=20,
    ...     use_parallel=True,
    ...     parallel_time_budget=0.03,
    ...     max_seeds=200,
    ...     aggregator="mean",
    ...     n_best_tracks=5,
    ...     tol=0.005,
    ...     rng_seed=123,
    ...     plot_path=Path("sa_progress.png"),
    ... )  # doctest: +SKIP
    '''
    rng = np.random.default_rng(rng_seed)
    space = _load_space(space_path, brancher_key)

    history: List[Dict[str, Any]] = []
    values: List[float] = []
    best_val = float("inf")
    best_params: Dict[str, Any] = {}

    n_trials_total = int(iterations)
    trial_idx = 0
    last3 = deque(maxlen=3)
    par_str = str(bool(use_parallel)).lower()
    bar_w = 24
    metric_lower = _normalize_metric_name(metric)

    def update_progress(val: float, dur: float) -> None:
        r'''
        Update the live progress line with current trial statistics.

        Given the newest objective value :math:`v_t` and wall-clock duration
        :math:`d_t` for trial :math:`t`, this helper:

        1. Appends :math:`v_t` to the running sequence ``values`` and :math:`d_t` to a
        bounded deque ``last3`` (size 3).
        2. Computes the running mean
        :math:`\overline{v} = \tfrac{1}{t}\sum_{i=1}^{t} v_i`.
        3. Estimates the remaining time (ETA) by multiplying the average of the most
        recent durations by the number of trials left:

        .. math::
            \mathrm{ETA} \;=\;
            \bigg(\frac{1}{K}\sum_{j=0}^{K-1} d_{t-j}\bigg)\;
            \max(0, N_{\text{tot}} - t),
            \qquad K=\min(t,3).

        4. Renders a compact progress bar with :func:`_render_bar` and writes a
        single in-place console line via :func:`_print_progress_line`.

        Parameters
        ----------
        val : float
            The objective value for the current trial (:math:`v_t`).
        dur : float
            The wall-clock duration (seconds) for the current trial (:math:`d_t`).

        Notes
        -----
        - This function **mutates** the surrounding closure: it appends to
        ``values`` and ``last3`` and uses read-only access to variables such as
        ``trial_idx``, ``n_trials_total``, ``bar_w``, ``metric_lower`` and
        ``par_str``.
        - Output formatting includes the running mean of the metric, the per-trial
        duration, a human-friendly ETA (via :func:`_fmt_time`), and a textual bar.
        - Coloring of the bar is applied only on ANSI-capable TTYs (see :func:`_isatty`).

        See Also
        --------
        _render_bar : Text progress bar renderer.
        _print_progress_line : In-place console line writer.
        _fmt_time : Seconds to ``H:MM:SS``/``MM:SS`` formatter.
        '''
        values.append(val)
        last3.append(dur)
        mean_val = float(np.mean(values))
        eta = (np.mean(last3) * max(n_trials_total - trial_idx, 0)) if len(last3) else 0.0

        # progress bar + explicit percent text
        bar = _render_bar(trial_idx, n_trials_total, width=bar_w)
        pct = int(round(100.0 * min(max(trial_idx, 0), n_trials_total) / max(n_trials_total, 1)))
        pct_str = f"{pct:3d}%"

        line = (f"[opt/{brancher_key}] "
                f"trial {trial_idx}/{n_trials_total} | "
                f"mean {metric_lower}={mean_val:.6g} | "
                f"{dur:.2f}s | parallel={par_str} | ETA {_fmt_time(eta)} | "
                f"{bar} {pct_str}")
        _print_progress_line(line)

    def run_params(pdict: Dict[str, Any]) -> Tuple[float, float]:
        r'''
        Evaluate a single parameter vector and update the running optimum.

        This helper performs one optimization **trial**:

        1. Increments the global trial index :math:`t \leftarrow t+1`.
        2. Measures start time, evaluates the scalar objective :math:`f(\theta)` for
        the provided parameter dictionary :math:`\theta` via :func:`evaluate_once`,
        and records the elapsed duration.
        3. Appends a history record ``{"value": f(θ), "params": θ, "time_s": Δt}``.
        4. Calls :func:`update_progress` to refresh the live console line.
        5. If :math:`f(\theta)` improves the best-so-far value, updates
        ``best_val`` and ``best_params``.

        Formally, letting :math:`f` denote the objective being minimized (e.g.,
        :math:`f=\mathrm{MSE}`, or :math:`f = 1/\text{TrackML score}`), the update is

        .. math::
        \text{if }\; f(\theta) < f^\star \;\text{ then }\;
        f^\star \leftarrow f(\theta), \quad \theta^\star \leftarrow \theta,

        where :math:`(f^\star,\theta^\star)` are the incumbent best value and params.

        Parameters
        ----------
        pdict : dict
            Parameter dictionary :math:`\theta` to evaluate. Keys and types must
            match the brancher's expected configuration (see the search space JSON).

        Returns
        -------
        value : float
            The scalar objective :math:`f(\theta)` returned by :func:`evaluate_once`.
        duration : float
            Wall-clock seconds taken to evaluate :math:`f(\theta)`.

        Side Effects
        ------------
        - Mutates the surrounding closure: ``trial_idx``, ``best_val``,
        ``best_params``, and the list ``history``.
        - Emits a debug log line upon improvement (via :mod:`logging`).

        Notes
        -----
        - The objective internally depends on the selected metric (e.g., ``"mse"``,
        ``"rmse"``, ``"recall"``, ``"precision"``, ``"f1"``, or ``"score"``) and on
        the aggregation policy passed to :func:`evaluate_once`.
        - Exceptions inside :func:`evaluate_once` are handled there by assigning a
        large penalty, allowing the search to continue robustly.

        See Also
        --------
        evaluate_once : Build tracks and compute the scalar objective for a parameter set.
        update_progress : Refresh the live progress display after each trial.
        '''
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
            suppress_builder_prints=True
        )
        dur = time.time() - t0
        history.append({"value": float(val), "params": dict(pdict), "time_s": float(dur)})
        update_progress(val, dur)
        if val < best_val:
            best_val, best_params = float(val), dict(pdict)
            logger.debug("[opt/%s] new best at trial %d/%d | %s=%.6g",
                         brancher_key, trial_idx, n_trials_total, metric_lower, best_val)
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
            val, _ = run_params(pdict)
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
            run_params(pdict)

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
            run_params(candidate)

    # finalize progress line
    _finish_progress_line()

    # Save history plot
    out_path = Path(plot_path) if plot_path else Path("opt_progress.png")
    try:
        _plot_history(history, metric_lower, out_path)
        logger.info("Optimization history plot saved to %s", out_path)
    except Exception:
        logger.exception("Failed to save optimization history plot to %s", out_path)

    scope = "all" if not n_best_tracks or n_best_tracks <= 0 else f"best {n_best_tracks}"
    logger.info(
        "[opt/%s] done | best %s(%s over %s)=%.6g | params: %s | parallel=%s",
        brancher_key, metric, aggregator, scope, best_val,
        _param_preview(best_params, limit=8), use_parallel
    )
    return OptResult(best_params=dict(best_params), best_value=float(best_val), history=history)