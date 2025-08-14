from __future__ import annotations

import cProfile
import pstats
import io
import time
import gc
import threading
import logging
from contextlib import contextmanager
from typing import Optional, Sequence, Union


def _resolve_sort_key(sort: Union[str, int, "pstats.SortKey"]) -> Union[int, "pstats.SortKey"]:
    r"""
    Normalize a sort key for :mod:`pstats` to a value accepted by ``Stats.sort_stats``.

    This helper accepts common string aliases (``"tottime"``, ``"cumtime"``,
    ``"calls"``, ``"name"``), a :class:`pstats.SortKey` enum (Python ≥ 3.8),
    or the legacy integer constants, and returns a value that can be passed
    directly to :meth:`pstats.Stats.sort_stats`.

    Parameters
    ----------
    sort : {str, int, pstats.SortKey}
        Desired sort criterion. Case-insensitive strings are mapped as follows::

            "tottime"  -> SortKey.TIME
            "cumtime"  -> SortKey.CUMULATIVE
            "calls"    -> SortKey.CALLS
            "pcalls"   -> SortKey.PCALLS
            "name"     -> SortKey.NAME
            "stdname"  -> SortKey.STDNAME
            "file"     -> SortKey.FILE
            "line"     -> SortKey.LINE
            "nfl"      -> SortKey.NFL
            "ncalls"   -> SortKey.CALLS  # alias

        If an unknown string is given, the function falls back to ``"tottime"``.

    Returns
    -------
    pstats.SortKey or int
        A key suitable for :meth:`pstats.Stats.sort_stats`.

    Notes
    -----
    - On Python versions where :class:`pstats.SortKey` is not available, the input
    is returned unchanged if it is an ``int``, otherwise ``"tottime"`` is used.
    - This function does **not** mutate global state; it only produces a value
    for consumption by :class:`pstats.Stats`.

    Examples
    --------
    >>> from pstats import SortKey
    >>> _resolve_sort_key("cumtime") is SortKey.CUMULATIVE
    True
    >>> _resolve_sort_key(SortKey.CALLS)  # passthrough
    <pstats.SortKey.CALLS: ...>
    >>> _resolve_sort_key("unknown")  # fallback
    <pstats.SortKey.TIME: ...>
    """
    # Python 3.8+: pstats.SortKey exists; still allow legacy int/str.
    try:
        SK = pstats.SortKey  # type: ignore[attr-defined]
        if isinstance(sort, SK.__class__) or isinstance(sort, SK):  # already a SortKey
            return sort
        if isinstance(sort, int):
            return sort
        s = str(sort).lower()
        mapping = {
            "tottime": SK.TIME,
            "cumtime": SK.CUMULATIVE,
            "calls":   SK.CALLS,
            "name":    SK.NAME,
            "nfl":     SK.NFL,          # filename:lineno
            "stdname": SK.STDNAME,
            "file":    SK.FILE,
            "line":    SK.LINE,
            "pcalls":  SK.PCALLS,
            "ncalls":  SK.CALLS,        # alias
        }
        return mapping.get(s, SK.TIME)
    except Exception:
        # Older Python: accept int or string understood by .sort_stats
        return sort if isinstance(sort, int) else (str(sort) if sort else "tottime")


@contextmanager
def prof(
    enable: bool = False,
    *,
    sort: Union[str, int, "pstats.SortKey"] = "tottime",
    limit: Optional[int] = 25,
    out_path: Optional[str] = None,
    dump_path: Optional[str] = None,
    strip_dirs: bool = True,
    profile_threads: bool = False,
    disable_gc: bool = False,
    include: Optional[Sequence[Union[str, int]]] = None,
    logger: Optional[logging.Logger] = None,
):
    r"""
    Lightweight, configurable CPU profiler context manager.

    Wraps :class:`cProfile.Profile` and :class:`pstats.Stats` to provide an
    ergonomic ``with`` block that can be toggled on/off, optionally profiles
    new threads, and emits both human-readable and binary outputs.

    Timing
    ------
    The wall-clock elapsed time of the profiled block is recorded as

    .. math::

    \Delta t \;=\; t_1 - t_0,

    where :math:`t_0` and :math:`t_1` are taken from :func:`time.perf_counter`
    immediately before enabling and right after disabling the profiler.

    Parameters
    ----------
    enable : bool, default: False
        If ``False``, the context is a no-op (near-zero overhead) and yields ``None``.
    sort : {str, int, pstats.SortKey}, default: ``"tottime"``
        Sorting criterion for ``Stats.sort_stats``. See :func:`_resolve_sort_key`
        for accepted values and aliases.
    limit : int or None, default: 25
        Row limit passed to ``Stats.print_stats``. ``None`` prints all rows.
    out_path : str or None, optional
        If set, write the formatted text profile to this UTF-8 file instead of
        printing or logging.
    dump_path : str or None, optional
        If set, write a binary ``.pstats`` file via ``Stats.dump_stats`` suitable
        for tools like *Snakeviz* or *gprof2dot*.
    strip_dirs : bool, default: True
        Apply ``Stats.strip_dirs()`` to shorten file paths in the textual output.
    profile_threads : bool, default: False
        If ``True``, install the profiler for **new threads** started inside the
        context using :func:`threading.setprofile`. Existing threads are not
        affected.
    disable_gc : bool, default: False
        Temporarily disable garbage collection within the context to reduce noise.
    include : sequence of {str, int} or None, optional
        Optional filters forwarded to ``Stats.print_stats(*include)`` (e.g. regex
        substrings or numeric limits).
    logger : logging.Logger or None, optional
        If provided and ``out_path`` is ``None``, emit the text profile via
        ``logger.info`` instead of printing to stdout.

    Yields
    ------
    cProfile.Profile or None
        The active profiler instance when ``enable=True``; otherwise ``None``.

    Side Effects
    ------------
    - When ``profile_threads=True``, ``threading.setprofile`` is set for the
    duration of the context and then restored.
    - If ``disable_gc=True`` and GC was enabled on entry, GC is re-enabled on exit.

    Output
    ------
    - Text profile is constructed with::

        ps = pstats.Stats(profile)
        ps.strip_dirs()   # if requested
        ps.sort_stats(key)
        ps.print_stats(limit or large_number)

    and is preceded by a header describing ``Δt``, sort key, limit, and flags.
    - Binary dumps are produced with ``ps.dump_stats(dump_path)`` when requested.

    Warnings
    --------
    - Thread profiling applies only to threads **created after** entering the
    context. Threads already running when the context is entered are not profiled.
    - If you disable GC, be aware that memory usage may increase temporarily.

    Examples
    --------
    Basic usage, printing top 25 functions by total time:

    >>> with prof(True, sort="tottime", limit=25):
    ...     _ = sum(i*i for i in range(10_000))

    Write text output and a ``.pstats`` file, sorted by cumulative time:

    >>> with prof(True, sort="cumtime", out_path="prof.txt", dump_path="prof.pstats"):
    ...     some_heavy_function()

    Profile threads spawned inside the block and log to a logger:

    >>> import logging
    >>> log = logging.getLogger("profile")
    >>> with prof(True, profile_threads=True, logger=log, limit=50):
    ...     run_threaded_workload()

    Filter to functions whose names match a substring (as in ``pstats``):

    >>> with prof(True, include=("my_package",)):
    ...     run_pipeline()

    See Also
    --------
    cProfile.Profile : Low-level deterministic profiler.
    pstats.Stats : Statistics and formatting for ``cProfile`` output.
    snakeviz : Browser-based visualization of ``.pstats`` files.
    gprof2dot : Convert profile data to Graphviz call graphs.
    """
    if not enable:
        yield None
        return

    pr = cProfile.Profile()

    # Optional: profile new threads that start inside this context
    prev_thread_prof = None
    if profile_threads:
        try:
            prev_thread_prof = threading.getprofile()
            # cProfile exposes a callable 'dispatcher' compatible with threading.setprofile
            threading.setprofile(pr.dispatcher)  # type: ignore[attr-defined]
        except Exception:
            prev_thread_prof = None  # fail-safe; still profile main thread

    # Optional: suspend GC while profiling
    gc_was_enabled = False
    if disable_gc:
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()

    t0 = time.perf_counter()
    pr.enable()
    try:
        yield pr
    finally:
        pr.disable()
        t1 = time.perf_counter()

        # Restore thread profiler & GC
        if profile_threads:
            try:
                threading.setprofile(prev_thread_prof)
            except Exception:
                pass
        if disable_gc and gc_was_enabled:
            gc.enable()

        # Build stats
        ps = pstats.Stats(pr)
        if strip_dirs:
            ps.strip_dirs()
        ps.sort_stats(_resolve_sort_key(sort))

        # Human-readable text
        s = io.StringIO()
        ps.stream = s
        header = f"[prof] elapsed={t1 - t0:.6f}s sort={sort} limit={limit} threads={profile_threads} gc_off={disable_gc}\n"
        if include:
            ps.print_stats(*include)
        else:
            # pstats accepts numeric limit or percentage string like '25%'
            ps.print_stats(limit if limit is not None else 1_000_000)
        text = header + s.getvalue()

        # Optional binary dump for visualization tools
        if dump_path:
            try:
                ps.dump_stats(dump_path)
            except Exception:
                pass

        # Output
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        elif logger is not None:
            logger.info(text)
        else:
            print(text, end="")

