from __future__ import annotations

import logging
from typing import Sequence, Optional

import warnings
import numpy as np
import pandas as pd
from trackml.dataset import load_dataset
from trackml.utils import add_position_quantities, add_momentum_quantities
from trackml.weights import weight_hits_phase1

import trackml_reco.hit_pool as trk_hit_pool


def _weight_hits_phase1_no_cow(truth: pd.DataFrame, particles: pd.DataFrame) -> pd.DataFrame:
    r"""
    Compute TrackML phase-1 hit weights with pandas **Copy-on-Write** temporarily disabled.

    This is a thin wrapper around :func:`trackml.weights.weight_hits_phase1` that:

    1. Temporarily turns off pandas *Copy-on-Write* (CoW) to avoid internal
       chained-assignment slow paths and warnings inside the TrackML utility.
    2. Silences the library's ``FutureWarning`` about chained assignment so the
       caller's logs remain clean.

    Parameters
    ----------
    truth : :class:`pandas.DataFrame`
        Event truth table with at least columns
        ``['hit_id', 'particle_id']`` (additional columns are passed through).
    particles : :class:`pandas.DataFrame`
        Particle table with at least ``['particle_id']`` and kinematic
        quantities used by TrackML for weighting.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame containing per-hit weights, typically including columns
        such as ``'hit_id'``, ``'weight'``, and/or ``'weight_pt'`` depending on
        the TrackML version.

    Notes
    -----
    The function restores the previous pandas CoW setting on exit (success or
    error). All changes to warnings are localized via ``warnings.catch_warnings``.

    See Also
    --------
    trackml.weights.weight_hits_phase1 : The underlying implementation used here.
    """
    prev = getattr(pd.options.mode, "copy_on_write", None)
    try:
        if prev is not None:
            pd.options.mode.copy_on_write = False
        # Hide the library’s chained-assignment FutureWarnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"A value is trying to be set on a copy.*",
                category=FutureWarning,
                module=r".*trackml\.weights",
            )
            return weight_hits_phase1(truth, particles)
    finally:
        if prev is not None:
            pd.options.mode.copy_on_write = prev


def load_and_preprocess(
    event_zip: str,
    *,
    pt_threshold: float = 2.0,
    prefer_weight_cols: Sequence[str] = ("weight", "weight_pt"),
) -> trk_hit_pool.HitPool:
    r"""
    Load a single TrackML event and produce a compact :class:`HitPool`
    with **metric-unit** coordinates, derived kinematics, and **per-hit weights**.

    Pipeline
    --------
    1. **Load** the first event from ``event_zip`` (``hits``, ``cells``, ``truth``,
       ``particles``).
    2. **Unit conversion**: cast ``x,y,z`` from millimeters to meters while
       preserving input dtype when possible (keeps ``float32`` if input is
       ``float32``; otherwise promotes to ``float64`` and writes back safely).
    3. **Derived features**:
       - :func:`trackml.utils.add_position_quantities` augments ``hits`` (e.g.,
         cylindrical coordinates).
       - :func:`trackml.utils.add_momentum_quantities` augments ``particles`` with
         transverse momentum :math:`p_T=\sqrt{p_x^2+p_y^2}` and related fields.
    4. **High-:math:`p_T` filtering**:
       Select particles with :math:`p_T \ge p_T^{\min}` and semi-join through
       ``truth`` into ``hits`` to obtain *truth-matched* high-:math:`p_T` hits.
    5. **Weights**:
       Compute per-hit weights using :func:`trackml.weights.weight_hits_phase1`
       via :func:`_weight_hits_phase1_no_cow`. Choose the first available column
       from ``prefer_weight_cols`` (e.g. ``'weight'`` then ``'weight_pt'``),
       mapping by ``hit_id``; default to weight ``1.0`` if none is present.
    6. **Return** a :class:`trackml_reco.hit_pool.HitPool` constructed from the
       full ``hits`` and the truth-matched, weighted subset.

    Parameters
    ----------
    event_zip : str
        Path to a TrackML event archive (``.zip``) readable by
        :func:`trackml.dataset.load_dataset`.
    pt_threshold : float, optional
        Momentum cut :math:`p_T^{\min}` in GeV/c used to filter particles before
        joining hits. Default is ``2.0``.
    prefer_weight_cols : sequence of str, optional
        Ordered preference of weight column names to use from the weighting
        DataFrame (first match wins). Default ``('weight','weight_pt')``.

    Returns
    -------
    pool : :class:`trackml_reco.hit_pool.HitPool`
        Container with

        - ``pool.hits``: all hits (meters, with derived position quantities),
        - ``pool.truth_hits``: truth-matched hits for particles with
          :math:`p_T \ge p_T^{\min}`, augmented with a numeric ``'weight'``.

    Notes
    -----
    - **Unit conversion.** Let the original coordinates be in millimeters.
      We write back :math:`(x,y,z) \leftarrow 10^{-3}\,(x,y,z)` so that all
      geometry is in meters. If the incoming columns are ``float32``, we keep
      them as ``float32`` to avoid dtype upcasts; otherwise we ensure the
      destination dtype is ``float64`` before assignment.
    - **High-:math:`p_T` semi-join.** With particles table :math:`\mathcal{P}` and
      truth table :math:`\mathcal{T}`, we take

      .. math::

          \mathcal{H}_\text{truth} \;=\;
          \bigl( \{\, (h,\pi)\in\mathcal{T} : \pi\in\mathcal{P},\; p_T(\pi)\ge p_T^{\min} \,\}
          \ \Join\ \text{hits on } \text{hit\_id} \bigr).

    - **Weights.** From the weight DataFrame :math:`W` we build a mapping
      :math:`w:\text{hit\_id}\mapsto \mathbb{R}_+` using the first available
      column in ``prefer_weight_cols``; if a hit is not found in :math:`W`,
      we assign :math:`w=1`. Duplicate ``hit_id`` rows in :math:`W`` are
      defensively reduced to the first occurrence.
    - **Stability.** We avoid chained-assignment and CoW pitfalls by
      (i) matching destination dtypes prior to assignment,
      (ii) using ``.to_numpy(..., copy=False)`` where applicable,
      (iii) performing Series mapping for weights instead of joining full frames.

    Examples
    --------
    >>> pool = load_and_preprocess("event000001000.zip", pt_threshold=1.5)
    >>> pool.truth_hits[['hit_id', 'weight']].head()
         hit_id  weight
    0   123456   0.84
    1   123789   1.00
    """
    # Load one event
    _, hits, _, truth, particles = next(
        load_dataset(event_zip, nevents=1, parts=["hits", "cells", "truth", "particles"])
    )

    # Units: mm -> m, without dtype-mismatch warnings
    xyz = hits[["x", "y", "z"]]
    # If input columns are float32, keep them float32; otherwise promote to float64 first.
    if (xyz.dtypes == np.float32).all():
        arr32 = xyz.to_numpy(dtype=np.float32, copy=False)
        arr32 *= np.float32(1e-3)
        hits.loc[:, ["x", "y", "z"]] = arr32
    else:
        if not (xyz.dtypes == np.float64).all():
            # promote destination cols to float64 to match the assigned array
            hits = hits.astype({"x": "float64", "y": "float64", "z": "float64"}, copy=True)
            xyz = hits[["x", "y", "z"]]
        hits.loc[:, ["x", "y", "z"]] = xyz.to_numpy(dtype=np.float64, copy=False) * 1e-3

    # Derived quantities
    hits = add_position_quantities(hits)
    particles = add_momentum_quantities(particles)

    # Filter high-pT truth rows via semi-join
    high_pt = particles.loc[particles["pt"] >= float(pt_threshold), ["particle_id"]]
    truth_hits = truth.merge(high_pt, on="particle_id", how="inner", sort=False)
    truth_hits = truth_hits.merge(hits, on="hit_id", how="inner", sort=False)

    # Compute weights with CoW disabled inside the library call
    wdf = _weight_hits_phase1_no_cow(truth, particles)

    # Choose the best available weight column
    weight_col: Optional[str] = next((c for c in prefer_weight_cols if c in wdf.columns), None)

    # Ensure we don’t have a stale/overlapping 'weight' column
    if "weight" in truth_hits.columns:
        truth_hits = truth_hits.drop(columns=["weight"])

    if weight_col is None:
        # Fallback: all ones
        truth_hits.loc[:, "weight"] = 1.0
    else:
        # Map a Series (no DataFrame join → no overlap)
        wser = (
            wdf[["hit_id", weight_col]]
            .drop_duplicates("hit_id")  # safety
            .set_index("hit_id")[weight_col]
            .astype("float64", copy=False)
        )
        truth_hits.loc[:, "weight"] = truth_hits["hit_id"].map(wser).fillna(1.0).to_numpy()

    logging.info(
        "Loaded %d hits; kept %d truth-matched hits with pT>=%.2f (unique particles: %d)",
        len(hits),
        len(truth_hits),
        pt_threshold,
        high_pt["particle_id"].nunique(),
    )

    return trk_hit_pool.HitPool(hits, truth_hits)
