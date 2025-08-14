from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def _make_submission(
    hit_ids: np.ndarray,
    track_ids: np.ndarray,
    *,
    renumber: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    r"""
    Construct a **TrackML-style submission** DataFrame mapping ``hit_id → track_id``,
    with an optional *stable* renumbering of track labels.

    The optional renumbering replaces the (possibly sparse) set of input
    ``track_ids`` by a random permutation of consecutive integers
    :math:`\{1,\dots,K\}`, where :math:`K` is the number of distinct tracks in
    the input. This is often useful to sanitize synthetic assignments before
    scoring.

    Let the original track labels be :math:`\{t_i\}_{i=1}^N` and
    :math:`U=\operatorname{unique}(\{t_i\})=\{u_1,\dots,u_K\}`. When
    ``renumber=True``, the output labels are

    .. math::

        t_i' \;=\; \pi\!\big(\,\operatorname{index}_U(t_i)\,\big),\qquad
        \pi \in \mathfrak{S}_K,

    where :math:`\pi` is a uniformly random permutation and
    ``index_U`` maps each original label to its position in the sorted unique set.

    Parameters
    ----------
    hit_ids : (N,) array_like of int
        Hit identifiers to appear in the ``hit_id`` column.
    track_ids : (N,) array_like of int
        Track identifiers aligned with ``hit_ids``.
    renumber : bool, optional
        If ``True`` (default), replace labels by a random permutation of
        ``1..K`` as described above. If ``False``, keep labels as provided.
    rng : numpy.random.Generator, optional
        Source of randomness for the permutation; a default generator is used
        if omitted.

    Returns
    -------
    df : pandas.DataFrame
        Two-column frame ``{'hit_id': int64, 'track_id': int64}``
        with one row per input pair.

    Raises
    ------
    ValueError
        If the input arrays are not 1D or have mismatched lengths.

    Notes
    -----
    - Input arrays are materialized to compact dtypes (``int64``) for robustness.
    - The renumbering is **label-stable** with respect to equality: equal
      input labels receive equal output labels.

    Examples
    --------
    >>> _make_submission([10, 11, 12], [100, 100, 42], renumber=False)
       hit_id  track_id
    0      10       100
    1      11       100
    2      12        42
    >>> df = _make_submission([10, 11, 12, 13], [7, 7, 9, 9], renumber=True,
    ...                       rng=np.random.default_rng(0))
    >>> sorted(df['track_id'].unique())
    [1, 2]
    """
    if hit_ids.ndim != 1 or track_ids.ndim != 1 or hit_ids.shape[0] != track_ids.shape[0]:
        raise ValueError("hit_ids and track_ids must be 1D arrays of the same length.")

    # Ensure compact dtypes (int64 for robustness across large IDs)
    hit_ids = np.asarray(hit_ids, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)

    if renumber:
        rng = np.random.default_rng() if rng is None else rng
        unique_ids, inverse = np.unique(track_ids, return_inverse=True)
        # 1..N_tracks permutation
        perm = np.arange(1, unique_ids.size + 1, dtype=np.int64)
        rng.shuffle(perm)
        track_ids = perm[inverse]

    # Construct without extra copies
    return pd.DataFrame(
        {"hit_id": hit_ids, "track_id": track_ids}
    )


def random_solution(
    hits: pd.DataFrame,
    ntracks: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    r"""
    Generate a **uniform random baseline** assignment of hits to tracks.

    For each hit :math:`h_i` (row in ``hits``), sample independently

    .. math::

        T_i \sim \operatorname{Unif}\{1,2,\dots,n_{\text{tracks}}\},

    and return the submission mapping ``hit_id → T_i``.

    Parameters
    ----------
    hits : pandas.DataFrame
        Must contain a ``'hit_id'`` column.
    ntracks : int
        Number of tracks to sample from (must be :math:`\ge 1`).
    rng : numpy.random.Generator, optional
        Random generator; a default generator is used if omitted.

    Returns
    -------
    df : pandas.DataFrame
        Two-column submission frame.

    Raises
    ------
    KeyError
        If ``'hit_id'`` is missing.
    ValueError
        If ``ntracks < 1``.

    Notes
    -----
    - Expected occupancy per track is :math:`\mathbb{E}[\#\{i:T_i=k\}] = N/n_{\text{tracks}}`.
    - This is intended as a quick control/baseline and will generally score poorly.

    Examples
    --------
    >>> hits = pd.DataFrame({'hit_id':[1,2,3,4]})
    >>> sol = random_solution(hits, ntracks=2, rng=np.random.default_rng(1))
    >>> set(sol.columns) == {'hit_id','track_id'}
    True
    """
    if "hit_id" not in hits.columns:
        raise KeyError("hits DataFrame must contain 'hit_id' column.")
    if ntracks < 1:
        raise ValueError("ntracks must be >= 1.")

    rng = np.random.default_rng() if rng is None else rng
    hit_ids = hits["hit_id"].to_numpy(dtype=np.int64, copy=False)
    track_ids = rng.integers(1, ntracks + 1, size=hit_ids.size, dtype=np.int64)
    return _make_submission(hit_ids, track_ids, renumber=False)


def drop_hits(
    truth: pd.DataFrame,
    probability: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    r"""
    Independently **drop/relable** each hit with probability :math:`p`.

    Given the true mapping ``hit_id → particle_id``, produce a submission in
    which each hit is, with probability :math:`p`, *removed* from its original
    track by assigning it a **new, unique** ``track_id`` not present in the
    original labels. Otherwise the original label is kept.

    Formally, for each hit :math:`i` with original label :math:`y_i`,

    .. math::

        \tilde{y}_i \;=\;
        \begin{cases}
          y_i, & \text{with prob. } 1-p,\\[3pt]
          z_i, & \text{with prob. } p,
        \end{cases}

    where the ``dropped'' labels :math:`z_i` are distinct and satisfy
    :math:`z_i \notin \{y_j\}`.

    Parameters
    ----------
    truth : pandas.DataFrame
        Must contain ``'hit_id'`` and ``'particle_id'`` columns.
    probability : float
        Drop probability :math:`p \in [0,1]`.
    rng : numpy.random.Generator, optional
        Random generator; a default generator is used if omitted.

    Returns
    -------
    df : pandas.DataFrame
        Two-column submission frame.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If ``probability`` is outside ``[0,1]``.

    Notes
    -----
    The expected number of relabeled hits is :math:`\mathbb{E}[K] = p\,N`.

    Examples
    --------
    >>> truth = pd.DataFrame({'hit_id':[1,2,3], 'particle_id':[10,10,20]})
    >>> sub = drop_hits(truth, 0.5, rng=np.random.default_rng(0))
    >>> set(sub.columns) == {'hit_id','track_id'}
    True
    """
    if not {"hit_id", "particle_id"} <= set(truth.columns):
        raise KeyError("truth DataFrame must contain 'hit_id' and 'particle_id' columns.")
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be in [0,1].")

    rng = np.random.default_rng() if rng is None else rng

    hit_ids = truth["hit_id"].to_numpy(dtype=np.int64, copy=False)
    out = truth["particle_id"].to_numpy(dtype=np.int64, copy=True)

    if out.size == 0 or probability == 0.0:
        return _make_submission(hit_ids, out)

    mask = rng.random(out.size) < probability
    k = int(mask.sum())
    if k:
        start = int(out.max()) + 1
        # assign distinct new ids to the dropped positions
        out[mask] = np.arange(start, start + k, dtype=np.int64)

    return _make_submission(hit_ids, out)


def shuffle_hits(
    truth: pd.DataFrame,
    probability: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    r"""
    With probability :math:`p` per hit, **reassign** to a *different existing*
    ``particle_id`` (label-preserving shuffle).

    For each hit with original label :math:`y_i` and the set of unique labels
    :math:`U=\{u_1,\dots,u_K\}` (with :math:`K\ge 2`), draw

    .. math::

        \tilde{y}_i \sim \operatorname{Unif}\big(U \setminus \{y_i\}\big)
        \quad\text{with prob. }p,\qquad
        \tilde{y}_i = y_i \text{ with prob. }1-p.

    If there is only one unique label, the mapping is returned unchanged.

    Parameters
    ----------
    truth : pandas.DataFrame
        Must contain ``'hit_id'`` and ``'particle_id'``.
    probability : float
        Shuffle probability :math:`p \in [0,1]` applied independently per hit.
    rng : numpy.random.Generator, optional
        Random generator; a default generator is used if omitted.

    Returns
    -------
    df : pandas.DataFrame
        Submission frame with possibly shuffled labels.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If ``probability`` is outside ``[0,1]``.

    Notes
    -----
    - Guarantees **no self-assignment** on shuffled positions.
    - The marginal distribution over new labels is uniform on the remaining
      :math:`K-1` classes for each shuffled hit.

    Examples
    --------
    >>> truth = pd.DataFrame({'hit_id':[1,2,3,4], 'particle_id':[10,10,20,30]})
    >>> sub = shuffle_hits(truth, 0.75, rng=np.random.default_rng(42))
    >>> set(sub.columns) == {'hit_id','track_id'}
    True
    """
    if not {"hit_id", "particle_id"} <= set(truth.columns):
        raise KeyError("truth DataFrame must contain 'hit_id' and 'particle_id' columns.")
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be in [0,1].")

    rng = np.random.default_rng() if rng is None else rng

    hit_ids = truth["hit_id"].to_numpy(dtype=np.int64, copy=False)
    out = truth["particle_id"].to_numpy(dtype=np.int64, copy=True)

    if out.size == 0 or probability == 0.0:
        return _make_submission(hit_ids, out)

    mask = rng.random(out.size) < probability
    if not mask.any():
        return _make_submission(hit_ids, out)

    unique_ids = np.unique(out)
    if unique_ids.size < 2:
        # Only one possible ID → cannot reassign to a different one
        return _make_submission(hit_ids, out)

    # Map originals to indices in unique_ids (unique is sorted)
    orig_idx = np.searchsorted(unique_ids, out[mask])
    n = unique_ids.size

    # Choose random indices in [0, n); then ensure different from orig_idx
    rand_idx = rng.integers(0, n, size=orig_idx.size, dtype=np.int64)
    # If equals, shift by +1 modulo n → guaranteed different
    rand_idx = (rand_idx + (rand_idx == orig_idx)) % n

    out[mask] = unique_ids[rand_idx]
    return _make_submission(hit_ids, out)


def jitter_seed_points(
    points: np.ndarray,
    sigma: float = 1e-3,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Add i.i.d. **Gaussian jitter** :math:`\mathcal{N}(0,\sigma^2)` to seed
    point coordinates (first 3 columns).

    For an input matrix :math:`P \in \mathbb{R}^{N\times M}` with
    :math:`M\ge 3`, return

    .. math::

        \tilde{P}_{:,0:3} \;=\; P_{:,0:3} + E,\qquad
        E_{ij} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2),

    while columns :math:`3..M-1` (if any) are left unchanged.

    Parameters
    ----------
    points : (N,3) or (N,M>=3) array_like
        Seed point coordinates. Only the first three columns are perturbed.
    sigma : float, optional
        Standard deviation :math:`\sigma` of the isotropic noise (default ``1e-3``).
    rng : numpy.random.Generator, optional
        Random generator; a default generator is used if omitted.

    Returns
    -------
    out : (N,M) ndarray (float64)
        Copy of the input with jitter added to columns 0..2.

    Raises
    ------
    ValueError
        If ``points`` is not 2D with at least 3 columns.

    Notes
    -----
    - The noise covariance on each row's spatial part is
      :math:`\sigma^2 I_3`.
    - If ``sigma <= 0``, the function returns a copy with no modification.

    Examples
    --------
    >>> pts = np.array([[0.,0.,0.],[1.,1.,1.]])
    >>> out = jitter_seed_points(pts, sigma=1e-6, rng=np.random.default_rng(0))
    >>> out.shape
    (2, 3)
    """
    pts = np.asarray(points, dtype=np.float64, order="C")
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, 3) or (N, M>=3).")
    if sigma <= 0.0:
        return pts.copy()

    rng = np.random.default_rng() if rng is None else rng
    noise = rng.normal(loc=0.0, scale=float(sigma), size=pts[:, :3].shape)
    out = pts.copy()
    out[:, :3] += noise
    return out
