import pandas as pd
import numpy as np

def _make_submission(hit_ids: np.ndarray, track_ids: np.ndarray, renumber: bool = True) -> pd.DataFrame:
    r"""
    Create a submission table mapping hit IDs to track IDs.

    Optionally, the track IDs can be randomly re-labeled (renumbered) to
    anonymize them while preserving the grouping structure.

    Parameters
    ----------
    hit_ids : numpy.ndarray
        Array of hit identifiers, shape ``(N,)``.
    track_ids : numpy.ndarray
        Array of track identifiers, shape ``(N,)``.
    renumber : bool, optional
        If ``True``, renumbers the track IDs to a random permutation of
        ``1, 2, \dots, N_{\text{tracks}}`` where :math:`N_{\text{tracks}}`
        is the number of unique tracks. Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:

        * ``hit_id`` — Hit identifiers.
        * ``track_id`` — Corresponding track identifiers (possibly renumbered).
    """
    if renumber:
        unique_ids, inverse = np.unique(track_ids, return_inverse=True)
        numbers = np.arange(1, len(unique_ids) + 1, dtype=unique_ids.dtype)
        np.random.shuffle(numbers)
        track_ids = numbers[inverse]
    return pd.DataFrame({'hit_id': hit_ids, 'track_id': track_ids})

def random_solution(hits: pd.DataFrame, ntracks: int) -> pd.DataFrame:
    r"""
    Assign random track IDs to hits to create a baseline or control submission.

    Each hit is assigned to a random track ID sampled uniformly from
    ``1, 2, \dots, n_{\text{tracks}}``.

    Parameters
    ----------
    hits : pandas.DataFrame
        Must contain a ``hit_id`` column.
    ntracks : int
        Total number of unique track IDs to use.

    Returns
    -------
    pandas.DataFrame
        Submission DataFrame with ``hit_id`` from ``hits`` and random
        ``track_id`` assignments.
    """
    ids = np.random.randint(1, ntracks + 1, size=len(hits), dtype='i4')
    return _make_submission(hits['hit_id'].values, ids, renumber=False)

def drop_hits(truth: pd.DataFrame, probability: float) -> pd.DataFrame:
    r"""
    Randomly remove (drop) a fraction of hits by assigning them to new
    "fake" track IDs.

    Each hit is dropped with probability :math:`p`, and assigned to a new
    unique track ID not present in the original truth.

    Parameters
    ----------
    truth : pandas.DataFrame
        Ground truth table containing:

        * ``hit_id`` — Hit identifiers.
        * ``particle_id`` — True particle identifiers.
    probability : float
        Drop probability :math:`p \in [0, 1]` for each hit.

    Returns
    -------
    pandas.DataFrame
        Submission DataFrame where dropped hits have new unique ``track_id``.
    """
    out = np.array(truth['particle_id'], copy=True)
    dropped_mask = (np.random.random_sample(len(out)) < probability)
    dropped_count = np.count_nonzero(dropped_mask)
    fakeid0 = np.max(out) + 1
    fakeids = np.arange(fakeid0, fakeid0 + dropped_count, dtype='i8')
    np.place(out, dropped_mask, fakeids)
    return _make_submission(truth['hit_id'].values, out)

def shuffle_hits(truth: pd.DataFrame, probability: float) -> pd.DataFrame:
    r"""
    Randomly reassign a fraction of hits to the wrong particle IDs.

    Each hit is shuffled with probability :math:`p` by assigning it a
    random particle ID sampled from the set of existing IDs.

    Parameters
    ----------
    truth : pandas.DataFrame
        Ground truth table containing:

        * ``hit_id`` — Hit identifiers.
        * ``particle_id`` — True particle identifiers.
    probability : float
        Shuffle probability :math:`p \in [0, 1]` for each hit.

    Returns
    -------
    pandas.DataFrame
        Submission DataFrame with some hits reassigned to incorrect ``track_id``.
    """
    out = np.array(truth['particle_id'], copy=True)
    shuffled_mask = (np.random.random_sample(len(out)) < probability)
    shuffled_count = np.count_nonzero(shuffled_mask)
    wrongparticles = np.random.choice(np.unique(out), size=shuffled_count)
    np.place(out, shuffled_mask, wrongparticles)
    return _make_submission(truth['hit_id'].values, out)

def jitter_seed_points(points: np.ndarray, sigma: float = 0.001) -> np.ndarray:
    r"""
    Apply Gaussian jitter to seed point coordinates to simulate
    measurement noise.

    Each coordinate :math:`x` is perturbed as:

    .. math::
        x' = x + \mathcal{N}(0, \sigma^2)

    Parameters
    ----------
    points : numpy.ndarray
        Array of seed point coordinates, shape ``(N, 3)``.
    sigma : float, optional
        Standard deviation :math:`\sigma` of the Gaussian noise (in the
        same units as ``points``). Default is ``0.001``.

    Returns
    -------
    numpy.ndarray
        Jittered coordinates, shape ``(N, 3)``.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=points.shape)
    return points + noise