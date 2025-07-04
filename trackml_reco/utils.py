import pandas as pd
import numpy as np

def _make_submission(hit_ids: np.ndarray, track_ids: np.ndarray, renumber: bool = True) -> pd.DataFrame:
    """
    Creates a submission DataFrame mapping hit IDs to track IDs.

    Parameters
    ----------
    hit_ids : np.ndarray
        Array of hit IDs.
    track_ids : np.ndarray
        Array of track IDs.
    renumber : bool, optional
        If True, randomly renumbers track IDs to anonymize them. Default is True.

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with 'hit_id' and 'track_id' columns.
    """
    if renumber:
        unique_ids, inverse = np.unique(track_ids, return_inverse=True)
        numbers = np.arange(1, len(unique_ids) + 1, dtype=unique_ids.dtype)
        np.random.shuffle(numbers)
        track_ids = numbers[inverse]
    return pd.DataFrame({'hit_id': hit_ids, 'track_id': track_ids})

def random_solution(hits: pd.DataFrame, ntracks: int) -> pd.DataFrame:
    """
    Assigns random track IDs to hits to create a baseline solution.

    Parameters
    ----------
    hits : pd.DataFrame
        DataFrame of hit information containing 'hit_id'.
    ntracks : int
        Number of unique track IDs to assign.

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with randomly assigned 'track_id'.
    """
    ids = np.random.randint(1, ntracks + 1, size=len(hits), dtype='i4')
    return _make_submission(hits['hit_id'].values, ids, renumber=False)

def drop_hits(truth: pd.DataFrame, probability: float) -> pd.DataFrame:
    """
    Randomly drops a fraction of hits by assigning fake track IDs.

    Parameters
    ----------
    truth : pd.DataFrame
        Ground truth DataFrame with 'hit_id' and 'particle_id'.
    probability : float
        Probability that a hit is dropped (reassigned to a fake ID).

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with dropped hits replaced by fake track IDs.
    """
    out = np.array(truth['particle_id'], copy=True)
    dropped_mask = (np.random.random_sample(len(out)) < probability)
    dropped_count = np.count_nonzero(dropped_mask)
    fakeid0 = np.max(out) + 1
    fakeids = np.arange(fakeid0, fakeid0 + dropped_count, dtype='i8')
    np.place(out, dropped_mask, fakeids)
    return _make_submission(truth['hit_id'].values, out)

def shuffle_hits(truth: pd.DataFrame, probability: float) -> pd.DataFrame:
    """
    Randomly reassigns some hits to incorrect track IDs.

    Parameters
    ----------
    truth : pd.DataFrame
        Ground truth DataFrame with 'hit_id' and 'particle_id'.
    probability : float
        Probability of a hit being reassigned to a different random particle.

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with shuffled track assignments.
    """
    out = np.array(truth['particle_id'], copy=True)
    shuffled_mask = (np.random.random_sample(len(out)) < probability)
    shuffled_count = np.count_nonzero(shuffled_mask)
    wrongparticles = np.random.choice(np.unique(out), size=shuffled_count)
    np.place(out, shuffled_mask, wrongparticles)
    return _make_submission(truth['hit_id'].values, out)

def jitter_seed_points(points: np.ndarray, sigma: float = 0.001) -> np.ndarray:
    """
    Adds Gaussian noise to seed points to simulate measurement uncertainty.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D seed point coordinates.
    sigma : float, optional
        Standard deviation of Gaussian noise. Default is 0.001.

    Returns
    -------
    np.ndarray
        Jittered seed points.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=points.shape)
    return points + noise