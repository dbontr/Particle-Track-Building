import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Dict
from scipy.spatial import cKDTree

def compute_metrics(xs: np.ndarray, true_points: np.ndarray, tol: float = 0.005) -> Tuple[float, float]:
    """
    Computes MSE and percentage of hits within a given distance tolerance.

    Parameters
    ----------
    xs : ndarray
        Predicted trajectory points (Nx3 or more).
    true_points : ndarray
        Ground truth hit coordinates.
    tol : float, optional
        Distance threshold for considering a hit correct. Default is 0.005.

    Returns
    -------
    tuple
        Mean squared error and percentage of matched hits.
    """
    true_tree = cKDTree(true_points)
    d_pred, _ = true_tree.query(xs[:, :3])
    mse = np.mean(d_pred**2)
    pred_tree = cKDTree(xs[:, :3])
    d_true, _ = pred_tree.query(true_points)
    pct_recovered = 100.0 * np.sum(d_true <= tol) / len(true_points)
    return mse, pct_recovered

def branch_mse(branch: Dict, true_xyz: np.ndarray) -> float:
    """
    Computes mean squared error between a branch and true hit positions.

    Parameters
    ----------
    branch : dict
        A branch dictionary containing 'traj' key.
    true_xyz : ndarray
        Ground truth hit coordinates.

    Returns
    -------
    float
        Mean squared distance between branch trajectory and true hits.
    """
    tree=cKDTree(true_xyz)
    traj=np.array(branch['traj'])
    d2,_=tree.query(traj,k=1)
    return np.mean(d2)

def branch_hit_stats(branch: Dict, true_xyz: np.ndarray, threshold: float = 1.0) -> Tuple[float, int]:
    """
    Computes hit recall statistics for a single branch.

    Parameters
    ----------
    branch : dict
        A branch dictionary containing 'traj' key.
    true_xyz : ndarray
        Ground truth hit coordinates.
    threshold : float, optional
        Maximum distance for a hit to be considered matched. Default is 1.0.

    Returns
    -------
    tuple
        Percentage of true hits matched and number of missed hits.
    """
    traj=np.array(branch['traj'][3:])
    true_points=np.array(true_xyz)
    if traj.shape[0] == 0 or true_points.size == 0:
        return 0.0, len(true_points)

    # if same length, compute pointwise distances
    if traj.shape[0] == true_points.shape[0]:
        dists = np.linalg.norm(traj - true_points, axis=1)
    else:
        # nearest‐neighbor distance from each true hit to any recon point
        dists = np.min(cdist(true_points, traj), axis=1)

    # count matched hits
    matched = np.sum(dists < threshold)
    total = len(true_points)
    pct_matched = 100.0 * matched / total
    missed = total - matched

    return pct_matched, missed