import numpy as np
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
    tree = cKDTree(true_points)
    d, _ = tree.query(xs[:,:3])
    mse = np.mean(d**2)
    pct = 100 * np.sum(d <= tol) / len(xs)
    return mse, pct

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
    dists=np.linalg.norm(traj-true_points,axis=1)
    true_hits=np.sum(dists<threshold)
    return true_hits/len(true_points)*100,len(true_points)-true_hits