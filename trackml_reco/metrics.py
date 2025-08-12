import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Dict
from scipy.spatial import cKDTree

def compute_metrics(xs: np.ndarray, true_points: np.ndarray, tol: float = 0.005) -> Tuple[float, float]:
    r"""
    Compute the **mean squared error (MSE)** and the **percentage of hits** within
    a given spatial tolerance.

    The MSE is defined as:

    .. math::

        \mathrm{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{x}_i - \mathbf{\hat{x}}_i \right\|_2^2

    where :math:`\mathbf{x}_i` is the predicted point and :math:`\mathbf{\hat{x}}_i` 
    is the nearest ground-truth hit.

    The **percentage recovered** is defined as:

    .. math::

        \%_{\mathrm{hits}} = \frac{\#\{ j : d_j \le \tau \}}{N_\text{true}} \times 100

    where :math:`d_j` is the distance from a true hit to the nearest predicted hit,
    and :math:`\tau` is the tolerance.

    Parameters
    ----------
    xs : ndarray of shape (N, 3) or (N, M)
        Predicted trajectory points. Only the first 3 coordinates (x, y, z) are used.
    true_points : ndarray of shape (N_true, 3)
        Ground truth hit coordinates.
    tol : float, optional
        Distance threshold :math:`\tau` (in the same units as coordinates) 
        for a hit to be considered correctly matched. Default is 0.005.

    Returns
    -------
    tuple of float
        ``(mse, pct_recovered)``, where:

        * ``mse`` — mean squared error in coordinate units.
        * ``pct_recovered`` — percentage of true hits within tolerance.
    """
    true_tree = cKDTree(true_points)
    d_pred, _ = true_tree.query(xs[:, :3])
    mse = np.mean(d_pred**2)
    pred_tree = cKDTree(xs[:, :3])
    d_true, _ = pred_tree.query(true_points)
    pct_recovered = 100.0 * np.sum(d_true <= tol) / len(true_points)
    return mse, pct_recovered

def branch_mse(branch: Dict, true_xyz: np.ndarray) -> float:
    r"""
    Compute the **mean squared error (MSE)** between a reconstructed branch
    trajectory and the ground-truth hit positions.

    The MSE is:

    .. math::

        \mathrm{MSE} = \frac{1}{N} \sum_{i=1}^{N} \min_{j} \left\| \mathbf{t}_i - \mathbf{g}_j \right\|_2^2

    where:
      * :math:`\mathbf{t}_i` is the i-th point on the branch trajectory.
      * :math:`\mathbf{g}_j` is a ground-truth hit.

    Parameters
    ----------
    branch : dict
        A branch dictionary containing a key ``'traj'`` with the predicted
        3D trajectory points.
    true_xyz : ndarray of shape (N_true, 3)
        Ground truth hit coordinates.

    Returns
    -------
    float
        Mean squared distance from each branch point to the nearest true hit.
    """
    tree=cKDTree(true_xyz)
    traj=np.array(branch['traj'])
    d2,_=tree.query(traj,k=1)
    return np.mean(d2)

def branch_hit_stats(branch: Dict, true_xyz: np.ndarray, threshold: float = 1.0) -> Tuple[float, int]:
    r"""
    Compute **hit recall statistics** for a reconstructed branch.

    Recall is computed as the fraction of ground-truth hits within
    a maximum distance threshold :math:`\delta` from any predicted point:

    .. math::

        \mathrm{Recall}(\%) = \frac{\#\{ j : \min_i \| \mathbf{g}_j - \mathbf{t}_i \|_2 < \delta \}}{N_\text{true}} \times 100

    Parameters
    ----------
    branch : dict
        A branch dictionary containing a key ``'traj'`` with predicted
        3D trajectory points.
    true_xyz : ndarray of shape (N_true, 3)
        Ground truth hit coordinates.
    threshold : float, optional
        Maximum allowable distance :math:`\delta` for a hit to be considered matched.
        Default is 1.0.

    Returns
    -------
    tuple
        ``(pct_matched, missed)``, where:

        * ``pct_matched`` — percentage of true hits matched within the threshold.
        * ``missed`` — number of true hits not matched.
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