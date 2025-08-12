from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def build_layer_trees(hits: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]], List[Tuple[int, int]]]:
    r"""
    Construct spatial search structures (KD-trees) for each detector layer.

    Each layer is identified by a tuple ``(volume_id, layer_id)``, and contains
    a set of hit points :math:`\mathbf{p}_i = (x_i, y_i, z_i)`.
    A :class:`scipy.spatial.cKDTree` is built for each such layer to enable
    :math:`O(\log N)` nearest-neighbor queries.

    Parameters
    ----------
    hits : pandas.DataFrame
        Table with at least the following columns:

        * ``volume_id`` : int — Detector volume identifier.
        * ``layer_id`` : int — Layer index within the volume.
        * ``hit_id`` : int — Unique hit identifier.
        * ``x, y, z`` : float — Hit coordinates in Cartesian space.

    Returns
    -------
    trees : dict
        Mapping ``(volume_id, layer_id) -> (tree, points, hit_ids)``, where:

        * ``tree`` — :class:`scipy.spatial.cKDTree` built on ``points``.
        * ``points`` — ndarray of shape ``(n_hits, 3)`` with hit coordinates.
        * ``hit_ids`` — ndarray of shape ``(n_hits,)`` with corresponding IDs.

    layers : list of tuple
        Sorted list of all layer keys ``(volume_id, layer_id)`` present in ``hits``.

    Notes
    -----
    The KD-tree enables fast neighbor queries:

    .. math::

        \text{query time} \;=\; O(\log N) \quad\text{for}\quad N\; \text{hits per layer}

    where :math:`N` is typically much smaller than the total number of hits.
    """
    hits['layer_key'] = list(zip(hits.volume_id, hits.layer_id))
    trees, layers = {}, []
    for key, grp in hits.groupby('layer_key'):
        points = grp[['x', 'y', 'z']].values
        ids = grp['hit_id'].values
        trees[key] = (cKDTree(points), points, ids)
        layers.append(key)
        print(f"Layer {key}: {len(points)} hits")
    layers = sorted(layers, key=lambda x: (x[0], x[1]))
    return trees, layers