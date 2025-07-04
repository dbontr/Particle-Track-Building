from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def build_layer_trees(hits: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]], List[Tuple[int, int]]]:
    """
    Builds KD-trees for each detector layer to enable fast hit lookup.

    Parameters
    ----------
    hits : pd.DataFrame
        DataFrame containing hit positions and detector layer IDs.

    Returns
    -------
    Tuple[dict, list]
        Dictionary mapping layer keys to KD-tree tuples (tree, points, hit_ids),
        and sorted list of layer keys.
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