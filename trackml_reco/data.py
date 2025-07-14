from trackml.dataset import load_dataset
from trackml.utils import add_position_quantities, add_momentum_quantities
from trackml.weights import weight_hits_phase1
from typing import Tuple
import pandas as pd
import trackml_reco.hit_pool as trk_hit_pool

def load_and_preprocess(event_zip: str, pt_threshold: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads an event and filters hits by transverse momentum threshold.

    Parameters
    ----------
    event_zip : str
        Path to event ZIP file.
    pt_threshold : float, optional
        Minimum transverse momentum (pT) to retain particles. Default is 2.0.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple of (hits, truth_hits, particles) DataFrames.
    """
    _, hits, _, truth, particles = next(
        load_dataset(event_zip, nevents=1,
                     parts=['hits','cells','truth','particles']))
    # convert to meters
    hits[['x', 'y', 'z']] /= 1000.0
    # derived position and momentum
    hits = add_position_quantities(hits)
    particles = add_momentum_quantities(particles)

    # filter high-pt particles
    valid_ids = set(particles[particles.pt >= pt_threshold].particle_id.astype(int))
    truth_hits = truth.merge(hits, on='hit_id', how='inner')
    pt_cut_hits = truth_hits[truth_hits.particle_id.isin(valid_ids)]

    # compute per-hit weights
    wdf = weight_hits_phase1(truth, particles)

    # choose a weight column (or default to 1.0)
    if 'weight' in wdf.columns:
        chosen = wdf.set_index('hit_id')['weight']
    elif 'weight_pt' in wdf.columns:
        chosen = wdf.set_index('hit_id')['weight_pt']
    else:
        # fallback: everyone weight=1.0
        chosen = pd.Series(1.0, index=wdf.hit_id)

    # map weights into truth_hits, filling any missing with 1.0
    pt_cut_hits['weight'] = pt_cut_hits['hit_id'].map(chosen).fillna(1.0)

    print(f"Loaded {len(hits)} hits, {len(pt_cut_hits)} selected-hits (pt>={pt_threshold})")

    return trk_hit_pool.HitPool(hits, pt_cut_hits)