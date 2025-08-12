from trackml.dataset import load_dataset
from trackml.utils import add_position_quantities, add_momentum_quantities
from trackml.weights import weight_hits_phase1
from typing import Tuple
import pandas as pd
import trackml_reco.hit_pool as trk_hit_pool

def load_and_preprocess(event_zip: str, pt_threshold: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""
    Load a particle collision event from a ZIP file, compute derived kinematic quantities,
    and filter hits by a transverse momentum threshold.

    This function:
      1. Loads hits, truth, and particle tables from an event file.
      2. Converts hit coordinates from millimeters to meters.
      3. Computes additional position and momentum quantities.
      4. Filters particles and their hits based on a minimum transverse momentum \( p_T \).
      5. Assigns per-hit weights for later reconstruction or training.

    The transverse momentum \( p_T \) is defined as:

    .. math::
        p_T = \sqrt{p_x^2 + p_y^2}

    where \( p_x \) and \( p_y \) are the momentum components in the transverse plane.

    Parameters
    ----------
    event_zip : str
        Path to the event `.zip` file containing the detector data.
    pt_threshold : float, optional
        Minimum transverse momentum \( p_T \) (in GeV/c) for a particle to be retained.
        Default is 2.0.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing:

        - **hits** : `pd.DataFrame`
          All hit records with added derived position quantities (in meters).
        - **truth_hits** : `pd.DataFrame`
          Truth-matched hits, merged with geometry, for all particles.
        - **particles** : `pd.DataFrame`
          Particle table with derived momentum quantities and \( p_T \).

    Notes
    -----
    - Input hit positions are converted from millimeters to meters by dividing by \( 10^3 \).
    - Hits are weighted according to the output of `weight_hits_phase1`:
      if the `"weight"` column exists, it is used;
      if not, `"weight_pt"` is used;
      otherwise, all hits are assigned weight \( 1.0 \).
    - Only hits belonging to particles with \( p_T \geq \text{pt_threshold} \) are kept
      in the filtered hit set.
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