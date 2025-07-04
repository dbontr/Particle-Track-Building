import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trackml.score import score_event

from trackml_reco import (
    load_and_preprocess,
    build_layer_trees,
    jitter_seed_points,
    HelixEKFBrancher,
    branch_mse,
    branch_hit_stats,
)
from trackml_reco.plotting import (
    plot_hits_colored_by_layer,
    plot_layer_boundaries,
    plot_truth_paths_rz,
    plot_truth_paths_3d,
    plot_seed_paths_rz,
    plot_seed_paths_3d,
)

PLOT = True
DEBUG_N = None  # Set to None to run through all seeds, or an int to limit
FILE_NAME = 'train_1.zip'

def main():
    parser = argparse.ArgumentParser(
        description="Run HelixEKFBrancher on a TrackML event"
    )
    parser.add_argument(
        '--file','-f',
        type=str,
        default='train_1.zip',
        help='input .zip event (default: train_1.zip)'
    )
    parser.add_argument(
        '--pt','-p',
        type=float,
        default=2.0,
        help='minimum pT threshold in GeV (default: 2.0)'
    )
    parser.add_argument(
        '--debug-n','-d',
        type=int,
        default=None,
        help='if set, only process this many seeds'
    )
    args = parser.parse_args()

    FILE_NAME = args.file
    DEBUG_N = args.debug_n

    # 1. Load & preprocess (unchanged)…
    hits, truth_hits, particles = load_and_preprocess(FILE_NAME, pt_threshold=args.pt)

    # Build layer list for coloring
    layer_tuples = sorted(set(zip(hits.volume_id, hits.layer_id)), key=lambda x:(x[0],x[1]))
    
    # 2. Plot the detector & truth before running EKF
    if PLOT:
        plot_hits_colored_by_layer(hits, layer_tuples)
        plot_layer_boundaries(hits, layer_tuples) 
        plot_truth_paths_rz(truth_hits, max_tracks=None)
        plot_truth_paths_3d(truth_hits, max_tracks=None)

    # 2. Build KD-trees & layers (unchanged)…
    trees, layers = build_layer_trees(hits)
    
    # 3. Build seeds _and_ capture the “future” true_xyzs for each seed
    seeds = []
    all_mses = []
    all_pcts = []
    for pid, grp in truth_hits.groupby('particle_id'):
        grp = grp.sort_values('r')
        if len(grp) < 4:  # need at least 3 for seed + 1 true layer
            continue
        seed_points = grp[['x','y','z']].values[:3]
        seed_points = jitter_seed_points(seed_points, sigma=0.001)
        
        # the “true” hits beyond the seed (for candidate injection)
        rest = grp[['x','y','z']].values[3:]
        # the layers corresponding to those rest hits
        true_layers = list(zip(grp.volume_id.values[3:], grp.layer_id.values[3:]))
        
        seeds.append((pid, seed_points, true_layers, rest))
        if DEBUG_N is not None and len(seeds) >= DEBUG_N:
            break
    seed_rows = []
    for pid, seed_points, _, _ in seeds:
        for pt in seed_points:
            seed_rows.append({
                'particle_id': pid,
                'x': pt[0], 'y': pt[1], 'z': pt[2]
            })
    seeds_df = pd.DataFrame(seed_rows)
    # now call your two helpers
    if PLOT:
        plot_seed_paths_rz(seeds_df, max_seeds=DEBUG_N)
        plot_seed_paths_3d(seeds_df, max_seeds=DEBUG_N)
    
    # 4. Loop over seeds—but now with HelixEKFBrancher
    submission_list = []
    for pid, seed_points, true_layers, true_xyzs in seeds:
        # instantiate your brancher
        ekf = HelixEKFBrancher(
            trees=trees,
            layers=true_layers,
            true_xyzs=true_xyzs,
            noise_std=1.0,
            B_z=0.002,
            num_branches=30,
            survive_top=12,
            max_cands=10,
            step_candidates=5
        )
        
        # create a time array matching the number of layers (plus a bit)
        t = np.linspace(0, 1, len(true_layers) + 3)
        
        # run the branching EKF
        branches, graph = ekf.run(seed_points, t, plot_tree=PLOT)
        
        # pick the best branch by lowest MSE to true_xyzs
        best = min(branches, key=lambda br: branch_mse(br, true_xyzs))
        best_traj = np.array(best['traj'])

        # you can compute metrics, plot, etc. just like before…
        mse = branch_mse(best, true_xyzs)
        pct_hits, _ = branch_hit_stats(best, true_xyzs)
        all_mses.append(mse)
        all_pcts.append(pct_hits)
        print(f"PID={pid}: MSE={mse:.3f}, %hits={pct_hits:.1f}")

        if PLOT:
            fig = plt.figure(figsize=(6,6))
            ax  = fig.add_subplot(111, projection='3d')
            ax.scatter(hits.x, hits.y, hits.z, c='gray', s=1, alpha=0.1)
            tp = truth_hits[truth_hits.particle_id==pid].sort_values('r')
            ax.plot(tp.x, tp.y, tp.z, '--k', label='truth')
            ax.plot(best_traj[:,0], best_traj[:,1], best_traj[:,2], '-b', label='best estimate')
            ax.set_title(f'PID {pid} (best branch)')
            ax.legend()
            plt.show()
        
        # collect hit IDs—brancher doesn’t track IDs by default, so if you need those:
        #   you could augment HelixEKFBrancher to also carry `hit_id` alongside xyz
        #   or simply project your best_traj points back onto the KD-tree:
        tree = trees[true_layers[-1]][0]  # example for last layer
        _, idxs = tree.query(best_traj, k=1)
        # map idxs back to hit_ids if you stored them in your tree
        # here I’m assuming your `trees` dict was (tree, points, ids) as in Script 1:
        ids = trees[true_layers[-1]][2][idxs]
        
        for hid in ids:
            submission_list.append({'hit_id': int(hid), 'track_id': pid})
    
    # 5. Final submission, scoring, etc. (unchanged)…
    submission_df = pd.DataFrame(submission_list).drop_duplicates('hit_id')
    score = score_event(
        truth_hits[['hit_id','particle_id','weight']],
        submission_df
    )
    # Print averages over all seeds processed
    if all_mses:
        print(f"\nAverage MSE over {len(all_mses)} seeds: {np.mean(all_mses):.3f}")
        print(f"Average %hits over {len(all_pcts)} seeds: {np.mean(all_pcts):.1f}%")
    print("Event score:", score)

main()