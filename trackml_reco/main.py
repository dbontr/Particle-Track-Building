import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trackml.score import score_event
from trackml_reco.branchers.ekf import HelixEKFBrancher
from trackml_reco.branchers.astar import HelixEKFAStarBrancher
from trackml_reco.branchers.aco import HelixEKFACOBrancher
from trackml_reco.branchers.pso import HelixEKFPSOBrancher
from trackml_reco.branchers.sa import HelixEKFSABrancher
from trackml_reco.branchers.ga import HelixEKFGABrancher
from trackml_reco.branchers.hungarian import HelixEKFHungarianBrancher
from trackml_reco.track_builder import TrackBuilder
from trackml_reco.parallel_track_builder import CollaborativeParallelTrackBuilder
import trackml_reco.data as trk_data
import trackml_reco.plotting as trk_plot
import trackml_reco.trees as trk_trees
import trackml_reco.hit_pool as trk_hit_pool
import trackml_reco.metrics as trk_metrics

def main():
    brancher_keys = ['ekf', 'astar', 'aco', 'pso', 'sa', 'ga', 'hungarian']
    brancher_cls_map = {
        'ekf': HelixEKFBrancher,
        'astar': HelixEKFAStarBrancher,
        'aco': HelixEKFACOBrancher,
        'pso': HelixEKFPSOBrancher,
        'sa': HelixEKFSABrancher,
        'ga': HelixEKFGABrancher,
        'hungarian': HelixEKFHungarianBrancher
    }
    parser = argparse.ArgumentParser(
        description="Run refactored track building on a TrackML event"
    )
    parser.add_argument('--file','-f', type=str, default='train_1.zip', 
                        help='input .zip event (default: train_1.zip)')
    parser.add_argument('--pt','-p', type=float, default=2.0,
                        help='minimum pT threshold in GeV (default: 2.0)')
    parser.add_argument('--debug-n','-d', type=int, default=None, 
                        help='if set, only process this many seeds (default: None)')
    parser.add_argument('--plot', type=bool, default=True, 
                        help='plot graphs of tracks (default: True)')
    parser.add_argument('--extra-plots', type=bool, default=False, 
                        help='displays extra presentation plots (default: False)')
    parser.add_argument('--parallel', type=bool, default=False, 
                        help='toggles parallel collaborative track building (default: False)')
    parser.add_argument('--brancher', '-b',type=str, choices=brancher_keys,
    default='ekf',metavar='BRANCHER',
    help=("""
        Branching strategy to use for the tracker.
        Options:
        ekf        - Extended Kalman Filter branching
        astar      - A* search-based branching
        aco        - Ant Colony Optimization-based branching
        pso        - Particle Swarm Optimization-based branching
        sa         - Simulated Annealing-based branching
        ga         - Genetic Algorithm-based branching
        hungarian  - Hungarian Algorithm for optimal hit-to-track assignment
        (default: 'ekf')
        """))
    
    args = parser.parse_args()

    # 1. Load & preprocess data
    print("Loading and preprocessing data...")
    hit_pool= trk_data.load_and_preprocess(args.file, pt_threshold=args.pt)
    
    hits, pt_cut_hits = hit_pool.hits, hit_pool.pt_cut_hits 
    layer_tuples = sorted(set(zip(hits.volume_id, hits.layer_id)), key=lambda x:(x[0],x[1]))
    layer_surfaces = {}
    for vol, lay in layer_tuples:
        df = hits[(hits.volume_id==vol)&(hits.layer_id==lay)]
        if df.empty: continue

        z_span = df.z.max() - df.z.min()
        r_vals = np.sqrt(df.x**2 + df.y**2)
        r_span = r_vals.max() - r_vals.min()

        if z_span < r_span * 0.1:
            plane_n = np.array([0.,0.,1.])
            plane_p = np.array([0.,0., df.z.mean()])
            layer_surfaces[(vol,lay)] = {
                'type': 'disk',
                'n'   : plane_n,
                'p'   : plane_p
            }
        else:
            R = r_vals.mean()
            layer_surfaces[(vol,lay)] = {
                'type': 'cylinder',
                'R'   : R
            }


    if args.extra_plots:
        print("Plotting detector and truth...")
        trk_plot.plot_hits_colored_by_layer(hits, layer_tuples)
        trk_plot.plot_layer_boundaries(hits, layer_tuples) 
        trk_plot.plot_truth_paths_rz(pt_cut_hits, max_tracks=None)
        trk_plot.plot_truth_paths_3d(pt_cut_hits, max_tracks=None)

    with open('config.json') as f:
        config = json.load(f)

    # Set layer_surfaces for each config
    for key in brancher_keys:
        config_key = f"ekf{key}_config" if key != "ekf" else "ekf_config"
        config[config_key]['layer_surfaces'] = layer_surfaces

    # Build the appropriate config and brancher class dynamically
    brancher_config_key = f"ekf{args.brancher}_config" if args.brancher != "ekf" else "ekf_config"
    
    
    track_builder_type = CollaborativeParallelTrackBuilder if args.parallel else TrackBuilder

    track_builder = track_builder_type(
        hit_pool=hit_pool,
        brancher_cls=brancher_cls_map[args.brancher],
        brancher_config=config[brancher_config_key]
    )

    # Build seeds and tracks
    print("Building seeds and tracks from truth hits...")
    completed_tracks = track_builder.build_tracks_from_truth(
        max_seeds=args.debug_n,
        max_tracks_per_seed=config['ekf_config']['num_branches'],  # Still from base EKF config
        max_branches=config['ekf_config']['survive_top']
    )

    # 6. Plot seeds if requested
    if args.plot:
        seeds_df = track_builder.get_seeds_dataframe()
        if not seeds_df.empty:
            # Group by particle_id to get 3-point seeds for plotting
            seed_groups = []
            for particle_id, group in seeds_df.groupby('particle_id'):
                if len(group) >= 3:
                    group_sorted = group.sort_values('seed_point_index')
                    seed_groups.append(group_sorted[['x', 'y', 'z']].values)
            
            if seed_groups:
                # Convert to DataFrame format expected by plotting functions
                plot_data = []
                for i, points in enumerate(seed_groups):
                    for j, point in enumerate(points):
                        plot_data.append({
                            'particle_id': i,
                            'x': point[0], 'y': point[1], 'z': point[2],
                            'r': np.sqrt(point[0]**2 + point[1]**2)
                        })
                
                plot_df = pd.DataFrame(plot_data)
                trk_plot.plot_seed_paths_rz(plot_df, max_seeds=args.debug_n)
                trk_plot.plot_seed_paths_3d(plot_df, max_seeds=args.debug_n)
    

    # 7. Get statistics and best tracks
    stats = track_builder.get_track_statistics()
    print(f"\nTrack building statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    best_tracks = track_builder.get_best_tracks(n=min(10, len(completed_tracks)))
    print(f"\nBest {len(best_tracks)} tracks:")
    
    # 8. Evaluate tracks against truth and create submission
    submission_list = []
    all_mses = []
    all_pcts = []
    
    for i, track in enumerate(best_tracks):
        # Get truth hits for this particle
        truth_particle = track_builder.hit_pool.pt_cut_hits[track_builder.hit_pool.pt_cut_hits.particle_id == track.particle_id]
        if len(truth_particle) == 0:
            continue
            
        # Convert track trajectory to numpy array for metrics
        track_traj = np.array(track.trajectory)
        truth_xyz = truth_particle[['x', 'y', 'z']].values
        
        # Calculate metrics
        mse = trk_metrics.branch_mse({'traj': track_traj}, truth_xyz)
        pct_hits, _ = trk_metrics.branch_hit_stats({'traj': track_traj}, truth_xyz)
        
        all_mses.append(mse)
        all_pcts.append(pct_hits)
        
        print(f"Track {i+1} (PID={track.particle_id}): MSE={mse:.3f}, %hits={pct_hits:.1f}%")
        
        # Add to submission
        for hit_id in track.hit_ids:
            submission_list.append({'hit_id': int(hit_id), 'track_id': track.particle_id})
        
        # Plot best tracks
        if args.plot and i < 3:  # Plot first 3 tracks
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(track_builder.hit_pool.hits.x, track_builder.hit_pool.hits.y, track_builder.hit_pool.hits.z, c='gray', s=1, alpha=0.1)
            
            # Plot truth
            tp = truth_particle.sort_values('r')
            ax.plot(tp.x, tp.y, tp.z, '--k', label='truth')
            
            # Plot reconstructed track
            ax.plot(track_traj[:,0], track_traj[:,1], track_traj[:,2], '-b', label='reconstructed')
            ax.set_title(f'Track {i+1} (PID {track.particle_id})')
            ax.legend()
            plt.show()
    
    if submission_list:
        submission_df = pd.DataFrame(submission_list).drop_duplicates('hit_id')
        score = score_event(
            track_builder.hit_pool.pt_cut_hits[['hit_id','particle_id','weight']],
            submission_df
        )
        
        # Print averages over all tracks processed
        if all_mses:
            print(f"\nAverage MSE over {len(all_mses)} tracks: {np.mean(all_mses):.3f}")
            print(f"Average %hits over {len(all_pcts)} tracks: {np.mean(all_pcts):.1f}%")
        print("Event score:", score)
    else:
        print("No tracks were successfully built")

if __name__ == '__main__':
    main()