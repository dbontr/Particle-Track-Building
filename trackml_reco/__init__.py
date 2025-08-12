__all__ = [
    "load_and_preprocess",
    "build_layer_trees",
    "random_solution", "drop_hits", "shuffle_hits", "jitter_seed_points",
    "Brancher", "HelixEKFBrancher", "HelixEKFAStarBrancher", 
    "HelixEKFACOBrancher", "HelixEKFPSOBrancher", "HelixEKFSABrancher", 
    "HelixEKFGABrancher", "HelixEKFHungarianBrancher",
    "compute_metrics", "branch_mse", "branch_hit_stats", 
    "plot_extras", "plot_seeds", "plot_best_track_3d",
    "check_seed_and_plot", "plot_hits_colored_by_layer", 
    "plot_layer_boundaries", "plot_track_building_debug",
    "plot_truth_paths_rz", "plot_truth_paths_3d",
    "plot_seed_paths_rz", "plot_seed_paths_3d",
    "plot_branches", "_make_submission", 
]

# Data & preprocessing
from .data import load_and_preprocess

# KD‑tree builder
from .trees import build_layer_trees

# Utilities
from .utils import (
    _make_submission,
    random_solution,
    drop_hits,
    shuffle_hits,
    jitter_seed_points
)

# Branchers
from .branchers.brancher import Brancher
from .branchers.ekf import HelixEKFBrancher
from .branchers.astar import HelixEKFAStarBrancher
from .branchers.aco import HelixEKFACOBrancher
from .branchers.pso import HelixEKFPSOBrancher
from .branchers.sa import HelixEKFSABrancher
from .branchers.ga import HelixEKFGABrancher
from .branchers.hungarian import HelixEKFHungarianBrancher

# Metrics
from .metrics import compute_metrics, branch_mse, branch_hit_stats

# Plotting
from .plotting import (
    plot_extras,
    plot_seeds,
    plot_best_track_3d,
    check_seed_and_plot,
    plot_hits_colored_by_layer,
    plot_layer_boundaries,
    plot_truth_paths_rz,
    plot_truth_paths_3d,
    plot_seed_paths_rz,
    plot_seed_paths_3d,
    plot_branches,
    plot_track_building_debug,
)
