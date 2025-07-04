__all__ = [
    "load_and_preprocess",
    "build_layer_trees",
    "random_solution", "drop_hits", "shuffle_hits", "jitter_seed_points",
    "HelixEKFBrancher",
    "compute_metrics", "branch_mse", "branch_hit_stats",
    "plot_hits_colored_by_layer", "plot_layer_boundaries",
    "plot_truth_paths_rz", "plot_truth_paths_3d",
    "plot_seed_paths_rz", "plot_seed_paths_3d",
    "_make_submission", 
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

# Tracker
from .brancher import HelixEKFBrancher

# Metrics
from .metrics import compute_metrics, branch_mse, branch_hit_stats

# Plotting
from .plotting import (
    plot_hits_colored_by_layer,
    plot_layer_boundaries,
    plot_truth_paths_rz,
    plot_truth_paths_3d,
    plot_seed_paths_rz,
    plot_seed_paths_3d,
)
