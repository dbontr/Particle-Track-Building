import time
from typing import Optional, List, Tuple, Any

import itertools
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from sklearn.metrics import mean_squared_error


class CombinatorialKalmanFilter:
    """
    Combinatorial Kalman Filter (CKF) for deterministic track finding.

    Attributes
    ----------
    Q : np.ndarray
        Process noise covariance (4x4).
    R : np.ndarray
        Measurement noise covariance (2x2).
    spatial_window : Tuple[float, float]
        Maximum allowed spatial deviation in x and y during gating.
    threshold : float
        Maximum Mahalanobis distance for gating.
    max_branches : int
        Maximum number of concurrent branches to keep.
    max_depth : int
        Maximum search depth (number of sequential updates).
    max_angle : float
        Maximum angular deviation (degrees) between segment directions.
    seed : np.ndarray
        Initial seed hits (N x 2).
    candidates : np.ndarray
        Candidate measurement points (M x 2).
    branches : List[List[Tuple[float, float]]]
        Final list of tracked branches (paths).
    mse_values : List[float]
        MSE for each branch if ground truth provided.
    best_idx : Optional[int]
        Index of branch with lowest MSE.

    Methods
    -------
    fit(seed, candidates, true_track=None) -> List[List[Tuple[float, float]]]
        Run CKF on a single seed and set of candidates.
    plot(true_track=None) -> None
        Plot all branches and overlay best branch if applicable.
    """

    def __init__(
        self,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        spatial_window: Tuple[float, float] = (0.8, 0.3),
        threshold: float = 2.5,
        max_branches: int = 10,
        max_depth: int = 22,
        max_angle: float = 30.0
    ) -> None:
        """
        Initialize CKF parameters.

        Parameters
        ----------
        Q : Optional[np.ndarray]
            Process noise covariance matrix (4x4). Defaults to 0.01*I.
        R : Optional[np.ndarray]
            Measurement noise covariance matrix (2x2). Defaults to 0.05*I.
        spatial_window : Tuple[float, float]
            (wx, wy) gating window around predicted position.
        threshold : float
            Mahalanobis distance gating threshold.
        max_branches : int
            Maximum number of branches to retain per iteration.
        max_depth : int
            Maximum tree depth (number of hits) to explore.
        max_angle : float
            Maximum angle (degrees) allowed between consecutive steps.
        """
        self.Q = Q if Q is not None else np.eye(4) * 0.01
        self.R = R if R is not None else np.eye(2) * 0.05
        self.spatial_window = spatial_window
        self.threshold = threshold
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.max_angle = max_angle
        self.seed = None  # type: Optional[np.ndarray]
        self.candidates = None  # type: Optional[np.ndarray]
        self.branches: List[List[Tuple[float, float]]] = []
        self.mse_values: List[float] = []
        self.best_idx: Optional[int] = None

    class KalmanFilter:
        """
        Simple linear Kalman Filter for 2D constant-velocity model.
        """

        def __init__(self, Q: np.ndarray, R: np.ndarray) -> None:
            """
            Parameters
            ----------
            Q : np.ndarray
                Process noise covariance (4x4).
            R : np.ndarray
                Measurement noise covariance (2x2).
            """
            self.Q = Q
            self.R = R

        def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Predict next state and covariance.

            Parameters
            ----------
            x : np.ndarray
                Current state vector [x, y, vx, vy].
            P : np.ndarray
                Current covariance matrix (4x4).

            Returns
            -------
            x_pred : np.ndarray
                Predicted state.
            P_pred : np.ndarray
                Predicted covariance.
            """
            F = np.eye(4)
            F[0, 2] = 1
            F[1, 3] = 1
            x_pred = F @ x
            P_pred = F @ P @ F.T + self.Q
            return x_pred, P_pred

        def update(
            self,
            x_pred: np.ndarray,
            P_pred: np.ndarray,
            z: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Perform measurement update.

            Parameters
            ----------
            x_pred : np.ndarray
                Predicted state.
            P_pred : np.ndarray
                Predicted covariance.
            z : np.ndarray
                Measurement [x, y].

            Returns
            -------
            x_upd : np.ndarray
                Updated state.
            P_upd : np.ndarray
                Updated covariance.
            """
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_upd = x_pred + K @ y
            P_upd = (np.eye(4) - K @ H) @ P_pred
            return x_upd, P_upd

    @staticmethod
    def mahalanobis_dist(
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
        R: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis distance between predicted position and measurement.

        Parameters
        ----------
        x_pred : np.ndarray
            Predicted state [x, y, vx, vy].
        P_pred : np.ndarray
            Predicted covariance (4x4).
        z : np.ndarray
            Measurement [x, y].
        R : np.ndarray
            Measurement noise covariance (2x2).

        Returns
        -------
        float
            Mahalanobis distance.
        """
        y = z - x_pred[:2]
        S = P_pred[:2, :2] + R
        return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute angle (degrees) between two vectors.

        Parameters
        ----------
        v1 : np.ndarray
            First vector.
        v2 : np.ndarray
            Second vector.

        Returns
        -------
        float
            Angle in degrees.
        """
        v1_u = v1 / (np.linalg.norm(v1) + 1e-9)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-9)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return float(np.degrees(np.arccos(dot)))

    @staticmethod
    def track_mse(
        track: List[Tuple[float, float]],
        true_track: np.ndarray
    ) -> float:
        """
        Compute sum of MSE on x and y between branch and ground truth.

        Parameters
        ----------
        track : List[Tuple[float, float]]
            Predicted branch points.
        true_track : np.ndarray
            Ground truth points (N x 2).

        Returns
        -------
        float
            Sum of x- and y-MSE.
        """
        length = min(len(track), len(true_track))
        rx, ry = zip(*track[:length])
        tx = true_track[:length, 0]
        ty = true_track[:length, 1]
        return mean_squared_error(tx, rx) + mean_squared_error(ty, ry)

    def _ckf_branches(
        self,
        seed: np.ndarray,
        candidates: np.ndarray
    ) -> List[List[Tuple[float, float]]]:
        """
        Internal search over candidate hits using a tree of Kalman
        filter branches with gating. Fixed to avoid array comparison errors by
        adding a tie-breaker counter in heap entries.
        """
        kf = self.KalmanFilter(self.Q, self.R)
        wx, wy = self.spatial_window
        # Initialize from last two seed points
        init_dir = seed[2] - seed[1]
        init_state = np.array([seed[2, 0], seed[2, 1], init_dir[0], init_dir[1]])
        init_cov = np.eye(4)

        # counter to break ties in heap
        counter = itertools.count()
        # heap entries: (score, tie_breaker, path_len, state, cov, path)
        heap: List[Any] = [(0.0, next(counter), len(seed), init_state, init_cov, [tuple(pt) for pt in seed])]

        for _depth in range(self.max_depth):
            new_heap: List[Any] = []
            expansions = 0
            while heap and expansions < self.max_branches:
                score, _, path_len, x, P, path = heappop(heap)
                expansions += 1
                x_pred, P_pred = kf.predict(x, P)
                for pt in candidates:
                    if abs(pt[0] - x_pred[0]) > wx or abs(pt[1] - x_pred[1]) > wy:
                        continue
                    dist = self.mahalanobis_dist(x_pred, P_pred, pt, self.R)
                    vec = np.array(path[-1])
                    ang = self.angle_between(x[2:], pt - vec)
                    if dist < self.threshold and ang < self.max_angle:
                        x_upd, P_upd = kf.update(x_pred, P_pred, pt)
                        new_score = score + dist + ang / 10.0
                        new_path = path + [tuple(pt)]
                        # include tie-breaker to avoid comparing arrays
                        new_heap.append((new_score, next(counter), path_len + 1, x_upd, P_upd, new_path))
            if not new_heap:
                break
            # sort by score then path_len (tie-breaker ignored in sort key)
            new_heap.sort(key=lambda t: (t[0], t[2]))
            # keep only top branches
            heap = new_heap[:self.max_branches]

        # extract just the path lists
        return [entry[5] for entry in heap]

    def fit(
        self,
        seed: np.ndarray,
        candidates: np.ndarray,
        true_track: Optional[np.ndarray] = None
    ) -> List[List[Tuple[float, float]]]:
        """
        Execute CKF and optionally compute best branch vs. ground truth.

        Parameters
        ----------
        seed : np.ndarray
            Initial seed hits (3 x 2).
        candidates : np.ndarray
            Candidate measurements (M x 2).
        true_track : Optional[np.ndarray]
            Ground truth points (N x 2).

        Returns
        -------
        branches : List of branch paths.
        """
        self.seed = seed
        self.candidates = candidates
        self.branches = self._ckf_branches(seed, candidates)
        self.mse_values = []
        self.best_idx = None
        if true_track is not None and len(self.branches) > 0:
            self.mse_values = [self.track_mse(branch, true_track) for branch in self.branches]
            if len(self.mse_values) > 0: self.best_idx = int(np.argmin(self.mse_values))
        else:
            if true_track is not None: print("⚠️  No branches generated; skipping MSE computation.")
        return self.branches
    
    def plot(
        self,
        true_track: Optional[np.ndarray] = None
    ) -> None:
        """
        Visualize CKF branches and optionally true track.

        Parameters
        ----------
        true_track : Optional[np.ndarray]
            Ground truth points to overlay.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 6))
        if true_track is not None:
            plt.plot(true_track[:, 0], true_track[:, 1], 'k--', label='True Track')
        cands = np.array(self.candidates)
        plt.scatter(cands[:, 0], cands[:, 1], alpha=0.5, label='Candidates')
        seeds = np.array(self.seed)
        plt.scatter(seeds[:, 0], seeds[:, 1], c='blue', label='Seed')
        for branch in self.branches:
            arr = np.array(branch)
            plt.plot(arr[:, 0], arr[:, 1], 'r-', alpha=0.2)
        if self.best_idx is not None:
            best = np.array(self.branches[self.best_idx])
            mse = self.mse_values[self.best_idx]
            plt.plot(best[:, 0], best[:, 1], 'r-', linewidth=2,
                     label=f'Best (MSE={mse:.3f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('CKF Branches')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)
    true_track = np.array([[i, 0.3 * i + np.random.normal(0, 0.05)] for i in range(25)])
    seed = true_track[:3]
    random_noise = np.random.uniform([0,0], [25, 8], size=(50,2))
    candidates = np.vstack([true_track[3:], random_noise])

    ckf = CombinatorialKalmanFilter()
    start = time.time()
    branches = ckf.fit(seed, candidates, true_track)
    print(f"Computed {len(branches)} branches in {time.time() - start:.2f}s")
    for idx, mse in enumerate(ckf.mse_values):
        print(f"Branch {idx:2d}: length={len(branches[idx])}, MSE={mse:.4f}")
    ckf.plot(true_track)
