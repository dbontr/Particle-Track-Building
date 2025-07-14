import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
import heapq
from trackml_reco.branchers.brancher import Brancher

class HelixAStarBrancher(Brancher):
    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 max_cands: int = 10):
        """
        Initializes the A* tracker with Kalman-filter-based gating for LHC tracks.

        Parameters
        ----------
        trees : dict
            Mapping (volume_id, layer_id) to (cKDTree, points array, hit IDs array) for spatial lookup.
        layers : list of tuple
            Ordered list of (volume_id, layer_id) specifying the detector layers.
        noise_std : float, optional
            Standard deviation of measurement noise (default: 2.0).
        B_z : float, optional
            Magnetic field strength along the z-axis in Tesla (default: 0.002).
        max_cands : int, optional
            Maximum number of nearest hit candidates to consider per layer (default: 10).

        Attributes
        ----------
        R : ndarray
            3x3 measurement covariance matrix.
        Q0 : ndarray
            7x7 process noise covariance matrix.
        gate : float
            χ² gating threshold.
        layer_z_positions : list of float
            z-coordinate of each layer plane, extracted from layer geometry.
        """
        self.trees = trees
        self.noise_std = noise_std
        self.B_z = B_z
        self.max_cands = max_cands
        self.R = (noise_std**2) * np.eye(3)
        self.Q0 = np.diag([1e-4]*3 + [1e-5]*3 + [1e-6]) * noise_std
        self.gate = 9.0
        self.layer_z_positions = [self.layer_points[layer][2] for layer in self.layers]

    def _edge_cost(self, br: Dict, z: np.ndarray, layer: Tuple[int,int], dt: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Performs KF predict+update for a candidate hit, returning cost increment.

        Parameters
        ----------
        br : dict
            Current branch state, containing 'state' and 'cov'.
        z : ndarray
            Candidate hit position (3D).
        layer : tuple
            Layer identifier (volume_id, layer_id).
        dt : float
            Time step between layers.

        Returns
        -------
        cost : float
            χ² increment plus any penalties (inf if gated out).
        x_upd : ndarray
            Updated state vector after measurement.
        P_upd : ndarray
            Updated covariance matrix.
        """
        x, P = br['state'], br['cov']
        F = self.compute_F(x, dt)
        x_pred = self.propagate(x, dt)
        P_pred = F @ P @ F.T + self.Q0 * dt
        pT = np.hypot(x_pred[3], x_pred[4])
        # project to plane
        plane_n = np.array([0,0,1]); plane_p = np.array([0,0,0])
        Huv = self.to_local_frame_jac(plane_n)
        meas_pred, Puv = self.to_local_frame(x_pred[:3], P_pred, plane_n, plane_p)
        meas_z, Ruv = self.to_local_frame(z, np.zeros((3,3)), plane_n, plane_p)
        Suv = Puv + Ruv
        res = meas_z - meas_pred
        chi2 = res @ np.linalg.solve(Suv, res)
        if chi2 > self.gate:
            return np.inf, x_pred, P_pred
        # Kalman update
        K = P_pred[:3,:3] @ Huv.T @ np.linalg.inv(Suv)
        x_upd = x_pred + K @ res
        P_upd  = (np.eye(7) - np.vstack([Huv, np.zeros((5,3))]).T @ K) @ P_pred
        κ_seed = br['state'][6]
        Δκ = x_upd[6] - κ_seed
        wκ = 10.0
        penalty = wκ * (Δκ / pT)**2
        return chi2 + penalty, x_upd, P_upd


    def _heuristic(self, state: np.ndarray, current_layer: int) -> float:
        """
        Computes an admissible χ² heuristic combining helix arc length and 3D distance.

        Parameters
        ----------
        state : ndarray
            Current 7D state vector.
        current_layer : int
            Index of the current layer in the tracking sequence.

        Returns
        -------
        h : float
            Estimated lower bound on remaining χ².
        """
        rem = len(self.layers) - current_layer
        if rem <= 0:
            return 0.0
        vx, vy, κ = state[3], state[4], state[6]
        pT = np.hypot(vx, vy) + 1e-12
        R  = pT / (abs(κ) * self.B_z + 1e-12)
        Δφ = rem * (2*np.pi / len(self.layers))
        L  = R * abs(Δφ)
        chi2_arc = (L / self.noise_std)**2
        z_now  = state[2]
        z_last = self.layer_z_positions[-1]
        Δz     = abs(z_last - z_now)
        d3     = np.hypot(L, Δz)
        chi2_3d = (d3 / self.noise_std)**2
        return max(chi2_arc, chi2_3d)

    def run(self, seed_xyz: np.ndarray, t: np.ndarray, plot: bool=False) -> Tuple[Dict, nx.DiGraph]:
        """
        Executes the A* search to build the best track from a 3-point seed.

        Parameters
        ----------
        seed_xyz : ndarray
            Initial three 3D hit coordinates for seeding the track.
        t : ndarray
            Time array corresponding to seed and layer steps.
        plot : bool, optional
            If True, plots the final XY trajectory (default: False).

        Returns
        -------
        best_branch : dict
            Dictionary containing the best track's state, trajectory, hit IDs, and score.
        G : networkx.DiGraph
            Graph of all expanded branches (for diagnostics).
        """
        dt = t[1] - t[0]
        # init seed
        v0, k0 = self._estimate_seed_helix(seed_xyz, dt, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, k0])
        P0 = np.eye(7)*0.1
        start = {'state':x0, 'cov':P0, 'layer_idx':0, 'score':0.0, 'traj':list(seed_xyz), 'hit_ids':[]}

        # A* open set
        open_heap = []  # tuples (f, g, id, branch)
        counter=0
        f0 = self._heuristic(x0,0)
        heapq.heappush(open_heap,(f0,0,counter,start)); counter+=1
        best_final=None
        best_score=np.inf
        G = nx.DiGraph();
        
        while open_heap:
            f, g, _, br = heapq.heappop(open_heap)
            i = br['layer_idx']
            if i >= len(self.layers):
                if g<best_score:
                    best_score=g; best_final=br
                break
            # expand
            layer = self.layers[i]
            candidates, ids = self._get_candidates(br['state'][:3], layer)
            for z, hid in zip(candidates, ids):
                cost, x_new, P_new = self._edge_cost(br, z, layer, dt)
                if cost==np.inf: continue
                g_new = g + cost
                h_new = self._heuristic(x_new, i+1)
                f_new = g_new + h_new
                new_branch = {
                    'state':x_new, 'cov':P_new,
                    'layer_idx':i+1,
                    'score':g_new,
                    'traj':br['traj']+[x_new[:3]],
                    'hit_ids':br['hit_ids']+[int(hid)]}
                heapq.heappush(open_heap,(f_new,g_new,counter,new_branch)); counter+=1
        
        if plot and best_final:
            traj = np.array(best_final['traj'])
            plt.plot(traj[:,0],traj[:,1],'o-'); plt.title('A* best path XY'); plt.show()
        return best_final, G
