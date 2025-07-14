import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from trackml_reco.branchers.brancher import Brancher

class HelixEKFBrancher(Brancher):
    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple['cKDTree', np.ndarray, np.ndarray]],
                 noise_std: float = 2.0,
                 B_z: float = 0.002,
                 num_branches: int = 30,
                 survive_top: int = 12,
                 max_cands: int = 10,
                 step_candidates: int = 5):
        """
        Initializes the HelixEKFBrancher with geometry, filter parameters, and branching configuration.

        Parameters
        ----------
        trees : dict
            Dictionary mapping (volume_id, layer_id) to (cKDTree, points, hit_ids) for fast spatial lookup.
        layers : list of tuple
            Ordered list of (volume_id, layer_id) specifying the tracking layers.
        true_xyzs : list of ndarray
            True hit coordinates for performance metrics and optional use in gating.
        noise_std : float, optional
            Standard deviation of measurement noise. Default is 2.0.
        B_z : float, optional
            Magnetic field strength along the z-axis (in Tesla). Default is 0.002.
        num_branches : int, optional
            Total number of branches maintained per layer. Default is 30.
        survive_top : int, optional
            Number of top-scoring branches to preserve per layer. Default is 12.
        max_cands : int, optional
            Maximum number of hit candidates considered per layer. Default is 10.
        step_candidates : int, optional
            Number of hit candidates used to branch at each step. Default is 5.
        """
        self.trees           = trees
        self.noise_std       = noise_std
        self.B_z             = B_z
        self.num_branches    = num_branches
        self.survive_top     = survive_top
        self.survive_rand    = num_branches - survive_top
        self.max_cands       = max_cands
        self.step_candidates = step_candidates
        self.state_dim       = 7

        # measurement & process noise
        self.R      = (noise_std**2) * np.eye(3)
        self.Q0     = np.diag([1e-4]*3 + [1e-5]*3 + [1e-6]) * noise_std
        
        # gating thresholds
        self.inner_gate = 9.0
        self.outer_gate = 16.0

        self.layer_normals = np.array([0,0,1])
        self.layer_points = np.array([0,0,0])

    def run(self, seed_xyz: np.ndarray, layers: List, t: np.ndarray, plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
        """
        Runs the full Extended Kalman Filter with branching on seed input.

        Parameters
        ----------
        seed_xyz : ndarray
            Initial three 3D hit coordinates.
        t : ndarray
            Time points associated with each seed position.
        plot_tree : bool, optional
            If True, plots the final branch tree.

        Returns
        -------
        tuple
            (List of final branches, networkx.DiGraph representing the branching tree).
        """
        dt = t[1] - t[0]
        v0, κ0 = self._estimate_seed_helix(seed_xyz, dt, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, κ0])
        P0 = np.eye(self.state_dim) * 0.1

        branches = [{'id':0,'parent':None,'traj':list(seed_xyz),'state':x0,'cov':P0,'score':0.0}]
        G = nx.DiGraph(); G.add_node(0, pos=seed_xyz[2]); next_id=1

        # uses truth level layers? 
        for i, layer in enumerate(layers):
            gate = self.inner_gate if i<3 else self.outer_gate
            plane_n = self.layer_normals
            plane_p = self.layer_points
            new_br = []
            for br in branches:
                F = self.compute_F(br['state'], dt)
                x_pred = self.propagate(br['state'], dt)
                P_pred = F @ br['cov'] @ F.T + self.Q0 * dt

                points_cand, id_cand = self._get_candidates(x_pred[:3], layer)
                for z, hid in zip(points_cand, id_cand):
                    meas_pred, Puv, H2 = self.to_local_frame(x_pred[:3], P_pred, plane_n, plane_p)
                    meas_z, Ruv, _ = self.to_local_frame(z, np.zeros((3,3)), plane_n, plane_p)
                    Suv  = Puv + Ruv
                    res_uv = meas_z - meas_pred
                    chi2 = res_uv @ np.linalg.solve(Suv, res_uv)

                    if chi2 < gate:
                        Huv    = self.to_local_frame_jac(plane_n)        # 2×3
                        H_full = np.zeros((2, self.state_dim))           # 2×7
                        H_full[:, :3] = Huv

                        K_uv = P_pred[:3, :3] @ Huv.T @ np.linalg.inv(Suv)  # 3×2
                        K    = np.zeros((self.state_dim, 2))               # 7×2
                        K[:3, :] = K_uv

                        x_upd = x_pred + K @ res_uv
                        P_upd = (np.eye(self.state_dim) - K @ H_full) @ P_pred
                        score = br['score'] + chi2
                    else:
                        x_upd = x_pred.copy()
                        x_upd[6] += np.random.randn() * 1e-4
                        P_upd = P_pred
                        score = br['score'] + chi2 + 5.0

                    traj    = br['traj'] + [x_upd[:3]]
                    node_id = next_id
                    G.add_node(node_id, pos=x_upd[:3])
                    G.add_edge(br['id'], node_id)
                    next_id += 1

                    new_br.append({
                        'id'      : node_id,
                        'parent'  : br['id'],
                        'traj'    : traj,
                        'state'   : x_upd,
                        'cov'     : P_upd,
                        'score'   : score,
                        'hit_ids' : br.get('hit_ids', []) + [int(hid)]
                    })
            if not new_br: break
            new_br.sort(key=lambda b: b['score'])
            branches = new_br[:self.num_branches]

        if plot_tree: self._plot_tree(G)
        return branches, G


    def _plot_tree(self, G: nx.DiGraph) -> None:
        """
        Plots the branch tree in XY view.

        Parameters
        ----------
        G : networkx.DiGraph
            Directed graph representing the branch tree.

        Returns
        -------
        None
        """
        pos={n:tuple(G.nodes[n]['pos'][:2]) for n in G.nodes()}
        plt.figure(figsize=(8,8)); nx.draw(G,pos,with_labels=True,node_size=50,arrowsize=10)
        plt.title('Branching tree (XY projection)'); plt.show()