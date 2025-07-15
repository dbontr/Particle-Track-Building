import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from scipy.optimize import newton
from trackml_reco.branchers.brancher import Brancher

class HelixEKFBrancher(Brancher):
    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple['cKDTree', np.ndarray, np.ndarray]],
                 layer_surfaces,
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
        layer_surfaces : list of dict
            Surface geometry for each tracking layer, each described as either a disk or cylinder with
            keys like 'type', 'n', 'p', or 'R'
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
        
        self.layer_surfaces = layer_surfaces
    
    def _solve_dt_to_surface(self, x0: np.ndarray, surf: dict, dt_init: float = 1.0) -> float:
        """
        Solve for the time step `dt` that brings the current state to intersect the given surface.

        Parameters
        ----------
        x0 : ndarray
            Initial state vector of shape (7,), representing the current particle state.
        surf : dict
            Surface geometry dictionary, either:
                - Disk: {'type': 'disk', 'n': normal vector, 'p': point on plane}
                - Cylinder: {'type': 'cylinder', 'R': radius}
        dt_init : float, optional
            Initial guess for the time step. Default is 1.0.

        Returns
        -------
        float
            Time step `dt` that brings the state to intersect the surface.
        """
        if surf['type']=='disk':
            n, p = surf['n'], surf['p']
            def f(dt):
                return (self.propagate(x0,dt)[:3]-p).dot(n)
        else:  # cylinder
            R = surf['R']
            def f(dt):
                xyt = self.propagate(x0,dt)[:2]
                return np.hypot(xyt[0], xyt[1]) - R

        return newton(f, dt_init, maxiter=20, tol=1e-6)

    def _get_candidates_in_gate(self, pred_pos: np.ndarray, layer: Tuple[int,int], radius: float):
        """
        Retrieve candidate hits within a gating radius from the predicted position.

        Parameters
        ----------
        pred_pos : ndarray
            Predicted 3D position (x, y, z) from the propagated particle state.
        layer : tuple of int
            (volume_id, layer_id) identifying the detector layer to search.
        radius : float
            Gating radius for spatial lookup.

        Returns
        -------
        points_sel : ndarray
            Array of selected hit positions closest to `pred_pos`.
        ids_sel : ndarray
            Corresponding hit IDs for the selected hits.
        """
        tree, points, ids = self.trees[layer]
        idxs = tree.query_ball_point(pred_pos, r=radius)
        points_sel = points[idxs]
        d2 = np.linalg.norm(points_sel - pred_pos, axis=1)
        # sort by distance and take up to step_candidates
        order = np.argsort(d2)[:self.step_candidates]
        sel = [idxs[i] for i in order]
        return points[sel], ids[sel]
    

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
        dt0 = t[1] - t[0] if t is not None else 1.0
        v0, κ0 = self._estimate_seed_helix(seed_xyz, dt0, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, κ0])
        P0 = np.eye(self.state_dim) * 0.1

        branches = [{'id':0,'parent':None,'traj':list(seed_xyz),'state':x0,'cov':P0,'score':0.0}]
        G = nx.DiGraph(); G.add_node(0, pos=seed_xyz[2]); next_id=1

        # uses truth level layers? 
        for i, layer in enumerate(layers):
            gate = self.inner_gate if i<3 else self.outer_gate
            surf = self.layer_surfaces[layer]
            new_br = []
            for br in branches:# 1) find the exact Δt so we hit this layer’s plane
                try:
                    dt_layer = self._solve_dt_to_surface(br['state'], surf, dt_init=dt0)
                except (RuntimeError, OverflowError):
                    continue

                F      = self.compute_F(br['state'], dt_layer)
                x_pred = self.propagate(br['state'], dt_layer)
                P_pred = F @ br['cov'] @ F.T + self.Q0 * dt_layer

                H = self.H_jac(None)  # 3×7
                S = H @ P_pred @ H.T + self.R
                
                gate_r = 3.0 * np.sqrt(np.max(np.linalg.eigvalsh(S)))

                points_cand, id_cand = self._get_candidates_in_gate(x_pred[:3], layer, gate_r)
                for z, hid in zip(points_cand, id_cand):
                    res = z - x_pred[:3]
                    chi2 = float(res @ np.linalg.solve(S, res))

                    if chi2 < gate:
                        K    = P_pred @ H.T @ np.linalg.inv(S)
                        x_upd = x_pred + K @ res
                        P_upd = (np.eye(self.state_dim) - K @ H) @ P_pred
                        score = br['score'] + chi2
                    else:
                        x_upd = x_pred.copy(); x_upd[6] += np.random.randn() * 1e-4
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