import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree

class HelixEKFBrancher:
    def __init__(self,
                 trees: Dict[Tuple[int, int], Tuple['cKDTree', np.ndarray, np.ndarray]],
                 layers: List[Tuple[int, int]],
                 true_xyzs: List[np.ndarray],
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
        self.trees = trees
        self.layers = layers
        self.true_xyzs = true_xyzs
        self.noise_std = noise_std
        self.B_z = B_z
        self.num_branches = num_branches
        self.survive_top = survive_top
        self.survive_rand = num_branches - survive_top
        self.max_cands = max_cands
        self.step_candidates = step_candidates
        self.state_dim = 7
        # measurement covariance
        self.R = (noise_std**2) * np.eye(3)
        # process noise: tighten kappa noise
        self.Q0 = np.diag([1e-4]*3 + [1e-5]*3 + [1e-6]) * noise_std
        # gating thresholds
        self.inner_gate = 9.0
        self.outer_gate = 16.0

        # --- user must fill these with per-layer geometry ---
        # e.g. dict mapping layer key -> normal vector
        self.layer_normals = {lay: np.array([0,0,1]) for lay in layers}
        # dict mapping layer key -> a point on that plane
        self.layer_points  = {lay: np.array([0,0,0]) for lay in layers}

    def H_jac(self, _: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian of the measurement model with respect to the state.

        Parameters
        ----------
        _ : ndarray
            Current state vector (7D).

        Returns
        -------
        ndarray
            3x7 Jacobian matrix that extracts (x, y, z) from the state.
        """
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1.0  # ∂x/∂x
        H[1, 1] = 1.0  # ∂y/∂y
        H[2, 2] = 1.0  # ∂z/∂z
        return H

    def compute_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Computes the analytic Jacobian of the helix state transition function.

        Parameters
        ----------
        x : ndarray
            State vector at current step (7D).
        dt : float
            Time step for propagation.

        Returns
        -------
        ndarray
            7x7 Jacobian matrix of the transition function.
        """
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pT = np.hypot(vx, vy)
        ω = self.B_z * κ * pT
        θ = ω * dt
        # initialize F
        F = np.eye(self.state_dim)
        # guard for small omega → near-linear motion
        if abs(ω) < 1e-6:
            # approximate helix as straight-line: x'=x+v*dt
            F[0,3] = dt  # ∂x/∂v_x
            F[1,4] = dt  # ∂y/∂v_y
            F[2,5] = dt  # ∂z/∂v_z
            return F
        # otherwise, use full analytic form
        c = np.cos(θ)
        s = np.sin(θ)
        # ∂pos/∂vel: from CMS eq(5)
        F[0,3] = c * dt - s/ω
        F[1,3] = s * dt + (1 - c)/ω
        F[0,4] = -s * dt - (1 - c)/ω
        F[1,4] = c * dt - s/ω
        F[2,5] = dt
        # ∂pos/∂κ : CMS eq(7) and small-θ Taylor (eqs 71-72)
        if abs(θ) < 1e-3:
            F[0,6] = -0.5 * self.B_z * pT * dt**2
            F[1,6] = 0.5 * self.B_z * pT * dt**2
        else:
            F[0,6] = (vy/κ) * (s/ω - c*dt)
            F[1,6] = (vx/κ) * (-(1 - c)/ω + s*dt)
        # velocity rotation derivatives
        F[3,3] = c;    F[4,4] = c;    F[5,5] = 1
        F[3,4] = -s;   F[4,3] = s
        F[3,6] = (-s * vx + c * vy) * self.B_z * dt * pT
        F[4,6] = (-c * vx - s * vy) * self.B_z * dt * pT
        return F

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagates the state forward using a helix model.

        Parameters
        ----------
        x : ndarray
            Current state vector (7D).
        dt : float
            Time increment.

        Returns
        -------
        ndarray
            New state vector after propagation.
        """
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pT = np.hypot(vx, vy)
        ω = self.B_z * κ * pT
        if abs(ω) < 1e-6:
            dx = np.array([vx, vy, vz]) * dt
            return x + np.hstack([dx, np.zeros(4)])
        θ = ω * dt
        c, s = np.cos(θ), np.sin(θ)
        vx2 = c*vx - s*vy
        vy2 = s*vx + c*vy
        pos2 = x[:3] + np.array([vx2, vy2, vz]) * dt
        return np.hstack([pos2, [vx2, vy2, vz, κ]])

    def _get_candidates(self, pred_xyz: np.ndarray, layer: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the nearest hit candidates on a given layer.

        Parameters
        ----------
        pred_xyz : ndarray
            Predicted 3D position of the particle.
        layer : tuple
            Tuple of (volume_id, layer_id).

        Returns
        -------
        tuple of (ndarray, ndarray)
            Candidate hit positions and their corresponding hit IDs.
        """
        # trees[layer] is (cKDTree, points, ids)
        tree, points, ids = self.trees[layer]

        # find k nearest on that layer
        dists, idxs = tree.query(pred_xyz, k=self.max_cands)
        idxs = np.atleast_1d(idxs)  # make array even if k=1

        # compute actual distances
        points_sel = points[idxs]
        d2 = np.linalg.norm(points_sel - pred_xyz, axis=1)

        # pick the best step_candidates indices
        best_local = np.argsort(d2)[:self.step_candidates]
        best_idxs = idxs[best_local]

        # return the points *and* their hit_ids
        return points[best_idxs], ids[best_idxs]

    def to_local_frame_jac(self, plane_normal: np.ndarray) -> np.ndarray:
        """
        Constructs a 2D basis for the plane defined by the normal vector.

        Parameters
        ----------
        plane_normal : ndarray
            Normal vector of the detector layer plane.

        Returns
        -------
        ndarray
            2x3 matrix transforming global coordinates to local (u, v).
        """
        w = plane_normal/np.linalg.norm(plane_normal)
        arbitrary = np.array([1,0,0]) if abs(w[0])<0.9 else np.array([0,1,0])
        u = np.cross(arbitrary, w);
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        return np.vstack([u, v])  # 2x3 Jacobian rows

    def to_local_frame(self, pos: np.ndarray, cov: np.ndarray,
                       plane_normal: np.ndarray, plane_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects a 3D position and covariance onto the 2D detector plane.

        Parameters
        ----------
        pos : ndarray
            3D global position.
        cov : ndarray
            3x3 covariance matrix of the position.
        plane_normal : ndarray
            Normal vector of the plane.
        plane_point : ndarray
            A point on the plane.

        Returns
        -------
        tuple of (ndarray, ndarray)
            2D measurement in local frame and its 2x2 covariance.
        """
        # compute measurement and covariance in local (u,v)
        H = self.to_local_frame_jac(plane_normal)
        meas = H @ (pos - plane_point)
        cov_local = H @ cov[:3, :3] @ H.T
        return meas, cov_local
    
    def _estimate_seed_helix(self, seed_xyz: np.ndarray, dt: float, B_z: float) -> Tuple[np.ndarray, float]:
        """
        Estimates initial velocity and curvature from a 3-point seed.

        Parameters
        ----------
        seed_xyz : ndarray
            Array of three 3D points.
        dt : float
            Time between each point.
        B_z : float
            Magnetic field along z-axis.

        Returns
        -------
        tuple
            Initial velocity vector and curvature scalar.
        """
        # three-point seed: discrete curvature and bisector tangent
        s0, s1, s2 = seed_xyz
        d1 = s1 - s0; d2 = s2 - s1; d02 = s2 - s0
        n1, n2, n02 = np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(d02)
        if min(n1,n2,n02) < 1e-6:
            v = (s2 - s0)/(2*dt); return v, 0.0
        cr = np.cross(d1, d2)
        kappa = 2*np.linalg.norm(cr)/(n1*n2*n02)
        if cr[2]*B_z < 0: kappa = -kappa
        t = (d1/n1 + d2/n2)
        tn = np.linalg.norm(t)
        if tn<1e-6: t = d2/n2
        else: t = t/tn
        v0 = t * (n2/dt)
        return v0, kappa

    def run(self, seed_xyz: np.ndarray, t: np.ndarray, plot_tree: bool = False) -> Tuple[List[Dict], nx.DiGraph]:
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
        v0, kappa0 = self._estimate_seed_helix(seed_xyz, dt, self.B_z)
        x0 = np.hstack([seed_xyz[2], v0, kappa0])
        P0 = np.eye(self.state_dim) * 0.1

        branches = [{'id':0,'parent':None,'traj':list(seed_xyz),'state':x0,'cov':P0,'score':0.0}]
        G = nx.DiGraph(); G.add_node(0, pos=seed_xyz[2]); next_id=1

        for i, layer in enumerate(self.layers):
            gate = self.inner_gate if i<3 else self.outer_gate
            true_hit = np.array(self.true_xyzs[i])
            plane_n = self.layer_normals[layer]
            plane_p = self.layer_points[layer]
            new_br = []
            for br in branches:
                F      = self.compute_F(br['state'], dt)
                x_pred = self.propagate(br['state'], dt)
                P_pred = F @ br['cov'] @ F.T + self.Q0 * dt

                points_cand, id_cand = self._get_candidates(x_pred[:3], layer)

                for z, hid in zip(points_cand, id_cand):
                    # 1) project into local (u,v), compute chi2
                    meas_pred, Puv = self.to_local_frame(x_pred[:3], P_pred, plane_n, plane_p)
                    meas_z,   Ruv = self.to_local_frame(z, np.zeros((3,3)), plane_n, plane_p)
                    Suv  = Puv + Ruv
                    res_uv = meas_z - meas_pred
                    chi2 = res_uv @ np.linalg.solve(Suv, res_uv)

                    # 2) Kalman update or penalize
                    if chi2 < gate:
                        Huv    = self.to_local_frame_jac(plane_n)        # 2×3
                        H_full = np.zeros((2, self.state_dim))           # 2×7
                        H_full[:, :3] = Huv

                        # Kalman gain
                        K_uv = P_pred[:3, :3] @ Huv.T @ np.linalg.inv(Suv)  # 3×2
                        K    = np.zeros((self.state_dim, 2))               # 7×2
                        K[:3, :] = K_uv

                        # state & cov updates
                        x_upd = x_pred + K @ res_uv
                        P_upd = (np.eye(self.state_dim) - K @ H_full) @ P_pred
                        score = br['score'] + chi2
                    else:
                        x_upd = x_pred.copy()
                        x_upd[6] += np.random.randn() * 1e-4
                        P_upd = P_pred
                        score = br['score'] + chi2 + 5.0

                    # 3) add new branch node
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

                    if chi2 < gate:
                        Huv = self.to_local_frame_jac(plane_n)
                        K_uv = P_pred[:3,:3] @ Huv.T @ np.linalg.inv(Suv)
                        K = np.zeros((7,2)); K[:3,:2] = K_uv
                        x_upd = x_pred + K @ res_uv
                        H_full = np.zeros((2, self.state_dim))
                        H_full[:, :3] = Huv

                        P_upd = (np.eye(self.state_dim) - K @ H_full) @ P_pred
                        score = br['score'] + chi2
                    else:
                        x_upd = x_pred.copy(); x_upd[6] += np.random.randn()*1e-4
                        P_upd = P_pred
                        score = br['score'] + chi2 + 5.0
                    traj = br['traj'] + [x_upd[:3]]
                    node_id = next_id
                    G.add_node(node_id, pos=x_upd[:3]); G.add_edge(br['id'], node_id)
                    next_id += 1
                    new_br.append({'id':node_id,'parent':br['id'],'traj':traj,
                                   'state':x_upd,'cov':P_upd,'score':score})
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