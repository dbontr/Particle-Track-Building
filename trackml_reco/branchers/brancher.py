import abc
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy.spatial import cKDTree

class Brancher(abc.ABC):
    """
    Abstract base class for any “branching” track finder.
    Defines the interface and holds common utilities:
      - measurement & process noise
      - get_candidates from KD-tree
      - to_local_frame and its jacobian
      - helix propagation Jacobian & state propagation
      - curvature-seed helper
    Subclasses must implement `run(...)`.
    """

    def __init__(self,
                 trees: Dict[Tuple[int,int], Tuple[cKDTree, np.ndarray, np.ndarray]],
                 layers: List[Tuple[int,int]],
                 noise_std: float=2.0,
                 B_z: float=0.002,
                 max_cands: int=10):
        """
        Initialize the track-building component with spatial search trees, detector layer info,
        and Kalman filter parameters.

        Parameters
        ----------
        trees : dict of tuple
            Dictionary mapping (volume_id, layer_id) to a tuple:
            (cKDTree, ndarray of hit coordinates, ndarray of hit IDs), used for spatial lookup.
        layers : list of tuple
            Ordered list of (volume_id, layer_id) tuples representing the tracking layers.
        noise_std : float, optional
            Standard deviation of the measurement noise (assumed isotropic in mm). Default is 2.0.
        B_z : float, optional
            Magnetic field strength along the z-axis in Tesla. Default is 0.002.
        max_cands : int, optional
            Maximum number of hit candidates to consider at each layer. Default is 10.
        """
        self.trees       = trees
        self.layers      = layers
        self.noise_std   = noise_std
        self.B_z         = B_z
        self.max_cands   = max_cands

        # measurement & process noise
        self.R   = (noise_std**2) * np.eye(3)
        self.Q0  = np.diag([1e-4]*3 + [1e-5]*3 + [1e-6]) * noise_std

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
        tree, points, ids = self.trees[layer]
        dists, idxs = tree.query(pred_xyz, k=self.max_cands)
        idxs = np.atleast_1d(idxs)  
        points_sel = points[idxs]
        d2 = np.linalg.norm(points_sel - pred_xyz, axis=1)
        best_local = np.argsort(d2)[:self.step_candidates]
        best_idxs = idxs[best_local]
        return points[best_idxs], ids[best_idxs]
    
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
        H = np.zeros((3,7)); H[0,0]=H[1,1]=H[2,2]=1.0; return H

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
        u = np.cross(arbitrary, w); u /= np.linalg.norm(u)
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
            2D local position and 2x2 covariance in local frame
        """
        H = self.to_local_frame_jac(plane_normal)
        meas = H @ (pos - plane_point)
        cov_local = H @ cov[:3, :3] @ H.T
        return meas, cov_local



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
        F = np.eye(self.state_dim)
        
        if abs(ω) < 1e-6:
            F[0,3] = F[1,4] = F[2,5] = dt
            return F
        
        c, s = np.cos(θ), np.sin(θ)
        # ∂pos/∂vel
        F[0,3] = c * dt - s/ω
        F[1,3] = s * dt + (1 - c)/ω
        F[0,4] = -s * dt - (1 - c)/ω
        F[1,4] = c * dt - s/ω
        F[2,5] = dt

        # ∂pos/∂κ
        if abs(θ) < 1e-3:
            F[0,6] = -0.5 * self.B_z * pT * dt**2
            F[1,6] = 0.5 * self.B_z * pT * dt**2
        else:
            F[0,6] = (vy/κ) * (s/ω - c*dt)
            F[1,6] = (vx/κ) * (-(1 - c)/ω + s*dt)
        
        # ∂vel_rotation & curvature
        F[3,3], F[4,4], F[5,5], F[3,4], F[4,3] = c, c, 1, -s, s
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
        s0, s1, s2 = seed_xyz
        d1, d2, d02 = s1 - s0, s2 - s1, s2 - s0
        n1, n2, n02 = np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(d02)
        if min(n1,n2,n02) < 1e-6:
            return (s2 - s0)/(2*dt), 0.0
        cr = np.cross(d1, d2)
        kappa = 2*np.linalg.norm(cr)/(n1*n2*n02)
        if cr[2]*B_z < 0: kappa = -kappa
        t = (d1/n1 + d2/n2)
        tn = np.linalg.norm(t)
        t = d2/n2 if tn<1e-6 else t/tn
        v0 = t * (n2/dt)
        return v0, kappa

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Must return branches, graph (or best branch, graph)."""

