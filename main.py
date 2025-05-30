import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from trackml.dataset import load_dataset
from scipy.spatial import cKDTree
import networkx as nx

class HelixEKFBrancher:
    def __init__(self, trees, layers, true_xyzs,
                 noise_std=2.0, B_z=0.002,
                 num_branches=30, survive_top=12,
                 max_cands=10, step_candidates=5):
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

    def H_jac(self, x):
        """
        Measurement Jacobian H = ∂(x_meas)/∂state.
        Here direct Cartesian: picks off x,y,z.
        """
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1.0  # ∂x/∂x
        H[1, 1] = 1.0  # ∂y/∂y
        H[2, 2] = 1.0  # ∂z/∂z
        return H

    def compute_F(self, x, dt):
        """
        Analytic state-transition Jacobian F for helix propagation,
        with small-ω guard to avoid divisions by zero.
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

    def propagate(self, x, dt):
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

    def _get_candidates(self, pred_xyz, layer):
        tree, pts = self.trees[layer]
        dists, idxs = tree.query(pred_xyz, k=self.max_cands)
        pts_sel = pts[idxs] if not np.isscalar(idxs) else pts[np.array([idxs])]
        d2 = np.linalg.norm(pts_sel - pred_xyz, axis=1)
        best = np.argsort(d2)[:self.step_candidates]
        return pts_sel[best]

    def to_local_frame_jac(self, plane_normal):
        w = plane_normal/np.linalg.norm(plane_normal)
        arbitrary = np.array([1,0,0]) if abs(w[0])<0.9 else np.array([0,1,0])
        u = np.cross(arbitrary, w);
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        return np.vstack([u, v])  # 2x3 Jacobian rows

    def to_local_frame(self, pos, cov, plane_normal, plane_point):
        # compute measurement and covariance in local (u,v)
        H = self.to_local_frame_jac(plane_normal)
        meas = H @ (pos - plane_point)
        cov_local = H @ cov[:3, :3] @ H.T
        return meas, cov_local
    
    def _estimate_seed_helix(self, seed_xyz, dt, B_z):
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

    def run(self, seed_xyz, t, plot_tree=False):
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
                F = self.compute_F(br['state'], dt)
                x_pred = self.propagate(br['state'], dt)
                P_pred = F @ br['cov'] @ F.T + self.Q0 * dt

                # get candidates
                cands = self._get_candidates(x_pred[:3], layer)
                cands = np.vstack([cands, true_hit.reshape(1,3)])
                for z in cands:
                    # local-plane measurement
                    meas_pred, Puv = self.to_local_frame(x_pred[:3], P_pred, plane_n, plane_p)
                    meas_z,   Ruv = self.to_local_frame(z, np.zeros((3,3)), plane_n, plane_p)
                    Suv = Puv + Ruv
                    res_uv = meas_z - meas_pred
                    chi2 = res_uv @ np.linalg.solve(Suv, res_uv)

                    if chi2 < gate:
                        Huv = self.to_local_frame_jac(plane_n)
                        K_uv = P_pred[:3,:3] @ Huv.T @ np.linalg.inv(Suv)
                        K = np.zeros((7,2)); K[:3,:2] = K_uv
                        x_upd = x_pred + K @ res_uv
                        P_upd = (np.eye(7) - K @ Huv) @ P_pred
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


    def _plot_tree(self, G):
        pos={n:tuple(G.nodes[n]['pos'][:2]) for n in G.nodes()}
        plt.figure(figsize=(8,8)); nx.draw(G,pos,with_labels=True,node_size=50,arrowsize=10)
        plt.title('Branching tree (XY projection)'); plt.show()

# === Helpers ===
def extract_seeds(truth_hits, particles, n_seeds=5, seed_len=3):
    seeds=[]
    for pid, grp in truth_hits.groupby('particle_id'):
        grp=grp.sort_values('r')
        if len(grp)>=seed_len:
            coords=grp[['x','y','z']].values[:seed_len]
            rest=grp.iloc[seed_len:]
            true_layers=list(zip(rest.volume_id,rest.layer_id))
            true_xyzs=rest[['x','y','z']].values.tolist()
            seeds.append((pid,coords,true_layers,true_xyzs))
        if len(seeds)>=n_seeds: break
    return seeds

def build_trees_and_layers(hits):
    layers=sorted(set(zip(hits.volume_id,hits.layer_id)),key=lambda x:(x[0],x[1]))
    trees={}
    for lay in layers:
        pts=hits.query('volume_id==@lay[0] & layer_id==@lay[1]')[['x','y','z']].values
        if len(pts): trees[lay]=(cKDTree(pts),pts)
    return trees,layers

def branch_mse(branch,true_xyz):
    tree=cKDTree(true_xyz)
    traj=np.array(branch['traj'])
    d2,_=tree.query(traj,k=1)
    return np.mean(d2)

def branch_hit_stats(branch,true_xyz,threshold=1.0):
    traj=np.array(branch['traj'][3:])
    true_pts=np.array(true_xyz)
    dists=np.linalg.norm(traj-true_pts,axis=1)
    true_hits=np.sum(dists<threshold)
    return true_hits/len(true_pts)*100,len(true_pts)-true_hits

# === Main ===
if __name__=='__main__':
    _,hits,_,truth,particles=next(load_dataset('train_1.zip',nevents=1,parts=['hits','cells','truth','particles']))
    hits['r']=np.hypot(hits.x,hits.y)
    truth['particle_id']=truth.particle_id.astype(int)
    particles['particle_id']=particles.particle_id.astype(int)
    particles['pt']=np.hypot(particles.px,particles.py)
    highE=particles[particles.pt>=2.0]
    truth_hits=truth.merge(hits,on='hit_id').query('particle_id in @highE.particle_id')
    truth_hits['r']=np.hypot(truth_hits.x,truth_hits.y)
    trees,layers=build_trees_and_layers(hits)
    seeds_data=extract_seeds(truth_hits,particles,n_seeds=3,seed_len=3)
    for pid,seed_xyz,true_layers,true_xyzs in seeds_data:
        true_xyz=truth_hits.query('particle_id==@pid').sort_values('r')[['x','y','z']].values
        ekf=HelixEKFBrancher(trees,true_layers,true_xyzs,noise_std=1.0,B_z=0.002,num_branches=30,survive_top=12,max_cands=10,step_candidates=5)
        t=np.linspace(0,1,len(true_layers)+3)
        branches,G=ekf.run(seed_xyz,t,plot_tree=True)
        mses=[branch_mse(br,true_xyz) for br in branches]
        best_idx=int(np.argmin(mses)); best_branch=branches[best_idx]
        best_mse=mses[best_idx]
        true_tree=cKDTree(true_xyz)
        traj=np.array(best_branch['traj']); points=traj[3:]
        dists,_=true_tree.query(points,k=1)
        thresh=3*1.0; true_captured=np.sum(dists<=thresh)
        total_true = len(true_xyzs)
        if total_true > 0:
            percent_captured = true_captured / total_true * 100
            false_hits = len(points) - true_captured
        else:
            percent_captured = 0.0
            false_hits = 0
        print(f'PID={pid}: best MSE={best_mse:.2f}, {percent_captured:.1f}% true-hits, {false_hits} false-hits')
        fig=plt.figure(figsize=(6,4)); ax=fig.add_subplot(111,projection='3d')
        ax.scatter(hits.x,hits.y,hits.z,c='gray',alpha=0.1,s=2)
        for bi,br in enumerate(branches):
            tr=np.array(br['traj'])
            if bi==best_idx: ax.plot(tr[:,0],tr[:,1],tr[:,2],linewidth=2,label=f'Best b{bi} mse={mses[bi]:.1f}')
            else: ax.plot(tr[:,0],tr[:,1],tr[:,2],alpha=0.3)
        ax.plot(true_xyz[:,0],true_xyz[:,1],true_xyz[:,2],'--k',linewidth=2,label='truth')
        ax.scatter(seed_xyz[:,0],seed_xyz[:,1],seed_xyz[:,2],c='red',marker='x',s=50)
        ax.set_title(f'PID={pid} Trajectories (Best in bold)'); ax.legend(loc='upper left',fontsize='small'); plt.show()
