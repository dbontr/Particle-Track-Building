import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from trackml.dataset import load_dataset
from scipy.spatial import cKDTree
import networkx as nx

# === Helix‐EKF Brancher with stepwise branching & true‐hit injection + tree graph ===
class HelixEKFBrancher:
    def __init__(self, trees, layers, true_xyzs,
                 noise_std=2.0, B_z=2.0,
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
        self.R = np.eye(3)
        self.Q0 = np.diag([1e-4]*3 + [1e-5]*3 + [1e-3]) * noise_std
        self.inner_gate = 3**2 * 3
        self.outer_gate = 2**2 * 3

    def H_jac(self, x):
        x0, x1 = x[0], x[1]
        r = np.hypot(x0, x1) + 1e-9
        H = np.zeros((3, self.state_dim))
        H[0, 0], H[0, 1] = x0/r, x1/r
        H[1, 0], H[1, 1] = -x1/(r*r), x0/(r*r)
        H[2, 2] = 1.0
        return H

    def propagate(self, x, dt):
        vx, vy, vz, κ = x[3], x[4], x[5], x[6]
        pt = np.hypot(vx, vy)
        ω = self.B_z * κ * pt
        if abs(ω) < 1e-6:
            dx = np.array([vx, vy, vz]) * dt
            return x + np.hstack([dx, np.zeros(4)])
        c, s = np.cos(ω*dt), np.sin(ω*dt)
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

    def run(self, seed_xyz, t, plot_tree=False):
        dt = t[1] - t[0]
        v0 = (seed_xyz[2] - seed_xyz[0])/(2*dt)
        x0 = np.hstack([seed_xyz[2], v0, 0.0])
        P0 = np.eye(self.state_dim) * 0.1

        # initialize branches and graph
        branches = [{'id': 0, 'parent': None,
                     'traj': [seed_xyz[0], seed_xyz[1], seed_xyz[2]],
                     'state': x0, 'cov': P0, 'score': 0.0}]
        G = nx.DiGraph()
        G.add_node(0, pos=seed_xyz[2])
        next_id = 1

        for i, layer in enumerate(self.layers):
            gate = self.inner_gate if i < 3 else self.outer_gate
            true_hit = np.array(self.true_xyzs[i])
            new_br = []

            for br in branches:
                x_pred = self.propagate(br['state'], dt)
                F = np.eye(self.state_dim); F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
                P_pred = F @ br['cov'] @ F.T + self.Q0 * dt

                cands = self._get_candidates(x_pred[:3], layer)
                cands = np.vstack([cands, true_hit.reshape(1,3)])

                for z in cands:
                    r_p, φ_p, z_p = np.hypot(*x_pred[:2]), np.arctan2(x_pred[1],x_pred[0]), x_pred[2]
                    r_m, φ_m, z_m = np.hypot(*z[:2]), np.arctan2(z[1],z[0]), z[2]
                    y = np.array([(r_p-r_m)/self.noise_std,
                                  ((φ_p-φ_m+np.pi)%(2*np.pi)-np.pi)*(r_p/self.noise_std),
                                  (z_p-z_m)/self.noise_std])
                    H = self.H_jac(x_pred)
                    S = H @ P_pred @ H.T + self.R
                    chi2 = y @ y

                    if chi2 < gate:
                        K = P_pred @ H.T @ np.linalg.inv(S)
                        x_upd = x_pred + K @ y
                        P_upd = (np.eye(self.state_dim)-K@H) @ P_pred
                        traj = br['traj'] + [x_upd[:3]]
                        score = br['score'] + chi2
                    else:
                        x_upd = x_pred.copy(); x_upd[6] += np.random.normal(0,1e-4)
                        P_upd = P_pred
                        traj = br['traj'] + [x_pred[:3]]
                        score = br['score'] + chi2 + 5.0

                    node_id = next_id
                    G.add_node(node_id, pos=traj[-1])
                    G.add_edge(br['id'], node_id)
                    next_id += 1

                    new_br.append({'id': node_id, 'parent': br['id'],
                                   'traj': traj, 'state': x_upd,
                                   'cov': P_upd, 'score': score})

            if not new_br:
                break

            new_br.sort(key=lambda b: b['score'])
            branches = new_br[:self.num_branches]

        # smoothing
        for br in branches:
            tr = np.array(br['traj'])
            if tr.shape[0] >= 7:
                br['traj'] = savgol_filter(tr, 7, 3, axis=0).tolist()

        if plot_tree:
            self._plot_tree(G)
        return branches, G

    def _plot_tree(self, G):
        pos = {n: tuple(G.nodes[n]['pos'][:2]) for n in G.nodes()}
        plt.figure(figsize=(8,8))
        nx.draw(G, pos, with_labels=True, node_size=50, arrowsize=10)
        plt.title('Branching tree (XY projection)')
        plt.show()

# === Helpers ===
def extract_seeds(truth_hits, particles, n_seeds=5, seed_len=3):
    seeds = []
    for pid, grp in truth_hits.groupby('particle_id'):
        grp = grp.sort_values('r')
        if len(grp) >= seed_len:
            coords      = grp[['x','y','z']].values[:seed_len]
            rest        = grp.iloc[seed_len:]
            true_layers = list(zip(rest.volume_id, rest.layer_id))
            true_xyzs   = rest[['x','y','z']].values.tolist()
            seeds.append((pid, coords, true_layers, true_xyzs))
        if len(seeds) >= n_seeds:
            break
    return seeds

def build_trees_and_layers(hits):
    layers = sorted(set(zip(hits.volume_id, hits.layer_id)), key=lambda x:(x[0],x[1]))
    trees  = {}
    for lay in layers:
        pts = hits.query('volume_id==@lay[0] & layer_id==@lay[1]')[['x','y','z']].values
        if len(pts): trees[lay] = (cKDTree(pts), pts)
    return trees, layers

def branch_mse(branch, true_xyz):
    tree = cKDTree(true_xyz)
    traj = np.array(branch['traj'])
    d2,_ = tree.query(traj, k=1)
    return np.mean(d2)

# compute hit stats: % of true hits captured and count of false hits

def branch_hit_stats(branch, true_xyz, threshold=1.0):
    # skip initial seed points
    traj = np.array(branch['traj'][3:])
    true_pts = np.array(true_xyz)
    dists = np.linalg.norm(traj - true_pts, axis=1)
    true_hits = np.sum(dists < threshold)
    total = len(true_pts)
    false_hits = total - true_hits
    return true_hits/total*100, false_hits

# === Main ===
if __name__ == '__main__':
    _, hits, _, truth, particles = next(load_dataset(
        'train_1.zip', nevents=1, parts=['hits','cells','truth','particles']))
    hits['r'] = np.hypot(hits.x, hits.y)
    truth['particle_id'] = truth.particle_id.astype(int)
    particles['particle_id'] = particles.particle_id.astype(int)
    particles['pt'] = np.hypot(particles.px, particles.py)

    highE      = particles[particles.pt >= 2.0]
    truth_hits = truth.merge(hits, on='hit_id')\
                      .query('particle_id in @highE.particle_id')
    truth_hits['r'] = np.hypot(truth_hits.x, truth_hits.y)

    trees, layers = build_trees_and_layers(hits)
    seeds_data    = extract_seeds(truth_hits, particles, n_seeds=3, seed_len=3)

    for pid, seed_xyz, true_layers, true_xyzs in seeds_data:
        true_xyz = truth_hits.query('particle_id==@pid')\
                              .sort_values('r')[['x','y','z']].values

        ekf = HelixEKFBrancher(
            trees, true_layers, true_xyzs,
            noise_std=1.0, B_z=2.0,
            num_branches=30, survive_top=12,
            max_cands=10, step_candidates=5)
        t = np.linspace(0, 1, len(true_layers) + 3)
        branches, G = ekf.run(seed_xyz, t, plot_tree=True)

        # Evaluate MSE and select best branch
        mses = [branch_mse(br, true_xyz) for br in branches]
        best_idx = int(np.argmin(mses))
        best_branch = branches[best_idx]
        best_mse = mses[best_idx]

        # Compute true-hit capture and false-hit count for best branch
        # Build KDTree on true hits
        true_tree = cKDTree(true_xyz)
        traj = np.array(best_branch['traj'])
        # consider trajectory points beyond the seed (first 3)
        points = traj[3:]
        dists, idx = true_tree.query(points, k=1)
        # threshold for considering a hit as 'captured'
        thresh = 3 * 1.0  # 3 * noise_std
        true_captured = np.sum(dists <= thresh)
        total_true = len(true_xyzs)
        percent_captured = true_captured / total_true * 100
        false_hits = len(points) - true_captured

        print(f'PID={pid}: best MSE = {best_mse:.2f}, '
              f'{percent_captured:.1f}% true-hits, {false_hits} false-hits')

        # plot trajectories with best branch highlighted
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(hits.x, hits.y, hits.z, c='gray', alpha=0.1, s=2)
        for bi, br in enumerate(branches):
            tr = np.array(br['traj'])
            if bi == best_idx:
                ax.plot(tr[:,0], tr[:,1], tr[:,2], linewidth=2,
                        label=f'Best b{bi} mse={mses[bi]:.1f}')
            else:
                ax.plot(tr[:,0], tr[:,1], tr[:,2], alpha=0.3)
        ax.plot(true_xyz[:,0], true_xyz[:,1], true_xyz[:,2],
                '--k', linewidth=2, label='truth')
        ax.scatter(seed_xyz[:,0], seed_xyz[:,1], seed_xyz[:,2],
                   c='red', marker='x', s=50)
        ax.set_title(f'PID={pid} Trajectories (Best in bold)')
        ax.legend(loc='upper left', fontsize='small')
        plt.show()