import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
import arlpy.uwapm as pm
import time

class VariationalAcousticSolver:
    def __init__(self, ranges, depths, c_grid, bottom_depth=2000.0):
        kx = min(1, len(ranges) - 1)
        ky = min(1, len(depths) - 1)
        self.ssp_spline = RectBivariateSpline(ranges, depths, c_grid, kx=kx, ky=ky)
        self.bottom_depth = bottom_depth
        
    def get_c(self, r, z):
        return self.ssp_spline(r, z, grid=False)

    def _calc_travel_time(self, r_nodes, z_nodes):
        T = 0.0
        for i in range(len(r_nodes) - 1):
            r1, z1 = r_nodes[i], z_nodes[i]
            r2, z2 = r_nodes[i+1], z_nodes[i+1]
            r_mid, z_mid = (r1 + r2) / 2.0, (z1 + z2) / 2.0
            c_mid = self.get_c(r_mid, z_mid)
            ds = np.sqrt((r2 - r1)**2 + (z2 - z1)**2)
            T += ds / c_mid
        return T

    def find_eigenray(self, start_pos, end_pos, topology="direct", num_nodes_per_seg=100):
        r_s, z_s = start_pos
        r_r, z_r = end_pos

        if topology == "direct":
            r_nodes = np.linspace(r_s, r_r, num_nodes_per_seg)
            z_init = np.linspace(z_s, z_r, num_nodes_per_seg)[1:-1]
            
            def objective(z_vars):
                z_full = np.concatenate(([z_s], z_vars, [z_r]))
                return self._calc_travel_time(r_nodes, z_full)
                
            res = minimize(objective, z_init, method='L-BFGS-B')
            opt_z = np.concatenate(([z_s], res.x, [z_r]))
            return r_nodes, opt_z

        elif topology == "surface":
            r_surf_init = (r_s + r_r) / 2.0
            z_left_init = np.linspace(z_s, 0.0, num_nodes_per_seg)[1:-1]
            z_right_init = np.linspace(0.0, z_r, num_nodes_per_seg)[1:-1]
            init_vars = np.concatenate(([r_surf_init], z_left_init, z_right_init))
            
            def objective(vars):
                r_surf = vars[0]
                if r_surf <= r_s or r_surf >= r_r: return 1e6 
                z_left = np.concatenate(([z_s], vars[1:num_nodes_per_seg-1], [0.0]))
                z_right = np.concatenate(([0.0], vars[num_nodes_per_seg-1:], [z_r]))
                r_left = np.linspace(r_s, r_surf, num_nodes_per_seg)
                r_right = np.linspace(r_surf, r_r, num_nodes_per_seg)
                return self._calc_travel_time(r_left, z_left) + self._calc_travel_time(r_right, z_right)

            res = minimize(objective, init_vars, method='L-BFGS-B')
            r_surf = res.x[0]
            opt_r = np.concatenate((np.linspace(r_s, r_surf, num_nodes_per_seg), 
                                    np.linspace(r_surf, r_r, num_nodes_per_seg)[1:]))
            opt_z = np.concatenate(([z_s], res.x[1:num_nodes_per_seg-1], [0.0], 
                                    res.x[num_nodes_per_seg-1:], [z_r]))
            return opt_r, opt_z

        elif topology == "bottom":
            r_bot_init = (r_s + r_r) / 2.0
            z_bot = self.bottom_depth
            z_left_init = np.linspace(z_s, z_bot, num_nodes_per_seg)[1:-1]
            z_right_init = np.linspace(z_bot, z_r, num_nodes_per_seg)[1:-1]
            init_vars = np.concatenate(([r_bot_init], z_left_init, z_right_init))
            
            def objective(vars):
                r_bot = vars[0]
                if r_bot <= r_s or r_bot >= r_r: return 1e6
                z_left = np.concatenate(([z_s], vars[1:num_nodes_per_seg-1], [z_bot]))
                z_right = np.concatenate(([z_bot], vars[num_nodes_per_seg-1:], [z_r]))
                r_left = np.linspace(r_s, r_bot, num_nodes_per_seg)
                r_right = np.linspace(r_bot, r_r, num_nodes_per_seg)
                return self._calc_travel_time(r_left, z_left) + self._calc_travel_time(r_right, z_right)

            res = minimize(objective, init_vars, method='L-BFGS-B')
            r_bot = res.x[0]
            opt_r = np.concatenate((np.linspace(r_s, r_bot, num_nodes_per_seg), 
                                    np.linspace(r_bot, r_r, num_nodes_per_seg)[1:]))
            opt_z = np.concatenate(([z_s], res.x[1:num_nodes_per_seg-1], [z_bot], 
                                    res.x[num_nodes_per_seg-1:], [z_r]))
            return opt_r, opt_z

def run_trajectory_comparison():
    print("=== Eigenray Trajectory Comparison ===")
    
    start_pos = (0.0, 50.0)
    end_pos = (2000.0, 200.0)
    bottom_depth = 1500.0
    
    # --- 【修正】同一のSVP (1D) を定義 ---
    base_depths = np.array([0.0, 20.0, 100.0, 2000.0])
    base_ssp = np.array([1500.0, 1495.0, 1490.0, 1510.0])
    
    # arlpy (BELLHOP) 用のSVP設定
    ssp_1d_arlpy = np.column_stack((base_depths, base_ssp))
    
    # 変分法ソルバー用のSVP設定 (1Dを2Dグリッドに拡張)
    ranges = np.array([0.0, 2000.0])
    c_grid = np.array([base_ssp, base_ssp]) # 距離によらず同じプロファイル
    
    solver = VariationalAcousticSolver(ranges, base_depths, c_grid, bottom_depth=bottom_depth)
    
    env = pm.create_env2d(
        depth=bottom_depth,
        soundspeed=ssp_1d_arlpy,
        bottom_soundspeed=1600.0,
        bottom_density=1.5,
        bottom_absorption=0.1,
        frequency=10000.0,
        rx_range=end_pos[0],
        rx_depth=end_pos[1],
        tx_depth=start_pos[1]
    )
    
    # --- BELLHOPの計算と時間計測 ---
    print("Computing eigenrays with BELLHOP...")
    start_time_bellhop = time.time()
    bellhop_eigenrays = pm.compute_eigenrays(env)
    elapsed_bellhop = time.time() - start_time_bellhop
    print(f"  -> BELLHOP time: {elapsed_bellhop:.5f} seconds")
    
    # --- 変分法ソルバーの計算と時間計測 ---
    print("Computing eigenrays with Variational Method...")
    var_results = {}
    start_time_var = time.time()
    for topo in ["direct", "surface", "bottom"]:
        try:
            r_path, z_path = solver.find_eigenray(start_pos, end_pos, topology=topo)
            var_results[topo] = (r_path, z_path)
        except Exception as e:
            pass
    elapsed_var = time.time() - start_time_var
    print(f"  -> Variational time: {elapsed_var:.5f} seconds")
    
    # --- グラフ描画 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    R_grid, Z_grid = np.meshgrid(np.linspace(0, end_pos[0], 100), np.linspace(0, bottom_depth, 100))
    C_grid_plot = solver.get_c(R_grid, Z_grid)
    c_contour = ax.contourf(R_grid, Z_grid, C_grid_plot, levels=30, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(c_contour, ax=ax)
    cbar.set_label('Sound Speed (m/s) [Unified 1D Profile]')
    
    ax.plot(start_pos[0], start_pos[1], 'r^', markersize=10, zorder=5, label='Tx')
    ax.plot(end_pos[0], end_pos[1], 'r*', markersize=12, zorder=5, label='Rx')
    ax.axhline(0, color='blue', linestyle='-', linewidth=2, label='Surface')
    ax.axhline(bottom_depth, color='brown', linestyle='--', linewidth=2, label='Bottom')
    
    var_colors = {"direct": "black", "surface": "cyan", "bottom": "orange"}
    
    # 変分法の結果をプロット
    for topo, (r_path, z_path) in var_results.items():
        ax.plot(r_path, z_path, color=var_colors[topo], linewidth=2.5, label=f'Variational: {topo.capitalize()}')
            
    # BELLHOPの結果をプロット
    if bellhop_eigenrays is not None and not bellhop_eigenrays.empty:
        for idx, row in bellhop_eigenrays.iterrows():
            ray_path = row['ray']
            bs, bb = row['surface_bounces'], row['bottom_bounces']
            
            if bs == 0 and bb == 0: topo, c = "Direct", var_colors["direct"]
            elif bs == 1 and bb == 0: topo, c = "Surface", var_colors["surface"]
            elif bs == 0 and bb == 1: topo, c = "Bottom", var_colors["bottom"]
            else: topo, c = "Multi", "magenta"
                
            ax.plot(ray_path[:, 0], ray_path[:, 1], color=c, linestyle='--', linewidth=1.5, label=f'BELLHOP: {topo}')
            
    # 重複する凡例をまとめる
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    
    ax.legend(
        unique.values(), 
        unique.keys(), 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=4
    )
    
    ax.set_title('Eigenray Trajectory Comparison (Same SVP)')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_xlim([0, end_pos[0]])
    ax.set_ylim([bottom_depth + 50, -50])
    
    plt.subplots_adjust(bottom=0.25)
    try:
     plt.show()
    except KeyboardInterrupt:
     pass

if __name__ == "__main__":
    run_trajectory_comparison()