import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt  # 追加

class VariationalAcousticSolver:
    def __init__(self, ranges, depths, c_grid, bottom_depth=2000.0):
        kx = min(3, len(ranges) - 1)
        ky = min(3, len(depths) - 1)
        self.ssp_spline = RectBivariateSpline(ranges, depths, c_grid, kx=kx, ky=ky)
        self.bottom_depth = bottom_depth
        
    def get_c(self, r, z):
        return self.ssp_spline(r, z, grid=False)
    
    def get_grad_c(self, r, z):
        dc_dr = self.ssp_spline(r, z, dx=1, dy=0, grid=False)
        dc_dz = self.ssp_spline(r, z, dx=0, dy=1, grid=False)
        return dc_dr, dc_dz

    def _calc_travel_time(self, r_nodes, z_nodes):
        """与えられた経路(r_nodes, z_nodes)の伝搬時間を計算"""
        T = 0.0
        for i in range(len(r_nodes) - 1):
            r1, z1 = r_nodes[i], z_nodes[i]
            r2, z2 = r_nodes[i+1], z_nodes[i+1]
            r_mid, z_mid = (r1 + r2) / 2.0, (z1 + z2) / 2.0
            c_mid = self.get_c(r_mid, z_mid)
            ds = np.sqrt((r2 - r1)**2 + (z2 - z1)**2)
            T += ds / c_mid
        return T

    def find_eigenray(self, start_pos, end_pos, topology="direct", num_nodes_per_seg=20):
        """
        topology: "direct" (直達), "surface" (海面1回), "bottom" (海底1回)
        """
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
            opt_r = r_nodes

        elif topology == "surface":
            # 変数: [反射点の距離 r_surf, 左セグメントのz, 右セグメントのz]
            r_surf_init = (r_s + r_r) / 2.0
            z_left_init = np.linspace(z_s, 0.0, num_nodes_per_seg)[1:-1]
            z_right_init = np.linspace(0.0, z_r, num_nodes_per_seg)[1:-1]
            init_vars = np.concatenate(([r_surf_init], z_left_init, z_right_init))
            
            def objective(vars):
                r_surf = vars[0]
                # 反射点が範囲外に出ないようペナルティ
                if r_surf <= r_s or r_surf >= r_r: return 1e6 
                
                z_left_vars = vars[1:num_nodes_per_seg-1]
                z_right_vars = vars[num_nodes_per_seg-1:]
                
                r_left = np.linspace(r_s, r_surf, num_nodes_per_seg)
                z_left = np.concatenate(([z_s], z_left_vars, [0.0])) # 海面 z=0
                
                r_right = np.linspace(r_surf, r_r, num_nodes_per_seg)
                z_right = np.concatenate(([0.0], z_right_vars, [z_r]))
                
                return self._calc_travel_time(r_left, z_left) + self._calc_travel_time(r_right, z_right)

            res = minimize(objective, init_vars, method='L-BFGS-B')
            r_surf = res.x[0]
            opt_r = np.concatenate((np.linspace(r_s, r_surf, num_nodes_per_seg), 
                                    np.linspace(r_surf, r_r, num_nodes_per_seg)[1:]))
            opt_z = np.concatenate(([z_s], res.x[1:num_nodes_per_seg-1], [0.0], 
                                    res.x[num_nodes_per_seg-1:], [z_r]))

        elif topology == "bottom":
            r_bot_init = (r_s + r_r) / 2.0
            z_bot = self.bottom_depth
            z_left_init = np.linspace(z_s, z_bot, num_nodes_per_seg)[1:-1]
            z_right_init = np.linspace(z_bot, z_r, num_nodes_per_seg)[1:-1]
            init_vars = np.concatenate(([r_bot_init], z_left_init, z_right_init))
            
            def objective(vars):
                r_bot = vars[0]
                if r_bot <= r_s or r_bot >= r_r: return 1e6
                
                z_left_vars = vars[1:num_nodes_per_seg-1]
                z_right_vars = vars[num_nodes_per_seg-1:]
                
                r_left = np.linspace(r_s, r_bot, num_nodes_per_seg)
                z_left = np.concatenate(([z_s], z_left_vars, [z_bot]))
                
                r_right = np.linspace(r_bot, r_r, num_nodes_per_seg)
                z_right = np.concatenate(([z_bot], z_right_vars, [z_r]))
                
                return self._calc_travel_time(r_left, z_left) + self._calc_travel_time(r_right, z_right)

            res = minimize(objective, init_vars, method='L-BFGS-B')
            r_bot = res.x[0]
            opt_r = np.concatenate((np.linspace(r_s, r_bot, num_nodes_per_seg), 
                                    np.linspace(r_bot, r_r, num_nodes_per_seg)[1:]))
            opt_z = np.concatenate(([z_s], res.x[1:num_nodes_per_seg-1], [z_bot], 
                                    res.x[num_nodes_per_seg-1:], [z_r]))
        else:
            raise ValueError("Unknown topology")

        optimal_delay = res.fun
        
        # --- AoA/AoD の算出 (隣接ノード間の直接微分) ---
        dz_dr_start = (opt_z[1] - opt_z[0]) / (opt_r[1] - opt_r[0])
        dz_dr_end = (opt_z[-1] - opt_z[-2]) / (opt_r[-1] - opt_r[-2])

        aod = np.arctan(dz_dr_start)
        aoa = np.arctan(dz_dr_end)
        
        return opt_r, opt_z, optimal_delay, aod, aoa

    def calculate_intensity(self, start_pos, end_pos, aod, topology="direct", d_theta=1e-4):
        """
        反射トポロジーに応じた強度計算。
        反射境界でスネルの法則(反射角=入射角)を適用して伝搬を継続する。
        """
        r_s, z_s = start_pos
        r_r, z_r = end_pos
        
        def ray_equations(r, y):
            z, theta = y
            c = self.get_c(r, z)
            dc_dr, dc_dz = self.get_grad_c(r, z)
            dz_dr = np.tan(theta)
            dtheta_dr = (dc_dz - np.tan(theta) * dc_dr) / c
            return [dz_dr, dtheta_dr]

        # 境界でのイベント検知 (ode_ivp用)
        def hit_surface(r, y): return y[0] - 0.0
        hit_surface.terminal = True
        
        def hit_bottom(r, y): return y[0] - self.bottom_depth
        hit_bottom.terminal = True

        theta_perturbed = aod + d_theta
        current_r = r_s
        current_z = z_s
        current_theta = theta_perturbed
        
        # 反射に応じたシミュレーションループ
        if topology == "direct":
            events = []
        elif topology == "surface":
            events = [hit_surface]
        elif topology == "bottom":
            events = [hit_bottom]

        sol = solve_ivp(ray_equations, [current_r, r_r], [current_z, current_theta], 
                        events=events, dense_output=True, method='RK45')
        
        # 反射が発生した場合、角度を反転させて再スタート
        if sol.status == 1: # イベント(反射)検知で停止
            current_r = sol.t[-1]
            current_z = sol.y[0][-1]
            current_theta = -sol.y[1][-1] # 角度反転 (反射)
            
            # 残りの距離を受信点まで進む
            sol2 = solve_ivp(ray_equations, [current_r, r_r], [current_z, current_theta], 
                             method='RK45', t_eval=[r_r])
            z_perturbed = sol2.y[0][-1]
        else:
            # 反射指定なのに境界に当たらずに到達した場合、または直達波
            z_perturbed = sol.y[0][-1]
            
        delta_L = abs(z_perturbed - z_r)
        if delta_L < 1e-8: delta_L = 1e-8
            
        intensity = (1.0 / r_r) * np.cos(aod) * (d_theta / delta_L)
        
        if topology in ["surface", "bottom"]:
            intensity *= 0.5 
            
        return intensity

# --- テスト実行およびプロット ---
if __name__ == "__main__":
    ranges = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
    depths = np.array([0.0, 20.0, 100.0, 2000.0])
    c_grid = np.array([
        [1500.0, 1495.0, 1490.0, 1510.0],
        [1502.5, 1496.5, 1491.0, 1512.5],
        [1505.0, 1498.0, 1492.0, 1515.0],
        [1507.5, 1499.5, 1493.0, 1517.5],
        [1510.0, 1501.0, 1494.0, 1520.0]
    ])

    # c_grid = np.array([
    #     [1500.0, 1500.0, 1500.0, 1500.0],  # range=0.0
    #     [1500.0, 1500.0, 1500.0, 1500.0],  # range=500.0
    #     [1500.0, 1500.0, 1500.0, 1500.0],  # range=1000.0
    #     [1500.0, 1500.0, 1500.0, 1500.0],  # range=1500.0
    #     [1500.0, 1500.0, 1500.0, 1500.0]   # range=2000.0
    # ])
    
    bottom_depth = 1500.0
    solver = VariationalAcousticSolver(ranges, depths, c_grid, bottom_depth=bottom_depth)
    
    start_pos = (0.0, 50.0)    
    end_pos = (1500.0, 200.0)  
    
    topologies = ["direct", "surface", "bottom"]
    
    # プロットのセットアップ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. 音速プロファイルの背景コンタープロット
    # c_gridの形状(len(ranges), len(depths))に合わせるため転置してプロット
    c_contour = ax.contourf(ranges, depths, c_grid.T, levels=30, cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(c_contour, ax=ax)
    cbar.set_label('Sound Speed (m/s)')
    
    # 2. 送受波器と境界のプロット
    ax.plot(start_pos[0], start_pos[1], 'r^', markersize=10, label='Tx (Start)')
    ax.plot(end_pos[0], end_pos[1], 'r*', markersize=12, label='Rx (End)')
    ax.axhline(0, color='blue', linestyle='-', linewidth=2, label='Surface')
    ax.axhline(bottom_depth, color='brown', linestyle='--', linewidth=2, label='Bottom')
    
    # プロット用のカラー設定
    colors = {"direct": "white", "surface": "cyan", "bottom": "orange"}
    
    for topo in topologies:
        print(f"\n--- Topology: {topo.upper()} ---")
        try:
            r_path, z_path, delay, aod, aoa = solver.find_eigenray(start_pos, end_pos, topology=topo, num_nodes_per_seg=30)
            intensity = solver.calculate_intensity(start_pos, end_pos, aod, topology=topo)
            amplitude = np.sqrt(intensity)
            
            print(f"Delay (s):       {delay:.4f}")
            print(f"AoD (deg):       {np.rad2deg(aod):.4f}")
            print(f"AoA (deg):       {np.rad2deg(aoa):.4f}")
            print(f"Intensity coeff: {intensity:.4e}")
            print(f"Amplitude:       {amplitude:.4e}")
            
            # 3. 計算された軌跡のプロット
            ax.plot(r_path, z_path, color=colors[topo], linewidth=2, label=f'{topo.capitalize()} Path')
            
        except Exception as e:
            print(f"Calculation failed for {topo}: {e}")

    # 軸の設定 (海面を上にするためY軸を反転)
    ax.set_title('Acoustic Ray Paths & Sound Speed Profile (Variational Method)')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_xlim([ranges[0], ranges[-1]])
    ax.set_ylim([bottom_depth + 100, -50]) # 少しマージンを持たせて反転
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()