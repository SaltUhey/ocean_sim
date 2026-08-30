import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator, RectBivariateSpline

class GaussianBeamPropagator:
    def __init__(self, ranges, depths, c_grid, bottom_ranges, bottom_depths, freq=1000.0):
        # 軌跡の屈折とビーム幅計算のため、滑らかな2階微分を保証する3次スプライン(kx=3, ky=3)を使用
        self.r_min, self.r_max = ranges[0], ranges[-1]
        self.z_min, self.z_max = depths[0], depths[-1]
       # 1. 距離と水深の高密度グリッドを生成 (例: 距離50m刻み, 水深5m刻み)
        r_dense = np.arange(self.r_min, self.r_max + 0.1, 50.0)
        z_dense = np.arange(self.z_min, self.z_max + 0.1, 5.0)
        
        # 2. 元の粗いデータから線形補間関数を作成
        lin_interp = RegularGridInterpolator((ranges, depths), c_grid, method='linear')
        
        # 3. 高密度グリッド上の音速場を計算
        R_dense, Z_dense = np.meshgrid(r_dense, z_dense, indexing='ij')
        c_grid_dense = lin_interp((R_dense, Z_dense))
        
        # 4. 高密度データに対して3次スプライン(kx=3, ky=3)を構築
        self.c_spline = RectBivariateSpline(r_dense, z_dense, c_grid_dense, kx=3, ky=3)
        
        self.bottom_func = interp1d(bottom_ranges, bottom_depths, kind='linear', fill_value="extrapolate")
        self.bottom_dzdr = 0.1 
        self.freq = freq
        self.omega = 2.0 * np.pi * freq

    def get_c_and_gradients(self, r, z):
        """位置(r, z)における音速c、1階偏微分、および2階偏微分をスプラインの解析的微分から取得"""
        # ODEソルバが一時的に領域外へステップした際、3次関数の発散を防ぐためのクリッピング
        r_clip = max(self.r_min, min(r, self.r_max))
        z_clip = max(self.z_min, min(z, self.z_max))
        
        # 有限差分による数値ノイズを排除し、SciPyの解析的微分(dx, dy)を直接使用
        c = self.c_spline(r_clip, z_clip, grid=False).item()
        dc_dr = self.c_spline(r_clip, z_clip, dx=1, dy=0, grid=False).item()
        dc_dz = self.c_spline(r_clip, z_clip, dx=0, dy=1, grid=False).item()
        
        c_rr = self.c_spline(r_clip, z_clip, dx=2, dy=0, grid=False).item()
        c_zz = self.c_spline(r_clip, z_clip, dx=0, dy=2, grid=False).item()
        c_rz = self.c_spline(r_clip, z_clip, dx=1, dy=1, grid=False).item()
        
        return c, dc_dr, dc_dz, c_rr, c_zz, c_rz

    def ray_odes(self, s, Y):
        r, z, pr, pz, tau, q_R, q_I, p_R, p_I = Y
        c, dc_dr, dc_dz, c_rr, c_zz, c_rz = self.get_c_and_gradients(r, z)

        dr_ds = c * pr
        dz_ds = c * pz
        dpr_ds = - (1.0 / c**2) * dc_dr
        dpz_ds = - (1.0 / c**2) * dc_dz
        dtau_ds = 1.0 / c

        nr = -c * pz
        nz = c * pr
        c_nn = c_rr * nr**2 + 2 * c_rz * nr * nz + c_zz * nz**2

        dqR_ds = c * p_R
        dqI_ds = c * p_I
        dpR_ds = - (c_nn / c**2) * q_R
        dpI_ds = - (c_nn / c**2) * q_I

        return [dr_ds, dz_ds, dpr_ds, dpz_ds, dtau_ds, dqR_ds, dqI_ds, dpR_ds, dpI_ds]

    def shoot_ray(self, start_pos, AoD_deg, max_range):
        r0, z0 = start_pos
        theta0 = np.radians(AoD_deg)
        c0, _, _, _, _, _ = self.get_c_and_gradients(r0, z0)

        pr0 = np.cos(theta0) / c0
        pz0 = np.sin(theta0) / c0

        q0_R, q0_I = 1.0, 0.0
        p0_R, p0_I = 0.0, 1.0

        Y0 = np.array([r0, z0, pr0, pz0, 0.0, q0_R, q0_I, p0_R, p0_I])

        def surface(s, Y): return Y[1]
        surface.terminal = True
        surface.direction = -1

        def bottom(s, Y): return Y[1] - self.bottom_func(Y[0])
        bottom.terminal = True
        bottom.direction = 1

        def target(s, Y): return Y[0] - max_range
        target.terminal = True

        trajectories = []
        current_s = 0.0
        max_s = max_range * 5.0 

        while Y0[0] < max_range:
            sol = solve_ivp(
                self.ray_odes,
                [current_s, max_s],
                Y0,
                events=[surface, bottom, target],
                max_step=10.0, 
                dense_output=True
            )
            
            trajectories.append(sol)
            
            if sol.status == 1: 
                events = sol.t_events
                
                if len(events[2]) > 0: # Target
                    break
                elif len(events[0]) > 0: # Surface
                    Y0 = sol.y[:, -1].copy()
                    Y0[1] = 0.0  
                    Y0[3] = -Y0[3] 
                elif len(events[1]) > 0: # Bottom
                    Y0 = sol.y[:, -1].copy()
                    r_col = Y0[0]
                    Y0[1] = self.bottom_func(r_col)
                    
                    dzb_dr = (self.bottom_func(r_col + self.bottom_dzdr) - 
                              self.bottom_func(r_col - self.bottom_dzdr)) / (2 * self.bottom_dzdr)
                    
                    norm_mag = np.sqrt(1.0 + dzb_dr**2)
                    nx, nz = dzb_dr / norm_mag, -1.0 / norm_mag
                    
                    p_dot_n = Y0[2]*nx + Y0[3]*nz
                    Y0[2] = Y0[2] - 2 * p_dot_n * nx
                    Y0[3] = Y0[3] - 2 * p_dot_n * nz
                
                current_s = sol.t[-1]
            else:
                break 

        return trajectories, AoD_deg

    def calculate_contribution(self, trajectories, rx_pos, beam_width_threshold=30.0):
        min_dist = float('inf')
        closest_idx = 0
        best_sol = None
        
        rx_r, rx_z = rx_pos
        
        for sol in trajectories:
            r_pts = sol.y[0]
            z_pts = sol.y[1]
            
            dists = np.sqrt((r_pts - rx_r)**2 + (z_pts - rx_z)**2)
            idx_min = np.argmin(dists)
            
            if dists[idx_min] < min_dist:
                min_dist = dists[idx_min]
                closest_idx = idx_min
                best_sol = sol
                
        is_hit = min_dist <= beam_width_threshold
        amplitude = 0.0
        closest_tau = 0.0
        
        if is_hit and best_sol is not None:
            r_c = best_sol.y[0, closest_idx]
            z_c = best_sol.y[1, closest_idx]
            closest_tau = best_sol.y[4, closest_idx]
            q_R = best_sol.y[5, closest_idx]
            q_I = best_sol.y[6, closest_idx]
            p_R = best_sol.y[7, closest_idx]
            p_I = best_sol.y[8, closest_idx]
            
            c_local, _, _, _, _, _ = self.get_c_and_gradients(r_c, z_c)
            
            q = complex(q_R, q_I)
            p = complex(p_R, p_I)
            
            imag_pq = np.imag(p / q)
            if imag_pq < 0: imag_pq = 0 
            
            geom_spreading = np.sqrt(c_local / max(abs(q), 1e-10))
            gaussian_envelope = np.exp(-0.5 * self.omega * imag_pq * (min_dist**2))
            
            amplitude = geom_spreading * gaussian_envelope
            
        return is_hit, min_dist, closest_tau, amplitude

def run_test():
    print("=== Gaussian Beam Tracer (Dynamic Ray Equations) ===")
    
    # 3次スプライン(kx=3)には各次元4点以上のデータが必要
    ranges = np.array([0.0, 1000.0, 2000.0, 3000.0])
    depths = np.array([0.0, 20.0, 100.0, 500, 2000.0])
    
    # 屈折が視覚的に分かりやすいよう、サウンドチャネルをやや強調
    base_ssp = [1510.0, 1490.0, 1470.0, 1500.0, 1530.0]
    c_grid = np.array([base_ssp, base_ssp, base_ssp, base_ssp])
     # c_grid = np.array([base_ssp, [1512.0, 1492.0, 1472.0, 1500, 1532.0], base_ssp, base_ssp])
    
    bottom_ranges = np.array([0.0, 1000.0, 2000.0, 3000.0])
    bottom_depths = np.array([1500.0, 1500.0, 1500.0, 1500.0])
    
    start_pos = (0.0, 50.0)
    end_pos = (2000.0, 200.0)
    target_range = 2500.0
    
    model = GaussianBeamPropagator(ranges, depths, c_grid, bottom_ranges, bottom_depths, freq=1000.0)

    angle_min, angle_max = -70, 70
    num_beams = 101
    test_angles = np.linspace(angle_min, angle_max, num_beams) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    R_grid, Z_grid = np.meshgrid(np.linspace(0, target_range, 50), np.linspace(0, 1500, 50))
    C_grid_plot = np.vectorize(lambda r, z: model.get_c_and_gradients(r, z)[0])(R_grid, Z_grid)
    c_contour = ax.contourf(R_grid, Z_grid, C_grid_plot, levels=30, cmap='viridis', alpha=0.3)
    plt.colorbar(c_contour, ax=ax, label='Sound Speed (m/s)')
    
    r_plot = np.linspace(0, target_range, 100)
    z_plot = model.bottom_func(r_plot)
    ax.fill_between(r_plot, z_plot, max(z_plot)+100, color='saddlebrown', alpha=0.4, label='Bottom')
    
    print(f"{'AoD (deg)':>10} | {'Hit?':>6} | {'Dist(m)':>9} | {'Delay(s)':>9} | {'Amplitude':>10}")
    print("-" * 55)
    
    for angle in test_angles:
        trajectories, AoD = model.shoot_ray(start_pos, angle, target_range)
        is_hit, min_dist, hit_tau, amplitude = model.calculate_contribution(trajectories, end_pos, beam_width_threshold=30.0)
        
        hit_str = "YES" if is_hit else "NO"
        delay_str = f"{hit_tau:.4f}" if is_hit else "-"
        amp_str = f"{amplitude:.2e}" if is_hit else "-"
        print(f"{AoD:10.2f} | {hit_str:>6} | {min_dist:9.2f} | {delay_str:>9} | {amp_str:>10}")
        
        color = 'red' if is_hit else 'blue'
        alpha = 0.9 if is_hit else 0.4
        linewidth = 1.5 if is_hit else 1.0
        
        for sol in trajectories:
            ax.plot(sol.y[0], sol.y[1], color=color, linewidth=linewidth, alpha=alpha)

    ax.plot(start_pos[0], start_pos[1], 'r^', markersize=10, zorder=5, label='Tx (Source)')
    ax.plot(end_pos[0], end_pos[1], 'y*', markersize=10, zorder=5, label='Rx (Receiver)')
    
    custom_lines = [
        Line2D([0], [0], color='blue', lw=1.0, alpha=0.4),
        Line2D([0], [0], color='red', lw=1.5, alpha=0.9)
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    labels.extend(['Ray (Miss)', 'Ray (Hit/Intersect)'])
    
    ax.set_title('Gaussian Beam with Dynamic Ray Tracing (p, q)')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_xlim([0, target_range])
    ax.set_ylim([1600, 0]) 
    ax.legend(handles=handles, labels=labels, loc='lower left')
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_test()