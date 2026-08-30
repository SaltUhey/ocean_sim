import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RectBivariateSpline

class GaussianBeamPropagator:
    def __init__(self, ranges, depths, c_grid, bottom_ranges, bottom_depths, freq=1000.0):
        """
        ranges: 距離の配列 (m)
        depths: 深度の配列 (m)
        c_grid: 2D音速プロファイル (m/s)
        bottom_ranges, bottom_depths: 海底地形データ (m)
        freq: 音源周波数 (Hz) - 振幅（ビーム幅）計算に使用
        """
        self.c_spline = RectBivariateSpline(ranges, depths, c_grid, kx=1, ky=1)
        self.bottom_func = interp1d(bottom_ranges, bottom_depths, kind='linear', fill_value="extrapolate")
        self.bottom_dzdr = 0.1 
        self.freq = freq
        self.omega = 2.0 * np.pi * freq

    def get_c_and_gradients(self, r, z):
        """位置(r, z)における音速c、1階偏微分、および2階偏微分を取得"""
        c = self.c_spline(r, z, grid=False).item()
        
        dr = 0.1
        dz = 0.1
        
        # 1階微分（中心差分）
        c_r_plus = self.c_spline(r + dr, z, grid=False).item()
        c_r_minus = self.c_spline(r - dr, z, grid=False).item()
        dc_dr = (c_r_plus - c_r_minus) / (2 * dr)
        
        c_z_plus = self.c_spline(r, z + dz, grid=False).item()
        c_z_minus = self.c_spline(r, z - dz, grid=False).item()
        dc_dz = (c_z_plus - c_z_minus) / (2 * dz)
        
        # 2階微分（中心差分）
        c_rr = (c_r_plus - 2*c + c_r_minus) / (dr**2)
        c_zz = (c_z_plus - 2*c + c_z_minus) / (dz**2)
        
        c_rz_pp = self.c_spline(r + dr, z + dz, grid=False).item()
        c_rz_mm = self.c_spline(r - dr, z - dz, grid=False).item()
        c_rz_pm = self.c_spline(r + dr, z - dz, grid=False).item()
        c_rz_mp = self.c_spline(r - dr, z + dz, grid=False).item()
        c_rz = (c_rz_pp - c_rz_mp - c_rz_pm + c_rz_mm) / (4 * dr * dz)
        
        return c, dc_dr, dc_dz, c_rr, c_zz, c_rz

    def ray_odes(self, s, Y):
        """
        運動学的＋動的音線方程式 (Kinematic & Dynamic Ray Equations)
        Y = [r, z, pr, pz, tau, q_R, q_I, p_R, p_I]
        """
        r, z, pr, pz, tau, q_R, q_I, p_R, p_I = Y
        c, dc_dr, dc_dz, c_rr, c_zz, c_rz = self.get_c_and_gradients(r, z)

        # --- 運動学の方程式 (軌跡) ---
        dr_ds = c * pr
        dz_ds = c * pz
        dpr_ds = - (1.0 / c**2) * dc_dr
        dpz_ds = - (1.0 / c**2) * dc_dz
        dtau_ds = 1.0 / c

        # --- 動的方程式 (ビームの広がり q とその微分 p) ---
        # 法線ベクトル成分
        nr = -c * pz
        nz = c * pr
        # 音線に垂直な方向の音速の2階微分 (c_nn)
        c_nn = c_rr * nr**2 + 2 * c_rz * nr * nz + c_zz * nz**2

        dqR_ds = c * p_R
        dqI_ds = c * p_I
        dpR_ds = - (c_nn / c**2) * q_R
        dpI_ds = - (c_nn / c**2) * q_I

        return [dr_ds, dz_ds, dpr_ds, dpz_ds, dtau_ds, dqR_ds, dqI_ds, dpR_ds, dpI_ds]

    def shoot_ray(self, start_pos, AoD_deg, max_range):
        """特定の放出角(AoD)で音線を撃ち、軌跡とビームパラメータを計算"""
        r0, z0 = start_pos
        theta0 = np.radians(AoD_deg)
        c0, _, _, _, _, _ = self.get_c_and_gradients(r0, z0)

        pr0 = np.cos(theta0) / c0
        pz0 = np.sin(theta0) / c0

        # ガウシアンビームの初期条件 (q(0)=1, p(0)=i 等)
        # q_R, q_I, p_R, p_I
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
        """レシーバーへの近接判定と、ガウシアンエンベロープに基づく振幅計算"""
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
            # 近接点でのパラメータ抽出
            r_c = best_sol.y[0, closest_idx]
            z_c = best_sol.y[1, closest_idx]
            closest_tau = best_sol.y[4, closest_idx]
            q_R = best_sol.y[5, closest_idx]
            q_I = best_sol.y[6, closest_idx]
            p_R = best_sol.y[7, closest_idx]
            p_I = best_sol.y[8, closest_idx]
            
            c_local, _, _, _, _, _ = self.get_c_and_gradients(r_c, z_c)
            
            # 複素数化
            q = complex(q_R, q_I)
            p = complex(p_R, p_I)
            
            # ガウシアンビーム振幅方程式: A ~ sqrt(c / |q|) * exp(-0.5 * omega * Im(p/q) * n^2)
            # nは中心音線からの法線距離(≒min_dist)
            imag_pq = np.imag(p / q)
            # 物理的に発散しないようエンベロープを保証
            if imag_pq < 0: imag_pq = 0 
            
            geom_spreading = np.sqrt(c_local / max(abs(q), 1e-10))
            gaussian_envelope = np.exp(-0.5 * self.omega * imag_pq * (min_dist**2))
            
            amplitude = geom_spreading * gaussian_envelope
            
        return is_hit, min_dist, closest_tau, amplitude

def run_test():
    print("=== Gaussian Beam Tracer (Dynamic Ray Equations) ===")
    
    ranges = np.array([0.0, 1000.0, 2000.0])
    depths = np.array([0.0, 20.0, 100.0, 2000.0])
    base_ssp = [1500.0, 1495.0, 1490.0, 1510.0]
    c_grid = np.array([base_ssp, [1502.0, 1496.0, 1491.0, 1512.0], base_ssp])
    
    bottom_ranges = np.array([0.0, 1000.0, 2000.0, 3000.0])
    bottom_depths = np.array([1500.0, 1500.0, 1500.0, 1500.0])
    
    start_pos = (0.0, 50.0)
    end_pos = (2000.0, 100.0)
    target_range = 2500.0
    
    # 1kHzの音源としてインスタンス化
    model = GaussianBeamPropagator(ranges, depths, c_grid, bottom_ranges, bottom_depths, freq=1000.0)
    
    test_angles = np.linspace(-15, 15, 15) # 音線を少し増やす
    
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