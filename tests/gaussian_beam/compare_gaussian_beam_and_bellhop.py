import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator
import arlpy.uwapm as pm
import time
import pandas as pd

class GaussianBeamPropagator:
    def __init__(self, ranges, depths, c_grid, bottom_ranges, bottom_depths, freq=1000.0):
        self.r_min, self.r_max = ranges[0], ranges[-1]
        self.z_min, self.z_max = depths[0], depths[-1]
        
        # 線形補間による高密度グリッドの生成（距離50m刻み、水深5m刻み）
        r_dense = np.arange(self.r_min, self.r_max + 0.1, 50.0)
        z_dense = np.arange(self.z_min, self.z_max + 0.1, 5.0)
        
        lin_interp = RegularGridInterpolator((ranges, depths), c_grid, method='linear')
        R_dense, Z_dense = np.meshgrid(r_dense, z_dense, indexing='ij')
        c_grid_dense = lin_interp((R_dense, Z_dense))
        
        # 高密度データに対して3次スプライン(kx=3, ky=3)を構築
        self.c_spline = RectBivariateSpline(r_dense, z_dense, c_grid_dense, kx=3, ky=3)
        
        self.bottom_func = interp1d(bottom_ranges, bottom_depths, kind='linear', fill_value="extrapolate")
        self.bottom_dzdr = 0.1 
        self.freq = freq
        self.omega = 2.0 * np.pi * freq

    def get_c_and_gradients(self, r, z):
        # 領域外エラー防止のためのクリッピング
        r_clip = max(self.r_min, min(r, self.r_max))
        z_clip = max(self.z_min, min(z, self.z_max))
        
        # RectBivariateSplineの解析的微分を使用（有限差分を廃止）
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

        nr, nz = -c * pz, c * pr
        c_nn = c_rr * nr**2 + 2 * c_rz * nr * nz + c_zz * nz**2

        dqR_ds, dqI_ds = c * p_R, c * p_I
        dpR_ds, dpI_ds = - (c_nn / c**2) * q_R, - (c_nn / c**2) * q_I

        return [dr_ds, dz_ds, dpr_ds, dpz_ds, dtau_ds, dqR_ds, dqI_ds, dpR_ds, dpI_ds]

    def shoot_ray(self, start_pos, AoD_deg, max_range):
        r0, z0 = start_pos
        theta0 = np.radians(AoD_deg)
        c0, _, _, _, _, _ = self.get_c_and_gradients(r0, z0)

        pr0, pz0 = np.cos(theta0) / c0, np.sin(theta0) / c0
        Y0 = np.array([r0, z0, pr0, pz0, 0.0, 1.0, 0.0, 0.0, 1.0])

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
                self.ray_odes, [current_s, max_s], Y0,
                events=[surface, bottom, target], max_step=10.0, dense_output=True
            )
            
            trajectories.append(sol)
            if sol.status == 1: 
                events = sol.t_events
                if len(events[2]) > 0: break
                elif len(events[0]) > 0:
                    Y0 = sol.y[:, -1].copy()
                    Y0[1] = 0.0  
                    Y0[3] = -Y0[3] 
                elif len(events[1]) > 0:
                    Y0 = sol.y[:, -1].copy()
                    r_col = Y0[0]
                    Y0[1] = self.bottom_func(r_col)
                    dzb_dr = (self.bottom_func(r_col + self.bottom_dzdr) - self.bottom_func(r_col - self.bottom_dzdr)) / (2 * self.bottom_dzdr)
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
        rx_r, rx_z = rx_pos
        
        for sol in trajectories:
            dists = np.sqrt((sol.y[0] - rx_r)**2 + (sol.y[1] - rx_z)**2)
            min_dist = min(min_dist, np.min(dists))
                
        is_hit = min_dist <= beam_width_threshold
        
        surface_bounces = sum(1 for sol in trajectories if sol.status == 1 and len(sol.t_events[0]) > 0)
        bottom_bounces = sum(1 for sol in trajectories if sol.status == 1 and len(sol.t_events[1]) > 0)
        
        topo = "multi"
        if surface_bounces == 0 and bottom_bounces == 0: topo = "direct"
        elif surface_bounces == 1 and bottom_bounces == 0: topo = "surface"
        elif surface_bounces == 0 and bottom_bounces == 1: topo = "bottom"
        
        return is_hit, topo


def run_trajectory_comparison():
    print("=== Eigenray Trajectory Comparison (GB vs BELLHOP) ===")
    
    start_pos = (0.0, 50.0)
    end_pos = (2000.0, 200.0)
    bottom_depth = 1500.0
    freq = 10000.0
    rx_threshold = 30.0
    angle_min, angle_max = -70, 70
    num_beams = 101
    test_angles = np.linspace(angle_min, angle_max, num_beams) 
    
    base_depths = np.array([0.0, 20.0, 100.0, 500.0, 2000.0])
    base_ssp = np.array([1510.0, 1490.0, 1470.0, 1500.0, 1530.0])
    
    ssp_1d_arlpy = np.column_stack((base_depths, base_ssp))
    env = pm.create_env2d(
        depth=bottom_depth,
        soundspeed=ssp_1d_arlpy,
        bottom_soundspeed=1600.0,
        bottom_density=1.5,
        bottom_absorption=0.1,
        frequency=freq,
        rx_range=end_pos[0],
        rx_depth=end_pos[1],
        tx_depth=start_pos[1],
        min_angle=angle_min,
        max_angle=angle_max,
        nbeams = num_beams
    )
    
    gb_ranges = np.array([0.0, 2500.0])
    gb_c_grid = np.array([base_ssp, base_ssp])
    gb_bottom_ranges = np.array([0.0, 3000.0])
    gb_bottom_depths = np.array([bottom_depth, bottom_depth])
    
    gb_solver = GaussianBeamPropagator(gb_ranges, base_depths, gb_c_grid, gb_bottom_ranges, gb_bottom_depths, freq=freq)
    
    print("Computing eigenrays with BELLHOP...")
    start_time_bellhop = time.time()
    bellhop_eigenrays = pm.compute_eigenrays(env)
    elapsed_bellhop = time.time() - start_time_bellhop
    bellhop_hits = len(bellhop_eigenrays) if bellhop_eigenrays is not None else 0
    print(f"  -> BELLHOP time: {elapsed_bellhop:.5f} seconds (Hits: {bellhop_hits})")
    
    print("Computing eigenrays with Custom Gaussian Beam...")
    start_time_gb = time.time()
    gb_hits = []
    
    for angle in test_angles:
        trajectories, _ = gb_solver.shoot_ray(start_pos, angle, end_pos[0] + 100)
        is_hit, topo = gb_solver.calculate_contribution(trajectories, end_pos, beam_width_threshold=rx_threshold)
        if is_hit:
            gb_hits.append((topo, trajectories))
            
    elapsed_gb = time.time() - start_time_gb
    print(f"  -> Gaussian Beam time: {elapsed_gb:.5f} seconds (Hits: {len(gb_hits)})")

    fig, ax = plt.subplots(figsize=(12, 7))
    
    R_grid, Z_grid = np.meshgrid(np.linspace(0, end_pos[0] + 200, 100), np.linspace(0, bottom_depth, 100))
    C_grid_plot = np.vectorize(lambda r, z: gb_solver.get_c_and_gradients(r, z)[0])(R_grid, Z_grid)
    c_contour = ax.contourf(R_grid, Z_grid, C_grid_plot, levels=30, cmap='viridis', alpha=0.4)
    cbar = plt.colorbar(c_contour, ax=ax)
    cbar.set_label('Sound Speed (m/s) [1D Profile]')
    
    ax.plot(start_pos[0], start_pos[1], 'r^', markersize=10, zorder=5, label='Tx')
    ax.plot(end_pos[0], end_pos[1], 'r*', markersize=12, zorder=5, label='Rx')
    ax.axhline(0, color='blue', linestyle='-', linewidth=2, label='Surface')
    ax.axhline(bottom_depth, color='brown', linestyle='--', linewidth=2, label='Bottom')
    
    topo_colors = {"direct": "black", "surface": "cyan", "bottom": "orange", "multi": "green"}
    
    for topo, trajectories in gb_hits:
        for sol in trajectories:
            ax.plot(sol.y[0], sol.y[1], color=topo_colors.get(topo, "gray"), linewidth=2.5, alpha=0.7, label=f'GB: {topo.capitalize()}')

    if bellhop_eigenrays is not None and not bellhop_eigenrays.empty:
        for idx, row in bellhop_eigenrays.iterrows():
            ray_path = row['ray']
            bs, bb = row['surface_bounces'], row['bottom_bounces']
            
            topo = "multi"
            if bs == 0 and bb == 0: topo = "direct"
            elif bs == 1 and bb == 0: topo = "surface"
            elif bs == 0 and bb == 1: topo = "bottom"
                
            ax.plot(ray_path[:, 0], ray_path[:, 1], color=topo_colors.get(topo, "gray"), linestyle='--', linewidth=1.5, label=f'BELLHOP: {topo.capitalize()}')
            
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    
    ax.legend(unique.values(), unique.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    ax.set_title('Eigenray Trajectory Comparison: Custom GB vs GB(BELLHOP)')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_xlim([0, end_pos[0] + 200])
    ax.set_ylim([bottom_depth + 50, -50])
    
    plt.subplots_adjust(bottom=0.25)
    plt.show()

if __name__ == "__main__":
    run_trajectory_comparison()