import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

class VariationalAcousticSolver:
    def __init__(self, ranges, depths, c_grid):
        """
        ranges: 距離グリッド (1D array)
        depths: 深度グリッド (1D array)
        c_grid: 音速の2次元配列 shape(len(ranges), len(depths))
        """
        # 2Dスプライン補間 (滑らかな音速場と空間微分を得るため)
        self.ssp_spline = RectBivariateSpline(ranges, depths, c_grid)
    
    def get_c(self, r, z):
        return self.ssp_spline(r, z, grid=False)
    
    def get_grad_c(self, r, z):
        # 音速の空間微分 (dc/dr, dc/dz)
        dc_dr = self.ssp_spline(r, z, dx=1, dy=0, grid=False)
        dc_dz = self.ssp_spline(r, z, dx=0, dy=1, grid=False)
        return dc_dr, dc_dz

    def find_eigenray(self, start_pos, end_pos, num_nodes=20):
        """
        変分法による固有音線の探索
        start_pos: (r_s, z_s), end_pos: (r_r, z_r)
        """
        r_s, z_s = start_pos
        r_r, z_r = end_pos
        
        # 経路の r 座標を固定グリッドとして分割
        self.r_nodes = np.linspace(r_s, r_r, num_nodes)
        
        # 初期パス（送信点と受信点を結ぶ直線）の z 座標
        # ※ 最適化変数は両端を除く中間の z 座標のみ
        z_init = np.linspace(z_s, z_r, num_nodes)[1:-1]
        
        # 目的関数（伝搬時間の線積分）
        def objective(z_vars):
            z_full = np.concatenate(([z_s], z_vars, [z_r]))
            T = 0.0
            for i in range(num_nodes - 1):
                r1, z1 = self.r_nodes[i], z_full[i]
                r2, z2 = self.r_nodes[i+1], z_full[i+1]
                
                # 中点での音速で近似
                r_mid, z_mid = (r1 + r2) / 2.0, (z1 + z2) / 2.0
                c_mid = self.get_c(r_mid, z_mid)
                
                ds = np.sqrt((r2 - r1)**2 + (z2 - z1)**2)
                T += ds / c_mid
            return T

        # 準ニュートン法 (L-BFGS-B) で最適化
        res = minimize(objective, z_init, method='L-BFGS-B', options={'disp': False})
        
        if not res.success:
            print("Warning: Optimization failed.")
            
        optimal_z = np.concatenate(([z_s], res.x, [z_r]))
        optimal_delay = res.fun
        
        # AoD (Angle of Departure) と AoA (Angle of Arrival) の算出 (単位: rad)
        # ※ 海面を0、下向きを正とする深度系での角度
        aod = np.arctan2(optimal_z[1] - optimal_z[0], self.r_nodes[1] - self.r_nodes[0])
        aoa = np.arctan2(optimal_z[-1] - optimal_z[-2], self.r_nodes[-1] - self.r_nodes[-2])
        
        return self.r_nodes, optimal_z, optimal_delay, aod, aoa

    def calculate_intensity(self, start_pos, end_pos, aod, d_theta=1e-4):
        """
        論文に基づく微小角レイトレーシングによる強度計算
        d_theta: 微小な射出角のズレ (rad)
        """
        r_s, z_s = start_pos
        r_r, z_r = end_pos
        
        # 距離 r を独立変数とした音線の連立微分方程式
        def ray_equations(r, y):
            z, theta = y
            c = self.get_c(r, z)
            dc_dr, dc_dz = self.get_grad_c(r, z)
            
            dz_dr = np.tan(theta)
            # スネルの法則の微分形 (r媒介変数版)
            dtheta_dr = (dc_dz - np.tan(theta) * dc_dr) / c
            return [dz_dr, dtheta_dr]

        # 微小角をズラして発射
        theta_perturbed = aod + d_theta
        
        # ODEソルバーで受信点の距離 r_r まで計算
        sol = solve_ivp(ray_equations, [r_s, r_r], [z_s, theta_perturbed], 
                        t_eval=[r_r], method='RK45')
        
        if not sol.success:
            return 0.0 # 計算失敗時は強度0
            
        z_perturbed = sol.y[0][0]
        
        # 断面積の広がり ΔL (受信点における到達深度のズレ)
        delta_L = abs(z_perturbed - z_r)
        
        if delta_L < 1e-8:
            delta_L = 1e-8 # ゼロ除算防止
            
        # 論文の公式に基づく信号強度係数 (円筒減衰 + 音線管面積)
        # intensity = (1.0 * 1.0 / r_r) * cos(theta) * (d_theta / delta_L)
        # ※ ここでは基準距離(1m)における強度を1とする
        intensity = (1.0 / r_r) * np.cos(aod) * (d_theta / delta_L)
        
        return intensity

# --- テスト実行 ---
if __name__ == "__main__":
    # 1. 2D環境データのモック作成
    ranges = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
    depths = np.array([0.0, 20.0, 100.0, 2000.0])
    
    # c_grid = np.array([
    #     [1500.0, 1495.0, 1490.0, 1510.0],  # range=0.0
    #     [1502.5, 1496.5, 1491.0, 1512.5],  # range=500.0
    #     [1505.0, 1498.0, 1492.0, 1515.0],  # range=1000.0
    #     [1507.5, 1499.5, 1493.0, 1517.5],  # range=1500.0
    #     [1510.0, 1501.0, 1494.0, 1520.0]   # range=2000.0
    # ])

    c_grid = np.array([
        [1500.0, 1500.0, 1500.0, 1500.0],  # range=0.0
        [1500.0, 1500.0, 1500.0, 1500.0],  # range=500.0
        [1500.0, 1500.0, 1500.0, 1500.0],  # range=1000.0
        [1500.0, 1500.0, 1500.0, 1500.0],  # range=1500.0
        [1500.0, 1500.0, 1500.0, 1500.0]   # range=2000.0
    ])
    
    solver = VariationalAcousticSolver(ranges, depths, c_grid)
    
    start_pos = (0.0, 50.0)    # 送信点 (r, z)
    end_pos = (1500.0, 200.0)  # 受信点 (r, z)
    
    # 2. 変分法による固有音線探索
    r_path, z_path, delay, aod, aoa = solver.find_eigenray(start_pos, end_pos, num_nodes=50)
    
    # 3. 音線管の断面積に基づく強度計算
    intensity = solver.calculate_intensity(start_pos, end_pos, aod)
    
    # 4. TVIR用のフォーマット化
    amplitude = np.sqrt(intensity) # パワーから振幅へ変換
    
    print(f"--- Eigenray Optimization Results ---")
    print(f"Delay (s):       {delay:.4f}")
    print(f"AoD (deg):       {np.rad2deg(aod):.4f}")
    print(f"AoA (deg):       {np.rad2deg(aoa):.4f}")
    print(f"Intensity coeff: {intensity:.4e}")
    print(f"Amplitude:       {amplitude:.4e}")