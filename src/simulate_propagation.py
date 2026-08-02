import pandas as pd
import numpy as np
import os
import sys
import time
import multiprocessing
from functools import partial
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import arlpy.uwapm as pm
from scipy.fft import fft, ifft

# 自作ユーティリティのインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env_setup.env_tools import load_env_config
from utils.motion.visualizer import UUVVisualizer
from utils.tvir.tvir_calculator import TVIRCalculator
from utils.signal.signal_generator import SignalGenerator
from utils.signal.signal_synthesizer import SignalSynthesizer
from utils.helpers import print_progress

# --- パス設定 ---
TX_CSV = os.path.join(project_root, 'data', 'uuv_trajectory_tx_fre100Hz_static_short.csv')
RX_CSV = os.path.join(project_root, 'data', 'uuv_trajectory_rx_fre100Hz_static_short.csv')
XML_CONFIG = os.path.join(project_root, 'data', 'env_config.xml')
OUTPUT_CSV = os.path.join(project_root, 'data', 'propagation_results.csv')

# --- 設定パラメータ(C-BELLHOP)---
RAY_UPDATE_INTERVAL = 10  # 音線の更新間隔（フレーム数）
NUM_RAYS = 0           # 可視化する音線の本数（この値を調整して密度を変える）

# =====================================================================
# 海面粗度（変位）データ生成クラス
# =====================================================================
class DynamicSurfaceModel:
    """時間・空間依存の海面波形を管理・生成するクラス"""
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

    def get_surface_displacement(self, r_grid, frame_time):
        """指定されたレンジ格子(r_grid)と時刻(frame_time)における海面変位 η(r) を返す"""
        eta_r = 0.35 * np.sin(2 * np.pi * 0.2 * frame_time - 2 * np.pi * 0.05 * r_grid)
        return eta_r

# =====================================================================
# WAPE型PEソルバークラス (Song et al., 2011 に基づくエッセンス)
# =====================================================================
class PESolver:
    """放物線方程式（PE）法による時間・空間前進計算エンジン"""

    @staticmethod
    def _compute_single_frequency(f, env_cfg, c0, Nz, Nr, dz, dr, z_grid, r_grid, idx_zs, idx_zr, eta_r):
        """1つの周波数における前進計算を行う独立したヘルパー関数 (各子プロセスで実行)"""
        k0 = 2 * np.pi * f / c0
        
        # 初期音場の構築 (ガウシアンスターター)
        psi = np.zeros(Nz, dtype=complex)
        psi[idx_zs] = 1.0
        
        # レンジ方向へのSplit-Step Fourier Marching
        for r_idx in range(Nr - 1):
            current_eta = eta_r[r_idx]
            
            # 1. 波数空間での自由空間伝搬 (WAPE演算子)
            kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
            with np.errstate(invalid='ignore', divide='ignore'):
                T_WAPE = np.sqrt(np.maximum(0, 1.0 - (kz/k0)**2)) - 1.0
            T_WAPE = np.nan_to_num(T_WAPE)
            
            psi_k = fft(psi)
            psi_k = psi_k * np.exp(-1j * k0 * T_WAPE * dr)
            psi = ifft(psi_k)
            
            # 2. 実空間での環境屈折および海面境界条件の反映
            ssp_val = env_cfg['ssp'](z_grid) if callable(env_cfg['ssp']) else c0
            n = c0 / ssp_val
            U_WAPE = n - 1.0
            
            # 海面境界条件 (海面より上 z < eta をゼロクリア)
            psi[z_grid < current_eta] = 0.0
            
            psi = psi * np.exp(-1j * k0 * U_WAPE * dr)
            
            # 海底の簡易減衰層 (偽像反射の防止)
            psi[int(Nz * 0.9):] *= 0.5
            
        return psi[idx_zr]

    @staticmethod
    def compute_single_frame_transmission(env_cfg, surface_model, p_tx, p_rx, dist_h, frame_time):
        f_c = env_cfg.get('frequency', 15000.0)      # 中心周波数
        bandwidth = env_cfg.get('bandwidth', 5000.0)  # 帯域幅
        N_f = env_cfg.get('num_frequencies', 16)      # テスト用に低めに設定 (本番は256など)
        freqs = np.linspace(f_c - bandwidth/2, f_c + bandwidth/2, N_f)
        
        z_s = p_tx[2]
        z_r = p_rx[2][0] if isinstance(p_rx[2], (np.ndarray, list)) else p_rx[2]
        c0 = 1500.0  # 基準音速
        
        # 空間グリッド解像度設定
        lambda_c = c0 / f_c
        dr = lambda_c
        dz = lambda_c / 10.0
        
        r_grid = np.arange(0, max(dist_h, dr), dr)
        z_grid = np.arange(0, env_cfg['max_depth'], dz)
        
        Nz = len(z_grid)
        Nr = len(r_grid)
        
        idx_zr = np.argmin(np.abs(z_grid - z_r))
        idx_zs = np.argmin(np.abs(z_grid - z_s))
        
        # 海面変位プロファイルを取得
        eta_r = surface_model.get_surface_displacement(r_grid, frame_time)
        
        # 引数の固定化 (凍結)
        worker_func = partial(
            PESolver._compute_single_frequency,
            env_cfg=env_cfg, c0=c0, Nz=Nz, Nr=Nr, dz=dz, dr=dr,
            z_grid=z_grid, r_grid=r_grid, idx_zs=idx_zs, idx_zr=idx_zr, eta_r=eta_r
        )
        
        # CPUマルチコアの並列化設定 (他作業用に2スレッド残す)
        max_cores = multiprocessing.cpu_count()
        num_cores = max(1, max_cores - 2)
        
        # 周波数ループを一気に並列処理
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(worker_func, freqs)
            
        H_f = np.array(results, dtype=complex)
        
        # 周波数軸IFFTによりインパルスレスポンスに変換
        cir_complex = ifft(H_f)
        
        # 代表値としての伝搬損失(TL)を算出
        tl_db = -20 * np.log10(np.abs(H_f[N_f // 2]) + 1e-10)
        
        return cir_complex, tl_db

def run_simulation():
    # 1. データのロード
    env_cfg = load_env_config(XML_CONFIG)
    sim_method = env_cfg['method']
    print(f"--- Simulation Mode: {sim_method} ---")

    df_tx = pd.read_csv(TX_CSV)
    df_rx = pd.read_csv(RX_CSV)
    min_len = min(len(df_tx), len(df_rx))

    simulation_log = []
    tvir_calc = TVIRCalculator()

    # ==========================================
    # 分岐A: BELLHOPモード（アニメーション実行）
    # ==========================================
    if sim_method == 'BELLHOP':
        all_x = np.concatenate([df_tx['x'], df_rx['x']])
        all_y = np.concatenate([df_tx['y'], df_rx['y']])
        viz = UUVVisualizer(
            x_limit=(np.min(all_x)-10, np.max(all_x)+10),
            y_limit=(np.min(all_y)-10, np.max(all_y)+10),
            max_depth=env_cfg['max_depth'],
            title=f"UUV Acoustic Simulation ({sim_method})"
        )

        def update_frame(frame):
            p_tx = df_tx.iloc[frame][['x', 'y', 'z']].values
            p_rx = df_rx.iloc[frame][['x', 'y', 'z']].values
            t = df_tx.iloc[frame]['time']
            dist_h = np.linalg.norm(p_tx[:2] - p_rx[:2])

            for line in viz.ray_lines:
                try:
                    line.remove()
                except:
                    pass

            env = pm.create_env2d(
                depth=[[0, env_cfg['max_depth']], [env_cfg['max_range'], env_cfg['max_depth']]],
                soundspeed=env_cfg['ssp'],
                tx_depth=p_tx[2],
                rx_depth=np.array([p_rx[2]]),
                rx_range=np.array([dist_h]),
                bottom_soundspeed=env_cfg['bottom']['sound_speed'],
                bottom_density=env_cfg['bottom']['density'],
                bottom_absorption=env_cfg['bottom']['attenuation'],
                frequency=env_cfg['frequency']
            )
            env['nbeams'] = NUM_RAYS

            tl_grid = pm.compute_transmission_loss(env, mode=pm.incoherent)
            current_tl = np.abs(tl_grid.iloc[0, 0])

            arrivals = pm.compute_arrivals(env)
            tvir_calc.add_frame(t, arrivals) 

            if frame % RAY_UPDATE_INTERVAL == 0:
                rays = pm.compute_eigenrays(env)
                if rays is not None and len(rays) > 0:
                    viz.update_rays(rays, p_tx, p_rx)
                else:
                    viz.update_rays([], p_tx, p_rx)

            simulation_log.append({
                'time': t, 'tx_x': p_tx[0], 'tx_y': p_tx[1], 'tx_z': p_tx[2],
                'rx_x': p_rx[0], 'rx_y': p_rx[1], 'rx_z': p_rx[2],
                'horizontal_range': dist_h, 'transmission_loss_db': current_tl
            })

            return viz.update(frame, df_tx, df_rx, current_tl, dist_h)
        
        ani = animation.FuncAnimation(
            viz.fig, update_frame, init_func=lambda: [], frames=min_len,
            interval=1, blit=False, repeat=False
        )
        
        print(f"Simulation started Mode: {sim_method}.")
        plt.show()

    # ==========================================
    # 分岐B: PEモード（アニメーションをスキップし一括並列計算）
    # ==========================================
    elif sim_method == 'PE':
        print("Computing PE frames in background (Multi-processing). Animation is skipped...")
        surface_model = DynamicSurfaceModel(env_cfg)
        
        start_time = time.time()
        print_progress(0, min_len)  # 開始直後に 0% を明示
        
        for frame in range(min_len):
            p_tx = df_tx.iloc[frame][['x', 'y', 'z']].values
            p_rx = df_rx.iloc[frame][['x', 'y', 'z']].values
            t = df_tx.iloc[frame]['time']
            dist_h = np.linalg.norm(p_tx[:2] - p_rx[:2])
            
            # 並列化されたエンジンを呼び出し
            cir_complex, current_tl = PESolver.compute_single_frame_transmission(
                env_cfg, surface_model, p_tx, p_rx, dist_h, t
            )
            
            # TVIR用パラメータの算出と格納
            bandwidth = env_cfg.get('bandwidth', 5000.0)
            N_f = env_cfg.get('num_frequencies', 1024)
            
            dt = 1.0 / bandwidth
            pe_delays = np.arange(N_f) * dt + (dist_h / 1500.0)
            
            pe_record = {
                'sim_time': t,
                'delays': pe_delays.tolist(),
                'amps': cir_complex.tolist() 
            }
            tvir_calc.records.append(pe_record)

            simulation_log.append({
                'time': t, 'tx_x': p_tx[0], 'tx_y': p_tx[1], 'tx_z': p_tx[2],
                'rx_x': p_rx[0], 'rx_y': p_rx[1], 'rx_z': p_rx[2],
                'horizontal_range': dist_h, 'transmission_loss_db': current_tl
            })

            # utilsからインポートした進捗バーを更新
            print_progress(frame + 1, min_len)
        
        elapsed_time = time.time() - start_time
        print(f"PE calculation completed in {elapsed_time:.2f} seconds.")

    # ==========================================
    # 共通処理: TVIRおよび信号の描画とCSV保存
    # ==========================================
    print("Generating TVIR Waterfall plot...")
    max_delay_ms = 1000
    tvir_calc.show_results(max_delay_ms)

    # 送信信号生成
    gen = SignalGenerator(fs=env_cfg['signal']['fs'])
    t_tx, s_tx = gen.generate_sin_wave(
        freq=env_cfg['frequency'], 
        start_time=env_cfg['signal']['start_time'], 
        duration=env_cfg['signal']['duration'], 
        total_sim_time=10.0
    )

    # 受信信号生成
    synth = SignalSynthesizer(fs=env_cfg['signal']['fs'])
    t_rx, s_rx = synth.synthesize(s_tx, tvir_calc.records, max_delay_ms)
    synth.plot_comparison(t_rx, s_tx, s_rx)

    # 5. CSV保存処理
    if simulation_log:
        output_df = pd.DataFrame(simulation_log)
        output_df.to_csv(OUTPUT_CSV, index=False)
        print(f"--- Results saved to {OUTPUT_CSV} ---")

# Windowsのマルチプロセス動作に必要なガードエントリーポイント
if __name__ == "__main__":
    run_simulation()