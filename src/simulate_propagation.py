import pandas as pd
import numpy as np
import os
import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import arlpy.uwapm as pm

# 自作ユーティリティのインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env_setup.env_tools import load_env_config
from utils.motion.visualizer import UUVVisualizer
from utils.tvir.tvir_calculator import TVIRCalculator

# --- パス設定 ---
TX_CSV = os.path.join(project_root, 'data', 'uuv_tx_trajectory.csv')
RX_CSV = os.path.join(project_root, 'data', 'uuv_rx_trajectory.csv')
XML_CONFIG = os.path.join(project_root, 'data', 'env_config.xml')
OUTPUT_CSV = os.path.join(project_root, 'data', 'propagation_results.csv')

# --- 設定パラメータ ---
RAY_UPDATE_INTERVAL = 10  # 音線の更新間隔（フレーム数）
NUM_RAYS = 50           # 可視化する音線の本数（この値を調整して密度を変える）

def run_simulation():
    # 1. データのロード
    env_cfg = load_env_config(XML_CONFIG)
    df_tx = pd.read_csv(TX_CSV)
    df_rx = pd.read_csv(RX_CSV)
    min_len = min(len(df_tx), len(df_rx))

    # 結果保存用のリスト
    simulation_log = []

    # TVIR計算機の初期化
    tvir_calc = TVIRCalculator()

    # 2. 可視化クラスの初期化
    all_x = np.concatenate([df_tx['x'], df_rx['x']])
    all_y = np.concatenate([df_tx['y'], df_rx['y']])
    viz = UUVVisualizer(
        x_limit=(np.min(all_x)-10, np.max(all_x)+10),
        y_limit=(np.min(all_y)-10, np.max(all_y)+10),
        max_depth=env_cfg['max_depth'],
        title=f"UUV Acoustic Simulation"
    )

    # 3. アニメーション用更新関数
    def update_frame(frame):
        p_tx = df_tx.iloc[frame][['x', 'y', 'z']].values
        p_rx = df_rx.iloc[frame][['x', 'y', 'z']].values
        t = df_tx.iloc[frame]['time']
        dist_h = np.linalg.norm(p_tx[:2] - p_rx[:2])

        # --- 物理計算 (BELLHOP) ---
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

        # --- 伝搬損失（TL）計算（毎フレーム） ---
        tl_grid = pm.compute_transmission_loss(env, mode=pm.incoherent)
        current_tl = np.abs(tl_grid.iloc[0, 0])

        # TVIR用のArrivals計算（新規追加）
        # 毎フレーム実行してデータを蓄積
        arrivals = pm.compute_arrivals(env)
        tvir_calc.add_frame(t, arrivals)

        # 古い音線を消去
        for line in viz.ray_lines:
            try:
                line.remove()
            except:
                pass
        # --- 音線（Ray Path）計算（数フレームに1回） ---
        # RAY_UPDATE_INTERVAL フレームごとに実行
        if frame % RAY_UPDATE_INTERVAL == 0:
            # BELLHOPで音線を計算 (DataFrameのリストが返る)
            print(f"[DEBUG] Frame {frame}: Computing {NUM_RAYS} rays for visualization...")
            #rays = pm.compute_rays(env)
            rays = pm.compute_eigenrays(env)
            # print(rays)
            # もし1本も到達していない場合は、空のリストが返る可能性があります
            if rays is not None and len(rays) > 0:
                viz.update_rays(rays, p_tx, p_rx)
            else:
                # 到達する音線がない場合は、古い線を消して何も表示しない
                viz.update_rays([], p_tx, p_rx)

        # --- データの蓄積 ---
        simulation_log.append({
            'time': t,
            'tx_x': p_tx[0], 'tx_y': p_tx[1], 'tx_z': p_tx[2],
            'rx_x': p_rx[0], 'rx_y': p_rx[1], 'rx_z': p_rx[2],
            'horizontal_range': dist_h,
            'transmission_loss_db': current_tl
        })

        # --- 描画更新 ---
        return viz.update(frame, df_tx, df_rx, current_tl, dist_h)

    def init_viz():
        # 何も返さない、あるいは背景要素の初期状態を返す
        return []
    
    # 4. アニメーション開始
    # repeat=False にして、終了時に保存処理が走るようにします
    ani = animation.FuncAnimation(
        viz.fig, update_frame, init_func=init_viz, frames=min_len,
        interval=1, #ms
        blit=False, repeat=False
    )
    
    print(f"Simulation started. Rays update every {RAY_UPDATE_INTERVAL} frames.")
    plt.show()

    print("Generating TVIR Waterfall plot...")
    tvir_calc.show_results(max_delay_ms=100) # 遅延時間の範囲は適宜調整

    # 5. CSV保存処理 (ウィンドウを閉じた後に実行)
    if simulation_log:
        output_df = pd.DataFrame(simulation_log)
        output_df.to_csv(OUTPUT_CSV, index=False)
        print(f"--- Results saved to {OUTPUT_CSV} ---")

if __name__ == "__main__":
    run_simulation()