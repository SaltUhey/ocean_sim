import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

# --- パス設定 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TX_CSV = os.path.join(PROJECT_ROOT, 'data', 'uuv_tx_trajectory.csv')
RX_CSV = os.path.join(PROJECT_ROOT, 'data', 'uuv_rx_trajectory.csv')

def run_dual_uuv_animation():
    # 1. データの読み込み
    try:
        df_tx = pd.read_csv(TX_CSV)
        df_rx = pd.read_csv(RX_CSV)
    except FileNotFoundError:
        print("Error: TX or RX CSV file not found.")
        return

    # データ長を揃える
    min_len = min(len(df_tx), len(df_rx))
    
    # 2. フィギュアと3D軸の設定
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 表示範囲の決定（両方のUUVをカバーする）
    all_x = np.concatenate([df_tx['x'], df_rx['x']])
    all_y = np.concatenate([df_tx['y'], df_rx['y']])
    ax.set_xlim(np.min(all_x) - 5, np.max(all_x) + 5)
    ax.set_ylim(np.min(all_y) - 5, np.max(all_y) + 5)
    ax.set_zlim(100, 0) # 0を海面、100mを海底とする

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Depth [m]')
    ax.set_title('UUV Communication Simulation: TX vs RX')

    # 3. プロット要素の定義
    # TX (送信側): 赤
    line_tx, = ax.plot([], [], [], 'r--', alpha=0.3, label='TX Path')
    uuv_tx, = ax.plot([], [], [], 'ro', markersize=10, label='TX (Source)')
    
    # RX (受信側): 青
    line_rx, = ax.plot([], [], [], 'b--', alpha=0.3, label='RX Path')
    uuv_rx, = ax.plot([], [], [], 'bo', markersize=10, label='RX (Receiver)')
    
    # 通信距離を表示するテキスト
    dist_text = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)
    ax.legend()

    # 4. アニメーション更新関数
    def update(frame):
        # TXの更新
        line_tx.set_data(df_tx['x'][:frame], df_tx['y'][:frame])
        line_tx.set_3d_properties(df_tx['z'][:frame])
        uuv_tx.set_data([df_tx['x'][frame]], [df_tx['y'][frame]])
        uuv_tx.set_3d_properties([df_tx['z'][frame]])

        # RXの更新
        line_rx.set_data(df_rx['x'][:frame], df_rx['y'][:frame])
        line_rx.set_3d_properties(df_rx['z'][:frame])
        uuv_rx.set_data([df_rx['x'][frame]], [df_rx['y'][frame]])
        uuv_rx.set_3d_properties([df_rx['z'][frame]])

        # 2台間の直線距離を計算
        pos_tx = np.array([df_tx['x'][frame], df_tx['y'][frame], df_tx['z'][frame]])
        pos_rx = np.array([df_rx['x'][frame], df_rx['y'][frame], df_rx['z'][frame]])
        distance = np.linalg.norm(pos_tx - pos_rx)
        dist_text.set_text(f'Time: {df_tx["time"][frame]:.1f}s\nDistance: {distance:.2f} m')

        return line_tx, uuv_tx, line_rx, uuv_rx, dist_text

    # 5. アニメーション実行
    # intervalはミリ秒単位。データが1秒刻みなら500〜1000程度が見やすいです。
    ani = animation.FuncAnimation(fig, update, frames=min_len, interval=500, blit=False)

    plt.show()

if __name__ == "__main__":
    run_dual_uuv_animation()