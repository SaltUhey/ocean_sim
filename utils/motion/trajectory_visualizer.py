import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class UUVVisualizer:
    def __init__(self, x_limit, y_limit, max_depth, title="UUV Propagation Simulation"):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 軸の設定（0を海面、max_depthを海底とする）
        self.ax.set_xlim(x_limit)
        self.ax.set_ylim(y_limit)
        self.ax.set_zlim(max_depth, 0)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Depth [m]')
        self.ax.set_title(title)

        # プロット要素の初期化
        self.line_tx, = self.ax.plot([], [], [], 'r--', alpha=0.3, label='TX Path')
        self.uuv_tx, = self.ax.plot([], [], [], 'ro', markersize=8, label='TX (Source)')
        self.line_rx, = self.ax.plot([], [], [], 'b--', alpha=0.3, label='RX Path')
        self.uuv_rx, = self.ax.plot([], [], [], 'bo', markersize=8, label='RX (Receiver)')
        
        # 2台を結ぶリンク線
        self.link_line, = self.ax.plot([], [], [], 'g-', lw=1, alpha=0.5)
        
        # 情報表示用テキスト
        self.info_text = self.ax.text2D(0.05, 0.90, '', transform=self.ax.transAxes, 
                                        fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.ax.legend(loc='upper right')

    def update(self, frame, df_tx, df_rx, current_tl, distance):
        """
        メインループから呼ばれ、描画を更新する関数
        """
        # 座標抽出
        p_tx = df_tx.iloc[frame][['x', 'y', 'z']].values
        p_rx = df_rx.iloc[frame][['x', 'y', 'z']].values
        t = df_tx.iloc[frame]['time']

        # TX更新
        self.line_tx.set_data(df_tx['x'][:frame+1], df_tx['y'][:frame+1])
        self.line_tx.set_3d_properties(df_tx['z'][:frame+1])
        self.uuv_tx.set_data([p_tx[0]], [p_tx[1]])
        self.uuv_tx.set_3d_properties([p_tx[2]])

        # RX更新
        self.line_rx.set_data(df_rx['x'][:frame+1], df_rx['y'][:frame+1])
        self.line_rx.set_3d_properties(df_rx['z'][:frame+1])
        self.uuv_rx.set_data([p_rx[0]], [p_rx[1]])
        self.uuv_rx.set_3d_properties([p_rx[2]])

        # リンク線更新
        self.link_line.set_data([p_tx[0], p_rx[0]], [p_tx[1], p_rx[1]])
        self.link_line.set_3d_properties([p_tx[2], p_rx[2]])

        # テキスト更新
        self.info_text.set_text(
            f'Time: {t:.1f}s\n'
            f'Horizontal Range: {distance:.2f} m\n'
            f'Transmission Loss: {current_tl:.2f} dB'
        )

        return self.line_tx, self.uuv_tx, self.line_rx, self.uuv_rx, self.link_line, self.info_text