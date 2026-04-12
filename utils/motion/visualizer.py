import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

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
        self.link_arrow, = self.ax.plot([], [], [], 'g-', lw=1.0, alpha=0.7, label='Direct Path')
        
        # 音線（Ray Path）用のリスト (数フレームに1回更新)
        self.ray_lines = []

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

        # 通信対象線更新
        if hasattr(self, 'link_arrow') and self.link_arrow is not None:
            self.link_arrow.remove()
        ux, uy, uz = p_rx[0] - p_tx[0], p_rx[1] - p_tx[1], p_rx[2] - p_tx[2]
        self.link_arrow = self.ax.quiver(p_tx[0], p_tx[1], p_tx[2], 
                                          ux, uy, uz, 
                                          color='green', alpha=0.8, linewidth=2,
                                          arrow_length_ratio=0.3) # 矢印の頭のサイズ
        # テキスト更新
        self.info_text.set_text(
            f'Time: {t:.1f}s\n'
            f'Horizontal Range: {distance:.2f} m\n'
            f'Transmission Loss: {current_tl:.2f} dB'
        )
        # 音線はここでは更新せず、戻り値リストに含めるだけにする
        artists = [self.line_tx, self.uuv_tx, self.line_rx, self.uuv_rx, self.link_arrow, self.info_text]
        artists.extend(self.ray_lines) # 現在描画されている音線も維持する
        
        return artists
    
    def update_rays(self, rays, p_tx, p_rx):
        """
        数フレームに1回呼ばれ、全音線を3D空間に再描画する
        """
        # 1. 古い音線を消去（simulate_propagation.py側ではなくこちらで行うのが安全）
        for line in self.ray_lines:
            try:
                line.remove()
            except:
                pass
        self.ray_lines = []

        # 2. 方位角（Azimuth）の計算
        dx = p_rx[0] - p_tx[0]
        dy = p_rx[1] - p_tx[1]
        angle = np.arctan2(dy, dx)

        # 3. rays の全行（全音線）をループして処理
        if isinstance(rays, pd.DataFrame):
            for i in range(len(rays)):
                # i番目の行（音線）を渡す
                self._plot_single_ray_row(rays.iloc[i], p_tx, angle)

    def _plot_single_ray_row(self, ray_series, p_tx, angle):
        """
        DataFrameの1行分（音線1本分）をプロットする
        """
        try:
            # Series内の 'ray' カラムには [[r0, z0], [r1, z1], ...] が入っている
            ray_data = np.array(ray_series['ray'])
            
            if ray_data.ndim != 2:
                return

            r = ray_data[:, 0]  # 水平距離
            z = ray_data[:, 1]  # 深さ
            
            # 3D座標への変換
            ray_x = p_tx[0] + r * np.cos(angle)
            ray_y = p_tx[1] + r * np.sin(angle)
            ray_z = z
            
            # 描画
            line, = self.ax.plot(
                ray_x, ray_y, ray_z, 
                color='yellow', 
                alpha=0.8, # 複数本重なるので透明度を少し下げる
                lw=0.6, 
                zorder=10
            )
            self.ray_lines.append(line)
            
        except Exception:
            pass