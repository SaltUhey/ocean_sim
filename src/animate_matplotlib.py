import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

# --- パス設定 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'uuv_initial_path.csv')

def run_matplotlib_animation():
    # 1. データの読み込み
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    times = df['time'].values
    points = df[['x', 'y', 'z']].values

    # 2. フィギュアと3D軸の設定
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 軸範囲の設定
    ax.set_xlim(np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1)
    ax.set_ylim(-5, 5) # Yは動かないが範囲を指定
    ax.set_zlim(100, 0) # 深度方向に反転 (0が海面、100mが深い)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Depth [m]')
    ax.set_title('UUV Trajectory Animation')

    # 3. アニメーション用のプロット要素（初期状態は空）
    # UUVのカレント位置（点）
    uuv_point, = ax.plot([], [], [], 'bo', markersize=10, label='Current UUV')
    
    # 軌跡（線）
    trace_line, = ax.plot([], [], [], 'y-', linewidth=2, alpha=0.6, label='Trace')
    
    # 時刻表示のテキスト
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    ax.legend(loc='upper right')

    # 4. アニメーションの更新関数（各フレームごとに呼ばれる）
    def update(frame_num):
        """
        frame_num: 現在のフレーム番号 (0 から len(points)-1 まで)
        """
        # --- 軌跡（線）の更新 ---
        # 0フレーム目から現在フレームまでの点をプロット
        current_trace = points[:frame_num+1]
        trace_line.set_data(current_trace[:, 0], current_trace[:, 1])
        trace_line.set_3d_properties(current_trace[:, 2])
        
        # --- UUV位置（点）の更新 ---
        current_pos = points[frame_num]
        uuv_point.set_data([current_pos[0]], [current_pos[1]])
        uuv_point.set_3d_properties([current_pos[2]])
        
        # --- テキストの更新 ---
        current_time = times[frame_num]
        time_text.set_text(f'Time: {current_time:.1f} s')
        
        # 更新した要素をリストで返す（blitting用）
        return uuv_point, trace_line, time_text

    # 5. アニメーションの実行
    # interval: フレーム間の遅延時間（ミリ秒）。データの1秒刻みに合わせて1000ms。
    ani = animation.FuncAnimation(fig, update, frames=len(df), 
                                  interval=1000, blit=True, repeat=True)

    # Jupyter Notebook上で表示する場合
    try:
        from IPython.display import HTML
        # HTML5ビデオとして埋め込む（これが一番綺麗で見やすい）
        display(HTML(ani.to_html5_video()))
        plt.close() # 静止画のプロットを閉じる
    except ImportError:
        # Notebook環境でない場合は plt.show() でウィンドウを開く
        print("Note: If you are in Jupyter, 'to_html5_video()' is recommended.")
        plt.show()

if __name__ == "__main__":
    run_matplotlib_animation()