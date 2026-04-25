import pandas as pd
import numpy as np
import os

def trajectory_generator(filename, start_pos, duration_sec, step_sec):
    """
    UUVの移動データを生成し、ocean_sim/dataに保存する
    """
    # 1. このスクリプト(trajectory_generator.py)の絶対パスを取得
    current_script_path = os.path.abspath(__file__)
    
    # 2. ocean_sim フォルダ（プロジェクトルート）のパスを取得
    # utils フォルダの1つ上の階層
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    
    # 3. 保存先のディレクトリパスを作成 (ocean_sim/data/raw)
    save_dir = os.path.join(project_root, 'data')
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    save_path = os.path.join(save_dir, filename)

    # --- データ生成ロジック ---
    times = np.arange(0, duration_sec + step_sec, step_sec)
    n_steps = len(times)
    velocity_x = 1.5 # m/s
    
    data = {
        'time': times,
        'x': start_pos['x'] + (velocity_x * times),
        'y': [start_pos['y']] * n_steps,
        'z': [start_pos['z']] * n_steps,
        'Roll': [start_pos['roll']] * n_steps,
        'Pitch': [start_pos['pitch']] * n_steps,
        'Yaw': [start_pos['yaw']] * n_steps
    }

    df = pd.DataFrame(data)
    
    # CSV保存
    df.to_csv(save_path, index=False)
    print(f"Success: Saved trajectory to {save_path}")

if __name__ == "__main__":
    initial_condition = {
        'x': 0.0, 'y': 0.0, 'z': 50.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }
    trajectory_generator('uuv_trajectory.csv', initial_condition, duration_sec=5, step_sec=(1/5000))