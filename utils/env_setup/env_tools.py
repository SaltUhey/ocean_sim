import xml.etree.ElementTree as ET
import numpy as np
import os

def load_env_config(xml_path):
    """
    XMLファイルから海洋環境設定を読み込み、辞書形式で返す
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Config file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. グローバル設定
    global_cfg = root.find('Global')
    max_depth = float(global_cfg.find('MaxDepth').text)
    max_range = float(global_cfg.find('MaxRange').text)
    frequency = float(global_cfg.find('Frequency').text)

    # 2. SSP (音速分布) の抽出
    ssp_points = []
    for pt in root.find('SSP').findall('Point'):
        d = float(pt.get('depth'))
        s = float(pt.text)
        ssp_points.append([d, s])
    ssp = np.array(ssp_points)

    # 3. 海底設定
    bottom_cfg = root.find('Bottom')
    bottom = {
        'sound_speed': float(bottom_cfg.find('SoundSpeed').text),
        'density': float(bottom_cfg.find('Density').text),
        'attenuation': float(bottom_cfg.find('Attenuation').text)
    }

    return {
        'max_depth': max_depth,
        'max_range': max_range,
        'frequency': frequency,
        'ssp': ssp,
        'bottom': bottom
    }

# テスト用メイン処理
if __name__ == "__main__":
    # パスは実行環境に合わせて調整
    try:
        config = load_env_config('../../data/env_config.xml')
        print("--- Loaded Environment Config ---")
        print(f"Frequency: {config['frequency']} Hz")
        print(f"Max Depth: {config['max_depth']} m")
        print(f"SSP Points:\n{config['ssp']}")
    except Exception as e:
        print(f"Error: {e}")