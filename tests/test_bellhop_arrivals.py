import numpy as np
import arlpy.uwapm as pm
import os

def calculate_thorp_attenuation_db_per_km(freq_hz):
    """
    Fortran コード (AttenMod) 内の Thorp (JKPS Eq. 1.34) 公式
    """
    f_khz = freq_hz / 1000.0
    f2 = f_khz ** 2
    thorp_db_km = 3.3e-3 + (0.11 * f2) / (1.0 + f2) + (44.0 * f2) / (4100.0 + f2) + 3e-4 * f2
    return thorp_db_km

def verify_bellhop_arrivals_complete():
    print("=== BELLHOP A-Mode Amplitude & Arrival Time Verification ===")

    # 検証条件
    target_dist = 3000.0   # 水平距離
    soundspeed = 1500.0    # 音速 1500 m/s
    freq_hz = 10000.0      # 周波数 10 kHz

    # 1. 環境設定 (反射のない超深海)
    env = pm.create_env2d(
        depth=10000.0,
        soundspeed=soundspeed,
        frequency=freq_hz
    )
    env['tx_depth'] = 5000.0
    env['rx_range'] = target_dist
    env['rx_depth'] = 5000.0

    # 2. BELLHOP の実行
    fname_base = "test_run_complete"
    print(f"Running BELLHOP... (Outputs: {fname_base}.*)")
    arrivals = pm.compute_arrivals(env, fname_base=fname_base)

    # 3. データの抽出
    df_sub = arrivals[np.isclose(arrivals['rx_range'], target_dist)] if 'rx_range' in arrivals.columns else arrivals
    if df_sub.empty:
        print("[ERROR] No arrival data found.")
        return
        
    amp_col = [c for c in df_sub.columns if 'amp' in c.lower()][0]
    time_col = [c for c in df_sub.columns if 'time' in c.lower() or 'delay' in c.lower()][0]
    
    # 最初のパス（直達波）を取得
    first_arrival = df_sub.sort_values(by=time_col).iloc[0]
    
    # --- [A] 到達時間 (Arrival Time) の検証 ---
    measured_time = first_arrival[time_col]
    # 直達波の理論時間 = 距離 / 音速
    theoretical_time = target_dist / soundspeed
    
    time_absolute_error = abs(measured_time - theoretical_time)
    time_error_percent = (time_absolute_error / theoretical_time) * 100

    # --- [B] 振幅 (Amplitude) の検証 ---
    measured_amp_mag = np.abs(first_arrival[amp_col])
    # 球面拡散 (1/R)
    spherical_amp = 1.0 / target_dist
    # Thorpの吸音減衰倍率の計算
    alpha_thorp = calculate_thorp_attenuation_db_per_km(freq_hz)
    dist_km = target_dist / 1000.0
    absorption_loss_db = alpha_thorp * dist_km
    absorption_factor = 10 ** (-absorption_loss_db / 20.0)
    # 総合理論振幅
    theoretical_amp_total = spherical_amp * absorption_factor
    
    amp_error_percent = abs(measured_amp_mag - theoretical_amp_total) / theoretical_amp_total * 100

    # 結果の表示
    print("\n" + "="*50)
    print(f" 条件: 距離 {target_dist:.1f} m | 音速 {soundspeed:.1f} m/s | 周波数 {freq_hz/1000.0:.1f} kHz")
    print("="*50)
    
    # 時間検証結果
    print("1. 到達時間の検証 (Arrival Time Verification)")
    print(f"  - BELLHOP 測定値 : {measured_time:.7f} 秒")
    print(f"  - 理論値 (D / C) : {theoretical_time:.7f} 秒")
    print(f"  - 絶対誤差       : {time_absolute_error:.7e} 秒")
    print(f"  - 相対誤差       : {time_error_percent:.5f} %")
    print("-" * 50)
    
    # 振幅検証結果（エラーのあったプリント文を修正しました）
    print("2. 振幅値の検証 (Amplitude Verification with Thorp)")
    print(f"  - BELLHOP 測定値 : {measured_amp_mag:.7f}")
    print(f"  - 理論値 (Total) : {theoretical_amp_total:.7f}")
    print(f"    (内訳 -> 球面拡散のみ: {spherical_amp:.7f} | Thorp減衰倍率: {absorption_factor:.4f})")
    print(f"  - 相対誤差       : {amp_error_percent:.5f} %")
    print("="*50)

if __name__ == "__main__":
    verify_bellhop_arrivals_complete()