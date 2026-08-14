import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt

class SignalSynthesizer:
    def __init__(self, fs=48000.0):
        self.fs = fs

    def synthesize(self, tx_signal, tvir_records, max_delay_ms, freq, sim_method, delay_res_s=None):
        """
        送信波形をベースバンドに変換し、TVIRと畳み込んだ後にアップコンバートして受信波形を生成する
        """
        if not tvir_records:
            raise ValueError("TVIR records are empty.")
        if sim_method not in ['BELLHOP', 'PE']:
            raise ValueError("sim_method must be either 'BELLHOP' or 'PE'")
        
        if delay_res_s is None:
            delay_res_s = 1.0 / self.fs

        # 1. 準備
        sim_times = np.array([r['sim_time'] for r in tvir_records])
        max_delay_s = max_delay_ms / 1000.0
        delay_bins = np.arange(0, max_delay_s, delay_res_s)
        n_delays = len(delay_bins)

        raw_matrix_c = np.zeros((len(sim_times), n_delays), dtype=complex)

        if sim_method == 'BELLHOP':
            for i, record in enumerate(tvir_records):
                for d, amp in zip(record['delays'], record['amps']):
                    if d < max_delay_s:
                        bin_idx = int(round(d / delay_res_s))
                        if bin_idx < n_delays:
                            raw_matrix_c[i, bin_idx] += amp

        elif sim_method == 'PE':
            # 【PE用】 連続的なベースバンド波形のリサンプリング（補間処理）
            for i, record in enumerate(tvir_records):
                pe_delays = np.array(record['delays'])
                pe_amps = np.array(record['amps'])
                
                # 遅延時間が単調増加になるようにソート（補間関数のエラー回避）
                sort_idx = np.argsort(pe_delays)
                pe_delays_sorted = pe_delays[sort_idx]
                pe_amps_sorted = pe_amps[sort_idx]
                
                # オーディオのサンプリング周波数(fs)のグリッドにマッピング
                interp_func_delay = interp1d(
                    pe_delays_sorted, pe_amps_sorted, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=0.0j  # 範囲外（波形が存在しない遅延帯）はゼロ埋め
                )
                raw_matrix_c[i, :] = interp_func_delay(delay_bins)

        total_samples = len(tx_signal)
        t_rx = np.arange(total_samples) / self.fs
        
        print("Synthesizing in baseband and upconverting...")

        # 2. 【ベースバンド変換】
        # 送信信号の解析信号（アナリティックシグナル）を求めてから、
        # exp(-j * 2 * pi * freq * t) を掛けてベースバンド（複素包絡線）に落とす
        tx_analytic = hilbert(tx_signal)
        tx_baseband = tx_analytic * np.exp(-1j * 2 * np.pi * freq * t_rx)

        # 受信ベースバンド信号の初期化
        rx_baseband = np.zeros(total_samples, dtype=complex)
        
        # 3. 【ベースバンド同士の畳み込み】
        for j in range(n_delays):
            raw_amplitudes_at_delay = raw_matrix_c[:, j]
            if not np.any(raw_amplitudes_at_delay):
                continue
                
            # 複素振幅の時間軸補間（extrapolateで端の値を維持）
            interp_func = interp1d(sim_times, raw_amplitudes_at_delay, kind='linear', 
                                   bounds_error=False, fill_value="extrapolate")
            amp_at_delay = interp_func(t_rx)  # 複素振幅
            
            # ベースバンド信号のインデックスシフト（遅延表現）
            # ベースバンドは変化が緩やかなため、整数シフトによる位相の乱れが起きにくい
            s_delayed_bb = np.zeros(total_samples, dtype=complex)
            if j < total_samples:
                s_delayed_bb[j:] = tx_baseband[:total_samples - j]
            
            # 複素振幅と遅延ベースバンド信号の足し合わせ
            rx_baseband += amp_at_delay * s_delayed_bb

        # 4. 【アップコンバート（パスバンドへ戻す）】
        # exp(j * 2 * pi * freq * t) を掛けて、実部を取ることで元のキャリア周波数に戻す
        rx_signal = np.real(rx_baseband * np.exp(1j * 2 * np.pi * freq * t_rx))

        print("Signal synthesis completed successfully.")
        return t_rx, rx_signal

    def plot_comparison(self, t, tx, rx):
        """送信信号と合成された受信信号の比較プロット"""
        plt.figure(figsize=(12, 6))
        
        # 上段：送信信号の全体像
        plt.subplot(2, 1, 1)
        plt.plot(t, tx, label="Transmitted Signal", alpha=0.7)
        plt.title("Comparison: Transmitted vs Received Signal (Baseband Convolved)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # 下段：受信信号の全体像
        plt.subplot(2, 1, 2)
        plt.plot(t, rx, label="Received Signal (Upconverted)", color='orange')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()