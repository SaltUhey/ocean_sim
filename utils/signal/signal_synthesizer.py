import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt

class SignalSynthesizer:
    def __init__(self, fs=48000.0):
        self.fs = fs

    def synthesize(self, tx_signal, tvir_records, max_delay_ms, delay_res_s=None):
        """
        送信波形をTVIRと畳み込んで受信波形を生成する（複素位相・正当対応版）
        """
        if not tvir_records:
            raise ValueError("TVIR records are empty.")
        
        if delay_res_s is None:
            delay_res_s = 1.0 / self.fs

        # 1. 準備：時間軸と遅延軸の設定
        sim_times = np.array([r['sim_time'] for r in tvir_records])
        max_delay_s = max_delay_ms / 1000.0
        delay_bins = np.arange(0, max_delay_s, delay_res_s)
        n_delays = len(delay_bins)

        # 【修正】複素数（Complex）のままマトリクスを保持する
        raw_matrix_c = np.zeros((len(sim_times), n_delays), dtype=complex)

        for i, record in enumerate(tvir_records):
            for d, amp in zip(record['delays'], record['amps']):
                if d < max_delay_s:
                    bin_idx = int(round(d / delay_res_s))
                    if bin_idx < n_delays:
                        raw_matrix_c[i, bin_idx] += amp  # 複素数のまま加算

        total_samples = len(tx_signal)
        t_rx = np.arange(total_samples) / self.fs
        
        # 【重要】送信実信号を複素解析信号（ヒルベルト変換）に変換する
        # これにより、送信信号が「振幅と位相の情報を持つ複素数」になります
        tx_complex = hilbert(tx_signal)
        
        # 受信信号の複素バッファ
        rx_signal_c = np.zeros(total_samples, dtype=complex)

        print("Creating 1D complex time interpolators per delay bin...")
        
        for j in range(n_delays):
            # 複素振幅の列を抽出
            raw_amplitudes_at_delay = raw_matrix_c[:, j]
            
            if not np.any(raw_amplitudes_at_delay):
                continue
                
            # 複素数のまま時間軸補間を行う（SciPyのinterp1dは複素数も自動対応します）
            interp_func = interp1d(sim_times, raw_amplitudes_at_delay, kind='linear', 
                                   bounds_error=False, fill_value=0.0)
            amp_at_delay = interp_func(t_rx)  # 型: complex
            
            # 遅延させた「複素」送信信号を作成
            s_delayed_c = np.zeros(total_samples, dtype=complex)
            if j < total_samples:
                s_delayed_c[j:] = tx_complex[:total_samples - j]
            
            # 複素数どうしの掛け算（振幅の変化と、正しい位相回転がここで合成される）
            rx_signal_c += amp_at_delay * s_delayed_c

        print("Signal synthesis completed. Converting to real passband signal...")
        
        # 最後に実部（Real part）を取ることで、物理的な実数信号（パスバンド）に戻す
        rx_signal = np.real(rx_signal_c)
        
        return t_rx, rx_signal
    def plot_comparison(self, t, tx, rx):
        """送信信号と合成された受信信号の比較プロット"""
        plt.figure(figsize=(12, 6))
        
        # 上段：送信信号の全体像
        plt.subplot(2, 1, 1)
        plt.plot(t, tx, label="Transmitted Signal", alpha=0.7)
        plt.title("Comparison: Transmitted vs Received Signal")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # 下段：受信信号の全体像
        plt.subplot(2, 1, 2)
        plt.plot(t, rx, label="Received Signal (Synthesized)", color='orange')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()