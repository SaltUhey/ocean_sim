import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt

class SignalSynthesizer:
    def __init__(self, fs=48000.0):
        self.fs = fs

    def synthesize(self, tx_signal, tvir_records, max_delay_ms, delay_res_s=None):
        """
        送信波形をTVIRと畳み込んで受信波形を生成する（arlpyパスバンド仕様・完全整合版）
        """
        if not tvir_records:
            raise ValueError("TVIR records are empty.")
        
        if delay_res_s is None:
            delay_res_s = 1.0 / self.fs

        # 1. 準備
        sim_times = np.array([r['sim_time'] for r in tvir_records])
        max_delay_s = max_delay_ms / 1000.0
        delay_bins = np.arange(0, max_delay_s, delay_res_s)
        n_delays = len(delay_bins)

        # arlpyの複素振幅出力をそのまま複素マトリクスとして保持
        raw_matrix_c = np.zeros((len(sim_times), n_delays), dtype=complex)

        for i, record in enumerate(tvir_records):
            for d, amp in zip(record['delays'], record['amps']):
                if d < max_delay_s:
                    bin_idx = int(round(d / delay_res_s))
                    if bin_idx < n_delays:
                        raw_matrix_c[i, bin_idx] += amp

        total_samples = len(tx_signal)
        t_rx = np.arange(total_samples) / self.fs
        rx_signal = np.zeros(total_samples)
        
        # 送信信号の90度移相信号（虚数部）を作るためにヒルベルト変換を使用
        tx_hilbert = hilbert(tx_signal)
        s_tx_real = np.real(tx_hilbert)  # 元の送信信号 s(t)
        s_tx_imag = np.imag(tx_hilbert)  # 90度移相した送信信号 s_hat(t)
        
        print("Synthesizing received signal via arlpy passband formula...")
        
        for j in range(n_delays):
            raw_amplitudes_at_delay = raw_matrix_c[:, j]
            if not np.any(raw_amplitudes_at_delay):
                continue
                
            # 複素振幅のま目で時間軸補間を実行
            interp_func = interp1d(sim_times, raw_amplitudes_at_delay, kind='linear', 
                                   bounds_error=False, fill_value=0.0)
            amp_at_delay = interp_func(t_rx)  # 型: complex (R + jI)
            
            # 遅延させた送信信号（実部と直交成分）をインデックスシフトで生成
            s_delayed_real = np.zeros(total_samples)
            s_delayed_imag = np.zeros(total_samples)
            if j < total_samples:
                s_delayed_real[j:] = s_tx_real[:total_samples - j]
                s_delayed_imag[j:] = s_tx_imag[:total_samples - j]
            
            # 【真のパスバンド合成式】
            # Re(Amp * s_complex) = Re(Amp)*Re(s) - Im(Amp)*Im(s)
            rx_signal += np.real(amp_at_delay) * s_delayed_real - np.imag(amp_at_delay) * s_delayed_imag

        print("Signal synthesis completed successfully.")
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