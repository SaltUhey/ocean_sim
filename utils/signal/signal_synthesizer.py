import numpy as np
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
import matplotlib.pyplot as plt

class SignalSynthesizer:
    def __init__(self, fs=48000.0):
        self.fs = fs

    def synthesize(self, tx_signal, tvir_records, max_delay_ms, delay_res_s=None):
        """
        送信波形をTVIRを畳み込んで受信波形を生成する
        """
        if not tvir_records:
            raise ValueError("TVIR records are empty.")
        
        if delay_res_s is None:
            # Delay方向の解像度をサンプリング周期 (1/fs) に合わせる (最強の解像度)
            delay_res_s = 1.0 / self.fs

        # 1. 準備：シミュレーション時間軸の抽出
        sim_times = np.array([r['sim_time'] for r in tvir_records])
        max_delay_s = max_delay_ms/1000
        delay_bins = np.arange(0, max_delay_s, delay_res_s)

        # 複素数（振幅と位相情報）を保持するマトリクス (Time x Delay)
        raw_matrix = np.zeros((len(sim_times), len(delay_bins)), dtype=complex)

        for i, record in enumerate(tvir_records):
            for d, amp in zip(record['delays'], record['amps']):
                if d < max_delay_s:
                    bin_idx = int(round(d / delay_res_s))
                    if bin_idx < len(delay_bins):
                        # BELLHOPの複素振幅をそのまま加算（干渉の再現のため）
                        raw_matrix[i, bin_idx] += amp

        print("Interpolating TVIR to sampling frequency level...")
        # -------------------------------------------------------------
        # STEP 2: 2次元補間関数の作成
        # -------------------------------------------------------------
        # RegularGridInterpolator を用いて、実部と虚部をそれぞれ補間
        r_interp = RegularGridInterpolator((sim_times, delay_bins), np.real(raw_matrix), 
                                           method='linear', bounds_error=False, fill_value=0.0)
        i_interp = RegularGridInterpolator((sim_times, delay_bins), np.imag(raw_matrix), 
                                           method='linear', bounds_error=False, fill_value=0.0)
        
        # 受信時間軸への完全アップサンプリング
        total_samples = len(tx_signal)
        t_rx = np.arange(total_samples) / self.fs
        rx_signal = np.zeros(total_samples)

        # 効率化のため、送信信号の補間関数を作成（分数遅延対策）
        t_tx_orig = np.arange(len(tx_signal)) / self.fs
        s_interp = interp1d(t_tx_orig, tx_signal, kind='linear', 
                            bounds_error=False, fill_value=0.0)
        
        print("Synthesizing received signal...")
        for j, tau in enumerate(delay_bins):
            # このDelayにおける、全受信時刻 t_rx での複素振幅を一括補間
            # グリッド点 (t_rx, tau) を作成
            points = np.vstack([t_rx, np.full_like(t_rx, tau)]).T
            
            amp_r = r_interp(points)
            amp_i = i_interp(points)
            amp_mesh = amp_r + 1j * amp_i
            
            # 遅延させた送信信号
            t_lookup = t_rx - tau
            s_delayed = s_interp(t_lookup)
            
            # 物理的な実数成分を足し合わせる
            rx_signal += np.real(amp_mesh * s_delayed)
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

        # 下段：受信信号の全体像（伝搬損失で小さくなっているため、単体でスケールを合わせる）
        plt.subplot(2, 1, 2)
        plt.plot(t, rx, label="Received Signal (Synthesized)", color='orange')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()