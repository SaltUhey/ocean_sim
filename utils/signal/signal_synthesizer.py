import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class SignalSynthesizer:
    def __init__(self, fs=48000.0):
        self.fs = fs

    def synthesize(self, tx_signal, tvir_records):
        """
        TVIRを用いて受信波形を合成する
        """
        if not tvir_records:
            raise ValueError("TVIR records are empty.")

        # 1. 準備：シミュレーション時間軸の抽出
        sim_times = np.array([r['sim_time'] for r in tvir_records])
        total_samples = len(tx_signal)
        t_rx = np.arange(total_samples) / self.fs
        rx_signal = np.zeros(total_samples)

        # パスごとのデータを補間用に整理
        # ※BELLHOPのパス数は変動するため、ここでは各フレームで最も強いN本のパスを
        # 「パスID」として連続的に扱う簡易的な補間、または近傍参照を行います。
        
        # 簡略化のため、各サンプル時刻において「最も近いTVIRフレーム」の
        # 遅延と振幅を適用する処理から始めます（これでも10fpsあれば十分動きます）
        
        print("Synthesizing received signal...")
        
        # 効率化のため、送信信号の補間関数を作成（分数遅延対策）
        t_tx_orig = np.arange(len(tx_signal)) / self.fs
        s_interp = interp1d(t_tx_orig, tx_signal, kind='linear', 
                            bounds_error=False, fill_value=0.0)

        # 全サンプルを回すと重いため、フレームごとにチャンク処理
        for i in range(len(sim_times)):
            # 現在のフレームの有効範囲（次のフレームまでの間）
            t_start = sim_times[i]
            t_end = sim_times[i+1] if i+1 < len(sim_times) else sim_times[i] + 0.1
            
            mask = (t_rx >= t_start) & (t_rx < t_end)
            t_chunk = t_rx[mask]
            
            if len(t_chunk) == 0:
                continue
                
            # このフレームのパス情報を取得
            delays = tvir_records[i]['delays']
            amps = tvir_records[i]['amps']
            
            chunk_result = np.zeros(len(t_chunk))
            
            for tau, amp in zip(delays, amps):
                # 受信時刻 t において、送信側のどの時刻の音を拾うべきか
                # t_lookup = t - tau
                t_lookup = t_chunk - tau
                
                # 送信信号から値をサンプリングして振幅（複素数）をかける
                # 物理的には実数部をとる
                path_contribution = np.real(amp * s_interp(t_lookup))
                chunk_result += path_contribution
            
            rx_signal[mask] = chunk_result

        return t_rx, rx_signal

    def plot_comparison(self, t, tx, rx):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, tx, label="Transmitted", alpha=0.7)
        plt.title("Transmitted vs Received Signal")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t, rx, label="Received", color='orange')
        plt.xlabel("Time [s]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()