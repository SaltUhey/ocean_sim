import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def __init__(self, fs=48000.0):
        self.fs = fs

    def generate_sin_wave(self, freq, start_time, duration, total_sim_time):
        """
        全シミュレーション時間の中で、特定のタイミングだけ音が鳴る波形を生成
        振幅は一定（矩形波的な切り出し）
        """
        # 全サンプルの時間軸を作成
        t_total = np.arange(0, total_sim_time, 1/self.fs)
        signal = np.zeros_like(t_total)

        # 音が鳴っている区間のインデックスを特定
        start_idx = int(round(start_time * self.fs))
        end_idx = int(round((start_time + duration) * self.fs))

        # 範囲外チェック
        if start_idx >= len(t_total):
            print("Warning: Start time is beyond total simulation time.")
            return t_total, signal

        # 音が鳴る部分の時間軸 (サンプルの個数を厳密に計算)
        num_samples = end_idx - start_idx
        t_pulse = np.arange(num_samples) / self.fs
        
        # 指定区間だけにSin波を挿入 (振幅1.0で固定)
        pulse = np.sin(2 * np.pi * freq * t_pulse)
        
        actual_end_idx = min(end_idx, len(signal))
        # 窓関数を削除し、そのまま代入
        signal[start_idx:actual_end_idx] = pulse[:actual_end_idx-start_idx]

        return t_total, signal

    def plot_signal(self, t, signal, title="Transmitted Signal"):
        plt.figure(figsize=(10, 4))
        # 全体を表示
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        plt.title(title)
        plt.ylabel("Amplitude")
        plt.grid(True)

        # 信号の開始部分を拡大して「振幅が一定」であることを確認
        plt.subplot(2, 1, 2)
        # 信号がない場合は 0付近を表示
        start_t = t[np.where(signal != 0)[0][0]] if np.any(signal != 0) else 0
        plt.plot(t, signal)
        plt.xlim(start_t, start_t + 0.01)
        plt.title("Zoomed View (Start of pulse)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# --- 使用例 ---
if __name__ == "__main__":
    fs = 48000
    freq = 5000      # 5kHz
    start_t = 0.0    # 0秒後に打ち出し
    dur = 7.5        # 7.5秒間
    sim_t = 10.0      # 全体10秒
    
    gen = SignalGenerator(fs)
    t, s = gen.generate_sin_wave(freq, start_t, dur, sim_t)
    gen.plot_signal(t, s)