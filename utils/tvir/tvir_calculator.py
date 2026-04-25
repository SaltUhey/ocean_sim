import numpy as np
import matplotlib.pyplot as plt

class TVIRCalculator:
    def __init__(self):
        self.records = []

    def add_frame(self, sim_time, arrivals):
        """BELLHOPの結果(DataFrame)をリストに蓄積"""
        if arrivals is not None and not arrivals.empty:
            # --- カラム名の不一致を吸収するロジック ---
            # 'delay' または 'arrival_time' または 'time' を探す
            d_cols = [c for c in arrivals.columns if 'time' in c.lower() or 'delay' in c.lower()]
            # 'amplitude' または 'coeff' または 'amp' を探す
            a_cols = [c for c in arrivals.columns if 'amp' in c.lower() or 'coeff' in c.lower()]

            if not d_cols or not a_cols:
                # 念のため、実行時にカラム名を確認できるようにする
                print(f"[ERROR] Required columns not found. Columns are: {list(arrivals.columns)}")
                return

            self.records.append({
                'sim_time': sim_time,
                'delays': arrivals[d_cols[0]].values,
                'amps': arrivals[a_cols[0]].values
            })

    def show_results(self, max_delay_ms=100, delay_res_ms=0.2):
        """蓄積したデータを画像として表示"""
        if not self.records:
            print("TVIRデータがありません。")
            return

        sim_times = np.array([r['sim_time'] for r in self.records])
        delay_bins = np.arange(0, max_delay_ms, delay_res_ms)
        
        matrix = np.zeros((len(sim_times), len(delay_bins)))

        for i, record in enumerate(self.records):
            for d, a in zip(record['delays'], record['amps']):
                d_ms = d * 1000
                if d_ms < max_delay_ms:
                    bin_idx = int(d_ms / delay_res_ms)
                    if bin_idx < len(delay_bins):
                        matrix[i, bin_idx] += np.abs(a)

        plt.figure(figsize=(10, 6))
        # ログスケール（dB）に変換。最小値を-60dB程度に制限して視認性を確保
        matrix_db = 20 * np.log10(matrix + 1e-6)
        
        im = plt.imshow(matrix_db.T, extent=[sim_times[0], sim_times[-1], delay_bins[-1], 0],
                        aspect='auto', cmap='magma', interpolation='nearest',
                        vmin=-60, vmax=matrix_db.max())
        
        plt.colorbar(im, label='Relative Amplitude [dB]')
        plt.xlabel('Simulation Time [s]')
        plt.ylabel('Delay [ms]')
        plt.title('Time-Varying Impulse Response (TVIR)')
        plt.gca().invert_yaxis()
        plt.show()