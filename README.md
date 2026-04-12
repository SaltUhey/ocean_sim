# Ocean Acoustic Propagation Simulator (ocean_sim) [Under Development]
Underwater acoustic propagation simulation between two UUVs using BELLHOP via arlpy.

## 📌 Project Overview
1対1のUUV（送信機・受信機）間における水中音波伝搬シミュレーターです。
移動する2台のUUVの座標データ（CSV）に基づき、各時刻の伝搬損失（Transmission Loss）や音線を計算します。

### Planned Features
- [x] Dynamic Path Integration: CSVから読み込んだ2台のUUV軌跡に基づく逐次シミュレーション
- [x] Acoustic Solver Integration: `arlpy` を介した BELLHOP による伝搬損失（TL）の精密計算
- [ ] 3D Visualization: UUVの移動、航跡、および音線の描画
- [ ] 送信波形・受信波形の出力
- [ ] ドップラーシフトの考慮
- [ ] 環境雑音の考慮
- [ ] 音響通信評価への拡張

## 🛠 Tech Stack & Dependencies
- **Language:** Python 3.10 (Conda environment: `ocean_sim`)
- **Acoustic Solver:** BELLHOP (Acoustics Toolbox)
- **Python Libraries:** `arlpy`, `pandas`, `matplotlib`, `numpy`
- **OS:** Windows 11

### ⚠️ External Tool Requirement (BELLHOP)
本シミュレーターの実行には、外部ツールの **Acoustics Toolbox (BELLHOP)** が必要です。
- **Source:** [Ocean Acoustics Library (OALIB)](http://oalib.hlsresearch.com/)
- **Setup:** ダウンロードした `at/bin` 内の実行ファイル（`bellhop.exe` 等）は、本プロジェクトディレクトリとは別の場所に配置し、Windowsの**システム環境変数（PATH）**に設定して使用してください。
- **Citation:** *Porter, M. B. (2011). The BELLHOP manual and user's guide: Preliminary draft.*

## 📂 Directory Structure
- `data/`: UUVの軌跡データ（uuv_tx_trajectory.csv, uuv_rx_trajectory.csv）
- `src/`: シミュレーション実行および可視化（Matplotlib）のメインスクリプト
- `utils/`: 座標変換、データ補間、音速プロファイル（SSP）生成などの補助関数群
- `tests/`: パラメータチューニングおよびプロトタイプ開発用

## 🚀 Getting Started
1. `data/env_config.xml` で海洋環境（水深、音速分布など）を設定します。
2. `data/uuv_tx_trajectory.csv` および uuv_rx_trajectory.csv にUUVのシナリオデータを配置します。
3. `src/simulate_propagation.py` を実行すると、3Dアニメーションが開始され、終了後に伝搬データが保存されます。

## ⚠️ License
**All Rights Reserved.**
Copyright (c) 2026 SalyUhey.

Unauthorized reproduction, redistribution, or modification of the code, data, and simulation results contained in this repository is strictly prohibited.
If you wish to use this project for academic purposes, collaborative research, or redistribution, please contact the author in advance to obtain explicit permission.

---
*This project is a work in progress. Features and documentation are subject to change.*
