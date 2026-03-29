# Ocean Communication Simulator (ocean_sim) [Under Development]
Underwater acoustic communication simulation between two UUVs using BELLHOP via arlpy.

## 📌 Project Overview
1対1のUUV（送信機・受信機）間における水中音響通信の品質評価シミュレーターです。
移動する2台のUUVの座標データ（CSV）に基づき、各時刻の伝搬損失（Transmission Loss）や音線を計算します。

### Planned Features
- [x] 2台のUUV軌跡データの生成と3D可視化（Matplotlib）
- [ ] 各時刻ごとの伝搬損失（TL）の逐次計算ロジック
- [ ] 送信波形の設定および受信側での波形歪み・復調シミュレーション（予定）

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
- `data/`: UUVの軌跡データ（uuv_tx.csv, uuv_rx.csv）
- `src/`: シミュレーション実行および可視化（Matplotlib）のメインスクリプト
- `utils/`: 座標変換、データ補間、音速プロファイル（SSP）生成などの補助関数群
- `notebooks/`: パラメータチューニングおよびプロトタイプ開発用

## ⚠️ License
**All Rights Reserved.**
Copyright (c) 2026 SalyUhey.

Unauthorized reproduction, redistribution, or modification of the code, data, and simulation results contained in this repository is strictly prohibited.
If you wish to use this project for academic purposes, collaborative research, or redistribution, please contact the author in advance to obtain explicit permission.

---
*This project is a work in progress. Features and documentation are subject to change.*
