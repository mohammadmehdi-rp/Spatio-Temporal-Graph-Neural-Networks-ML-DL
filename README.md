# Spatio-Temporal Graph Neural Networks for Network Traffic Prediction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the complete reproducible research package for **Spatio-Temporal Graph Neural Networks for Network Traffic Estimation and Prediction**, developed as part of a Master's thesis in Information Engineering at the University of Trento (2026).

**Supervisor:** Professor Fabrizio Granelli

## 📖 Overview

This work presents a novel approach to network traffic prediction using **Graph Neural Networks (GNNs)** on realistic network topologies emulated with **ComNetsEmu/Containernet**. The system predicts multiple network performance metrics including queue backlog, throughput, link utilization, and end-to-end latency using spatio-temporal features extracted from network telemetry.

### Key Contributions

- **Multi-target prediction framework**: Simultaneous estimation of queue backlog, throughput, utilization, and RTT
- **Graph-based approach**: Leverages network topology structure through GraphSAGE and RouteNet-lite encoders
- **Realistic emulation**: Uses ComNetsEmu with Docker containers for reproducible network experiments
- **Sensor optimization**: Intelligent sensor placement strategies to minimize monitoring overhead
- **Calibration techniques**: Post-training calibration to reduce false positives in idle/busy state detection
- **Cross-topology generalization**: Evaluation on both Dumbbell and NSFNET (14-node) topologies

## 🏗️ Architecture

The system architecture consists of:

1. **Network Emulation Layer** (ComNetsEmu/Containernet)
   - Dumbbell topology (canonical bottleneck scenario)
   - NSFNET topology (14 nodes, 21 links with realistic propagation delays)
   - Docker-based hosts for realistic traffic generation

2. **Data Collection Pipeline**
   - High-frequency (5 Hz) telemetry collection
   - Interface counters, queue statistics, and latency measurements
   - Automated processing and feature engineering

3. **GNN Model**
   - GraphSAGE or RouteNet-lite encoders
   - Temporal feature aggregation (multi-lag support)
   - Multi-target prediction heads
   - Calibration modules for improved accuracy

4. **Evaluation Framework**
   - Nowcast (current state) and Lead-1 (next timestep) prediction
   - Ensemble methods for stability
   - Comprehensive baseline comparisons (AR, ARIMA, global models)

## 📁 Repository Structure

```
├── src/                          # Python source code
│   ├── run_dumbbell_capture.py   # Dumbbell topology emulation & capture
│   ├── run_nsfnet_capture_plus.py # NSFNET topology emulation & capture
│   ├── simple_dumbbell.py        # Dumbbell topology builder
│   ├── simple_nsfnet.py          # NSFNET topology builder
│   ├── collector_5hz.py          # High-frequency telemetry collector
│   ├── models_gnn.py             # GNN model architectures
│   ├── train_*.py                # Training scripts
│   ├── eval_*.py                 # Evaluation scripts
│   └── calibrate_*.py            # Calibration methods
├── scripts/                      # Automation scripts
│   ├── reproduce_all_fast.sh     # Quick reproduction (minutes)
│   ├── reproduce_all_full.sh     # Full reproduction (hours)
│   ├── run_dumbbell_full.sh      # Dumbbell experiments
│   └── run_nsfnet_multitarget.sh # NSFNET experiments
├── data/npz/                     # NPZ datasets (input)
├── outputs/                      # Generated outputs
│   ├── figures/                  # Plots (PDF/PNG)
│   ├── tables/                   # LaTeX tables
│   ├── models/                   # Trained checkpoints
│   ├── summaries/                # CSV/JSON results
│   └── logs/                     # Training logs
├── runs/                         # Experiment runs
│   ├── dumbbell_seed*/           # Dumbbell experiments
│   └── nsfnet_seed*/             # NSFNET experiments
├── docs/                         # Documentation
├── latex/                        # Thesis LaTeX snippets
├── environment.yml               # Conda environment
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **Docker**: For ComNetsEmu/Containernet
- **Root access**: Required for Mininet network emulation
- **Optional**: CUDA-capable GPU for faster training

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/mohammadmehdi-rp/Spatio-Temporal-Graph-Neural-Networks-ML-DL.git
cd Spatio-Temporal-Graph-Neural-Networks-ML-DL
```

#### 2. Set up Python environment

**Option A: Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate thesis-repro
```

**Option B: Pip**

```bash
pip install -r requirements.txt
```

#### 3. Build Docker images for ComNetsEmu

```bash
bash scripts/build_ndt_host_image.sh
```

This creates the `ndt/host:focal-nettools` Docker image used for network hosts.

## 📊 Reproducing Results

### Fast Mode (5-10 minutes)

Regenerates plots and tables from pre-computed summaries and cached artifacts:

```bash
bash scripts/reproduce_all_fast.sh
```

This will generate:

- Figures in `outputs/figures/`
- LaTeX tables in `outputs/tables/`
- Summary CSVs in `outputs/summaries/`

### Full Mode (Several hours)

Runs complete pipeline including network emulation, training, and evaluation:

```bash
bash scripts/reproduce_all_full.sh
```

**⚠️ Warning**: Requires sudo/root access for network emulation. Estimated time: 4-8 hours depending on hardware.

### Individual Experiments

**Dumbbell topology experiments:**

```bash
bash scripts/run_dumbbell_full.sh
```

**NSFNET topology experiments:**

```bash
bash scripts/run_nsfnet_multitarget.sh
```

**Multi-target estimation:**

```bash
bash scripts/reproduce_multi_target_estimation.sh
```

## 🧪 Running Custom Experiments

### 1. Capture Network Data

**Dumbbell topology:**

```bash
sudo -E python3 src/run_dumbbell_capture.py \
    --outdir runs/my_experiment \
    --n 4 \
    --img ndt/host:focal-nettools \
    --duration 180
```

**NSFNET topology:**

```bash
sudo -E python3 src/run_nsfnet_capture_plus.py \
    --outdir runs/my_nsfnet \
    --img ndt/host:focal-nettools \
    --duration 180
```

### 2. Train Models

```bash
python3 src/train_nowcast_sparse.py \
    --npz runs/my_experiment/dataset.npz \
    --out runs/my_experiment/models/nowcast.pt \
    --epochs 100
```

### 3. Evaluate Models

```bash
python3 src/eval_gnn.py \
    --npz runs/my_experiment/dataset.npz \
    --ckpt runs/my_experiment/models/nowcast.pt
```

## 📈 Key Results

The proposed GraphSAGE-based approach achieves:

- **Queue Backlog Estimation**: Significant improvement over AR/ARIMA baselines
- **Multi-target Prediction**: Simultaneous accurate estimation of 4 network metrics
- **Sensor Optimization**: 70%+ monitoring cost reduction with k=10 sensors
- **Calibration Benefits**: ~50% reduction in idle false positives with minimal impact on busy-state accuracy
- **Cross-topology Generalization**: Robust performance on both Dumbbell and NSFNET topologies

See `docs/RESULTS_OVERVIEW.md` for detailed results and figures.

## 🔧 Technologies Used

- **Network Emulation**: [ComNetsEmu](https://git.comnets.net/public-repo/comnetsemu)/Containernet, Mininet
- **Deep Learning**: PyTorch, PyTorch Geometric
- **GNN Architectures**: GraphSAGE, RouteNet-lite
- **Traffic Generation**: iperf3, Docker containers
- **Telemetry**: Linux tc/netlink, interface statistics
- **Visualization**: Matplotlib, Seaborn
- **Baseline Models**: statsmodels (AR/ARIMA), scikit-learn

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{rajabpourshirazy2026spatiotemporal,
  author = {Rajabpourshirazy, Mohammad Mehdi},
  title = {Spatio-Temporal Graph Neural Networks for Network Traffic Estimation and Prediction},
  school = {University of Trento},
  year = {2026},
  type = {Master's Thesis},
  department = {Information Engineering}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Professor Fabrizio Granelli for his invaluable supervision and guidance
- ComNetsEmu team for the network emulation framework
- PyTorch Geometric developers
- University of Trento, Department of Information Engineering

## 📧 Contact

**Mohammad Mehdi Rajabpourshirazy**

- GitHub: [@mohammadmehdi-rp](https://github.com/mohammadmehdi-rp)
- University of Trento, Information Engineering

## 🐛 Troubleshooting

### Common Issues

**1. "Mininet/ComNetsEmu must run as root"**

```bash
sudo -E python3 src/run_dumbbell_capture.py ...
```

The `-E` flag preserves your environment variables.

**2. Docker image not found**

```bash
bash scripts/build_ndt_host_image.sh
```

**3. CUDA out of memory**
Reduce batch size in training scripts or use CPU:

```bash
python3 src/train_nowcast_sparse.py --device cpu ...
```

**4. Path issues**
Always run scripts from the repository root directory.

**5. Large files warning on GitHub**
Some output CSVs exceed 50 MB. Consider using Git LFS for frequent modifications:

```bash
git lfs install
git lfs track "*.csv"
```

For additional help, please open an issue on GitHub.
