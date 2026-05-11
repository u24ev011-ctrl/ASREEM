# EcoFuse-µNet 🌊🔬

**Reliability-Aware Optical–Ultrasonic Sensor Fusion for Real-Time Microplastic Detection on Low-Power FPGA**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Xilinx Vitis](https://img.shields.io/badge/Xilinx-Vivado_HLS-black.svg?style=flat&logo=Xilinx)](https://www.xilinx.com/)
[![PYNQ](https://img.shields.io/badge/PYNQ-v3.0-ff69b4.svg)](http://www.pynq.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EcoFuse-µNet is an end-to-end multimodal AI system designed to detect tiny microplastics in hostile wastewater environments. Traditional optical sensors fail under high turbidity and false positives (bubbles, organic debris). EcoFuse-µNet solves this by fusing streaming camera data with ultrasonic scattering profiles using a novel, lightweight fusion mechanism: **RIME-Fuse**. 

Designed strictly for edge deployment, the entire INT8 quantized system runs on a **Zynq-7020 (PYNQ-Z2)** FPGA within a <3W power budget at 15–30 FPS.

## 🚀 Key Innovations

*   **RIME-Fuse (Reliability-Aware Intermodal Microplastic Evidence Fusion):** A feature-level gating mechanism that uses acoustic scattering confidence to suppress optical false positives (like bubbles) and boost true plastic detections.
*   **Echo-to-Grid Low-Rank Projection:** Converts 1D ultrasonic temporal sequences into 2D optical feature spaces without the massive memory overhead of cross-attention transformers or dense full-frame concatenation.
*   **Asynchronous Sensor Decay:** Handles temporal drift and jitter between the camera and ultrasonic Analog-to-Digital Converter (ADC) natively within the fusion math.
*   **FPGA-First Architecture:** Built entirely around depthwise line-buffers, INT8 DSP MAC mapping, and minimal DRAM traffic (~150–350 MB/s). 

---

## 📂 Repository Structure

```text
EcoFuse-uNet/
├── model/                 # Algorithm design & training
│   ├── architectures/     # PyTorch definitions for Edge-Ghost Encoder & RIME-Fuse
│   ├── quantization/      # QAT (Quantization-Aware Training) scripts
│   └── augmentations/     # Synthetic turbidity and acoustic noise models
├── firmware/              # Vitis HLS C++ source and Vivado constraints
│   ├── mac_engine/        # INT8 matrix multiply accumulator IP
│   └── fusion_gate/       # Hard-sigmoid & low-rank projection IP
├── hardware/              # Pre-compiled bitstreams & overlays
│   └── pynq_z2/           # .bit and .hwh files for PYNQ deployment
├── notebooks/             # Jupyter notebooks for running on the PYNQ-Z2 PS
├── data/                  # Sample synchronized optical/ultrasonic data arrays
└── requirements.txt       # Python dependencies for host machine
