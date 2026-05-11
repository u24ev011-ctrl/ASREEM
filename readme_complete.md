# EcoFuse-µNet 🌊🔬

**Reliability-Aware Optical–Ultrasonic Sensor Fusion for Real-Time Microplastic Detection on Low-Power FPGA**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Xilinx Vitis](https://img.shields.io/badge/Xilinx-Vivado_HLS-black.svg?style=flat&logo=Xilinx)](https://www.xilinx.com/)
[![PYNQ](https://img.shields.io/badge/PYNQ-v3.0-ff69b4.svg)](http://www.pynq.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EcoFuse-µNet is an end-to-end multimodal AI system designed to detect tiny microplastics in hostile wastewater environments. Traditional optical sensors fail under high turbidity and suffer from false positives caused by bubbles and organic debris. Ultrasonic sensors capture physical density but lack spatial resolution. 

EcoFuse-µNet solves this by fusing streaming camera data with ultrasonic scattering profiles using a novel, lightweight fusion mechanism: **RIME-Fuse**. Designed strictly for edge deployment, the entire INT8-quantized system runs on a **Zynq-7020 (PYNQ-Z2)** FPGA within a <3W power budget at 15–30 FPS.

---

## 🚀 Core Innovations

*   **RIME-Fuse (Reliability-Aware Intermodal Microplastic Evidence Fusion):** A feature-level gating mechanism that uses acoustic scattering confidence to suppress optical false positives and boost true plastic detections.
*   **Echo-to-Grid Low-Rank Projection:** Converts 1D ultrasonic temporal sequences into 2D optical feature spaces without the massive memory overhead of cross-attention transformers or dense full-frame concatenation.
*   **Asynchronous Sensor Decay:** Handles temporal drift and jitter between the camera and ultrasonic Analog-to-Digital Converter (ADC) natively within the fusion math.
*   **Temporal Stabilization on Logits:** Avoids flickering detections caused by flow/bubbles by applying temporal smoothing to the lightweight output logits rather than heavy intermediate feature maps.

---

## 🧠 Model Architecture: EcoFuse-µNet

The model contains ~1.05M–1.25M parameters, requiring roughly 160M–190M MACs/frame and peaking at ~200–300 KB of tiled activation memory.

### 1. Optical Branch: Edge-Ghost Encoder
To detect tiny particles under low contrast while remaining FPGA-friendly, the input ($160 \times 160 \times 2$ containing luminance + Sobel edge) passes through lightweight depthwise-separable "Ghost" blocks.

| Stage | Output | Operator | Params | MACs/frame |
| :--- | :--- | :--- | :--- | :--- |
| **Stem** | $80 \times 80 \times 16$ | 3x3 Conv (s=2) | 0.3K | 1.8M |
| **O1** | $80 \times 80 \times 16$ | 2x Ghost-DS block | 5K | 18M |
| **O2** | $40 \times 40 \times 32$ | 3x Ghost-DS block | 28K | 25M |
| **O3** | $20 \times 20 \times 64$ | 4x Ghost-DS block | 120K | 35M |
| **O4** | $10 \times 10 \times 96$ | 3x Ghost-DS block | 180K | 14M |
| **O5** | $5 \times 5 \times 128$ | 2x Ghost-DS block | 150K | 3M |

*A compact Feature Pyramid Network (FPN) uses addition (not concatenation) to generate $P_1 (80\times80\times32)$ for tiny particles and $P_2 (40\times40\times48)$ for medium particles.*

### 2. Ultrasonic Branch: Scattering Encoder
The input ($256 \times 3$) consists of the echo envelope, local attenuation estimate, and temporal derivative. It uses 1D Convolutions and a fixed Spectral Bank to estimate acoustic scattering strength and attenuation trend.

| Stage | Output | Operator | Params | MACs/frame |
| :--- | :--- | :--- | :--- | :--- |
| **U-Stem** | $256 \times 16$ | 1D Conv (k=7) | 0.4K | 0.1M |
| **U1, U2** | $128 \times 32, 64 \times 48$ | DS-TCN (stride 2) | 11K | 0.6M |
| **U3** | $32 \times 64$ | Dilated DS-TCN | 15K | 0.35M |
| **Spectral Bank** | $16 \times 32$ | Goertzel/FIR energy bank | 2K | 0.15M |
| **Global Vec** | $1 \times 96$ | Pooling + Projection | 12K | <0.1M |

---

## 🧮 Sensor Fusion: RIME-Fuse Mathematics

RIME-Fuse treats ultrasound as a reliability-conditioned evidence signal, avoiding naive concatenation.

**1. Reliability Estimation:**
Optical quality ($q_o$) and Ultrasonic quality ($q_u$) are estimated via MLPs. The reliability is bounded using an FPGA-friendly hard sigmoid:
`hard_sigmoid(x) = clamp((x + 3) / 6, 0, 1)`

To handle asynchronous sensing, ultrasonic reliability decays over time if a pulse is delayed:
`r_u' = r_u * clamp(1 - Δt / τ, 0, 1)`

**2. Echo-to-Grid Projection:**
Ultrasonic features ($U_t$) are projected into the optical space without dense 2D expansion. 
`E_l[h, w, c] = a_l[h] * b_l[c]`
*Where `a_l` is a 1D spatial projection and `b_l` is a channel projection. `E_l` is broadcast along the width, as ultrasound provides range (depth) evidence.*

**3. Gated Fusion:**
Final features ($F_l$) are modulated based on agreement:
`F_l = P_l * (1 + λ * r_u' * C_u,l) + E_l * (r_u' * A_l) - P_l * μ * (1 - C_u,l) * r_u'`
*If sensors agree, detections are boosted. If optical suggests a particle but ultrasound disagrees (with high reliability), the false positive is suppressed.*

---

## 🛠️ FPGA Hardware Co-Design (PYNQ-Z2)

The system is partitioned between the ARM Cortex-A9 Processing System (PS) and the Programmable Logic (PL).

### PS-PL Partitioning
*   **PS (ARM):** Sensor control, DMA orchestration, NMS, tracking, and network I/O.
*   **PL (FPGA):** Streaming camera acquisition, optical/ultrasonic preprocessing, INT8 CNN inference, RIME-Fuse, and temporal logits buffering.

### Hardware Optimizations
*   **AXI4-Stream Dataflow:** Video and ADC data bypass the DDR where possible, streaming directly into PL line buffers.
*   **Dual Compute Engines:** 
    *   *Pointwise Engine:* 64–128 parallel INT8 MACs per cycle mapped to DSP48E1 slices for $1\times1$ convolutions.
    *   *Depthwise Engine:* LUT-multiplier mapping with line-buffered $3\times3$ windows to save DSPs.
*   **Memory Bandwidth:** Model avoids full-frame feature buffering. BRAM (~85–115 used out of 140) stores only 3–5 rows per layer and ping-pong tile buffers. Total DDR traffic is restricted to ~150–350 MB/s.

### Estimated Resource Utilization
| Resource | Estimate | Capacity | Status |
| :--- | :--- | :--- | :--- |
| **DSP** | 120 – 150 | 220 | 🟢 Safe |
| **BRAM** | 85 – 115 | 140 | 🟡 Tight (Tiling Req.) |
| **LUT** | 32K – 42K | 53K | 🟢 Safe |
| **Target Clock** | 100 – 125 MHz | - | 🟢 Stable |

---

## 🏋️ Training & Quantization Pipeline

### 1. Quantization-Aware Training (QAT)
Post-Training Quantization (PTQ) causes a 3–8 mAP drop under high turbidity. Therefore, QAT is mandatory.
*   **Weights:** INT8 symmetric, per-channel.
*   **Activations:** INT8 asymmetric or symmetric, per-tensor.
*   **Accumulators:** INT32.
*   **Final/Reliability Logits:** INT16.

### 2. Loss Function
The total loss is a combination of Detection, Reliability, Cross-Modal Consistency, and Temporal smoothness:
`L_total = L_det + (λ_cls * L_focal) + (λ_box * L_CIoU) + (λ_rel * L_reliability) + (λ_cons * L_consistency) + (λ_temp * L_temporal)`

### 3. Wastewater Augmentations
Models are trained with domain-specific synthetic augmentations:
*   **Optical:** Synthetic Turbidity model ($I_{turbid} = I_{clean} * e^{-\beta d} + A * (1 - e^{-\beta d}) + noise$), forward scattering blur, bubble overlays, and motion blur along flow direction.
*   **Ultrasonic:** Additive echo noise, pulse dropout, time-of-flight jitter, and attenuation slope perturbation.
*   **Multimodal:** Randomly corrupting one modality to prevent the fusion gate from over-relying on a single sensor.

---

## 📂 Repository Structure

```text
EcoFuse-uNet/
├── model/                 # Algorithm design & training (PyTorch)
│   ├── architectures/     # Edge-Ghost Encoder & RIME-Fuse modules
│   ├── quantization/      # QAT scripts, FakeQuant insertion
│   └── augmentations/     # Synthetic turbidity and acoustic noise models
├── firmware/              # Vitis HLS C++ source and Vivado constraints
│   ├── mac_engine/        # INT8 matrix multiply accumulator IP
│   ├── fusion_gate/       # Hard-sigmoid & low-rank projection IP
│   └── tcl/               # Block design generation scripts
├── hardware/              # Pre-compiled bitstreams & overlays
│   └── pynq_z2/           # .bit and .hwh files for PYNQ deployment
├── notebooks/             # PYNQ Jupyter notebooks for live edge inference
├── data/                  # Sample synchronized optical/ultrasonic data arrays
└── requirements.txt       # Python dependencies for host machine
