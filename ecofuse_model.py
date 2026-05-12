"""
EcoFuse-µNet: Reliability-Aware Optical-Ultrasonic Microplastic Detection Network
Complete spec-faithful PyTorch implementation for Kaggle training.

Target: ~1.05M-1.25M params, ~160-190M MACs/frame, INT8-friendly ops only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. FPGA-Friendly Activation
# =============================================================================
def hard_sigmoid(x):
    """FPGA-friendly hard sigmoid: clamp((x+3)/6, 0, 1)"""
    return torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


# =============================================================================
# 2. Ghost-DS Block (Core building block)
# =============================================================================
class GhostDSBlock(nn.Module):
    """
    Ghost Depthwise-Separable Block.
    Flow: PW1x1 expand -> DW3x3 ghost -> concat -> DW3x3 spatial (stride) -> PW1x1 project -> residual
    Uses ReLU6, BN-foldable, line-buffer friendly DW convs.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        half = max(out_c // 2, 8)
        exp_c = half * 2

        # 1x1 pointwise expansion
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_c, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU6(inplace=True)
        )
        # DW 3x3 ghost feature generation
        self.dw_ghost = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU6(inplace=True)
        )
        # DW 3x3 spatial filtering (carries stride)
        self.dw_spatial = nn.Sequential(
            nn.Conv2d(exp_c, exp_c, 3, stride=stride, padding=1, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            nn.ReLU6(inplace=True)
        )
        # 1x1 pointwise projection
        self.pw_project = nn.Sequential(
            nn.Conv2d(exp_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.use_res = (in_c == out_c) and (stride == 1)

    def forward(self, x):
        e = self.pw_expand(x)
        g = self.dw_ghost(e)
        cat = torch.cat([e, g], dim=1)
        spatial = self.dw_spatial(cat)
        out = self.pw_project(spatial)
        if self.use_res:
            out = out + x
        return F.relu6(out)


def _make_stage(in_c, out_c, num_blocks, stride_first=1):
    """Build a stage of Ghost-DS blocks. First block may have stride."""
    layers = [GhostDSBlock(in_c, out_c, stride=stride_first)]
    for _ in range(num_blocks - 1):
        layers.append(GhostDSBlock(out_c, out_c, stride=1))
    return nn.Sequential(*layers)


# =============================================================================
# 3. Edge-Ghost Optical Encoder
# =============================================================================
class OpticalEncoder(nn.Module):
    """
    Input: [B, 2, 160, 160] (luminance + Sobel edge)
    Outputs: O1(80x80x16), O2(40x40x32), O3(20x20x64), O4(10x10x96), O5(5x5x128)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )  # -> 80x80x16
        self.o1 = _make_stage(16, 16, 2)        # 80x80x16
        self.o2 = _make_stage(16, 32, 3, stride_first=2)   # 40x40x32
        self.o3 = _make_stage(32, 64, 4, stride_first=2)   # 20x20x64
        self.o4 = _make_stage(64, 96, 3, stride_first=2)   # 10x10x96
        self.o5 = _make_stage(96, 128, 2, stride_first=2)  # 5x5x128

    def forward(self, x):
        x = self.stem(x)
        o1 = self.o1(x)
        o2 = self.o2(o1)
        o3 = self.o3(o2)
        o4 = self.o4(o3)
        o5 = self.o5(o4)
        return o1, o2, o3, o4, o5


# =============================================================================
# 4. DS-TCN Ultrasonic Encoder
# =============================================================================
class DSTCNBlock(nn.Module):
    """1D Depthwise-Separable Temporal Convolution block."""
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        pad = dilation
        self.dw = nn.Conv1d(in_c, in_c, 3, stride=stride, padding=pad,
                            dilation=dilation, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)
        self.pw = nn.Conv1d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.use_res = (in_c == out_c) and (stride == 1) and (dilation == 1)

    def forward(self, x):
        out = F.relu6(self.bn1(self.dw(x)))
        out = F.relu6(self.bn2(self.pw(out)))
        if self.use_res:
            out = out + x
        return out


class SpectralEnergyBank(nn.Module):
    """Fixed Goertzel/FIR spectral energy bank (non-learnable filters + learnable projection)."""
    def __init__(self, in_c=64, num_bands=8, out_c=32):
        super().__init__()
        self.num_bands = num_bands
        # Fixed bandpass filters approximated as grouped 1D convs
        self.filters = nn.Conv1d(in_c, in_c * num_bands, kernel_size=5,
                                 padding=2, groups=in_c, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(2)  # energy per band
        self.proj = nn.Sequential(
            nn.Conv1d(in_c * num_bands, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        # x: [B, 64, 32]
        energy = self.filters(x)        # [B, 64*8, 32]
        energy = self.pool(energy)       # [B, 512, 2]
        energy = self.proj(energy)       # [B, 32, 2]
        return energy


class UltrasonicEncoder(nn.Module):
    """
    Input: [B, 3, 256] (envelope, attenuation slope, temporal derivative)
    Outputs: U_t [B, 64, 32], U_g [B, 96]
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU6(inplace=True)
        )  # -> 256x16
        self.u1 = DSTCNBlock(16, 32, stride=2)    # 128x32
        self.u2 = DSTCNBlock(32, 48, stride=2)    # 64x48
        self.u3 = DSTCNBlock(48, 64, dilation=2)  # 64x64
        self.temporal_pool = nn.AdaptiveAvgPool1d(32)  # -> 32x64

        self.spectral = SpectralEnergyBank(64, num_bands=4, out_c=32)  # -> 32x2 -> pooled

        # Global projection: pool temporal + spectral -> 96
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(64 + 32, 96)

    def forward(self, x):
        x = self.stem(x)
        x = self.u1(x)
        x = self.u2(x)
        x = self.u3(x)
        u_t = self.temporal_pool(x)  # [B, 64, 32]

        spec = self.spectral(u_t)     # [B, 32, 2]
        spec_g = self.global_pool(spec).squeeze(-1)  # [B, 32]
        temp_g = self.global_pool(u_t).squeeze(-1)    # [B, 64]
        u_g = self.global_proj(torch.cat([temp_g, spec_g], dim=1))  # [B, 96]

        return u_t, u_g


# =============================================================================
# 5. Compact 2-Scale Feature Pyramid
# =============================================================================
class CompactFPN(nn.Module):
    """
    Inputs: O1(80x80x16), O2(40x40x32), O3(20x20x64), O5(5x5x128)
    Outputs: P1(80x80x32), P2(40x40x48), Pg(1x1x128)
    Uses add-not-concat to save memory.
    """
    def __init__(self):
        super().__init__()
        # Lateral projections for P2
        self.lat_o2 = nn.Conv2d(32, 48, 1, bias=False)
        self.lat_o3 = nn.Conv2d(64, 48, 1, bias=False)
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True)
        )
        # Lateral projections for P1
        self.lat_o1 = nn.Conv2d(16, 32, 1, bias=False)
        self.lat_p2 = nn.Conv2d(48, 32, 1, bias=False)
        self.smooth_p1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        # Pg: global average pool of O5
        self.pg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, o1, o2, o3, o5):
        # P2 = O2 + upsample(O3)  -> 40x40x48
        p2 = self.lat_o2(o2) + F.interpolate(self.lat_o3(o3), size=o2.shape[2:], mode='nearest')
        p2 = self.smooth_p2(p2)

        # P1 = O1 + upsample(P2)  -> 80x80x32
        p1 = self.lat_o1(o1) + F.interpolate(self.lat_p2(p2), size=o1.shape[2:], mode='nearest')
        p1 = self.smooth_p1(p1)

        # Pg = global pool of O5  -> 1x1x128
        pg = self.pg_pool(o5).squeeze(-1).squeeze(-1)  # [B, 128]

        return p1, p2, pg


# =============================================================================
# 6. RIME-Fuse Module
# =============================================================================
class RIMEFuse(nn.Module):
    """
    Reliability-aware Intermodal Microplastic Evidence Fusion.
    Per-scale module: fuses optical feature map P_l with ultrasonic evidence.

    Uses low-rank echo-to-grid projection (no dense 2D expansion).
    All ops: 1x1 conv, DW 3x3, elementwise, hard_sigmoid. No softmax/attention.
    """
    def __init__(self, opt_channels, height, width, ult_temporal_c=64, ult_global_c=96):
        super().__init__()
        self.C = opt_channels
        self.H = height
        self.W = width

        # --- Optical reliability MLP ---
        self.opt_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_o = nn.Sequential(
            nn.Linear(opt_channels, 16), nn.ReLU6(), nn.Linear(16, 1)
        )
        # --- Ultrasonic reliability MLP ---
        self.mlp_u = nn.Sequential(
            nn.Linear(ult_global_c, 16), nn.ReLU6(), nn.Linear(16, 1)
        )

        # --- Echo-to-grid low-rank projection ---
        # a_l: Conv1D maps U_t temporal to spatial height dimension
        self.proj_a = nn.Conv1d(ult_temporal_c, 1, kernel_size=1, bias=False)
        # b_l: Linear maps U_g to channel dimension
        self.proj_b = nn.Linear(ult_global_c, opt_channels, bias=False)

        # --- Gating convolutions ---
        self.dw_opt_gate = nn.Conv2d(opt_channels, opt_channels, 3, padding=1,
                                     groups=opt_channels, bias=False)
        self.pw_ult_gate = nn.Conv2d(opt_channels, opt_channels, 1, bias=False)

        # --- Learnable fusion scalars ---
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(1))
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.tensor(0.3))

    def forward(self, P_l, U_t, U_g, pg_feat=None, delta_t=None):
        B = P_l.size(0)

        # 1. Optical reliability r_o
        opt_pooled = self.opt_pool(P_l).view(B, -1)
        r_o = hard_sigmoid(self.mlp_o(opt_pooled))  # [B, 1]

        # 2. Ultrasonic reliability r_u (with optional async decay)
        r_u = hard_sigmoid(self.mlp_u(U_g))  # [B, 1]
        if delta_t is not None:
            decay = torch.clamp(1.0 - delta_t / 10.0, 0.0, 1.0)
            r_u = r_u * decay.view(B, 1)

        # 3. Echo-to-grid low-rank projection
        # a_l: temporal -> spatial height
        a_l = self.proj_a(U_t)  # [B, 1, T]
        a_l = F.interpolate(a_l, size=self.H, mode='linear', align_corners=False)  # [B, 1, H]
        a_l = a_l.view(B, 1, self.H, 1)  # broadcast over W and C

        # b_l: global -> channel
        b_l = self.proj_b(U_g).view(B, self.C, 1, 1)  # broadcast over H and W

        # E_l[h,w,c] = a_l[h] * b_l[c], broadcast along width
        E_l = a_l * b_l  # [B, C, H, 1]
        E_l = E_l.expand(-1, -1, -1, self.W)  # [B, C, H, W]

        # 4. Reliability-gated fusion
        C_o = hard_sigmoid(self.dw_opt_gate(P_l))   # optical confidence
        C_u = hard_sigmoid(self.pw_ult_gate(E_l))    # ultrasonic confidence

        # Attention map
        r_o_b = r_o.view(B, 1, 1, 1)
        r_u_b = r_u.view(B, 1, 1, 1)

        A_l = hard_sigmoid(
            self.alpha * C_o + self.beta * C_u +
            self.gamma * r_o_b + self.delta * r_u_b
        )

        # Fused feature
        term1 = P_l * (1.0 + self.lam * r_u_b * C_u)   # boost if ultrasound agrees
        term2 = E_l * (r_u_b * A_l)                      # inject ultrasonic evidence
        term3 = P_l * self.mu * (1.0 - C_u) * r_u_b     # suppress if ultrasound disagrees

        F_l = term1 + term2 - term3

        return F_l, r_o.squeeze(-1), r_u.squeeze(-1)


# =============================================================================
# 7. Temporal Stabilization Gate
# =============================================================================
class TemporalStabilizationGate(nn.Module):
    """
    Operates on logits, not features. ~70-90KB storage.
    S_t = blend(L_t, S_{t-1}) based on reliability and velocity.
    """
    def __init__(self, base_momentum=0.7):
        super().__init__()
        self.base_momentum = base_momentum
        self.gate_fc = nn.Linear(2, 1)  # takes [r_combined, velocity_est]

    def forward(self, current_logits, prev_logits, r_o, r_u, velocity=None):
        if prev_logits is None:
            return current_logits

        B = current_logits.size(0)
        r_combined = (r_o + r_u) / 2.0  # [B]

        if velocity is None:
            velocity = torch.zeros(B, device=current_logits.device)

        gate_input = torch.stack([r_combined, velocity], dim=1)  # [B, 2]
        momentum = torch.sigmoid(self.gate_fc(gate_input))  # [B, 1]
        momentum = momentum.view(B, 1, 1, 1)

        stabilized = momentum * prev_logits + (1.0 - momentum) * current_logits
        return stabilized


# =============================================================================
# 8. Anchor-Free Detection Head
# =============================================================================
class DetectionHead(nn.Module):
    """
    DW3x3 + PW1x1 per scale.
    Output per cell: [obj, cx, cy, w, h, cls_fiber, cls_fragment, cls_film] = 8 channels
    """
    def __init__(self, in_c, num_classes=3):
        super().__init__()
        out_c = 5 + num_classes  # obj + 4 box + num_classes
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True)
        )
        self.pw = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


# =============================================================================
# 9. Full EcoFuse-µNet
# =============================================================================
class EcoFuseUNet(nn.Module):
    """
    Complete EcoFuse-µNet with all components.
    Input: optical [B,2,160,160], ultrasonic [B,3,256]
    Output: det_p1 [B,8,80,80], det_p2 [B,8,40,40], r_o, r_u
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.optical_encoder = OpticalEncoder()
        self.ultrasonic_encoder = UltrasonicEncoder()
        self.fpn = CompactFPN()

        # RIME-Fuse for each scale
        self.rime_p1 = RIMEFuse(opt_channels=32, height=80, width=80)
        self.rime_p2 = RIMEFuse(opt_channels=48, height=40, width=40)

        # Detection heads
        self.head_p1 = DetectionHead(32, num_classes)  # tiny particles, 80x80
        self.head_p2 = DetectionHead(48, num_classes)  # medium particles, 40x40

        # Temporal stabilization
        self.temporal_gate_p1 = TemporalStabilizationGate()
        self.temporal_gate_p2 = TemporalStabilizationGate()

    def forward(self, optical, ultrasonic, delta_t=None,
                prev_logits_p1=None, prev_logits_p2=None):
        # 1. Encode
        o1, o2, o3, o4, o5 = self.optical_encoder(optical)
        u_t, u_g = self.ultrasonic_encoder(ultrasonic)

        # 2. Feature Pyramid
        p1, p2, pg = self.fpn(o1, o2, o3, o5)

        # 3. RIME-Fuse
        f1, r_o1, r_u1 = self.rime_p1(p1, u_t, u_g, pg, delta_t)
        f2, r_o2, r_u2 = self.rime_p2(p2, u_t, u_g, pg, delta_t)

        # Average reliability across scales
        r_o = (r_o1 + r_o2) / 2.0
        r_u = (r_u1 + r_u2) / 2.0

        # 4. Detection heads
        det_p1 = self.head_p1(f1)  # [B, 8, 80, 80]
        det_p2 = self.head_p2(f2)  # [B, 8, 40, 40]

        # 5. Temporal stabilization (optional, for inference)
        if prev_logits_p1 is not None:
            det_p1 = self.temporal_gate_p1(det_p1, prev_logits_p1, r_o, r_u)
        if prev_logits_p2 is not None:
            det_p2 = self.temporal_gate_p2(det_p2, prev_logits_p2, r_o, r_u)

        return det_p1, det_p2, r_o, r_u


# =============================================================================
# 10. Model Info Utility
# =============================================================================
def print_model_info(model):
    """Print parameter counts per component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    components = {
        'Optical Encoder': model.optical_encoder,
        'Ultrasonic Encoder': model.ultrasonic_encoder,
        'Feature Pyramid': model.fpn,
        'RIME-Fuse P1': model.rime_p1,
        'RIME-Fuse P2': model.rime_p2,
        'Det Head P1': model.head_p1,
        'Det Head P2': model.head_p2,
        'Temporal Gate P1': model.temporal_gate_p1,
        'Temporal Gate P2': model.temporal_gate_p2,
    }

    print("=" * 60)
    print("EcoFuse-µNet Architecture Summary")
    print("=" * 60)
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:.<40s} {params:>8,d}")
    print("-" * 60)
    print(f"  {'TOTAL':.<40s} {total:>8,d}")
    print(f"  {'Trainable':.<40s} {trainable:>8,d}")
    print(f"  {'INT8 Weight Size':.<40s} {total/1e6:>7.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    model = EcoFuseUNet()
    print_model_info(model)

    # Shape verification
    opt = torch.randn(1, 2, 160, 160)
    ult = torch.randn(1, 3, 256)
    det_p1, det_p2, r_o, r_u = model(opt, ult)
    print(f"\nShape verification:")
    print(f"  det_p1: {det_p1.shape}  (expect [1, 8, 80, 80])")
    print(f"  det_p2: {det_p2.shape}  (expect [1, 8, 40, 40])")
    print(f"  r_o: {r_o.shape}, r_u: {r_u.shape}")
