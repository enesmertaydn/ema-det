# -*- coding: utf-8 -*-
"""
ema_det.py — EMA-Det: Hardware-Aware Edge Verification Network for Two-Stage UAV Detection

Reference architecture (Fig. 1 of the manuscript):

    Input 1x128x128 (grayscale / thermal)
      -> Block1 (Laplacian+Sobel init, 1->16)         -> 16x64x64
      -> Block2 (16->32)                               -> 32x32x32  --> Gate 1
      -> Block3 (32->64)                                -> 64x16x16
      -> Block4 (64->128)                                -> 128x8x8  --> Gate 2
      -> Block5 (multi-scale, dilation 1 & 2, fusion)     -> 128x4x4
      -> Block6 (128->256)                                -> 256x4x4
      -> Verification Head: GAP -> Dropout(0.4) -> FC(1) cls, FC(4) bbox

Design notes tied to the manuscript:
  * Block1 weights are initialized from a deterministic Laplacian/Sobel filter
    bank (Eq. 2) but are left UNFROZEN — they are ordinary nn.Parameter tensors
    and receive gradient updates like any other layer. See `LaplacianSobelInit`.
  * Gate 1 / Gate 2 implement the asymptotic early-exit rule of Eq. 4:
        result = background (halt)   if  p_hat <= 1 - tau
                 continue to C_{i+1}  if  p_hat >  1 - tau
    This is only meaningful for single-sample (batch=1) edge inference, which
    is how the FPGA/Jetson verifier actually consumes candidate regions
    (one RoI at a time). `forward(x, tau=...)` therefore asserts batch=1.
  * CBAM is deliberately NOT part of the default forward path. Section III-C /
    Table II of the manuscript shows CBAM's global-pooling attention degrades
    small-target features at 128x128 (mAP@.5:.95 drops from 62.4% to 58.8%
    when CBAM is present), so the final EMA-Det omits it. It is kept here as
    an optional module (`use_attention=True`) purely so the ablation study
    (Table II, row "A) CBAM + EE") stays reproducible from this same file.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn as nn


# ==============================================================================
# Laplacian / Sobel physical kernel bank (Eq. 2)
# ==============================================================================
def make_edge_kernels(out_channels: int) -> torch.Tensor:
    """
    Build the deterministic prior bank K_c used to initialize Block 1.

    Channels 0..11 are exact 3x3 Laplacian / Sobel operators (and their signed
    combinations); any remaining channels (if out_channels > 12) are drawn
    from N(0, 0.1) as in Eq. 2's random-prior term R_c, so the layer keeps
    its capacity to generalize beyond the fixed operators.
    """
    lap  = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    lap2 = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    sx   = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    sy   = sx.T
    base = [lap, lap2, -lap, -lap2,
            sx, -sx, sy, -sy,
            sx + sy, sx - sy, -sx + sy, -sx - sy]

    kernels = []
    for i in range(out_channels):
        k = base[i].unsqueeze(0).unsqueeze(0) if i < len(base) \
            else torch.randn(1, 1, 3, 3) * 0.1
        kernels.append(k)
    return torch.cat(kernels, dim=0)


class LaplacianSobelInit(nn.Module):
    """Block 1: 1->out_ch conv, physically initialized, trainable (unfrozen)."""

    def __init__(self, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(make_edge_kernels(out_ch))
        # No requires_grad_(False) call anywhere: weights update during training.
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ==============================================================================
# CBAM — kept only for the ablation study (see module docstring)
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(ch // r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, mid, bias=False), nn.ReLU(),
            nn.Linear(mid, ch, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * w.view(x.size(0), -1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], dim=1)
        return x * self.sig(self.conv(desc))


class CBAM(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class Identity2(nn.Module):
    """No-op stand-in used when use_attention=False, so block code stays uniform."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ==============================================================================
# Early-exit gate (Gate 1 / Gate 2)
# ==============================================================================
class EarlyExitGate(nn.Module):
    """1x1 conv -> BN -> ReLU -> GAP -> FC(1), as drawn in Fig. 1."""

    def __init__(self, in_ch: int):
        super().__init__()
        mid = max(in_ch // 2, 8)
        self.conv = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(mid)
        self.act  = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(mid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        return self.head(self.pool(x).view(x.size(0), -1)).squeeze(1)


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 conv + BN + LeakyReLU layers, matching Blocks 2/3/4/6."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1),
        nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1),
    )


# ==============================================================================
# Structured output
# ==============================================================================
class EMADetOutput(NamedTuple):
    """
    cls_logit  : (B,)   raw logit — BCEWithLogitsLoss target during training;
                 already the gate's own logit if exit_stage != 'main'.
    bbox       : (B,4) or None — sigmoid-normalized (cx,cy,w,h) in [0,1];
                 None when the sample halted at Gate 1/2 (no bbox refinement).
    gate1_logit: (B,) or None — always populated once Block 2 has run.
    gate2_logit: (B,) or None — populated only if execution reached Block 4.
    exit_stage : 'gate1' | 'gate2' | 'main'
    """
    cls_logit: torch.Tensor
    bbox: Optional[torch.Tensor]
    gate1_logit: Optional[torch.Tensor]
    gate2_logit: Optional[torch.Tensor]
    exit_stage: str


# ==============================================================================
# EMA-Det
# ==============================================================================
class EMADet(nn.Module):
    """
    Hardware-aware RoI verification network.

    Args:
        use_attention: reproduces the rejected CBAM variant (Table II, row A)
                       for ablation purposes. Default False = final EMA-Det.
        dropout:       verification-head dropout rate (paper: 0.4).
    """

    def __init__(self, use_attention: bool = False, dropout: float = 0.4):
        super().__init__()
        self.use_attention = use_attention

        # Block 1 — Laplacian/Sobel init, 1 -> 16, 128 -> 64
        self.block1 = LaplacianSobelInit(16)
        self.pool1  = nn.MaxPool2d(2)

        # Block 2 — 16 -> 32, 64 -> 32
        self.block2 = conv_block(16, 32)
        self.pool2  = nn.MaxPool2d(2)
        self.gate1  = EarlyExitGate(32)

        # Block 3 — 32 -> 64, 32 -> 16
        self.block3 = conv_block(32, 64)
        self.attn3  = CBAM(64) if use_attention else Identity2()
        self.pool3  = nn.MaxPool2d(2)

        # Block 4 — 64 -> 128, 16 -> 8
        self.block4 = conv_block(64, 128)
        self.attn4  = CBAM(128) if use_attention else Identity2()
        self.pool4  = nn.MaxPool2d(2)
        self.gate2  = EarlyExitGate(128)

        # Block 5 — multi-scale (dilation 1 & 2) fusion, 8 -> 4
        self.block5a = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
        )
        self.block5b = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
        )
        self.attn5 = CBAM(128) if use_attention else Identity2()
        self.pool5 = nn.MaxPool2d(2)

        # Block 6 — 128 -> 256, spatial stays 4x4
        self.block6 = conv_block(128, 256)
        self.attn6  = CBAM(256) if use_attention else Identity2()

        # Verification head
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.cls_head  = nn.Linear(256, 1)
        self.bbox_head = nn.Linear(256, 4)

    # -- shared feature path, reused by forward() and by the FPS/FLOPs tooling --
    def _stem_to_gate1(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool1(self.block1(x))
        f = self.pool2(self.block2(f))
        return f

    def _gate1_to_gate2(self, f: torch.Tensor) -> torch.Tensor:
        f = self.pool3(self.attn3(self.block3(f)))
        f = self.pool4(self.attn4(self.block4(f)))
        return f

    def _gate2_to_head(self, f: torch.Tensor) -> torch.Tensor:
        f = self.attn5(self.fusion(torch.cat([self.block5a(f), self.block5b(f)], dim=1)))
        f = self.pool5(f)
        f = self.attn6(self.block6(f))
        return self.dropout(self.gap(f).view(f.size(0), -1))

    def forward(self, x: torch.Tensor, tau: Optional[float] = None) -> EMADetOutput:
        """
        tau=None  : training mode. Runs the full network unconditionally and
                    returns every head so BCE(cls) + BCE(gate1) + BCE(gate2) +
                    CIoU(bbox) (Eq. 3) can all receive gradients.
        tau=float : inference mode, single sample only (asserts x.size(0)==1),
                    applies the cascading Eq. 4 rule with the literal
                    `if prob < threshold` gating structure.
        """
        f = self._stem_to_gate1(x)
        gate1_logit = self.gate1(f)

        if tau is not None:
            assert x.size(0) == 1, "tau-gated inference is single-sample (batch=1) only"
            exit_threshold = 1.0 - tau
            prob1 = torch.sigmoid(gate1_logit)
            if prob1 < exit_threshold:
                return EMADetOutput(gate1_logit, None, gate1_logit, None, "gate1")

        f = self._gate1_to_gate2(f)
        gate2_logit = self.gate2(f)

        if tau is not None:
            prob2 = torch.sigmoid(gate2_logit)
            if prob2 < exit_threshold:
                return EMADetOutput(gate2_logit, None, gate1_logit, gate2_logit, "gate2")

        feat = self._gate2_to_head(f)
        cls_logit = self.cls_head(feat).squeeze(1)
        bbox      = torch.sigmoid(self.bbox_head(feat))
        return EMADetOutput(cls_logit, bbox, gate1_logit, gate2_logit, "main")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ==============================================================================
# Self-test
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    model = EMADet(use_attention=False)
    n_params = count_parameters(model)
    print(f"EMA-Det parameters: {n_params:,} ({n_params / 1e6:.3f} M)")
    print(f"  (manuscript Table IV reports 1.46 M for EMA-Det)")

    # --- Block 1 init sanity check ---
    # Channels 0..11 are the deterministic Laplacian/Sobel bank (Eq. 2, K_c
    # term) and must match exactly. Channels 12..15 (R_c term) are drawn from
    # N(0, 0.1) independently each call, so only their statistics are checked.
    assert model.block1.conv.weight.requires_grad, "Block1 must stay unfrozen"
    deterministic_bank = make_edge_kernels(12)
    assert torch.allclose(model.block1.conv.weight.detach()[:12], deterministic_bank), \
        "Block1 channels 0..11 must start as the Laplacian/Sobel prior bank"
    random_tail = model.block1.conv.weight.detach()[12:]
    assert random_tail.std() < 0.5, "Block1 random-prior channels should be small-scale"
    print("Block1 Laplacian/Sobel init verified, weights unfrozen (requires_grad=True).")

    # --- Training-mode forward/backward shape check ---
    x = torch.rand(8, 1, 128, 128)
    out = model(x)
    assert out.cls_logit.shape == (8,)
    assert out.bbox.shape == (8, 4)
    assert out.gate1_logit.shape == (8,)
    assert out.gate2_logit.shape == (8,)
    assert out.exit_stage == "main"
    loss = out.cls_logit.sum() + out.bbox.sum() + out.gate1_logit.sum() + out.gate2_logit.sum()
    loss.backward()
    assert model.block1.conv.weight.grad is not None, "gradients must reach Block1"
    print(f"Training-mode forward/backward OK. cls={tuple(out.cls_logit.shape)} "
          f"bbox={tuple(out.bbox.shape)}")

    # --- Inference-mode early exit check (batch=1) ---
    model.eval()
    with torch.no_grad():
        x1 = torch.rand(1, 1, 128, 128)
        for tau in (0.85, 0.90, 0.92, 0.95):
            out = model(x1, tau=tau)
            print(f"  tau={tau:.2f} -> exit_stage={out.exit_stage:<5} "
                  f"bbox={'yes' if out.bbox is not None else 'no'}")

    print("\nAll self-tests passed.")
