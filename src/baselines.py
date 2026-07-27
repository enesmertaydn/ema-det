# -*- coding: utf-8 -*-
"""
baselines.py — lightweight comparison models for Table III (paper plan:
"YOLOv8n yanına MobileNetV3 gibi 2 hafif model daha ekle").

Both are torchvision backbones adapted the same way EMA-Det and the paper's
YOLOv8n RoI-mode baseline are: trained from scratch (no ImageNet weights,
for a fair from-scratch comparison), first conv patched to 1 input channel
(grayscale, matching this project's images), and a verification head (cls
logit + sigmoid bbox) replacing the original ImageNet classifier.
"""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm


class VerifierOutput(NamedTuple):
    cls_logit: torch.Tensor
    bbox: torch.Tensor


class VerificationHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(in_features, 1)
        self.bbox_head = nn.Linear(in_features, 4)

    def forward(self, feat: torch.Tensor) -> VerifierOutput:
        feat = self.dropout(feat)
        cls_logit = self.cls_head(feat).squeeze(1)
        bbox = torch.sigmoid(self.bbox_head(feat))
        return VerifierOutput(cls_logit, bbox)


class MobileNetV3Verifier(nn.Module):
    """torchvision mobilenet_v3_small, 1-channel input, scratch-trained."""

    def __init__(self):
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights=None)
        backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.head = VerificationHead(backbone.classifier[0].in_features)

    def forward(self, x: torch.Tensor) -> VerifierOutput:
        f = self.features(x)
        f = self.avgpool(f).flatten(1)
        return self.head(f)


class ShuffleNetV2Verifier(nn.Module):
    """torchvision shufflenet_v2_x0_5, 1-channel input, scratch-trained."""

    def __init__(self):
        super().__init__()
        backbone = tvm.shufflenet_v2_x0_5(weights=None)
        backbone.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5
        self.head = VerificationHead(backbone.fc.in_features)

    def forward(self, x: torch.Tensor) -> VerifierOutput:
        f = self.conv1(x)
        f = self.maxpool(f)
        f = self.stage2(f)
        f = self.stage3(f)
        f = self.stage4(f)
        f = self.conv5(f)
        f = f.mean([2, 3])  # global average pool, matches torchvision's own forward
        return self.head(f)


def build_model(name: str) -> nn.Module:
    if name == "mobilenet_v3_small":
        return MobileNetV3Verifier()
    if name == "shufflenet_v2_x0_5":
        return ShuffleNetV2Verifier()
    raise ValueError(f"Unknown baseline model: {name}")


if __name__ == "__main__":
    for name in ("mobilenet_v3_small", "shufflenet_v2_x0_5"):
        model = build_model(name)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.rand(4, 1, 128, 128)
        out = model(x)
        assert out.cls_logit.shape == (4,)
        assert out.bbox.shape == (4, 4)
        print(f"{name}: {n_params:,} params, cls={tuple(out.cls_logit.shape)} bbox={tuple(out.bbox.shape)}")
