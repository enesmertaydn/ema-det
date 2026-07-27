# -*- coding: utf-8 -*-
"""
losses.py — Loss terms for EMA-Det (Eq. 3 of the manuscript).

    L_total = lambda_cls * L_BCE(p_hat, p) + lambda_bbox * L_CIoU(b_hat, b)

Gate 1 / Gate 2 are supervised with their own BCE terms against the same
binary label as the main classification head (the gate's job is to reject
confident negatives early, not to solve a different task), weighted by
lambda_gate1 / lambda_gate2. These weights live in train.py's Config, not
here — this module only implements the two loss primitives themselves.
"""

import math

import torch


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete-IoU loss (Zheng et al., AAAI 2020), used for L_CIoU in Eq. 3.

    Args:
        pred_boxes:   (N, 4) predicted (cx, cy, w, h), normalized to [0, 1].
        target_boxes: (N, 4) ground-truth (cx, cy, w, h), normalized to [0, 1].
    Returns:
        scalar loss = mean(1 - CIoU) over the N positive samples.
    """
    def to_xyxy(b: torch.Tensor):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return x1, y1, x2, y2

    px1, py1, px2, py2 = to_xyxy(pred_boxes)
    tx1, ty1, tx2, ty2 = to_xyxy(target_boxes)

    # Intersection / union -> IoU
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    p_area = (px2 - px1) * (py2 - py1)
    t_area = (tx2 - tx1) * (ty2 - ty1)
    union  = p_area + t_area - inter + eps
    iou    = inter / union

    # Smallest enclosing box diagonal
    cx1 = torch.min(px1, tx1); cx2 = torch.max(px2, tx2)
    cy1 = torch.min(py1, ty1); cy2 = torch.max(py2, ty2)
    c_diag_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    # Center distance term
    pcx, pcy = pred_boxes[:, 0], pred_boxes[:, 1]
    tcx, tcy = target_boxes[:, 0], target_boxes[:, 1]
    rho_sq = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # Aspect-ratio consistency term
    pw, ph = pred_boxes[:, 2], pred_boxes[:, 3]
    tw, th = target_boxes[:, 2], target_boxes[:, 3]
    v = (4 / (math.pi ** 2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho_sq / c_diag_sq - alpha * v
    return (1 - ciou).mean()


def box_iou_elementwise(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Plain (non-complete) IoU per row, used for the mIoU diagnostic metric, not for training."""
    px1 = pred[:, 0] - pred[:, 2] / 2;   px2 = pred[:, 0] + pred[:, 2] / 2
    py1 = pred[:, 1] - pred[:, 3] / 2;   py2 = pred[:, 1] + pred[:, 3] / 2
    tx1 = target[:, 0] - target[:, 2] / 2; tx2 = target[:, 0] + target[:, 2] / 2
    ty1 = target[:, 1] - target[:, 3] / 2; ty2 = target[:, 1] + target[:, 3] / 2

    ix1 = torch.max(px1, tx1); ix2 = torch.min(px2, tx2)
    iy1 = torch.max(py1, ty1); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    p_area = (px2 - px1) * (py2 - py1)
    t_area = (tx2 - tx1) * (ty2 - ty1)
    union  = p_area + t_area - inter + eps
    return inter / union
