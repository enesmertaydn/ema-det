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


def center_head_loss(hm_logit: torch.Tensor, reg: torch.Tensor, gt: torch.Tensor, pos: torch.Tensor):
    """
    Loss of the centre-based localization head (src/ema_det.py CenterHead) for one box per patch.
      hm_logit (B,1,H,W), reg (B,4,H,W), gt (B,4) normalized (cx, cy, w, h), pos (B,) bool.
    Heat map: CenterNet focal loss with a Gaussian around the centre cell (all-zero target for background patches).
    Regression: on the 3x3 cells around the centre cell, CIoU of the decoded box plus L1 of offset and size.
    Returns (heat-map loss, box loss).
    """
    B, _, H, W = hm_logit.shape
    dev = hm_logit.device
    posf = pos.float()
    xs = torch.arange(W, device=dev).view(1, 1, W).float()
    ys = torch.arange(H, device=dev).view(1, H, 1).float()
    gx, gy = gt[:, 0] * W, gt[:, 1] * H
    cxc = gx.floor().clamp(0, W - 1)
    cyc = gy.floor().clamp(0, H - 1)
    sigma = (torch.minimum(gt[:, 2], gt[:, 3]) * W / 3).clamp(min=0.8).view(B, 1, 1)
    target = torch.exp(-((xs - cxc.view(B, 1, 1)) ** 2 + (ys - cyc.view(B, 1, 1)) ** 2) / (2 * sigma ** 2)) * posf.view(B, 1, 1)
    peak_cell = (target == 1.0) & pos.view(B, 1, 1)
    p = torch.sigmoid(hm_logit[:, 0]).clamp(1e-4, 1 - 1e-4)
    pos_loss = -((1 - p) ** 2) * torch.log(p) * peak_cell
    neg_loss = -((1 - target) ** 4) * (p ** 2) * torch.log(1 - p) * (~peak_cell)
    n_pos = pos.sum().clamp(min=1)
    l_hm = (pos_loss.sum() + neg_loss.sum()) / n_pos

    l_box = hm_logit.new_zeros(())
    if pos.any():
        boxes, gts, l1 = [], [], []
        ar = torch.arange(B, device=dev)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                iy = (cyc + dy).long()
                ix = (cxc + dx).long()
                valid = pos & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
                if not valid.any():
                    continue
                iy, ix = iy.clamp(0, H - 1), ix.clamp(0, W - 1)
                r = reg[ar, :, iy, ix][valid]
                g = gt[valid]
                cx = (ix[valid].float() + r[:, 0]) / W
                cy = (iy[valid].float() + r[:, 1]) / H
                boxes.append(torch.stack([cx, cy, torch.sigmoid(r[:, 2]), torch.sigmoid(r[:, 3])], dim=1))
                gts.append(g)
                off_t = torch.stack([gx[valid] - ix[valid].float(), gy[valid] - iy[valid].float()], dim=1)
                l1.append((r[:, :2] - off_t).abs().mean(1) + (torch.sigmoid(r[:, 2:]) - g[:, 2:]).abs().mean(1))
        l_box = ciou_loss(torch.cat(boxes), torch.cat(gts)) + torch.cat(l1).mean()
    return l_hm, l_box
