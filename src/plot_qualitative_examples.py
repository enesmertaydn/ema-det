# -*- coding: utf-8 -*-
"""
plot_qualitative_examples.py — Qualitative detection-example figure for
the paper's Cross-Sensor Generalization section.

Builds a 2x2 grid, each panel a single 128x128 test-set crop with the
ground-truth box (green) and the model's predicted box (orange, only
drawn when the classification head predicts positive at conf_thresh):
  (a) EO in-domain      — emadet_final_best.pt on the EO test set
  (b) Thermal in-domain — results/thermal/emadet_final_best.pt on the
                          thermal test set
  (c) Zero-shot success — the EO checkpoint (a), no fine-tuning,
                          evaluated on a thermal test image it still
                          gets right
  (d) Zero-shot failure — the same EO checkpoint on a thermal test
                          image it misses -- this is the failure mode
                          quantified numerically in Table 6 (recall
                          falling to 80-86% zero-shot)

This directly illustrates the zero-shot transfer gap already reported
numerically in Section 5.6, rather than showing arbitrary correct/wrong
examples.
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import DroneDetectDataset
from ema_det import EMADet

EO_TEST_DIR = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\shuffled_dataset_detect_clean\test"
THERMAL_TEST_DIR = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\antiuav410_clean\test"
EO_CKPT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\emadet_final_best.pt"
THERMAL_CKPT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\thermal\emadet_final_best.pt"
OUT_DIR = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\paper"
CONF_THRESH = 0.5
IMG_SIZE = 128


def cxcywh_to_xyxy(box, size=IMG_SIZE):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * size
    y1 = (cy - h / 2) * size
    x2 = (cx + w / 2) * size
    y2 = (cy + h / 2) * size
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def find_example(model, dataset, device, want="success", max_scan=400):
    """Scans positive-labeled samples for one matching the requested case.
    want='success': confident positive prediction with IoU>=0.5.
    want='failure': confident-background prediction (score<CONF_THRESH) on a
                     true-positive sample, or IoU<0.3 if none found.
    Returns (img_uint8, gt_xyxy, pred_xyxy_or_None, score) or None.
    """
    model.eval()
    fallback = None
    n = min(len(dataset), max_scan)
    for i in range(n):
        img_t, label_t, bbox_t = dataset[i]
        if label_t.item() != 1.0:
            continue
        out = model(img_t.unsqueeze(0).to(device))
        score = torch.sigmoid(out.cls_logit)[0].item()
        gt_xyxy = cxcywh_to_xyxy(bbox_t.numpy())
        img_uint8 = (img_t.squeeze(0).numpy() * 255).astype(np.uint8)

        if out.bbox is not None:
            pred_xyxy = cxcywh_to_xyxy(out.bbox[0].cpu().numpy())
            iou = iou_xyxy(gt_xyxy, pred_xyxy)
        else:
            pred_xyxy, iou = None, 0.0

        is_positive_pred = (score >= CONF_THRESH) and (out.bbox is not None)

        if want == "success" and is_positive_pred and iou >= 0.5:
            return img_uint8, gt_xyxy, pred_xyxy, score
        if want == "failure" and not is_positive_pred:
            return img_uint8, gt_xyxy, None, score
        if want == "failure" and is_positive_pred and iou < 0.3 and fallback is None:
            fallback = (img_uint8, gt_xyxy, pred_xyxy, score)
    return fallback


def draw_panel(ax, img_uint8, gt_xyxy, pred_xyxy, title):
    ax.imshow(img_uint8, cmap="gray", vmin=0, vmax=255)
    gx1, gy1, gx2, gy2 = gt_xyxy
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                                edgecolor="#2ca02c", facecolor="none", linewidth=2,
                                label="Ground truth"))
    if pred_xyxy is not None:
        px1, py1, px2, py2 = pred_xyxy
        ax.add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1,
                                    edgecolor="#ff7f0e", facecolor="none", linewidth=2,
                                    linestyle="--", label="Prediction"))
    else:
        ax.text(4, 12, "no detection", color="#ff7f0e", fontsize=9,
                 bbox=dict(facecolor="black", alpha=0.6, pad=1))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eo_model = EMADet(use_attention=False, use_spatial_bbox=True).to(device)
    eo_model.load_state_dict(torch.load(EO_CKPT, map_location=device))
    thermal_model = EMADet(use_attention=False, use_spatial_bbox=True).to(device)
    thermal_model.load_state_dict(torch.load(THERMAL_CKPT, map_location=device))

    eo_ds = DroneDetectDataset(EO_TEST_DIR, augment=False, img_size=IMG_SIZE, cache_images=True)
    thermal_ds = DroneDetectDataset(THERMAL_TEST_DIR, augment=False, img_size=IMG_SIZE, cache_images=True)

    panel_a = find_example(eo_model, eo_ds, device, want="success")
    panel_b = find_example(thermal_model, thermal_ds, device, want="success")
    panel_c = find_example(eo_model, thermal_ds, device, want="success")
    panel_d = find_example(eo_model, thermal_ds, device, want="failure")

    panels = [
        (panel_a, "(a) EO in-domain (correct)"),
        (panel_b, "(b) Thermal in-domain (correct)"),
        (panel_c, "(c) EO$\\to$thermal zero-shot (correct)"),
        (panel_d, "(d) EO$\\to$thermal zero-shot (missed)"),
    ]
    for data, title in panels:
        if data is None:
            raise RuntimeError(f"Could not find a qualifying example for: {title}")

    fig, axes = plt.subplots(2, 2, figsize=(6, 6.4))
    for ax, (data, title) in zip(axes.flat, panels):
        img_uint8, gt_xyxy, pred_xyxy, score = data
        draw_panel(ax, img_uint8, gt_xyxy, pred_xyxy, f"{title}\nscore={score:.2f}")

    handles = [
        plt.Line2D([0], [0], color="#2ca02c", lw=2, label="Ground truth"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=2, linestyle="--", label="Prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, "qualitative_examples.png")
    pdf_path = os.path.join(OUT_DIR, "qualitative_examples.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved -> {png_path}\nSaved -> {pdf_path}")


if __name__ == "__main__":
    main()
