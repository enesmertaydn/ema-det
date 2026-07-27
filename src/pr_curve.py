# -*- coding: utf-8 -*-
"""
pr_curve.py — Precision-Recall curve for trained EMA-Det checkpoints.

Reviewer-requested metric (see project history): the conference draft only
reported point precision/recall at a single confidence threshold (0.5). This
sweeps the threshold to produce a full PR curve, for both the final
(no-attention) and cbam checkpoints produced by train.py's
run_controlled_comparison(), evaluated on the clean, leakage-free test set.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DroneDetectDataset
from ema_det import EMADet


def collect_scores_labels(model, dataset, device, batch_size=256):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls, _ in dl:
            imgs = imgs.to(device)
            out = model(imgs)  # tau=None -> full batch
            scores = torch.sigmoid(out.cls_logit).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(lbls.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def pr_curve(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 101):
    """Classification PR curve (cls head only, independent of bbox/IoU) —
    matches the conf_thresh sweep semantics used elsewhere in this project."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precisions, recalls = [], []
    n_pos = (labels == 1).sum()
    for t in thresholds:
        pred_pos = scores >= t
        tp = int(((pred_pos) & (labels == 1)).sum())
        fp = int(((pred_pos) & (labels == 0)).sum())
        fn = int(n_pos - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    ap = float(trapezoid(np.array(precisions)[::-1], np.array(recalls)[::-1]))
    return thresholds.tolist(), precisions, recalls, ap


def main():
    ap_parser = argparse.ArgumentParser()
    ap_parser.add_argument("--dataset_root",
                            default=r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\shuffled_dataset_detect_clean")
    ap_parser.add_argument("--results_dir",
                            default=r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results")
    ap_parser.add_argument("--img_size", type=int, default=128)
    args = ap_parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = os.path.join(args.dataset_root, "test")
    test_ds = DroneDetectDataset(test_dir, augment=False, img_size=args.img_size, cache_images=True)

    configs = {
        "final": (False, os.path.join(args.results_dir, "emadet_final_best.pt")),
        "cbam": (True, os.path.join(args.results_dir, "emadet_cbam_best.pt")),
    }

    plt.figure(figsize=(6, 5))
    out = {}
    for tag, (use_attention, ckpt_path) in configs.items():
        if not os.path.exists(ckpt_path):
            print(f"[skip] {tag}: checkpoint not found at {ckpt_path}")
            continue
        model = EMADet(use_attention=use_attention).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        scores, labels = collect_scores_labels(model, test_ds, device)
        thresholds, precisions, recalls, ap = pr_curve(scores, labels)
        out[tag] = {"thresholds": thresholds, "precision": precisions, "recall": recalls, "ap": ap}
        print(f"{tag}: AP = {ap:.4f}")
        plt.plot(recalls, precisions, label=f"EMA-Det ({tag}), AP={ap:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (RoI Verification, clean test set)")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(args.results_dir, exist_ok=True)
    png_path = os.path.join(args.results_dir, "pr_curve.png")
    json_path = os.path.join(args.results_dir, "pr_curve.json")
    plt.savefig(png_path, dpi=150)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {png_path}\nSaved -> {json_path}")


if __name__ == "__main__":
    main()
