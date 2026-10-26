# -*- coding: utf-8 -*-
"""
threshold_free_cls.py -- is EMA-Det's higher verifier recall just a calibration effect?

Compares the classification score of EMA-Det, MobileNetV3-Small and ShuffleNetV2 x0.5 (same head,
same protocol) with threshold-free measures on the test set: AUROC and recall at a fixed false-positive
rate (patches without a target), plus recall at the fixed 0.5 threshold used in the paper.
Writes results/threshold_free_cls.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
SEEDS = [42, 7, 123, 2024, 31415]


def scores(model, ds, dev):
    model.eval()
    s, y = [], []
    with torch.no_grad():
        for i in range(0, len(ds), 256):
            batch = [ds[j] for j in range(i, min(i + 256, len(ds)))]
            x = torch.stack([b[0] for b in batch]).to(dev)
            out = model(x)
            s.append(torch.sigmoid(out.cls_logit.view(-1)).cpu().numpy())
            y.append(np.array([float(b[1]) for b in batch]))
    return np.concatenate(s), np.concatenate(y)


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(s, y, fpr):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean() * 100)


def main():
    from ema_det import EMADet
    from baselines import build_model
    from dataset import DroneDetectDataset
    from train_baseline import _NoGateAdapter
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    makers = {
        "EMA-Det (final)": (lambda: EMADet(use_attention=False, use_spatial_bbox=True), "emadet_final_seed%d.pt"),
        "MobileNetV3-Small": (lambda: build_model("mobilenet_v3_small"), "mobilenet_v3_small_seed%d.pt"),
        "ShuffleNetV2 x0.5": (lambda: build_model("shufflenet_v2_x0_5"), "shufflenet_v2_x0_5_seed%d.pt"),
    }
    out = {}
    for name, (mk, pat) in makers.items():
        rows = []
        for sd in SEEDS:
            m = mk().to(dev)
            m.load_state_dict(torch.load(os.path.join(RES, pat % sd), map_location=dev))
            m = m if name.startswith("EMA") else _NoGateAdapter(m)
            s, y = scores(m, ds, dev)
            rows.append({"auroc": auroc(s, y), "recall@0.5": float((s[y == 1] >= 0.5).mean() * 100),
                         "recall@FPR5%": recall_at_fpr(s, y, 0.05), "recall@FPR1%": recall_at_fpr(s, y, 0.01)})
        out[name] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])), [r[k] for r in rows]]
                     for k in rows[0]}
        print(name, {k: "%.3f +/- %.3f" % (v[0], v[1]) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "threshold_free_cls.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
