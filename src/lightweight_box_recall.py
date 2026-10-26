# -*- coding: utf-8 -*-
"""Box-level recall (IoU>=0.5) at score>=0.5 for the lightweight baselines, to put them on the same footing as
EMA-Det and YOLOv8n in the revised comparison table. Runs on the CPU. Writes results/lightweight_box_recall.json."""
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


def xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def main():
    from ema_det import EMADet
    from baselines import build_model
    from dataset import DroneDetectDataset
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = np.array([float(ds[i][1]) for i in range(len(ds))])
    B = np.stack([ds[i][2].numpy() for i in range(len(ds))])
    gt = xyxy(B)
    pos = Y == 1
    makers = {
        "EMA-Det (final)": (lambda: EMADet(use_attention=False, use_spatial_bbox=True), "emadet_final_seed%d.pt", False),
        "EMA-Det (legacy pooled-vector bbox)": (lambda: EMADet(use_attention=False, use_spatial_bbox=False), "emadet_final_legacygapbbox_seed%d.pt", False),
        "MobileNetV3-Small": (lambda: build_model("mobilenet_v3_small"), "mobilenet_v3_small_seed%d.pt", True),
        "ShuffleNetV2 x0.5": (lambda: build_model("shufflenet_v2_x0_5"), "shufflenet_v2_x0_5_seed%d.pt", True),
    }
    out = {}
    for name, (mk, pat, _) in makers.items():
        rows = []
        for sd in SEEDS:
            m = mk()
            m.load_state_dict(torch.load(os.path.join(RES, pat % sd), map_location="cpu"))
            m.eval()
            sc, bb = [], []
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    o = m(X[i:i + 256])
                    sc.append(torch.sigmoid(o.cls_logit.view(-1)).numpy())
                    bb.append(o.bbox.numpy())
            sc, bb = np.concatenate(sc), np.concatenate(bb)
            p = xyxy(bb)
            ix = np.clip(np.minimum(p[:, 2], gt[:, 2]) - np.maximum(p[:, 0], gt[:, 0]), 0, None)
            iy = np.clip(np.minimum(p[:, 3], gt[:, 3]) - np.maximum(p[:, 1], gt[:, 1]), 0, None)
            inter = ix * iy
            iou = inter / ((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter + 1e-9)
            flag = sc >= 0.5
            rows.append({"patch_recall": 100 * float((flag & pos).sum() / pos.sum()),
                         "box_recall": 100 * float((flag & pos & (iou >= 0.5)).sum() / pos.sum()),
                         "precision": 100 * float((flag & pos).sum() / max(1, flag.sum()))})
        out[name] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
        print(name, {k: "%.2f +/- %.2f" % tuple(v) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "lightweight_box_recall.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
