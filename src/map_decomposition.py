# -*- coding: utf-8 -*-
"""
map_decomposition.py -- why is EMA-Det's mAP far below YOLOv8n's although its classification is better?

For each of the five seeds of the default EMA-Det, mAP@0.5 and mAP@0.5:0.95 on the test set are recomputed with
torchmetrics (the same metric as src/train.py) for four variants of the single predicted box per patch:
  A  as is: predicted box, classification score
  B  predicted box, score x IoU(predicted box, ground truth)   (oracle: scores that know the localization quality)
  C  ground-truth box, classification score                     (oracle: perfect localization)
  D  ground-truth box, score x 1                                 (same as C; kept for the negatives: score unchanged)
Comparing A with B isolates the loss caused by scores that ignore localization; A with C isolates the loss caused by boxes.
Runs on the CPU. Writes results/map_decomposition.json.
"""
import json
import os
import sys

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
SEEDS = [42, 7, 123, 2024, 31415]


def xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def iou_rows(p, g):
    ix = np.clip(np.minimum(p[:, 2], g[:, 2]) - np.maximum(p[:, 0], g[:, 0]), 0, None)
    iy = np.clip(np.minimum(p[:, 3], g[:, 3]) - np.maximum(p[:, 1], g[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter + 1e-9)


def map_of(boxes, scores, gt, pos):
    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    preds, tgts = [], []
    for i in range(len(boxes)):
        preds.append({"boxes": torch.tensor(boxes[i:i + 1], dtype=torch.float32), "scores": torch.tensor(scores[i:i + 1], dtype=torch.float32),
                      "labels": torch.zeros(1, dtype=torch.int64)})
        tgts.append({"boxes": torch.tensor(gt[i:i + 1], dtype=torch.float32) if pos[i] else torch.zeros((0, 4)),
                     "labels": torch.zeros(1 if pos[i] else 0, dtype=torch.int64)})
    m.update(preds, tgts)
    r = m.compute()
    return float(r["map_50"]) * 100, float(r["map"]) * 100


def main():
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    pos = np.array([float(ds[i][1]) == 1.0 for i in range(len(ds))])
    gt = xyxy(np.stack([ds[i][2].numpy() for i in range(len(ds))]))
    rows = []
    for sd in SEEDS:
        m = EMADet(use_attention=False, use_spatial_bbox=True)
        m.load_state_dict(torch.load(os.path.join(ROOT, "results", "emadet_final_seed%d.pt" % sd), map_location="cpu"))
        m.eval()
        sc, bb = [], []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                o = m(X[i:i + 256])
                sc.append(torch.sigmoid(o.cls_logit.view(-1)).numpy())
                bb.append(o.bbox.numpy())
        sc, bb = np.concatenate(sc), np.concatenate(bb)
        p = xyxy(bb)
        iou = iou_rows(p, gt)
        row = {}
        row["A_as_is"] = map_of(p, sc, gt, pos)
        row["B_score_x_iou_oracle"] = map_of(p, np.where(pos, sc * iou, sc), gt, pos)
        row["C_gt_box"] = map_of(gt, sc, gt, pos)
        row["mean_iou_positives"] = float(iou[pos].mean() * 100)
        row["frac_iou_ge_0.5"] = float((iou[pos] >= 0.5).mean() * 100)
        rows.append(row)
        print(sd, {k: (tuple(round(x, 1) for x in v) if isinstance(v, tuple) else round(v, 1)) for k, v in row.items()}, flush=True)
    out = {"per_seed": rows}
    for k in ("A_as_is", "B_score_x_iou_oracle", "C_gt_box"):
        out[k] = {"map50": [float(np.mean([r[k][0] for r in rows])), float(np.std([r[k][0] for r in rows]))],
                  "map50_95": [float(np.mean([r[k][1] for r in rows])), float(np.std([r[k][1] for r in rows]))]}
    for k in ("mean_iou_positives", "frac_iou_ge_0.5"):
        out[k] = [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
    json.dump(out, open(os.path.join(ROOT, "results", "map_decomposition.json"), "w"), indent=2)
    print({k: v for k, v in out.items() if k != "per_seed"})


if __name__ == "__main__":
    main()
