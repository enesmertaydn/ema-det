# -*- coding: utf-8 -*-
"""
main_model_numbers.py -- every EMA-Det number of the manuscript for the final (centre-based head) model, five seeds, EO test split.
  * patch-level recall / precision and box-level recall (IoU >= 0.5) at score thresholds 0.5 and 0.1
  * AUROC and recall at 5% false-positive rate (from results/variants_eval.json) with Welch t-tests against the other models
Writes results/main_model_numbers.json. CPU.
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
THR = [0.5, 0.1]


def main():
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    from map_decomposition import iou_rows, xyxy
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    pos = np.array([float(ds[i][1]) == 1.0 for i in range(len(ds))])
    gt = xyxy(np.stack([ds[i][2].numpy() for i in range(len(ds))]))
    rows = []
    for sd in SEEDS:
        m = EMADet(center_head=True)
        m.load_state_dict(torch.load(os.path.join(RES, "emadet_final_centerhead_seed%d.pt" % sd), map_location="cpu"))
        m.eval()
        sc, bb = [], []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                o = m(X[i:i + 256])
                sc.append(torch.sigmoid(o.cls_logit.view(-1)).numpy())
                bb.append(o.bbox.numpy())
        sc, bb = np.concatenate(sc), np.concatenate(bb)
        iou = iou_rows(xyxy(bb), gt)
        row = {}
        for t in THR:
            flag = sc >= t
            row[str(t)] = {"patch_recall": 100 * float((flag & pos).sum() / pos.sum()),
                           "precision": 100 * float((flag & pos).sum() / max(1, flag.sum())),
                           "box_recall": 100 * float((flag & pos & (iou >= 0.5)).sum() / pos.sum())}
        rows.append(row)
    out = {"thresholds": {str(t): {k: [float(np.mean([r[str(t)][k] for r in rows])), float(np.std([r[str(t)][k] for r in rows]))]
                                   for k in ("patch_recall", "precision", "box_recall")} for t in THR}}
    # threshold-free comparison with Welch tests
    from math import sqrt
    var = json.load(open(os.path.join(RES, "variants_eval.json")))["Centre-based box head"]["per_seed"]
    ours = {"auroc": [r["auroc"] for r in var], "recall_fpr5": [r["recall_fpr5"] for r in var]}
    others = {}
    for n, f in (("MobileNetV3-Small", "threshold_free_cls.json"), ("ShuffleNetV2", "threshold_free_cls.json"),
                 ("YOLOv8n scratch", "yolo_threshold_free.json"), ("YOLOv8n COCO", "yolo_threshold_free.json")):
        d = json.load(open(os.path.join(RES, f)))
        key = {"MobileNetV3-Small": "MobileNetV3-Small", "ShuffleNetV2": "ShuffleNetV2 x0.5", "YOLOv8n scratch": "YOLOv8n (scratch)",
               "YOLOv8n COCO": "YOLOv8n (COCO-pretrained)"}[n]
        others[n] = {"auroc": d[key]["auroc"][2], "recall_fpr5": d[key]["recall@FPR5%"][2]}

    def welch(a, b):
        a, b = np.array(a), np.array(b)
        t = (a.mean() - b.mean()) / sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        v = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
        xs = np.linspace(0, abs(t), 20001)
        from math import lgamma, pi
        pdf = np.exp(lgamma((v + 1) / 2) - lgamma(v / 2)) / np.sqrt(v * pi) * (1 + xs ** 2 / v) ** (-(v + 1) / 2)
        integ = getattr(np, "trapezoid", None) or np.trapz
        return float(1 - 2 * integ(pdf, xs))
    out["ours"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in ours.items()}
    out["welch_p"] = {n: {k: welch(ours[k], others[n][k]) for k in ours} for n in others}
    out["others"] = {n: {k: [float(np.mean(v)), float(np.std(v))] for k, v in others[n].items()} for n in others}
    json.dump(out, open(os.path.join(RES, "main_model_numbers.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
