# -*- coding: utf-8 -*-
"""
yolo_threshold_free.py -- threshold-free classification measures for YOLOv8n, on the same footing as
src/threshold_free_cls.py: the score of a patch is the confidence of its highest-scoring detection (0 if none).
Reports AUROC and recall at 5% / 1% false-positive rate (patches without a target) over the five seeds.
Runs on the CPU. Writes results/yolo_threshold_free.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
SEEDS = [42, 7, 123, 2024, 31415]
# run folders whose Ultralytics recall matches the published per-seed values (see src/recall_definitions.py)
SCRATCH = {42: "yolov8n_scratch_seed423", 7: "yolov8n_scratch_seed7", 123: "yolov8n_scratch_seed123",
           2024: "yolov8n_scratch_seed2024", 31415: "yolov8n_scratch_seed31415"}


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    # average ranks for ties (many negatives score exactly 0)
    allv = np.concatenate([pos, neg])
    sorted_v = allv[order]
    r = np.arange(1, len(order) + 1, dtype=float)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(s, y, fpr):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean() * 100)


def scores(weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    paths = sorted(os.path.join(TEST, "images", f) for f in os.listdir(os.path.join(TEST, "images")))
    s, y = [], []
    for r, p in zip(m.predict(paths, imgsz=128, conf=0.001, verbose=False, stream=True, device="cpu"), paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(TEST, "labels", stem + ".txt")
        y.append(1.0 if (os.path.exists(lp) and os.path.getsize(lp) > 0) else 0.0)
        s.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(s), np.array(y)


def main():
    out = {}
    for name, pat in (("YOLOv8n (scratch)", lambda sd: os.path.join(RES, "yolov8n_clean", SCRATCH[sd], "weights", "best.pt")),
                      ("YOLOv8n (COCO-pretrained)", lambda sd: os.path.join(RES, "yolov8n_pretrained", "yolov8n_coco_seed%d" % sd, "weights", "best.pt"))):
        rows = []
        for sd in SEEDS:
            s, y = scores(pat(sd))
            rows.append({"auroc": auroc(s, y), "recall@FPR5%": recall_at_fpr(s, y, 0.05), "recall@FPR1%": recall_at_fpr(s, y, 0.01)})
        out[name] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])), [r[k] for r in rows]] for k in rows[0]}
        print(name, {k: "%.3f +/- %.3f" % (v[0], v[1]) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "yolo_threshold_free.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
