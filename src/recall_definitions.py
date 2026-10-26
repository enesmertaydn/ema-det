# -*- coding: utf-8 -*-
"""
recall_definitions.py -- are the recall numbers of EMA-Det and YOLOv8n comparable?

EMA-Det's P/R/F1 in src/train.py are PATCH-LEVEL: a positive patch counts as found when the
classification score >= conf (0.5), without checking the box. Ultralytics' reported recall
(metrics.box.mr) is BOX-LEVEL (IoU >= 0.5) at the confidence of maximum F1. This script measures
all models under the SAME definitions and at several confidence thresholds, on the same test set:
  patch recall    = positives with at least one detection / score >= t
  patch precision = positives among patches flagged at t
  box recall      = positives whose detection / box with score >= t has IoU >= 0.5 with the ground truth
Writes results/recall_definitions.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
YAML = os.path.join(ROOT, "data", "clean_data.yaml")
SEEDS = [42, 7, 123, 2024, 31415]
THR = [0.5, 0.25, 0.1, 0.05, 0.01]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cxcywh_to_xyxy(b):
    return [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]


def gt_for(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    lp = os.path.join(TEST, "labels", stem + ".txt")
    if os.path.exists(lp) and os.path.getsize(lp) > 0:
        parts = open(lp).readline().split()
        return cxcywh_to_xyxy([float(x) for x in parts[1:5]])
    return None


def summarize(records):
    """records: per image list of (is_positive, [(score, iou_with_gt or 0), ...])"""
    out = {}
    npos = sum(1 for p, _ in records if p)
    for t in THR:
        flagged = [(p, [d for d in dets if d[0] >= t]) for p, dets in records]
        flagged = [(p, d) for p, d in flagged if d]
        tp = sum(1 for p, d in flagged if p)
        box_tp = sum(1 for p, d in flagged if p and any(iou >= 0.5 for _, iou in d))
        miou = np.mean([max([iou for _, iou in d], default=0.0) if any(sc >= t for sc, _ in d) else 0.0 for p, d in records if p])
        out[str(t)] = {"mean_iou": 100.0 * float(miou), "patch_recall": 100.0 * tp / npos, "patch_precision": 100.0 * tp / max(1, len(flagged)),
                       "box_recall": 100.0 * box_tp / npos}
    return out


def yolo_records(weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    paths = sorted(os.path.join(TEST, "images", f) for f in os.listdir(os.path.join(TEST, "images")))
    recs = []
    for r, p in zip(m.predict(paths, imgsz=128, conf=0.001, verbose=False, stream=True, device=0 if torch.cuda.is_available() else "cpu"), paths):
        gt = gt_for(p)
        dets = []
        if len(r.boxes):
            xy = r.boxes.xyxyn.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            for b, c in zip(xy, cf):
                dets.append((float(c), iou_xyxy(list(b), gt) if gt is not None else 0.0))
        recs.append((gt is not None, dets))
    return recs


def emadet_records(ckpt, spatial=True):
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    m = EMADet(use_attention=False, use_spatial_bbox=spatial).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    recs = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, lbl, bb = ds[i]
            out = m(img.unsqueeze(0).to(dev))
            sc = float(torch.sigmoid(out.cls_logit).view(-1)[0])
            pb = cxcywh_to_xyxy(out.bbox[0].cpu().tolist())
            gt = cxcywh_to_xyxy(bb.tolist()) if float(lbl) == 1.0 else None
            recs.append((gt is not None, [(sc, iou_xyxy(pb, gt) if gt is not None else 0.0)]))
    return recs


def mean_std(per_seed):
    keys = list(per_seed[0].keys())
    return {t: {k: [float(np.mean([s[t][k] for s in per_seed])), float(np.std([s[t][k] for s in per_seed]))]
                for k in per_seed[0][t]} for t in keys}


def main():
    results = {}
    # EMA-Det
    results["EMA-Det (final)"] = mean_std([summarize(emadet_records(os.path.join(RES, "emadet_final_seed%d.pt" % s))) for s in SEEDS])
    print("EMA-Det done", flush=True)
    results["EMA-Det (legacy pooled-vector bbox)"] = mean_std([summarize(emadet_records(os.path.join(RES, "emadet_final_legacygapbbox_seed%d.pt" % s), spatial=False)) for s in SEEDS])
    print("legacy done", flush=True)
    # YOLOv8n pretrained
    results["YOLOv8n (COCO-pretrained)"] = mean_std(
        [summarize(yolo_records(os.path.join(RES, "yolov8n_pretrained", "yolov8n_coco_seed%d" % s, "weights", "best.pt"))) for s in SEEDS])
    print("YOLO pretrained done", flush=True)
    # YOLOv8n scratch: several run folders per seed exist; keep the one whose Ultralytics recall matches the published JSON
    base = json.load(open(os.path.join(RES, "baseline_yolov8n_results.json")))
    want = {r["seed"]: r["recall"] for r in base["per_seed"]}
    scratch = []
    from ultralytics import YOLO
    for s in SEEDS:
        best, best_gap = None, 1e9
        for suf in ("", "2", "3"):
            w = os.path.join(RES, "yolov8n_clean", "yolov8n_scratch_seed%d%s" % (s, suf), "weights", "best.pt")
            if not os.path.exists(w):
                continue
            mr = float(YOLO(w).val(data=YAML, split="test", imgsz=128, verbose=False, plots=False).box.mr) * 100
            if abs(mr - want[s]) < best_gap:
                best, best_gap = w, abs(mr - want[s])
        print("scratch seed", s, "->", os.path.basename(os.path.dirname(os.path.dirname(best))), "gap %.3f" % best_gap, flush=True)
        scratch.append(summarize(yolo_records(best)))
    results["YOLOv8n (scratch)"] = mean_std(scratch)

    json.dump(results, open(os.path.join(RES, "recall_definitions.json"), "w"), indent=2)
    print("\n%-28s %-5s %18s %18s %18s" % ("model", "conf", "patch recall", "patch precision", "box recall (IoU.5)"))
    for name, d in results.items():
        for t in THR:
            r = d[str(t)]
            print("%-28s %-5s %9.2f +/- %4.2f %9.2f +/- %4.2f %9.2f +/- %4.2f" % (
                name, t, r["patch_recall"][0], r["patch_recall"][1], r["patch_precision"][0], r["patch_precision"][1],
                r["box_recall"][0], r["box_recall"][1]))



if __name__ == "__main__":
    main()
