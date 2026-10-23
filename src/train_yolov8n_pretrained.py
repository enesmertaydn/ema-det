# -*- coding: utf-8 -*-
"""
train_yolov8n_pretrained.py -- COCO-pretrained YOLOv8n baseline on the clean
(leakage-free) RoI dataset, N=5 seeds, identical protocol to the from-scratch
baseline in train_yolov8n_clean.py (150 epochs, patience 20, batch 16, 128 px,
same seeds). The ONLY difference is the initialization: yolov8n.pt (COCO)
instead of yolov8n.yaml (random). Written for the SIVP revision
(Reviewer 1 point 3, Reviewer 3 attachment point 1).

Run (needs the ultralytics package, CUDA PyTorch and yolov8n.pt, which
Ultralytics downloads automatically on first use):
    python src/train_yolov8n_pretrained.py
Results: results/baseline_yolov8n_pretrained_results.json
         results/yolov8n_pretrained/yolov8n_coco_seed<seed>/
"""
import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    data_yaml: str = os.path.join(ROOT, "data", "clean_data.yaml")
    img_size: int = 128
    epochs: int = 150
    batch: int = 16
    patience: int = 20
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123, 2024, 31415])
    workers: int = 4
    device: object = 0 if torch.cuda.is_available() else "cpu"
    project: str = os.path.join(ROOT, "results", "yolov8n_pretrained")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(seed: int, cfg: Config):
    set_seed(seed)
    print(f"\n{'='*60}\n  YOLOv8n (COCO-pretrained) -- seed={seed}\n{'='*60}")

    model = YOLO("yolov8n.pt")  # COCO-pretrained weights
    run_name = f"yolov8n_coco_seed{seed}"
    model.train(
        data=cfg.data_yaml, epochs=cfg.epochs, imgsz=cfg.img_size, batch=cfg.batch,
        patience=cfg.patience, device=cfg.device, seed=seed, verbose=True,
        workers=cfg.workers, project=cfg.project, name=run_name, pretrained=True,
    )

    metrics = model.val(data=cfg.data_yaml, split="test", imgsz=cfg.img_size)
    result = {
        "seed": seed,
        "map50": float(metrics.box.map50) * 100,
        "map75": float(metrics.box.map75) * 100 if hasattr(metrics.box, "map75") else None,
        "map50_95": float(metrics.box.map) * 100,
        "precision": float(metrics.box.mp) * 100,
        "recall": float(metrics.box.mr) * 100,
    }
    print(f"  Test: mAP@0.5={result['map50']:.1f}%  mAP@0.5:0.95={result['map50_95']:.1f}%  "
          f"Recall={result['recall']:.1f}%")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 42,7")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, separate output dir")
    args = ap.parse_args()
    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.seeds:
        cfg.seeds = [int(x) for x in args.seeds.split(",")]
    if args.smoke:
        cfg.epochs, cfg.seeds = 1, [42]
        cfg.project = os.path.join(ROOT, "results", "yolov8n_pretrained_smoke")
    print(f"Device: {cfg.device}  Seeds: {cfg.seeds}  Epochs: {cfg.epochs}")

    per_seed = [train_one_seed(seed, cfg) for seed in cfg.seeds]

    keys = ["map50", "map50_95", "precision", "recall"]
    summary = {k: {"mean": float(np.mean([r[k] for r in per_seed if r[k] is not None])),
                   "std": float(np.std([r[k] for r in per_seed if r[k] is not None]))}
               for k in keys}

    out = {"model": "yolov8n_coco_pretrained_clean", "per_seed": per_seed, "summary": summary}
    out_path = os.path.join(ROOT, "results", "baseline_yolov8n_pretrained_results%s.json" % ("_smoke" if args.smoke else ""))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}\n  YOLOv8n (COCO-pretrained, clean data) -- mean +/- std, N={len(cfg.seeds)}\n{'='*60}")
    for k in keys:
        print(f"  {k:<12}: {summary[k]['mean']:6.2f} +/- {summary[k]['std']:5.2f}")
    print(f"\nSaved -> {out_path}")
