# -*- coding: utf-8 -*-
"""
train_yolov8n_clean.py — YOLOv8n RoI-mode baseline, retrained from scratch
on the clean/leakage-free dataset, N=5 seeds, for a fair Table III (the
conference draft's YOLOv8n numbers were trained/evaluated on the leaky
dataset, same issue as EMA-Det's — see src/train.py's module docstring).
"""
import json
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Config:
    data_yaml: str = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\clean_data.yaml"
    img_size: int = 128
    epochs: int = 150
    batch: int = 16
    patience: int = 20
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123, 2024, 31415])
    workers: int = 4
    device: object = 0 if torch.cuda.is_available() else "cpu"
    project: str = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\yolov8n_clean"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(seed: int, cfg: Config):
    set_seed(seed)
    print(f"\n{'='*60}\n  YOLOv8n (scratch) — seed={seed}\n{'='*60}")

    model = YOLO("yolov8n.yaml")  # scratch architecture, no COCO weights
    run_name = f"yolov8n_scratch_seed{seed}"
    model.train(
        data=cfg.data_yaml, epochs=cfg.epochs, imgsz=cfg.img_size, batch=cfg.batch,
        patience=cfg.patience, device=cfg.device, seed=seed, verbose=True,
        workers=cfg.workers, project=cfg.project, name=run_name,
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
    print(f"  Test: mAP@.5={result['map50']:.1f}%  mAP@.5:.95={result['map50_95']:.1f}%  "
          f"Recall={result['recall']:.1f}%")
    return result


if __name__ == "__main__":
    cfg = Config()
    print(f"Device: {cfg.device}  Seeds: {cfg.seeds}  Epochs: {cfg.epochs}")

    per_seed = [train_one_seed(seed, cfg) for seed in cfg.seeds]

    keys = ["map50", "map50_95", "precision", "recall"]
    summary = {k: {"mean": float(np.mean([r[k] for r in per_seed if r[k] is not None])),
                   "std": float(np.std([r[k] for r in per_seed if r[k] is not None]))}
               for k in keys}

    out = {"model": "yolov8n_scratch_clean", "per_seed": per_seed, "summary": summary}
    out_path = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\baseline_yolov8n_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}\n  YOLOv8n (scratch, clean data) — mean +/- std, N={len(cfg.seeds)}\n{'='*60}")
    for k in keys:
        print(f"  {k:<12}: {summary[k]['mean']:6.2f} +/- {summary[k]['std']:5.2f}")
    print(f"\nSaved -> {out_path}")
