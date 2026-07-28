# -*- coding: utf-8 -*-
"""
cross_sensor_eval.py -- zero-shot cross-sensor generalization check.

Evaluates the EO-trained EMA-Det "final" checkpoint (spatial-bbox default,
no CBAM, results/emadet_final_best.pt) directly on the thermal-IR
Anti-UAV410 test split (data/antiuav410_clean/test) with NO retraining or
fine-tuning -- this measures how much of the EO-domain performance
transfers to an unseen sensor modality.
"""
import json
import os

import torch

from dataset import DroneDetectDataset
from ema_det import EMADet
from train import Config, evaluate

THERMAL_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\antiuav410_clean"
EO_CKPT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\emadet_final_best.pt"
OUT_PATH = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\cross_sensor_zeroshot.json"


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    model = EMADet(use_attention=False, use_spatial_bbox=True).to(device)
    model.load_state_dict(torch.load(EO_CKPT, map_location=device))
    model.eval()

    results = {}
    for split in ("val", "test"):
        split_dir = os.path.join(THERMAL_ROOT, split)
        ds = DroneDetectDataset(split_dir, augment=False, img_size=cfg.img_size, cache_images=True)
        res = evaluate(model, split_dir, device, cfg.img_size, cfg.conf_thresh,
                        eval_batch_size=cfg.eval_batch_size, dataset=ds)
        results[split] = {
            "map50": res.map50, "map75": res.map75, "map50_95": res.map50_95,
            "precision": res.precision, "recall": res.recall, "f1": res.f1,
            "mean_iou": res.mean_iou, "n_samples": len(ds),
        }
        print(f"  [{split}] mAP@.5:{res.map50:.2f}%  mAP@.75:{res.map75:.2f}%  "
              f"mAP@.5:.95:{res.map50_95:.2f}%  P:{res.precision:.2f}%  "
              f"R:{res.recall:.2f}%  F1:{res.f1:.2f}%  IoU:{res.mean_iou:.2f}%  (n={len(ds)})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Zero-shot: EO-trained emadet_final_best.pt (spatial-bbox, no CBAM), "
                    "no fine-tuning, evaluated directly on thermal-IR Anti-UAV410 patches.",
            "checkpoint": EO_CKPT, "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
