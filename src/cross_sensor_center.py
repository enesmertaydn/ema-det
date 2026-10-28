# -*- coding: utf-8 -*-
"""
cross_sensor_center.py -- zero-shot EO -> thermal transfer of the centre-head EMA-Det (all five EO-trained seeds, no fine-tuning).
Evaluates results/emadet_final_centerhead_seed*.pt on the Anti-UAV410 val and test splits and writes
results/cross_sensor_zeroshot_center.json (per seed, mean and std). CPU.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
SEEDS = [42, 7, 123, 2024, 31415]
KEYS = ("map50", "map75", "map50_95", "precision", "recall", "f1", "mean_iou")


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    out = {}
    for split in ("val", "test"):
        d = os.path.join(ROOT, "data", "antiuav410_clean", split)
        ds = DroneDetectDataset(d, augment=False, img_size=128, cache_images=True)
        rows = []
        for sd in SEEDS:
            m = EMADet(center_head=True)
            m.load_state_dict(torch.load(os.path.join(ROOT, "results", "emadet_final_centerhead_seed%d.pt" % sd), map_location="cpu"))
            m.eval()
            r = train.evaluate(m, d, torch.device("cpu"), 128, 0.5, dataset=ds)
            rows.append({k: float(getattr(r, k)) for k in KEYS})
            print(split, sd, {k: round(v, 2) for k, v in rows[-1].items()}, flush=True)
        out[split] = {"per_seed": rows, "summary": {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in KEYS}}
        print(split, {k: "%.2f +/- %.2f" % tuple(v) for k, v in out[split]["summary"].items()}, flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "cross_sensor_zeroshot_center.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
