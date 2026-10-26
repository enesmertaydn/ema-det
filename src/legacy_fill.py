# -*- coding: utf-8 -*-
"""Precision / F1 / mean IoU of the legacy pooled-vector configuration (5 seeds, test set), using the same
evaluate() as every other row of the ablation table. CPU only. Writes results/legacy_fill.json."""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    import train
    from ema_det import EMADet
    dev = torch.device("cpu")
    D = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
    rows = []
    for sd in [42, 7, 123, 2024, 31415]:
        m = EMADet(use_attention=False, use_spatial_bbox=False)
        m.load_state_dict(torch.load(os.path.join(ROOT, "results", "emadet_final_legacygapbbox_seed%d.pt" % sd), map_location=dev))
        m.eval()
        r = train.evaluate(m, D, dev, 128, 0.5)
        rows.append({k: float(getattr(r, k)) for k in ("map50", "map75", "map50_95", "precision", "recall", "f1", "mean_iou")})
    out = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
    print({k: "%.2f +/- %.2f" % tuple(v) for k, v in out.items()})
    json.dump({"summary": out, "per_seed": rows}, open(os.path.join(ROOT, "results", "legacy_fill.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
