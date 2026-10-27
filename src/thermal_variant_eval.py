# -*- coding: utf-8 -*-
"""
thermal_variant_eval.py -- independent test of the centre-based box head on the Anti-UAV410 thermal data.

The centre-based head (src/ema_det.py CenterHead) was designed and selected on the EO data. The thermal data played no role in
its design or selection, so thermal-from-scratch training is an independent test of whether it improves localization.
Both models are trained from scratch with the same protocol (src/train.py, 150 epochs, patience 20, seeds 42, 7, 123, 2024, 31415):
  default          results/thermal/emadet_final_seed*.pt           (src/train_thermal_scratch.py)
  centre-based     results/thermal/emadet_final_centerhead_seed*.pt (scratch_runner3.sh)
Per seed, on the thermal test split, with train.evaluate (CPU): mAP@0.5, mAP@0.75, mAP@0.5:0.95, precision, recall, F1, mean IoU.
Mean +- std over the available seeds and a paired t-test. Writes results/thermal_variant_eval.json.
"""
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
TEST = os.path.join(ROOT, "data", "antiuav410_clean", "test")
RES = os.path.join(ROOT, "results", "thermal")
SEEDS = [42, 7, 123, 2024, 31415]
KEYS = ("map50", "map75", "map50_95", "precision", "recall", "f1", "mean_iou")


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    dev = torch.device("cpu")
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    out = {}
    for name, tag, kw in (("default", "final", {}), ("centre-based head", "final_centerhead", dict(center_head=True))):
        rows = {}
        for sd in SEEDS:
            ck = os.path.join(RES, "emadet_%s_seed%d.pt" % (tag, sd))
            if not os.path.exists(ck) or (tag != "final" and not os.path.exists(os.path.join(RES, "emadet_%s_seed%d.DONE" % (tag, sd)))):
                continue
            m = EMADet(**kw)
            m.load_state_dict(torch.load(ck, map_location=dev))
            m.eval()
            r = train.evaluate(m, TEST, dev, 128, 0.5, dataset=ds)
            rows[sd] = {k: float(getattr(r, k)) for k in KEYS}
            print(name, sd, {k: round(v, 2) for k, v in rows[sd].items()}, flush=True)
        out[name] = {"per_seed": rows}
        if rows:
            out[name]["summary"] = {k: [float(np.mean([r[k] for r in rows.values()])), float(np.std([r[k] for r in rows.values()]))] for k in KEYS}
    common = sorted(set(out["default"]["per_seed"]) & set(out["centre-based head"]["per_seed"]))
    if len(common) >= 2:
        from scipy import stats
        out["paired_t"] = {}
        for k in KEYS:
            a = [out["centre-based head"]["per_seed"][s][k] for s in common]
            b = [out["default"]["per_seed"][s][k] for s in common]
            t, p = stats.ttest_rel(a, b)
            out["paired_t"][k] = [float(np.mean(a) - np.mean(b)), float(p)]
        out["seeds_compared"] = common
    json.dump(out, open(os.path.join(ROOT, "results", "thermal_variant_eval.json"), "w"), indent=2)
    for name in ("default", "centre-based head"):
        if "summary" in out[name]:
            print(name, {k: "%.2f +/- %.2f" % tuple(v) for k, v in out[name]["summary"].items()})
    if "paired_t" in out:
        print("paired (centre - default):", {k: "%+.2f (p=%.4f)" % tuple(v) for k, v in out["paired_t"].items()}, "seeds", common)


if __name__ == "__main__":
    main()
