# -*- coding: utf-8 -*-
"""Does the centre-head gain also appear on the EO validation split (not used in the mAP decomposition that motivated the head)? CPU, 5 seeds."""
import json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
D = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "val")
SEEDS = [42, 7, 123, 2024, 31415]
KEYS = ("map50", "map75", "map50_95", "mean_iou", "recall")


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    ds = DroneDetectDataset(D, augment=False, img_size=128, cache_images=True)
    out = {}
    for name, tag, kw in (("default", "final", {}), ("centre", "final_centerhead", dict(center_head=True))):
        rows = []
        for sd in SEEDS:
            m = EMADet(**kw)
            m.load_state_dict(torch.load(os.path.join(ROOT, "results", "emadet_%s_seed%d.pt" % (tag, sd)), map_location="cpu"))
            m.eval()
            r = train.evaluate(m, D, torch.device("cpu"), 128, 0.5, dataset=ds)
            rows.append({k: float(getattr(r, k)) for k in KEYS})
        out[name] = rows
    from variants_eval import paired_t
    for k in KEYS:
        a = [r[k] for r in out["centre"]]; b = [r[k] for r in out["default"]]
        print("%-9s default %.2f+-%.2f  centre %.2f+-%.2f  diff %+.2f  paired p=%.4f  per-seed diff %s" % (
            k, np.mean(b), np.std(b), np.mean(a), np.std(a), np.mean(a) - np.mean(b), paired_t(a, b)[1],
            [round(x - y, 1) for x, y in zip(a, b)]))
    json.dump(out, open(os.path.join(ROOT, "results", "eo_val_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
