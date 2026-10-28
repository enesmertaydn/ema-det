# -*- coding: utf-8 -*-
"""
external_eval_by_scale.py -- external DUT Anti-UAV patches (build_dut_dataset.py) split by the drone's relative size in the patch,
sqrt(w*h) / patch side. The EO training patches have a median of 0.45 (10th percentile 0.27); DUT drones are mostly smaller
because the 128 px context floor dominates for tiny drones. Evaluates the five final (centre-head) EMA-Det seeds, zero-shot, per
size bin (positives of the bin plus their same-image hard negatives). Writes results/external_dut_by_scale.json. CPU.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
SRC = os.path.join(ROOT, "data", "dut_antiuav_test")
RES = os.path.join(ROOT, "results")
SEEDS = [42, 7, 123, 2024, 31415]
KEYS = ("map50", "map75", "map50_95", "recall", "precision")
EDGES = [0.0, 0.18, 0.24, 0.27, 1.01]   # relative size bins; last bin = inside the EO training range (>= 0.27)


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    sizes = {}
    for f in glob.glob(os.path.join(SRC, "labels", "p_*.txt")):
        t = open(f).read().split()
        sizes[os.path.basename(f)[2:-4]] = float(np.sqrt(float(t[3]) * float(t[4])))
    out = {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        stems = [s for s, v in sizes.items() if lo <= v < hi]
        d = os.path.join(ROOT, "data", "dut_scale_tmp")
        shutil.rmtree(d, ignore_errors=True)
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(d, sub))
        for s in stems:
            for pre in ("p_", "n_"):
                shutil.copy(os.path.join(SRC, "images", pre + s + ".jpg"), os.path.join(d, "images"))
                shutil.copy(os.path.join(SRC, "labels", pre + s + ".txt"), os.path.join(d, "labels"))
        ds = DroneDetectDataset(d, augment=False, img_size=128, cache_images=True)
        rows = []
        for sd in SEEDS:
            m = EMADet(center_head=True)
            m.load_state_dict(torch.load(os.path.join(RES, "emadet_final_centerhead_seed%d.pt" % sd), map_location="cpu"))
            m.eval()
            r = train.evaluate(m, d, torch.device("cpu"), 128, 0.5, dataset=ds)
            rows.append({k: float(getattr(r, k)) for k in KEYS})
        key = "%.2f-%.2f" % (lo, min(hi, 1.0))
        out[key] = {"n_pos": len(stems), "summary": {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in KEYS}}
        print(key, len(stems), {k: "%.1f+/-%.1f" % tuple(v) for k, v in out[key]["summary"].items()}, flush=True)
    shutil.rmtree(os.path.join(ROOT, "data", "dut_scale_tmp"), ignore_errors=True)
    json.dump(out, open(os.path.join(RES, "external_dut_by_scale.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
