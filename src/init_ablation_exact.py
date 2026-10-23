# -*- coding: utf-8 -*-
"""
init_ablation_exact.py -- exact (unrounded) test metrics of the five physical-init
and the five random-init EMA-Det checkpoints, same evaluate() as training.
Writes results/init_ablation_exact.json (used by init_ablation_analysis.py).
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ema_det import EMADet      # noqa: E402
from train import evaluate      # noqa: E402

SEEDS = [42, 7, 123, 2024, 31415]
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(path, physical):
    m = EMADet(use_attention=False, use_spatial_bbox=True, physical_init=physical).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    r = evaluate(m, TEST, dev, 128, 0.5)
    return {"map50": r.map50, "map75": r.map75, "map50_95": r.map50_95, "recall": r.recall, "precision": r.precision,
            "f1": r.f1, "mean_iou": r.mean_iou}


out = {"physical": {}, "random": {}}
for s in SEEDS:
    out["physical"][s] = run(os.path.join(RES, "emadet_final_seed%d.pt" % s), True)
    rp = os.path.join(RES, "emadet_final_randinit_seed%d.pt" % s)
    if not os.path.exists(rp):
        rp = os.path.join(RES, "randinit_seed31415", "emadet_final_randinit_seed%d.pt" % s)
    out["random"][s] = run(rp, False)
    print(s, out["physical"][s]["map50"], out["random"][s]["map50"], flush=True)
json.dump(out, open(os.path.join(RES, "init_ablation_exact.json"), "w"), indent=2)
