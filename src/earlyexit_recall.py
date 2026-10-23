# -*- coding: utf-8 -*-
"""
earlyexit_recall.py -- recall/precision across the early-exit threshold tau
(revision, Reviewer 1 point 1). Re-evaluates the same EMA-Det (final,
spatial-bbox, no CBAM) checkpoints used for Table 4 on the test set:
  * the single best checkpoint (the one behind the published tau sweep), and
  * all five seed checkpoints (mean +/- std),
for the full network (tau=None) and tau in {0.85, 0.90, 0.92, 0.95}.
Writes results/early_exit_recall.json.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

# cv2.imread cannot open paths containing non-ASCII characters on Windows
# (this checkout lives under ".../Masaustu" with a Turkish "u-umlaut").
_orig_imread = cv2.imread
def _imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
cv2.imread = _imread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ema_det import EMADet          # noqa: E402
from train import evaluate          # noqa: E402

DATA = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
RES = os.path.join(ROOT, "results")
SEEDS = [42, 7, 123, 2024, 31415]
TAUS = [None, 0.85, 0.90, 0.92, 0.95]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(ckpt):
    m = EMADet(use_attention=False, use_spatial_bbox=True).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    out = {}
    for tau in TAUS:
        r = evaluate(m, DATA, dev, 128, 0.5, early_exit_tau=tau)
        out["full" if tau is None else str(tau)] = {
            "map50": r.map50, "recall": r.recall, "precision": r.precision}
        print(os.path.basename(ckpt), tau, out["full" if tau is None else str(tau)], flush=True)
    return out


best = run(os.path.join(RES, "emadet_final_best.pt"))
per_seed = {s: run(os.path.join(RES, f"emadet_final_seed{s}.pt")) for s in SEEDS}
summary = {}
for k in best:
    summary[k] = {}
    for met in ("map50", "recall", "precision"):
        v = [per_seed[s][k][met] for s in SEEDS]
        summary[k][met] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
json.dump({"best_ckpt": best, "per_seed": per_seed, "summary": summary},
          open(os.path.join(RES, "early_exit_recall.json"), "w"), indent=2)
print(json.dumps(summary, indent=1))
