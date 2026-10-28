# -*- coding: utf-8 -*-
"""
earlyexit_center.py -- early-exit analysis of the final (centre-based head) EMA-Det.
  part "recall": mAP@0.5 / recall / precision at tau in {full, 0.85, 0.90, 0.92, 0.95} for the five EO seeds (mean +- std); CPU is fine.
  part "table":  the tau table (mAP, FLOPs saved, FPS, exit distribution) for the seed with the best VALIDATION mAP@0.5;
                 FPS needs an idle GPU, so run it when nothing else trains.
Writes results/early_exit_center_recall.json and results/early_exit_center_table.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
EO = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean")
SEEDS = [42, 7, 123, 2024, 31415]
TAUS = [None, 0.85, 0.90, 0.92, 0.95]


def main(part):
    import train
    from ema_det import EMADet
    if part == "recall":
        dev = torch.device("cpu")
        per_seed = {}
        for sd in SEEDS:
            m = EMADet(center_head=True)
            m.load_state_dict(torch.load(os.path.join(RES, "emadet_final_centerhead_seed%d.pt" % sd), map_location=dev))
            m.eval()
            per_seed[sd] = {}
            for tau in TAUS:
                r = train.evaluate(m, os.path.join(EO, "test"), dev, 128, 0.5, early_exit_tau=tau)
                per_seed[sd]["full" if tau is None else str(tau)] = {"map50": r.map50, "recall": r.recall, "precision": r.precision}
                print(sd, tau, per_seed[sd]["full" if tau is None else str(tau)], flush=True)
        summary = {k: {met: [float(np.mean([per_seed[s][k][met] for s in SEEDS])), float(np.std([per_seed[s][k][met] for s in SEEDS]))]
                       for met in ("map50", "recall", "precision")} for k in per_seed[SEEDS[0]]}
        json.dump({"per_seed": per_seed, "summary": summary}, open(os.path.join(RES, "early_exit_center_recall.json"), "w"), indent=2)
        for k, v in summary.items():
            print(k, {m: "%.2f +/- %.2f" % tuple(x) for m, x in v.items()})
    else:
        val = json.load(open(os.path.join(RES, "eo_val_check.json")))["centre"]
        best = SEEDS[int(np.argmax([r["map50"] for r in val]))]
        print("best validation seed:", best)
        cfg = train.Config()
        cfg.dataset_root = EO
        cfg.device = os.environ.get("EE_DEVICE", cfg.device)
        dev = torch.device(cfg.device)
        m = EMADet(center_head=True).to(dev)
        m.load_state_dict(torch.load(os.path.join(RES, "emadet_final_centerhead_seed%d.pt" % best), map_location=dev))
        m.eval()
        rows = train.analyze_early_exit_table(m, cfg)
        json.dump({"seed": best, "rows": rows}, open(os.path.join(RES, "early_exit_center_table%s.json" % os.environ.get("EE_TAG", "")), "w"), indent=2)
        for r in rows:
            print(r)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recall")
