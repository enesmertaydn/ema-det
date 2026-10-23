# -*- coding: utf-8 -*-
"""
frozen_ablation.py -- frozen vs. trainable Laplacian/Sobel kernels (N=5 seeds).

Stage 1 (needs PyTorch, run in the training environment):
    python src/frozen_ablation.py eval
  evaluates the five emadet_final_frozen_seed<seed>.pt checkpoints with the same
  evaluate() as training, checks that the 12 fixed kernels are bit-identical to the
  Laplacian/Sobel bank, and writes results/frozen_ablation_exact.json.
Stage 2 (needs scipy):
    python src/frozen_ablation.py stats
  compares them with the trainable physical-init run (results/init_ablation_exact.json)
  using paired t-tests, and with the training log for convergence; writes
  results/frozen_ablation.json.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEEDS = [42, 7, 123, 2024, 31415]
NAMES = ["map50", "map75", "map50_95", "recall", "precision", "f1", "mean_iou"]


def stage_eval():
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from ema_det import EMADet, make_edge_kernels
    from train import evaluate
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
    bank = make_edge_kernels(16)[:12]
    out = {"frozen": {}, "fixed_kernels_bit_identical": {}}
    for s in SEEDS:
        m = EMADet(use_attention=False, use_spatial_bbox=True).to(dev)
        sd = torch.load(os.path.join(RES, "emadet_final_frozen_seed%d.pt" % s), map_location=dev)
        m.load_state_dict(sd)
        m.eval()
        out["fixed_kernels_bit_identical"][s] = bool(torch.equal(m.block1.conv.weight[:12].detach().cpu(), bank))
        r = evaluate(m, test, dev, 128, 0.5)
        out["frozen"][s] = {"map50": r.map50, "map75": r.map75, "map50_95": r.map50_95, "recall": r.recall,
                            "precision": r.precision, "f1": r.f1, "mean_iou": r.mean_iou}
        print(s, out["frozen"][s]["map50"], out["frozen"][s]["recall"], out["fixed_kernels_bit_identical"][s], flush=True)
    json.dump(out, open(os.path.join(RES, "frozen_ablation_exact.json"), "w"), indent=2)


def parse_log(path, config):
    res, cur = {}, None
    re_cfg = re.compile(r"config=(\S+)\s+seed=(\d+)")
    re_ep = re.compile(r"Ep\s+(\d+)/\d+ \|.*?Val mAP@\.5:([0-9.]+)%")
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re_cfg.search(line)
        if m:
            cur = int(m.group(2)) if m.group(1) == config else None
            if cur is not None:
                res[cur] = []
            continue
        m = re_ep.search(line)
        if cur is not None and m:
            res[cur].append(float(m.group(2)))
    return res


def stage_stats():
    from scipy import stats
    fz = json.load(open(os.path.join(RES, "frozen_ablation_exact.json"), encoding="utf-8"))
    ph = json.load(open(os.path.join(RES, "init_ablation_exact.json"), encoding="utf-8"))["physical"]
    summary = {}
    for n in NAMES:
        a = np.array([ph[str(s)][n] for s in SEEDS])
        b = np.array([fz["frozen"][str(s)][n] for s in SEEDS])
        t, p = stats.ttest_rel(a, b)
        summary[n] = {"trainable_mean": float(a.mean()), "trainable_std": float(a.std()),
                      "frozen_mean": float(b.mean()), "frozen_std": float(b.std()),
                      "diff_trainable_minus_frozen": float((a - b).mean()), "paired_t_p": float(p)}
    logs = parse_log(os.path.join(RES, "full_run_spatialbbox.log"), "final_spatialbbox")
    import glob
    flogs = {}
    # full_run_frozen.log (seeds 42, 7 and a crashed seed 123) then the per-seed re-runs, which override earlier entries
    for lp in sorted(glob.glob(os.path.join(RES, "full_run_frozen*.log"))):
        for sd, v in parse_log(lp, "final_frozen").items():
            if lp.endswith("full_run_frozen.log") and sd not in (42, 7):
                continue  # seed 123 in the first log was cut short by a CUDA crash
            if v:
                flogs[sd] = v

    def conv(d):
        r = {}
        for k in (5, 10, 20):
            v = [d[s][k - 1] for s in SEEDS]
            r["val_map50_ep%d" % k] = [float(np.mean(v)), float(np.std(v))]
        return r
    out = {"summary": summary, "all_fixed_kernels_bit_identical": all(fz["fixed_kernels_bit_identical"].values()),
           "convergence": {"trainable": conv(logs), "frozen": conv(flogs)}}
    json.dump(out, open(os.path.join(RES, "frozen_ablation.json"), "w"), indent=2)
    print("frozen vs trainable (mean +/- std, 5 seeds); paired t-test p; fixed kernels bit-identical:", out["all_fixed_kernels_bit_identical"])
    for n in NAMES:
        s = summary[n]
        print("  %-9s trainable %6.2f +/- %4.2f | frozen %6.2f +/- %4.2f | diff %+5.2f | p=%.3f" % (
            n, s["trainable_mean"], s["trainable_std"], s["frozen_mean"], s["frozen_std"], s["diff_trainable_minus_frozen"], s["paired_t_p"]))
    print("convergence", {k: {a: [round(x, 1) for x in b] for a, b in v.items()} for k, v in out["convergence"].items()})


if __name__ == "__main__":
    {"eval": stage_eval, "stats": stage_stats}[sys.argv[1]]()
