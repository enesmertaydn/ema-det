# -*- coding: utf-8 -*-
"""
init_ablation_analysis.py -- Laplacian/Sobel (physical) vs. random first-layer
initialization, N=5 seeds (revision, Reviewer 3 / reviewer-style critique).

Reads the training logs of the two runs (identical protocol; only the Block 1
initialization differs) and reports, per seed and as mean +/- std:
  * test mAP@0.5, mAP@0.75, mAP@0.5:0.95 and recall,
  * paired t-tests across the 5 matched seeds,
  * convergence: validation mAP@0.5 at epochs 5/10/20, and the epoch of the
    best validation mAP.
Writes results/init_ablation.json.
Logs: results/full_run_spatialbbox.log  (physical init, config=final_spatialbbox)
      results/full_run_randinit.log + results/full_run_randinit_seed31415.log
"""
import json
import os
import re

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEEDS = [42, 7, 123, 2024, 31415]

RE_CFG = re.compile(r"config=(\S+)\s+seed=(\d+)")
RE_EP = re.compile(r"Ep\s+(\d+)/\d+ \|.*?Val mAP@\.5:([0-9.]+)%")
RE_TEST = re.compile(r"Test mAP@\.5:([0-9.]+)%\s+mAP@\.75:([0-9.]+)%\s+mAP@\.5:\.95:([0-9.]+)%\s+Recall:([0-9.]+)%")


def parse(path, config):
    out = {}
    cur = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = RE_CFG.search(line)
        if m:
            cur = int(m.group(2)) if m.group(1) == config else None
            if cur is not None:
                out[cur] = {"val": [], "test": None}
            continue
        if cur is None:
            continue
        m = RE_EP.search(line)
        if m:
            out[cur]["val"].append(float(m.group(2)))
        m = RE_TEST.search(line)
        if m:
            out[cur]["test"] = [float(x) for x in m.groups()]
    return out


phys = parse(os.path.join(RES, "full_run_spatialbbox.log"), "final_spatialbbox")
rand = parse(os.path.join(RES, "full_run_randinit.log"), "final_randinit")
rand.update({k: v for k, v in parse(os.path.join(RES, "full_run_randinit_seed31415.log"), "final_randinit").items()
             if v["test"] is not None})
for name, d in (("physical", phys), ("random", rand)):
    missing = [s for s in SEEDS if s not in d or d[s]["test"] is None]
    assert not missing, "%s run missing seeds %s" % (name, missing)

names = ["map50", "map75", "map50_95", "recall", "precision", "f1", "mean_iou"]


def table(d):
    return {n: [d[s]["test"][i] for s in SEEDS] for i, n in enumerate(names[:4])}  # logs only carry the first four


tp, tr = table(phys), table(rand)
_exact = os.path.join(RES, "init_ablation_exact.json")
if os.path.exists(_exact):  # unrounded test metrics from init_ablation_exact.py (preferred over the 0.1-rounded logs)
    ex = json.load(open(_exact, encoding="utf-8"))
    tp = {n: [ex["physical"][str(s)][n] for s in SEEDS] for n in names}
    tr = {n: [ex["random"][str(s)][n] for s in SEEDS] for n in names}
summary = {}
for n in names:
    a, b = np.array(tp[n]), np.array(tr[n])
    t, p = stats.ttest_rel(a, b)
    summary[n] = {"physical_mean": float(a.mean()), "physical_std": float(a.std()),
                  "random_mean": float(b.mean()), "random_std": float(b.std()),
                  "diff_phys_minus_rand": float((a - b).mean()), "paired_t_p": float(p)}


def conv(d):
    res = {}
    for k in (5, 10, 20):
        v = [d[s]["val"][k - 1] if len(d[s]["val"]) >= k else np.nan for s in SEEDS]
        res["val_map50_ep%d" % k] = [float(np.nanmean(v)), float(np.nanstd(v))]
    best = [int(np.argmax(d[s]["val"])) + 1 for s in SEEDS]
    res["best_epoch"] = [float(np.mean(best)), float(np.std(best))]
    res["epochs_run"] = [float(np.mean([len(d[s]["val"]) for s in SEEDS])), float(np.std([len(d[s]["val"]) for s in SEEDS]))]
    res["best_val_map50"] = [float(np.mean([max(d[s]["val"]) for s in SEEDS])), float(np.std([max(d[s]["val"]) for s in SEEDS]))]
    return res


out = {"seeds": SEEDS, "per_seed_test": {"physical": tp, "random": tr}, "summary": summary,
       "convergence": {"physical": conv(phys), "random": conv(rand)}}
with open(os.path.join(RES, "init_ablation.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("TEST (mean +/- std over 5 seeds); paired t-test p")
for n in names:
    s = summary[n]
    print("  %-9s physical %6.2f +/- %4.2f | random %6.2f +/- %4.2f | diff %+5.2f | p=%.3f" % (
        n, s["physical_mean"], s["physical_std"], s["random_mean"], s["random_std"], s["diff_phys_minus_rand"], s["paired_t_p"]))
print("CONVERGENCE")
for k, v in out["convergence"].items():
    print(" ", k, {a: [round(x, 2) for x in b] for a, b in v.items()})
