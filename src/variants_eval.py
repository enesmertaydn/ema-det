# -*- coding: utf-8 -*-
"""
variants_eval.py -- five-seed comparison of architecture variants against the default EMA-Det.

Variants (all trained with src/train.py under the same protocol, one seed per process):
  final               default EMA-Det (1.53M parameters)
  final_dsdeep        depthwise-separable Blocks 4-6 (--ds_deep)
  final_c6128         Block 6 narrowed to 128 channels (--c6 128)
  final_fusionbbox    multi-scale box head (--fusion_bbox)

Per seed, on the test set: mAP@0.5, mAP@0.75, mAP@0.5:0.95, precision, recall (score>=0.5, box not checked),
F1, mean IoU (train.evaluate definitions), box recall (IoU>=0.5), AUROC and recall at a 5% false-positive rate.
Then mean/std over the seeds and a paired t-test against the default model.
Runs on the CPU. Writes results/variants_eval.json.
"""
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
SEEDS = [42, 7, 123, 2024, 31415]
VARIANTS = {
    "EMA-Det (default)": ("final", {}),
    "EMA-Det-DS (depthwise Blocks 4-6)": ("final_dsdeep", dict(ds_deep=True)),
    "Block 6 narrowed (128 ch)": ("final_c6128", dict(c6=128)),
    "Multi-scale box head": ("final_fusionbbox", dict(fusion_bbox=True)),
}


def xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired_t(a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    if d.std(ddof=1) == 0:
        return 0.0, 1.0
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    from math import lgamma, exp
    df = n - 1
    # two-sided p from the t distribution with df degrees of freedom (numerical integration)
    xs = np.linspace(0, abs(t), 20001)
    pdf = np.exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / np.sqrt(df * math.pi) * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    integ = getattr(np, "trapezoid", None) or np.trapz
    return float(t), float(1 - 2 * integ(pdf, xs))


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    dev = torch.device("cpu")
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = np.array([float(ds[i][1]) for i in range(len(ds))])
    gt = xyxy(np.stack([ds[i][2].numpy() for i in range(len(ds))]))
    pos = Y == 1
    out = {}
    for name, (tag, kw) in VARIANTS.items():
        rows = []
        for sd in SEEDS:
            ck = os.path.join(RES, "emadet_%s_seed%d.pt" % (tag, sd))
            if not os.path.exists(ck):
                print("missing", ck)
                continue
            m = EMADet(**kw)
            m.load_state_dict(torch.load(ck, map_location=dev))
            m.eval()
            r = train.evaluate(m, TEST, dev, 128, 0.5)
            sc, bb = [], []
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    o = m(X[i:i + 256])
                    sc.append(torch.sigmoid(o.cls_logit.view(-1)).numpy())
                    bb.append(o.bbox.numpy())
            sc, bb = np.concatenate(sc), np.concatenate(bb)
            p = xyxy(bb)
            ix = np.clip(np.minimum(p[:, 2], gt[:, 2]) - np.maximum(p[:, 0], gt[:, 0]), 0, None)
            iy = np.clip(np.minimum(p[:, 3], gt[:, 3]) - np.maximum(p[:, 1], gt[:, 1]), 0, None)
            inter = ix * iy
            iou = inter / ((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter + 1e-9)
            thr = np.quantile(sc[~pos], 0.95)
            row = {k: float(getattr(r, k)) for k in ("map50", "map75", "map50_95", "precision", "recall", "f1", "mean_iou")}
            row["box_recall"] = 100 * float(((sc >= 0.5) & pos & (iou >= 0.5)).sum() / pos.sum())
            row["auroc"] = auroc(sc, Y)
            row["recall_fpr5"] = 100 * float((sc[pos] > thr).mean())
            row["n_params"] = int(sum(q.numel() for q in m.parameters()))
            rows.append(row)
            print(name, sd, {k: round(v, 3) for k, v in row.items()}, flush=True)
        out[name] = {"per_seed": rows}
        if rows:
            out[name]["summary"] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
    base = out["EMA-Det (default)"]["per_seed"]
    for name, v in out.items():
        if name == "EMA-Det (default)" or len(v["per_seed"]) != len(base):
            continue
        v["paired_t_vs_default"] = {k: paired_t([r[k] for r in v["per_seed"]], [r[k] for r in base])
                                    for k in ("map50", "map75", "map50_95", "recall", "box_recall", "auroc", "recall_fpr5", "mean_iou")}
    json.dump(out, open(os.path.join(RES, "variants_eval.json"), "w"), indent=2)
    for name, v in out.items():
        if "summary" in v:
            print("\n" + name, {k: "%.2f +/- %.2f" % tuple(x) for k, x in v["summary"].items()})
            if "paired_t_vs_default" in v:
                print("  paired p vs default:", {k: round(x[1], 3) for k, x in v["paired_t_vs_default"].items()})


if __name__ == "__main__":
    main()
