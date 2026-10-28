# -*- coding: utf-8 -*-
"""
new_variants_eval.py -- two follow-up variants of the final (centre-based head) EMA-Det, five seeds, EO test split, paired against the final model:
  depthwise Blocks 2-6   results/emadet_final_centerhead_dsall_seed*.pt     (--center_head --ds_all)
  fitness selection      results/emadet_final_centerhead_fit_seed*.pt       (--center_head --select fitness: checkpoint chosen by
                         0.1 mAP@0.5 + 0.9 mAP@0.5:0.95 on the validation split, the rule Ultralytics uses for YOLOv8n)
Metrics per seed: mAP@0.5, mAP@0.75, mAP@0.5:0.95, precision, recall, F1, mean IoU (train.evaluate), box recall (IoU >= 0.5), AUROC,
recall at 5% false-positive rate, parameters. Mean +- std and paired t-test against the final model.
Writes results/new_variants_eval.json. CPU.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from variants_eval import auroc, paired_t, xyxy  # noqa: E402

RES = os.path.join(ROOT, "results")
TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
SEEDS = [42, 7, 123, 2024, 31415]
VARIANTS = {
    "Centre head, depthwise Blocks 2-6": ("final_centerhead_dsall", dict(center_head=True, ds_all=True)),
    "Centre head, fitness selection": ("final_centerhead_fit", dict(center_head=True)),
}


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = np.array([float(ds[i][1]) for i in range(len(ds))])
    gt = xyxy(np.stack([ds[i][2].numpy() for i in range(len(ds))]))
    pos = Y == 1
    ref = json.load(open(os.path.join(RES, "variants_eval.json")))["Centre-based box head"]["per_seed"]
    out = {}
    for name, (tag, kw) in VARIANTS.items():
        rows = []
        for sd in SEEDS:
            ck = os.path.join(RES, "emadet_%s_seed%d.pt" % (tag, sd))
            if not os.path.exists(os.path.join(RES, "emadet_%s_seed%d.DONE" % (tag, sd))):
                continue
            m = EMADet(**kw)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m.eval()
            r = train.evaluate(m, TEST, torch.device("cpu"), 128, 0.5)
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
            if len(rows) == len(ref):
                out[name]["paired_t_vs_final"] = {k: paired_t([r[k] for r in rows], [r[k] for r in ref])
                                                  for k in ("map50", "map75", "map50_95", "recall", "box_recall", "auroc", "recall_fpr5", "mean_iou")}
    json.dump(out, open(os.path.join(RES, "new_variants_eval.json"), "w"), indent=2)
    for name, v in out.items():
        if "summary" in v:
            print("\n" + name, {k: "%.2f +/- %.2f" % tuple(x) if k != "auroc" else "%.4f +/- %.4f" % tuple(x) for k, x in v["summary"].items()})
            if "paired_t_vs_final" in v:
                print("  paired vs final (diff, p):", {k: (round(x[0], 2), round(x[1], 3)) for k, x in v["paired_t_vs_final"].items()})


if __name__ == "__main__":
    main()
