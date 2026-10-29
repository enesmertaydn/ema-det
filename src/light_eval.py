# -*- coding: utf-8 -*-
"""
light_eval.py -- parameter-reduced variants of the final centre-head EMA-Det, five seeds each, paired against the final model.
  Block 6 narrowed to 128 ch   results/emadet_final_c6128_centerhead_seed*.pt   (--center_head --c6 128)
  depthwise Blocks 4-6         results/emadet_final_dsdeep_centerhead_seed*.pt  (--center_head --ds_deep)
Sets: EO validation (the selection set) and EO test (reported only). Metrics as in scaleaug_eval.py. Writes results/light_eval.json. CPU.
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
D = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean")
SETS = {"EO val": os.path.join(D, "val"), "EO test": os.path.join(D, "test")}
MODELS = {
    "final": ("final_centerhead", dict(center_head=True)),
    "Block 6 128 ch": ("final_c6128_centerhead", dict(center_head=True, c6=128)),
    "depthwise Blocks 4-6": ("final_dsdeep_centerhead", dict(center_head=True, ds_deep=True)),
}
SEEDS = [42, 7, 123, 2024, 31415]
KEYS = ("map50", "map75", "map50_95", "precision", "recall", "mean_iou", "box_recall", "auroc", "recall_fpr5")


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    out = {}
    for sname, sdir in SETS.items():
        ds = DroneDetectDataset(sdir, augment=False, img_size=128, cache_images=True)
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        Y = np.array([float(ds[i][1]) for i in range(len(ds))])
        gt = xyxy(np.stack([ds[i][2].numpy() for i in range(len(ds))]))
        pos = Y == 1
        out[sname] = {}
        for mname, (tag, kw) in MODELS.items():
            rows = []
            for sd in SEEDS:
                m = EMADet(**kw)
                m.load_state_dict(torch.load(os.path.join(RES, "emadet_%s_seed%d.pt" % (tag, sd)), map_location="cpu"))
                m.eval()
                n_params = int(sum(q.numel() for q in m.parameters()))
                r = train.evaluate(m, sdir, torch.device("cpu"), 128, 0.5, dataset=ds)
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
                row = {k: float(getattr(r, k)) for k in ("map50", "map75", "map50_95", "precision", "recall", "mean_iou")}
                row["box_recall"] = 100 * float(((sc >= 0.5) & pos & (iou >= 0.5)).sum() / pos.sum())
                row["auroc"] = auroc(sc, Y)
                row["recall_fpr5"] = 100 * float((sc[pos] > thr).mean())
                row["n_params"] = n_params
                rows.append(row)
            out[sname][mname] = {"per_seed": rows, "summary": {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in KEYS}, "n_params": rows[0]["n_params"]}
            print(sname, mname, out[sname][mname]["n_params"], {k: ("%.2f+/-%.2f" % tuple(v)) for k, v in out[sname][mname]["summary"].items()}, flush=True)
        for mname in list(MODELS)[1:]:
            a, b = out[sname][mname]["per_seed"], out[sname]["final"]["per_seed"]
            out[sname][mname]["paired_t_vs_final"] = {k: paired_t([r[k] for r in a], [r[k] for r in b]) for k in ("map50", "map50_95", "recall", "box_recall", "auroc", "recall_fpr5")}
            diff = {k: round(float(np.mean([x[k] for x in a]) - np.mean([x[k] for x in b])), 2) for k in ("map50", "map50_95", "recall", "auroc", "recall_fpr5")}
            print("  %s vs final: diff %s | p %s" % (mname, diff, {k: round(v[1], 3) for k, v in out[sname][mname]["paired_t_vs_final"].items()}), flush=True)
    json.dump(out, open(os.path.join(RES, "light_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
