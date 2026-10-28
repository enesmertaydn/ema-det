# -*- coding: utf-8 -*-
"""
external_eval.py -- zero-shot evaluation of the five EO-trained final (centre-head) EMA-Det seeds on the external DUT Anti-UAV test
patches built by build_dut_dataset.py (no fine-tuning; none of these images was seen in training or model selection).
Same metrics as new_variants_eval.py. Optional second row: the earlier spatial regression head (CBAM, from variants_eval).
Writes results/external_dut_eval.json. CPU.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from variants_eval import auroc, xyxy  # noqa: E402

RES = os.path.join(ROOT, "results")
TEST = os.path.join(ROOT, "data", "dut_antiuav_test")
SEEDS = [42, 7, 123, 2024, 31415]
KEYS = ("map50", "map75", "map50_95", "precision", "recall", "f1", "mean_iou")


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = np.array([float(ds[i][1]) for i in range(len(ds))])
    gt = xyxy(np.stack([ds[i][2].numpy() for i in range(len(ds))]))
    pos = Y == 1
    rows = []
    for sd in SEEDS:
        m = EMADet(center_head=True)
        m.load_state_dict(torch.load(os.path.join(RES, "emadet_final_centerhead_seed%d.pt" % sd), map_location="cpu"))
        m.eval()
        r = train.evaluate(m, TEST, torch.device("cpu"), 128, 0.5, dataset=ds)
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
        row = {k: float(getattr(r, k)) for k in KEYS}
        row["box_recall"] = 100 * float(((sc >= 0.5) & pos & (iou >= 0.5)).sum() / pos.sum())
        row["auroc"] = auroc(sc, Y)
        row["recall_fpr5"] = 100 * float((sc[pos] > thr).mean())
        rows.append(row)
        print(sd, {k: round(v, 3) for k, v in row.items()}, flush=True)
    out = {"n_patches": int(len(ds)), "n_pos": int(pos.sum()), "per_seed": rows,
           "summary": {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}}
    json.dump(out, open(os.path.join(RES, "external_dut_eval.json"), "w"), indent=2)
    print({k: "%.2f +/- %.2f" % tuple(v) if k != "auroc" else "%.4f +/- %.4f" % tuple(v) for k, v in out["summary"].items()})


if __name__ == "__main__":
    main()
