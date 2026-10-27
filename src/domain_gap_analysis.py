# -*- coding: utf-8 -*-
"""
domain_gap_analysis.py -- why does the centre-based head help much more on the EO data than on the thermal data?

For the EO test split and the thermal test split, and for the default and the centre-based EMA-Det (all available seeds), reports:
  - ground-truth box size (median side length in pixels of the 128x128 patch) and the share of boxes below 16 px,
  - mean IoU of the predicted box over the positive patches, and the shares with IoU >= 0.5 and >= 0.75,
  - mAP@0.5 / mAP@0.5:0.95 as is, with an oracle score (score x IoU), and with the ground-truth box (see src/map_decomposition.py).
CPU only. Writes results/domain_gap_analysis.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from map_decomposition import iou_rows, map_of, xyxy  # noqa: E402

SEEDS = [42, 7, 123, 2024, 31415]
SETS = {
    "EO": (os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test"), os.path.join(ROOT, "results"), "emadet_%s_seed%d.pt"),
    "thermal": (os.path.join(ROOT, "data", "antiuav410_clean", "test"), os.path.join(ROOT, "results", "thermal"), "emadet_%s_seed%d.pt"),
}


def main():
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    out = {}
    for dname, (root, rdir, pat) in SETS.items():
        ds = DroneDetectDataset(root, augment=False, img_size=128, cache_images=True)
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        pos = np.array([float(ds[i][1]) == 1.0 for i in range(len(ds))])
        bb = np.stack([ds[i][2].numpy() for i in range(len(ds))])
        gt = xyxy(bb)
        side = np.sqrt(bb[pos, 2] * bb[pos, 3]) * 128
        out[dname] = {"n_patches": int(len(ds)), "n_pos": int(pos.sum()), "gt_side_px_median": float(np.median(side)),
                      "gt_side_px_p10_p90": [float(np.percentile(side, 10)), float(np.percentile(side, 90))], "share_below_16px": float((side < 16).mean() * 100)}
        print(dname, out[dname], flush=True)
        for mname, tag, kw in (("default", "final", {}), ("centre", "final_centerhead", dict(center_head=True))):
            rows = []
            for sd in SEEDS:
                ck = os.path.join(rdir, pat % (tag, sd))
                done = os.path.join(rdir, "emadet_%s_seed%d.DONE" % (tag, sd))
                if not os.path.exists(ck) or (tag != "final" and dname == "thermal" and not os.path.exists(done)):
                    continue
                m = EMADet(**kw)
                m.load_state_dict(torch.load(ck, map_location="cpu"))
                m.eval()
                sc, bx, ds_ = [], [], []
                with torch.no_grad():
                    for i in range(0, len(X), 256):
                        o = m(X[i:i + 256])
                        sc.append(torch.sigmoid(o.cls_logit.view(-1)).numpy())
                        bx.append(o.bbox.numpy())
                        ds_.append((o.det_score if o.det_score is not None else torch.sigmoid(o.cls_logit.view(-1))).numpy())
                sc, bx, dsc = np.concatenate(sc), np.concatenate(bx), np.concatenate(ds_)
                p = xyxy(bx)
                iou = iou_rows(p, gt)
                row = {"mean_iou": float(iou[pos].mean() * 100), "iou_ge_0.5": float((iou[pos] >= 0.5).mean() * 100),
                       "iou_ge_0.75": float((iou[pos] >= 0.75).mean() * 100)}
                row["map_as_is"] = map_of(p, dsc, gt, pos)
                row["map_oracle_score"] = map_of(p, np.where(pos, dsc * iou, dsc), gt, pos)
                row["map_gt_box"] = map_of(gt, dsc, gt, pos)
                rows.append(row)
                print(dname, mname, sd, {k: (tuple(round(x, 1) for x in v) if isinstance(v, tuple) else round(v, 1)) for k, v in row.items()}, flush=True)
            if rows:
                out[dname][mname] = {"n_seeds": len(rows),
                                     "mean_iou": float(np.mean([r["mean_iou"] for r in rows])),
                                     "iou_ge_0.5": float(np.mean([r["iou_ge_0.5"] for r in rows])),
                                     "iou_ge_0.75": float(np.mean([r["iou_ge_0.75"] for r in rows])),
                                     "map50_as_is": float(np.mean([r["map_as_is"][0] for r in rows])),
                                     "map50_95_as_is": float(np.mean([r["map_as_is"][1] for r in rows])),
                                     "map50_oracle_score": float(np.mean([r["map_oracle_score"][0] for r in rows])),
                                     "map50_95_oracle_score": float(np.mean([r["map_oracle_score"][1] for r in rows])),
                                     "map50_gt_box": float(np.mean([r["map_gt_box"][0] for r in rows]))}
                print(dname, mname, out[dname][mname], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "domain_gap_analysis.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
