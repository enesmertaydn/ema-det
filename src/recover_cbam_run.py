# -*- coding: utf-8 -*-
"""
recover_cbam_run.py — one-off recovery after a CUDA crash mid-way through the
CBAM+spatial-bbox N=5 run (crashed during seed=2024 at epoch 62/150, an
"unspecified launch failure" -- GPU responded normally afterward via
nvidia-smi, consistent with a transient driver hiccup, not a code bug or
persistent hardware fault).

Seeds 42, 7, 123 completed fully and their checkpoints are on disk --
re-evaluating them (rather than re-training) gets the complete metric set
(precision/F1/mean_iou weren't in the truncated console log) without
wasting ~45 min of compute redoing work that already finished cleanly.
Seed 2024's on-disk checkpoint is from a crash-truncated run (stopped at
epoch 62, not by convergence/patience) and is NOT reused -- it and seed
31415 (never started) are retrained from scratch here.

Wrapped in main()/`if __name__ == "__main__"` -- required on Windows since
train_one_run()'s DataLoader uses num_workers>0, which spawns worker
processes that re-import this module; without the guard, the first crash
attempt (seed=2024) hit exactly the RuntimeError this pattern exists to
prevent.
"""
import copy
import json
import os

import numpy as np
import torch

from dataset import DroneDetectDataset
from ema_det import EMADet
from train import Config, evaluate, train_one_run

SAVE_DIR = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results"
GOOD_SEEDS = [42, 7, 123]
REDO_SEEDS = [2024, 31415]


def main():
    cfg = Config()
    device = torch.device(cfg.device)
    test_dir = os.path.join(cfg.dataset_root, "test")
    val_dir = os.path.join(cfg.dataset_root, "val")
    test_ds = DroneDetectDataset(test_dir, augment=False, img_size=cfg.img_size, cache_images=True)
    val_ds = DroneDetectDataset(val_dir, augment=False, img_size=cfg.img_size, cache_images=True)

    per_seed = []

    # --- Re-evaluate the 3 checkpoints that finished cleanly ---
    for seed in GOOD_SEEDS:
        ckpt_path = os.path.join(SAVE_DIR, f"emadet_cbam_seed{seed}.pt")
        model = EMADet(use_attention=True, use_spatial_bbox=True).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        res = evaluate(model, test_dir, device, cfg.img_size, cfg.conf_thresh,
                       eval_batch_size=cfg.eval_batch_size, dataset=test_ds)
        val_res = evaluate(model, val_dir, device, cfg.img_size, cfg.conf_thresh,
                            eval_batch_size=cfg.eval_batch_size, dataset=val_ds)
        n_params = sum(p.numel() for p in model.parameters())
        per_seed.append({
            "config": "cbam", "seed": seed, "n_params": n_params, "flops": "423.677M",
            "test_map50": res.map50, "test_map75": res.map75, "test_map50_95": res.map50_95,
            "test_precision": res.precision, "test_recall": res.recall, "test_f1": res.f1,
            "test_mean_iou": res.mean_iou,
            "best_val_map50": val_res.map50,
        })
        print(f"seed={seed} (re-evaluated): mAP50={res.map50:.2f} mAP50:95={res.map50_95:.2f} "
              f"Recall={res.recall:.2f} F1={res.f1:.2f} val_mAP50={val_res.map50:.2f}")

    # --- Retrain the crash-truncated / never-started seeds ---
    best_map50, best_state = -1.0, None
    for m in per_seed:
        if m["test_map50"] > best_map50:
            best_map50 = m["test_map50"]

    for seed in REDO_SEEDS:
        model, m = train_one_run(seed, True, cfg, device)
        per_seed.append(m)
        if m["test_map50"] > best_map50:
            best_map50, best_state = m["test_map50"], copy.deepcopy(model.state_dict())

    if best_state is None:
        best_seed = max(per_seed[:len(GOOD_SEEDS)], key=lambda m: m["test_map50"])["seed"]
        best_state = torch.load(os.path.join(SAVE_DIR, f"emadet_cbam_seed{best_seed}.pt"), map_location=device)

    torch.save(best_state, os.path.join(SAVE_DIR, "emadet_cbam_best.pt"))

    keys = ["test_map50", "test_map75", "test_map50_95", "test_precision",
            "test_recall", "test_f1", "test_mean_iou", "best_val_map50"]
    summary = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                   "std": float(np.std([m[k] for m in per_seed]))} for k in keys}

    print(f"\n{'='*65}\n  CBAM (recovered) -- mean +/- std over N={len(per_seed)} seeds\n{'='*65}")
    for k in keys:
        print(f"  {k:<18}: {summary[k]['mean']:6.2f} +/- {summary[k]['std']:5.2f}")

    out = {"cbam": {"per_seed": per_seed, "summary": summary,
                    "best_ckpt": os.path.join(SAVE_DIR, "emadet_cbam_best.pt"),
                    "n_params": per_seed[0]["n_params"], "flops": per_seed[0]["flops"]}}
    out_path = os.path.join(SAVE_DIR, "controlled_comparison_cbam_recovered.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
