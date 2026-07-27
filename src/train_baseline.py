# -*- coding: utf-8 -*-
"""
train_baseline.py — N=5-seed, 150-epoch training for the lightweight
comparison models (baselines.py), on the same clean/leakage-free dataset
and identical protocol as train.py's EMA-Det runs, for a fair Table III.

No gate losses (these architectures have no early-exit heads) — just
BCE(cls) + lambda_bbox * CIoU(bbox), same lambda_bbox=2.0 as EMA-Det.
"""
import argparse
import copy
import functools
import json
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from thop import profile as thop_profile
    from thop import clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from baselines import build_model
from dataset import DroneDetectDataset
from ema_det import EMADetOutput
from losses import ciou_loss
from train import WarmupCosineScheduler, EarlyStopping, evaluate, set_seed, _worker_seed_fn


class _NoGateAdapter(nn.Module):
    """
    Wraps a gate-less baseline (VerifierOutput: cls_logit, bbox) so it can
    be passed to train.py's evaluate(), which expects an EMADetOutput
    (cls_logit, bbox, gate1_logit, gate2_logit, exit_stage) — avoids
    duplicating evaluate()'s torchmetrics/precision/recall/IoU bookkeeping
    for a second time here.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, tau=None) -> EMADetOutput:
        assert tau is None, "baseline models have no early-exit gates"
        out = self.model(x)
        return EMADetOutput(out.cls_logit, out.bbox, None, None, "main")


@dataclass
class BaselineConfig:
    model_name: str = "mobilenet_v3_small"
    dataset_root: str = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\shuffled_dataset_detect_clean"
    save_dir: str = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results"

    epochs: int = 150
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_bbox: float = 2.0

    warmup_epochs: int = 5
    patience: int = 20
    min_delta: float = 1e-4

    seeds: List[int] = field(default_factory=lambda: [42, 7, 123, 2024, 31415])
    img_size: int = 128
    num_workers: int = 2
    eval_batch_size: int = 256
    conf_thresh: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_run(seed: int, cfg: BaselineConfig, device: torch.device):
    print(f"\n{'='*70}\n  model={cfg.model_name}  seed={seed}\n{'='*70}")
    set_seed(seed)
    model = build_model(cfg.model_name).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    flops_str = "N/A"
    if THOP_AVAILABLE:
        dummy = torch.zeros(1, 1, cfg.img_size, cfg.img_size, device=device)
        model_copy = copy.deepcopy(model).eval()
        with torch.no_grad():
            macs, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)[:2]
        del model_copy
        # clever_format returns a bare string (not a list) for single-element
        # input, unlike its 2+-element behavior — indexing [0] on that would
        # silently grab just the first character (e.g. "3" instead of
        # "3.256M"). Pass n_params alongside it, matching train.py's usage,
        # to get the documented tuple-return behavior instead.
        flops_str, _ = clever_format([macs * 2, n_params], "%.3f")
    print(f"  Params: {n_params:,}   FLOPs: {flops_str}")

    train_dir = os.path.join(cfg.dataset_root, "train")
    val_dir = os.path.join(cfg.dataset_root, "val")
    test_dir = os.path.join(cfg.dataset_root, "test")

    tr_dl = DataLoader(
        DroneDetectDataset(train_dir, augment=True, img_size=cfg.img_size, cache_images=True),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=functools.partial(_worker_seed_fn, base_seed=seed),
    )
    val_ds = DroneDetectDataset(val_dir, augment=False, img_size=cfg.img_size, cache_images=True)
    test_ds = DroneDetectDataset(test_dir, augment=False, img_size=cfg.img_size, cache_images=True)

    bce = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = WarmupCosineScheduler(opt, cfg.warmup_epochs, cfg.epochs)
    stopper = EarlyStopping(cfg.patience, cfg.min_delta)

    ckpt_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_seed{seed}.pt")

    ep = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tloss = 0.0
        n_batches = 0
        for imgs, lbls, bboxes in tr_dl:
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            lbls = lbls.view(-1).float().to(device)

            opt.zero_grad(set_to_none=True)
            out = model(imgs)
            cls_out = out.cls_logit.view(-1)

            loss_cls = bce(cls_out, lbls)
            pos_mask = (lbls == 1.0)
            loss_bbox = (ciou_loss(out.bbox[pos_mask], bboxes[pos_mask])
                         if pos_mask.sum() > 0 else torch.tensor(0.0, device=device))
            loss = loss_cls + cfg.lambda_bbox * loss_bbox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            tloss += loss.item()
            n_batches += 1

        sched.step()
        avg_loss = tloss / max(n_batches, 1)

        val_res = evaluate(_NoGateAdapter(model), val_dir, device, cfg.img_size, cfg.conf_thresh,
                           eval_batch_size=cfg.eval_batch_size, dataset=val_ds)

        cur_lr = sched.get_last_lr()[0]
        print(f"  Ep {ep:3d}/{cfg.epochs} | Loss:{avg_loss:.4f} | "
              f"Val mAP@.5:{val_res.map50:.1f}% | LR:{cur_lr:.1e}", end="")

        stop = stopper.step(val_res.map50, model)
        if val_res.map50 >= stopper.best:
            torch.save(stopper.best_state or model.state_dict(), ckpt_path)
            print("  * saved", end="")
        print()

        if stop:
            print(f"  Early stopping: no improvement for {cfg.patience} epochs.")
            break

    stopper.restore(model)
    print(f"  Best Val mAP@.5: {stopper.best:.1f}%")

    print("  Evaluating test set...")
    test_res = evaluate(_NoGateAdapter(model), test_dir, device, cfg.img_size, cfg.conf_thresh,
                        eval_batch_size=cfg.eval_batch_size, dataset=test_ds)

    metrics = {
        "model": cfg.model_name, "seed": seed, "n_params": n_params, "flops": flops_str,
        "best_val_map50": stopper.best,
        "test_map50": test_res.map50, "test_map75": test_res.map75,
        "test_map50_95": test_res.map50_95, "test_precision": test_res.precision,
        "test_recall": test_res.recall, "test_f1": test_res.f1,
        "test_mean_iou": test_res.mean_iou, "n_epochs": ep,
    }
    print(f"  Test mAP@.5:{test_res.map50:.1f}%  mAP@.5:.95:{test_res.map50_95:.1f}%  "
          f"Recall:{test_res.recall:.1f}%")
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mobilenet_v3_small", "shufflenet_v2_x0_5"], required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = BaselineConfig(model_name=args.model)
    if args.quick:
        cfg.epochs, cfg.patience, cfg.seeds = 2, 1, [42]

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Device: {device}  Model: {cfg.model_name}  Seeds: {cfg.seeds}")

    per_seed = []
    best_map50, best_state = -1.0, None
    for seed in cfg.seeds:
        model, m = train_one_run(seed, cfg, device)
        per_seed.append(m)
        if m["test_map50"] > best_map50:
            best_map50, best_state = m["test_map50"], copy.deepcopy(model.state_dict())

    keys = ["test_map50", "test_map75", "test_map50_95", "test_precision",
            "test_recall", "test_f1", "test_mean_iou", "best_val_map50"]
    summary = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                   "std": float(np.std([m[k] for m in per_seed]))} for k in keys}

    torch.save(best_state, os.path.join(cfg.save_dir, f"{cfg.model_name}_best.pt"))

    out = {"model": cfg.model_name, "n_params": per_seed[0]["n_params"], "flops": per_seed[0]["flops"],
           "per_seed": per_seed, "summary": summary}
    out_path = os.path.join(cfg.save_dir, f"baseline_{cfg.model_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*65}\n  {cfg.model_name} — mean +/- std over N={len(cfg.seeds)} seeds\n{'='*65}")
    for k in keys:
        print(f"  {k:<18}: {summary[k]['mean']:6.2f} +/- {summary[k]['std']:5.2f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
