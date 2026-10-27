# -*- coding: utf-8 -*-
"""
train.py — Controlled reproducible training for EMA-Det.

This script exists specifically to resolve a methodological inconsistency
found while preparing the SIVP extension: the conference draft's headline
comparison (Table III: 90.9/67.7/99.2 mAP50/mAP50:95/Recall) was produced by
a checkpoint that still had CBAM wired into blocks 3/4/5/6
(`emadetforsiu/ieee_results/dynet_best_overall.pt`), while the CBAM ablation
that is used to justify removing it (Table II) was run under a *different,
much smaller* budget (N=1 seed, 50 epochs) than the main result (N=5 seeds,
150 epochs + early stopping). That mismatch confounds the effect of CBAM
with the effect of training budget, so no clean conclusion can be drawn from
the conference numbers alone.

`run_controlled_comparison()` retrains BOTH variants —
`EMADet(use_attention=False)` (final, Fig. 1) and `EMADet(use_attention=True)`
(CBAM variant, Table II row A) — under the identical N=5-seed / 150-epoch /
early-stopping protocol, so the comparison is apples-to-apples. Results
supersede the conference Table II/III numbers; see README for the write-up
note explaining the correction.
"""

import argparse
import copy
import functools
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DroneDetectDataset
from ema_det import EMADet, EMADetOutput
from losses import ciou_loss, box_iou_elementwise, center_head_loss

try:
    from thop import profile as thop_profile
    from thop import clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class Config:
    dataset_root: str = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\shuffled_dataset_detect_clean"
    save_dir:     str = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results"

    epochs:     int   = 150
    batch_size: int   = 16
    lr:         float = 1e-3
    weight_decay: float = 1e-4   # code-verified value; manuscript text says 0.01 (needs correcting, see README)

    lambda_bbox: float = 2.0     # code-verified value; manuscript Eq. 3 says 2.5 (needs correcting, see README)
    lambda_e1:   float = 0.3
    lambda_e2:   float = 0.4

    physical_init: bool = True      # False: random first-layer init (ablation only, tag "_randinit")
    c6: int = 256                   # Block-6 width (pilot: 128 = narrow variant, tag "_c6<w>")
    ds_deep: bool = False           # depthwise-separable Blocks 4-6 (pilot, tag "_dsdeep")
    aug_v2: bool = False            # label-consistent scale/rotation/translation + target-safe occlusion (tag "_augv2")
    center_grid: int = 16           # 32: fine-grid centre head for small targets (tag "_centerhead32")
    center_head: bool = False       # centre-based localization head (tag "_centerhead")
    kd_w: float = 0.0               # >0: feature distillation from a COCO YOLOv8n teacher (train only, tag "_kd")
    fusion_bbox: bool = False       # True: multi-scale box head (pilot, tag "_fusionbbox")
    freeze_kernels: bool = False    # True: keep the 12 fixed Laplacian/Sobel kernels frozen (ablation only, tag "_frozen")
    use_spatial_bbox: bool = True   # default architecture (see EMADet's docstring):
                                     # bbox head reads the pre-GAP 256x4x4 feature map.
                                     # False reproduces the legacy (pre-ablation) head.

    warmup_epochs: int = 5
    patience:      int = 20
    min_delta:     float = 1e-4

    seeds: List[int] = field(default_factory=lambda: [42, 7, 123, 2024, 31415])
    tau_values: List[float] = field(default_factory=lambda: [0.85, 0.90, 0.92, 0.95])

    img_size:    int   = 128
    num_workers: int   = 2      # Windows uses spawn; each worker gets a copy of the cached
                                 # dataset, so keep this modest rather than os.cpu_count().
    eval_batch_size: int = 256   # inference-only (no grad), safe to go large regardless of
                                 # the training batch_size above.
    conf_thresh: float = 0.5
    device:      str   = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureTeacher:
    """COCO-pretrained YOLOv8n (trained on the TRAIN split only) used for feature distillation.
    Returns its P3/P4/P5 neck outputs (64x16x16, 128x8x8, 256x4x4 at 128x128 input)."""

    def __init__(self, seed: int, device):
        from ultralytics import YOLO
        w = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "yolov8n_pretrained",
                         "yolov8n_coco_seed%d" % seed, "weights", "best.pt")
        self.m = YOLO(w).model.to(device).float().eval()
        for p_ in self.m.parameters():
            p_.requires_grad_(False)
        self.feats = {}
        for idx in (15, 18, 21):
            self.m.model[idx].register_forward_hook(lambda mod, i, o, k=idx: self.feats.__setitem__(k, o))

    @torch.no_grad()
    def __call__(self, imgs):
        self.m(imgs.repeat(1, 3, 1, 1))
        return [self.feats[15], self.feats[18], self.feats[21]]


def _kd_loss(student_feats, adapters, teacher_feats):
    """Mean over levels of 1 - cosine similarity (per location) between adapted student and teacher features."""
    tot = 0.0
    for sf, ad, tf in zip(student_feats, adapters, teacher_feats):
        a = torch.nn.functional.normalize(ad(sf), dim=1)
        b = torch.nn.functional.normalize(tf, dim=1)
        tot = tot + (1.0 - (a * b).sum(1)).mean()
    return tot / len(student_feats)


def _arch_suffix(cfg) -> str:
    """Tag suffix distinguishing architecture variants (legacy head, random init)."""
    return ("_legacygapbbox" if not cfg.use_spatial_bbox else "") + ("_randinit" if not cfg.physical_init else "") + ("_frozen" if cfg.freeze_kernels else "") + ("_fusionbbox" if cfg.fusion_bbox else "") + ("_c6%d" % cfg.c6 if cfg.c6 != 256 else "") + ("_dsdeep" if cfg.ds_deep else "") + ("_kd" if cfg.kd_w > 0 else "") + ("_augv2" if cfg.aug_v2 else "") + (("_centerhead" + ("32" if cfg.center_grid == 32 else "")) if cfg.center_head else "")


# ==============================================================================
# DETERMINISM
# ==============================================================================
def _worker_seed_fn(worker_id: int, base_seed: int):
    """
    Module-level (not a lambda/closure) so it survives pickling under
    Windows' `spawn` multiprocessing start method, required whenever
    num_workers > 0.
    """
    np.random.seed(base_seed + worker_id)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================================
# COMPLEXITY MEASUREMENT (thop)
# ==============================================================================
def measure_complexity(model: nn.Module, device: torch.device, img_size: int = 128) -> Tuple[str, str, int]:
    n_params = sum(p.numel() for p in model.parameters())
    if not THOP_AVAILABLE:
        return "N/A (thop missing)", f"{n_params/1e6:.3f}M", n_params

    dummy = torch.zeros(1, 1, img_size, img_size, device=device)
    model_copy = copy.deepcopy(model).eval()
    with torch.no_grad():
        macs, params = thop_profile(model_copy, inputs=(dummy,), verbose=False)[:2]
    del model_copy
    flops_str, params_str = clever_format([macs * 2, params], "%.3f")
    return flops_str, params_str, n_params


# ==============================================================================
# LR SCHEDULE — linear warmup + cosine annealing
# ==============================================================================
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min_ratio: float = 1e-3, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if ep < self.warmup_epochs:
                lr = base_lr * (ep + 1) / max(self.warmup_epochs, 1)
            else:
                progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                lr = base_lr * self.eta_min_ratio + 0.5 * base_lr * (1 - self.eta_min_ratio) * (
                    1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs


# ==============================================================================
# EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.best_state = -float("inf"), 0, None

    def step(self, metric: float, model: nn.Module) -> bool:
        if metric > self.best + self.min_delta:
            self.best, self.counter = metric, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ==============================================================================
# EVALUATION
# ==============================================================================
@dataclass
class EvalResult:
    map50: float = 0.0
    map75: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0


def _cxcywh_to_xyxy(boxes: torch.Tensor, img_size: int = 1) -> torch.Tensor:
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_size
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_size
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_size
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_size
    return torch.stack([x1, y1, x2, y2], dim=1).clamp(0, img_size)


def _accumulate_batch(metric, tp, fp, fn, iou_sum, iou_cnt, scores: torch.Tensor,
                       pred_bbox: torch.Tensor, lbls: torch.Tensor, gt_bbox: torch.Tensor,
                       img_size: int, conf_thresh: float, has_pred: Optional[torch.Tensor] = None,
                       map_scores: Optional[torch.Tensor] = None):
    """
    Vectorized per-batch bookkeeping shared by both evaluate() paths.

    has_pred: (B,) bool mask, default all-True. False marks a sample that
    halted at Gate 1/2 with no bbox at all — it must be submitted to
    torchmetrics as an EMPTY detection (not a zero-box prediction with the
    gate's score), and can never count as a positive prediction for P/R/F1.
    """
    if has_pred is None:
        has_pred = torch.ones_like(lbls, dtype=torch.bool)

    gt_pos = lbls == 1.0
    pred_pos = (scores >= conf_thresh) & has_pred

    pred_xyxy = _cxcywh_to_xyxy(pred_bbox, img_size)
    gt_xyxy = _cxcywh_to_xyxy(gt_bbox, img_size)

    preds_tm, targets_tm = [], []
    for i in range(scores.size(0)):
        if has_pred[i]:
            preds_tm.append({"boxes": pred_xyxy[i:i+1], "scores": (map_scores if map_scores is not None else scores)[i:i+1], "labels": torch.tensor([0])})
        else:
            preds_tm.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
        if gt_pos[i]:
            targets_tm.append({"boxes": gt_xyxy[i:i+1], "labels": torch.tensor([0])})
        else:
            targets_tm.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)})
    metric.update(preds_tm, targets_tm)

    tp_mask = pred_pos & gt_pos
    fp += int((pred_pos & ~gt_pos).sum().item())
    fn += int((~pred_pos & gt_pos).sum().item())
    tp += int(tp_mask.sum().item())
    if tp_mask.any():
        ious = box_iou_elementwise(pred_bbox[tp_mask], gt_bbox[tp_mask])
        iou_sum += float(ious.sum().item())
        iou_cnt += int(tp_mask.sum().item())
    return tp, fp, fn, iou_sum, iou_cnt


def evaluate(model: nn.Module, data_dir: str, device: torch.device, img_size: int = 128,
             conf_thresh: float = 0.5, early_exit_tau: Optional[float] = None,
             eval_batch_size: int = 256, num_workers: int = 0,
             dataset: Optional[DroneDetectDataset] = None) -> EvalResult:
    """
    tau=None (used every epoch during training): batched, GPU-resident path —
    the dataset's cached/resized images are loaded through a DataLoader so
    the GPU isn't starved waiting on per-image disk reads.
    tau=<float> (early-exit analysis only, run a handful of times total):
    single-sample loop, required by EMADet.forward's batch=1 tau-gating.

    Pass a pre-built `dataset` (e.g. the val set) when calling this every
    epoch — otherwise a fresh DroneDetectDataset is constructed per call,
    which re-runs its image-caching pass from disk every single epoch and
    defeats the point of caching.
    """
    if not TORCHMETRICS_AVAILABLE:
        raise RuntimeError("torchmetrics is required for evaluate(); pip install torchmetrics")

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    tp = fp = fn = 0
    iou_sum, iou_cnt = 0.0, 0

    if early_exit_tau is None:
        ds = dataset if dataset is not None else DroneDetectDataset(
            data_dir, augment=False, img_size=img_size, cache_images=True)
        dl = DataLoader(ds, batch_size=eval_batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device.type == "cuda"))
        with torch.no_grad():
            for imgs, lbls, bboxes in dl:
                imgs, lbls, bboxes = imgs.to(device), lbls.to(device), bboxes.to(device)
                out: EMADetOutput = model(imgs)  # tau=None -> full batch, bbox always present
                scores = torch.sigmoid(out.cls_logit)
                tp, fp, fn, iou_sum, iou_cnt = _accumulate_batch(
                    metric, tp, fp, fn, iou_sum, iou_cnt, scores, out.bbox, lbls, bboxes,
                    img_size, conf_thresh, map_scores=out.det_score)
    else:
        img_paths = sorted(
            glob.glob(os.path.join(data_dir, "images", "*.jpg")) +
            glob.glob(os.path.join(data_dir, "images", "*.png"))
        )
        lbl_dir = os.path.join(data_dir, "labels")
        with torch.no_grad():
            for p in img_paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                lp = os.path.join(lbl_dir, stem + ".txt")
                gt_bbox_np, lbl_val = None, 0.0
                if os.path.exists(lp) and os.path.getsize(lp) > 0:
                    parts = open(lp).readline().strip().split()
                    if len(parts) >= 5:
                        gt_bbox_np = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
                        lbl_val = 1.0

                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                inp = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

                out: EMADetOutput = model(inp, tau=early_exit_tau)
                score_t = torch.sigmoid(out.cls_logit)
                has_pred_t = torch.tensor([out.bbox is not None], device=device)
                bbox_t = out.bbox if out.bbox is not None else torch.zeros(1, 4, device=device)
                lbl_t = torch.tensor([lbl_val], device=device)
                gt_t = torch.tensor(gt_bbox_np if gt_bbox_np is not None else np.zeros(4, dtype=np.float32),
                                     device=device).unsqueeze(0)
                tp, fp, fn, iou_sum, iou_cnt = _accumulate_batch(
                    metric, tp, fp, fn, iou_sum, iou_cnt, score_t, bbox_t, lbl_t, gt_t,
                    img_size, conf_thresh, has_pred=has_pred_t)

    tm_res = metric.compute()
    map50    = float(tm_res.get("map_50", torch.tensor(0.0))) * 100
    map75    = float(tm_res.get("map_75", torch.tensor(0.0))) * 100
    map50_95 = float(tm_res.get("map", torch.tensor(0.0))) * 100

    precision = tp / (tp + fp + 1e-7) * 100
    recall    = tp / (tp + fn + 1e-7) * 100
    f1        = 2 * precision * recall / (precision + recall + 1e-7)
    mean_iou  = iou_sum / (iou_cnt + 1e-7) * 100

    return EvalResult(map50, map75, map50_95, precision, recall, f1, mean_iou)


# ==============================================================================
# SINGLE-SEED / SINGLE-CONFIG TRAINING
# ==============================================================================
def train_one_run(seed: int, use_attention: bool, cfg: Config, device: torch.device) -> Tuple[nn.Module, Dict]:
    tag = ("cbam" if use_attention else "final") + _arch_suffix(cfg)
    print(f"\n{'='*70}\n  config={tag}  seed={seed}\n{'='*70}")

    set_seed(seed)
    model = EMADet(use_attention=use_attention, use_spatial_bbox=cfg.use_spatial_bbox,
                   physical_init=cfg.physical_init, fusion_bbox=cfg.fusion_bbox, c6=cfg.c6, ds_deep=cfg.ds_deep, center_head=cfg.center_head, center_grid=cfg.center_grid).to(device)

    flops_str, params_str, n_params = measure_complexity(model, device, cfg.img_size)
    print(f"  Params: {params_str} ({n_params:,})   FLOPs: {flops_str}")

    train_dir = os.path.join(cfg.dataset_root, "train")
    val_dir   = os.path.join(cfg.dataset_root, "val")
    test_dir  = os.path.join(cfg.dataset_root, "test")

    tr_dl = DataLoader(
        DroneDetectDataset(train_dir, augment=True, img_size=cfg.img_size, cache_images=True, aug_v2=cfg.aug_v2),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=functools.partial(_worker_seed_fn, base_seed=seed),
    )
    # Built once and reused every epoch/at test time — evaluate()'s internal
    # DataLoader wraps these directly, so val images are cached in RAM exactly
    # once rather than re-read from disk on every one of the 150 epochs.
    val_ds  = DroneDetectDataset(val_dir, augment=False, img_size=cfg.img_size, cache_images=True)
    test_ds = DroneDetectDataset(test_dir, augment=False, img_size=cfg.img_size, cache_images=True)

    bce = nn.BCEWithLogitsLoss()
    if cfg.freeze_kernels:
        model.block1.freeze_fixed_kernels()
    teacher, adapters = None, None
    if cfg.kd_w > 0:
        teacher = FeatureTeacher(42, device)          # same fixed teacher for every student seed
        adapters = nn.ModuleList([nn.Conv2d(64, 64, 1), nn.Conv2d(128, 128, 1), nn.Conv2d(model.c6, 256, 1)]).to(device)
    params = list(model.parameters()) + (list(adapters.parameters()) if adapters is not None else [])
    opt = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = WarmupCosineScheduler(opt, cfg.warmup_epochs, cfg.epochs)
    stopper = EarlyStopping(cfg.patience, cfg.min_delta)

    ckpt_path = os.path.join(cfg.save_dir, f"emadet_{tag}_seed{seed}.pt")
    history = {"train_loss": [], "val_map50": []}

    ep = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tloss = tloss_cls = tloss_bbox = tloss_e1 = tloss_e2 = 0.0
        n_batches = 0

        for imgs, lbls, bboxes in tr_dl:
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            lbls = lbls.view(-1).float().to(device)

            opt.zero_grad(set_to_none=True)
            out: EMADetOutput = model(imgs)  # tau=None -> full network, all four heads

            cls_out = out.cls_logit.view(-1)
            e1_out  = out.gate1_logit.view(-1)
            e2_out  = out.gate2_logit.view(-1)

            loss_cls = bce(cls_out, lbls)
            pos_mask = (lbls == 1.0)
            if cfg.center_head:
                l_hm, l_cbox = center_head_loss(out.aux["hm"], out.aux["reg"], bboxes, pos_mask)
                loss_bbox = l_cbox + l_hm / max(cfg.lambda_bbox, 1e-6)      # heat-map term enters with weight 1 after the lambda_bbox factor
            else:
                loss_bbox = (ciou_loss(out.bbox[pos_mask], bboxes[pos_mask])
                             if pos_mask.sum() > 0 else torch.tensor(0.0, device=device))
            loss_e1 = bce(e1_out, lbls)
            loss_e2 = bce(e2_out, lbls)

            loss = loss_cls + cfg.lambda_bbox * loss_bbox + cfg.lambda_e1 * loss_e1 + cfg.lambda_e2 * loss_e2
            if teacher is not None:
                loss = loss + cfg.kd_w * _kd_loss([model._p3, model._p4, model._p5], adapters, teacher(imgs))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if cfg.freeze_kernels:
                model.block1.restore_frozen()

            tloss += loss.item(); tloss_cls += loss_cls.item(); tloss_bbox += loss_bbox.item()
            tloss_e1 += loss_e1.item(); tloss_e2 += loss_e2.item(); n_batches += 1

        sched.step()
        avg_loss = tloss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        val_res = evaluate(model, val_dir, device, cfg.img_size, cfg.conf_thresh,
                           eval_batch_size=cfg.eval_batch_size, dataset=val_ds)
        history["val_map50"].append(val_res.map50)
        cur_lr = sched.get_last_lr()[0]

        print(f"  Ep {ep:3d}/{cfg.epochs} | Loss:{avg_loss:.4f} "
              f"(cls:{tloss_cls/n_batches:.3f} bbox:{tloss_bbox/n_batches:.3f} "
              f"e1:{tloss_e1/n_batches:.3f} e2:{tloss_e2/n_batches:.3f}) | "
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
    test_res = evaluate(model, test_dir, device, cfg.img_size, cfg.conf_thresh,
                        eval_batch_size=cfg.eval_batch_size, dataset=test_ds)

    metrics = {
        "config": tag, "seed": seed, "n_params": n_params, "flops": flops_str,
        "best_val_map50": stopper.best,
        "test_map50": test_res.map50, "test_map75": test_res.map75,
        "test_map50_95": test_res.map50_95, "test_precision": test_res.precision,
        "test_recall": test_res.recall, "test_f1": test_res.f1,
        "test_mean_iou": test_res.mean_iou, "n_epochs": ep, "history": history,
    }
    print(f"  Test mAP@.5:{test_res.map50:.1f}%  mAP@.75:{test_res.map75:.1f}%  "
          f"mAP@.5:.95:{test_res.map50_95:.1f}%  Recall:{test_res.recall:.1f}%")
    return model, metrics


# ==============================================================================
# CONTROLLED COMPARISON — the actual fix for the CBAM/budget confound
# ==============================================================================
def run_controlled_comparison(cfg: Config, configs_to_run: List[bool]) -> Dict:
    """
    Runs both use_attention in {False, True} (or a subset via configs_to_run)
    under the identical N=5-seed / cfg.epochs / early-stopping protocol, so
    any difference between them is attributable to CBAM itself, not budget.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    all_results = {}

    for use_attention in configs_to_run:
        tag = ("cbam" if use_attention else "final") + _arch_suffix(cfg)
        per_seed, best_map50, best_state = [], -1.0, None

        for seed in cfg.seeds:
            model, m = train_one_run(seed, use_attention, cfg, torch.device(cfg.device))
            per_seed.append(m)
            if m["test_map50"] > best_map50:
                best_map50, best_state = m["test_map50"], copy.deepcopy(model.state_dict())

        keys = ["test_map50", "test_map75", "test_map50_95", "test_precision",
                "test_recall", "test_f1", "test_mean_iou", "best_val_map50"]
        summary = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                       "std": float(np.std([m[k] for m in per_seed]))} for k in keys}

        best_path = os.path.join(cfg.save_dir, f"emadet_{tag}_best.pt")
        torch.save(best_state, best_path)

        all_results[tag] = {"per_seed": per_seed, "summary": summary, "best_ckpt": best_path,
                             "n_params": per_seed[0]["n_params"], "flops": per_seed[0]["flops"]}
        print_summary(tag, summary)

    out_name = "controlled_comparison" + _arch_suffix(cfg) + ".json"
    out_path = os.path.join(cfg.save_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Both configs trained under identical N={}-seed / {}-epoch / "
                     "early-stopping(patience={}) protocol — supersedes the conference "
                     "Table II/III numbers, which were confounded by mismatched budgets."
                     ).format(len(cfg.seeds), cfg.epochs, cfg.patience),
            "config": {"seeds": cfg.seeds, "epochs": cfg.epochs, "patience": cfg.patience,
                       "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                       "lambda_bbox": cfg.lambda_bbox, "lambda_e1": cfg.lambda_e1, "lambda_e2": cfg.lambda_e2},
            "results": {tag: {"summary": r["summary"], "n_params": r["n_params"], "flops": r["flops"],
                              "per_seed": r["per_seed"]}
                        for tag, r in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved controlled comparison -> {out_path}")
    return all_results


def print_summary(tag: str, summary: Dict):
    print(f"\n{'='*65}\n  {tag.upper()} — mean ± std over N seeds\n{'='*65}")
    labels = [("Test mAP@0.5", "test_map50"), ("Test mAP@0.75", "test_map75"),
              ("Test mAP@.5:.95", "test_map50_95"), ("Precision", "test_precision"),
              ("Recall", "test_recall"), ("F1", "test_f1"), ("Mean IoU", "test_mean_iou")]
    for label, key in labels:
        m, s = summary[key]["mean"], summary[key]["std"]
        print(f"  {label:<18}: {m:6.2f} +/- {s:5.2f} %")


# ==============================================================================
# EARLY-EXIT (tau) ANALYSIS — reuses the model's own submodules for thop
# ==============================================================================
class _Gate1Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.gate1(self.m._stem_to_gate1(x))


class _Gate2Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.gate2(self.m._gate1_to_gate2(x))


class _MainWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        feat_map = self.m._gate2_to_feature_map(x)
        pooled = self.m._pool_and_dropout(feat_map)
        if self.m.center_head:
            hm, reg = self.m.ctr(self.m._p3, self.m._p4, feat_map, getattr(self.m, '_p2', None))
            return torch.cat([self.m.cls_head(pooled), hm.flatten(1), reg.flatten(1)], dim=1)
        return torch.cat([self.m.cls_head(pooled), self.m.bbox_head(self.m._bbox_input(feat_map, pooled))], dim=1)


def measure_block_flops(model: nn.Module, device: torch.device, img_size: int = 128) -> Tuple[float, float, float]:
    if not THOP_AVAILABLE:
        print("[WARN] thop not available; falling back to fixed 0.25/0.50 ratios.")
        return 0.25, 0.50, 1.0

    model = copy.deepcopy(model).eval()
    dummy = torch.zeros(1, 1, img_size, img_size, device=device)

    macs_g1 = thop_profile(_Gate1Wrapper(model).to(device), inputs=(dummy,), verbose=False)[0]
    with torch.no_grad():
        f1 = model._stem_to_gate1(dummy)
    macs_g2 = thop_profile(_Gate2Wrapper(model).to(device), inputs=(f1,), verbose=False)[0]
    with torch.no_grad():
        f2 = model._gate1_to_gate2(f1)
    macs_gm = thop_profile(_MainWrapper(model).to(device), inputs=(f2,), verbose=False)[0]

    total = macs_g1 + macs_g2 + macs_gm + 1e-9
    r1 = macs_g1 / total
    r2 = (macs_g1 + macs_g2) / total
    return r1, r2, 1.0


def analyze_early_exit_table(model: nn.Module, cfg: Config) -> List[Dict]:
    device = torch.device(cfg.device)
    r1, r2, r3 = measure_block_flops(model, device, cfg.img_size)
    test_dir = os.path.join(cfg.dataset_root, "test")
    img_paths = sorted(glob.glob(os.path.join(test_dir, "images", "*.jpg")) +
                        glob.glob(os.path.join(test_dir, "images", "*.png")))

    rows = []
    model.eval()
    for tau in cfg.tau_values:
        res = evaluate(model, test_dir, device, cfg.img_size, cfg.conf_thresh, early_exit_tau=tau)

        c1 = c2 = cm = 0
        times = []
        n_warmup = 50
        with torch.no_grad():
            for i, p in enumerate(img_paths):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (cfg.img_size, cfg.img_size))
                inp = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(inp, tau=tau)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                if out.exit_stage == "gate1": c1 += 1
                elif out.exit_stage == "gate2": c2 += 1
                else: cm += 1

                if i >= n_warmup:
                    times.append(time.perf_counter() - t0)

        total = c1 + c2 + cm + 1e-7
        flops_ratio = (c1 * r1 + c2 * r2 + cm * r3) / total
        saving_pct = (1 - flops_ratio) * 100
        fps = 1.0 / np.mean(times) if times else 0.0

        rows.append({"tau": tau, "map50": res.map50, "saving_%": saving_pct, "fps": fps,
                      "gate1_%": c1 / total * 100, "gate2_%": c2 / total * 100, "main_%": cm / total * 100,
                      "flops_r1": round(r1, 4), "flops_r2": round(r2, 4)})
    return rows


def print_early_exit_table(rows: List[Dict]):
    print(f"\n{'='*70}\n  EARLY-EXIT TABLE\n{'='*70}")
    print(f"  {'tau':>6} {'mAP@.5':>9} {'FLOPs saved':>13} {'FPS':>8} | "
          f"{'Gate1%':>7} {'Gate2%':>7} {'Main%':>7}")
    for r in rows:
        print(f"  {r['tau']:>6.2f} {r['map50']:>8.1f}% {r['saving_%']:>12.1f}% {r['fps']:>8.1f} | "
              f"{r['gate1_%']:>6.1f}% {r['gate2_%']:>6.1f}% {r['main_%']:>6.1f}%")


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EMA-Det controlled training / CBAM-budget-fix comparison")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated, e.g. 42,7,123,2024,31415")
    parser.add_argument("--only", choices=["final", "cbam", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 2 epochs")
    parser.add_argument("--random_init", action="store_true",
                        help="ablation: random (Kaiming-uniform) first-layer init instead of Laplacian/Sobel")
    parser.add_argument("--freeze_kernels", action="store_true",
                        help="ablation: keep the 12 fixed Laplacian/Sobel kernels frozen during training")
    parser.add_argument("--legacy_gap_bbox", action="store_true",
                        help="reproduce the pre-ablation architecture (bbox head on the "
                             "GAP-pooled vector) instead of the default spatial bbox head")
    parser.add_argument("--fusion_bbox", action="store_true", help="pilot: multi-scale (16x16 + 4x4) box head")
    parser.add_argument("--c6", type=int, default=256, help="pilot: Block-6 output width")
    parser.add_argument("--ds_deep", action="store_true", help="pilot: depthwise-separable Blocks 4-6")
    parser.add_argument("--center_grid", type=int, default=16, choices=[16, 32], help="grid of the centre head (32: finer, for small targets)")
    parser.add_argument("--center_head", action="store_true", help="pilot: centre-based localization head (heat map + offset + size)")
    parser.add_argument("--kd_w", type=float, default=0.0, help="pilot: feature-distillation weight (0 = off)")
    parser.add_argument("--aug_v2", action="store_true", help="pilot: label-consistent geometric augmentation")
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_root: cfg.dataset_root = args.dataset_root
    if args.save_dir: cfg.save_dir = args.save_dir
    if args.epochs: cfg.epochs = args.epochs
    if args.patience: cfg.patience = args.patience
    if args.seeds: cfg.seeds = [int(s) for s in args.seeds.split(",")]
    cfg.use_spatial_bbox = not args.legacy_gap_bbox
    cfg.physical_init = not args.random_init
    cfg.freeze_kernels = args.freeze_kernels
    cfg.fusion_bbox = args.fusion_bbox
    cfg.c6 = args.c6
    cfg.kd_w = args.kd_w
    cfg.center_head = args.center_head
    cfg.center_grid = args.center_grid
    cfg.aug_v2 = args.aug_v2
    cfg.ds_deep = args.ds_deep

    if args.quick:
        cfg.epochs, cfg.patience, cfg.seeds = 2, 1, [42]

    configs_to_run = {"final": [False], "cbam": [True], "both": [False, True]}[args.only]

    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"Device: {cfg.device}  Dataset: {cfg.dataset_root}")
    print(f"Seeds: {cfg.seeds}  Epochs: {cfg.epochs}  Patience: {cfg.patience}")

    results = run_controlled_comparison(cfg, configs_to_run)

    if False in configs_to_run:
        final_tag = "final" + _arch_suffix(cfg)
        best_final = EMADet(use_attention=False, use_spatial_bbox=cfg.use_spatial_bbox,
                            physical_init=cfg.physical_init, fusion_bbox=cfg.fusion_bbox, c6=cfg.c6, ds_deep=cfg.ds_deep, center_head=cfg.center_head, center_grid=cfg.center_grid).to(torch.device(cfg.device))
        best_final.load_state_dict(torch.load(results[final_tag]["best_ckpt"], map_location=cfg.device))
        ee_rows = analyze_early_exit_table(best_final, cfg)
        print_early_exit_table(ee_rows)
        ee_suffix = _arch_suffix(cfg)
        ee_path = os.path.join(cfg.save_dir, f"early_exit_table{ee_suffix}.json")
        with open(ee_path, "w", encoding="utf-8") as f:
            json.dump(ee_rows, f, indent=2)


if __name__ == "__main__":
    main()
