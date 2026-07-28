# -*- coding: utf-8 -*-
"""
train_thermal_scratch.py -- thermal-only reference model.

Trains EMA-Det (final config: no CBAM, spatial-bbox default) from scratch
on the Anti-UAV410 thermal-IR dataset only, under the same N=5-seed /
150-epoch / early-stopping protocol used for the EO results. This is the
"upper bound" comparison point for cross_sensor_eval.py's zero-shot
EO->thermal number: how well EMA-Det performs when it actually gets to
learn the thermal domain, versus transferring untuned from EO.

Writes to results/thermal/ (separate save_dir) so it never touches the
EO checkpoints (results/emadet_final_best.pt etc).
"""
import os

from train import Config, run_controlled_comparison


def main():
    cfg = Config()
    cfg.dataset_root = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\antiuav410_clean"
    cfg.save_dir = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\thermal"
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Device: {cfg.device}  Dataset: {cfg.dataset_root}")
    print(f"Seeds: {cfg.seeds}  Epochs: {cfg.epochs}  Patience: {cfg.patience}")

    run_controlled_comparison(cfg, [False])  # final config only; no CBAM ablation needed here


if __name__ == "__main__":
    main()
