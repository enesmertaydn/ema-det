# EMA-Det: Hardware-Aware Edge Verification for Two-Stage UAV Detection

EMA-Det is a lightweight RoI-verification network for two-stage Low-Slow-Small (LSS) UAV
detection pipelines on edge hardware. Given a 128×128 candidate region cropped by an
upstream proposal stage (radar, motion detector, etc.), EMA-Det answers "is there a UAV
here, and if so, refine the box" — with a Laplacian/Sobel-initialized first layer and two
early-exit gates that let confidently-background candidates skip the rest of the network.

This repository is the Springer SIVP journal extension of a national conference paper
(SIU 2026). It supersedes the conference version's results — see
[**Corrections vs. the conference draft**](#corrections-vs-the-conference-draft) below for
why the numbers differ and why that's a good thing.

## Architecture

```
Input 1x128x128 (grayscale / thermal)
  -> Block1 (Laplacian+Sobel init, 1->16, trainable/unfrozen)  -> 16x64x64
  -> Block2 (16->32)                                            -> 32x32x32  --> Gate 1
  -> Block3 (32->64)                                            -> 64x16x16
  -> Block4 (64->128)                                           -> 128x8x8   --> Gate 2
  -> Block5 (multi-scale, dilation 1 & 2, fusion)                -> 128x4x4
  -> Block6 (128->256)                                           -> 256x4x4
  -> Verification Head: GAP -> Dropout(0.4) -> FC(1) cls, FC(4) bbox
```

Gate 1 / Gate 2 implement an asymptotic early-exit rule: a candidate halts (classified
background, no bbox refinement) once its confidence is below `1 - tau`; otherwise it
proceeds to the next block. See [`src/ema_det.py`](src/ema_det.py) for the full
implementation, including a self-test (`python src/ema_det.py`).

## Results (clean, leakage-free dataset, N=5 seeds, 150 epochs, identical protocol)

| Config | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Precision | Recall | F1 | Mean IoU | Params | FLOPs |
|---|---|---|---|---|---|---|---|---|---|
| **EMA-Det (final)** | 56.64±2.44% | 18.33±2.67% | 25.18±2.08% | 94.03±1.51% | **96.33±2.54%** | **95.14±1.51%** | 62.53±1.45% | 1.517M | 423.1M |
| EMA-Det + CBAM | 60.70±2.98% | 23.88±2.29% | 28.93±1.67% | 96.79±0.89% | 87.28±**8.85%** | 91.51±4.70% | 66.43±0.88% | 1.543M | 423.6M |

CBAM improves point-estimate mAP/IoU but roughly triples recall's run-to-run standard
deviation (2 of 5 seeds collapse to 74-79% recall) — an unacceptable reliability profile
for a safety-relevant edge verifier. That instability, not a uniform accuracy penalty, is
why the final architecture omits it. See `results/controlled_comparison.json` for the raw
per-seed data.

### Early-exit (τ) trade-off, final config

| τ | mAP@0.5 | FLOPs saved | FPS | Gate1% | Gate2% | Main% |
|---|---|---|---|---|---|---|
| 0.85 | 61.1% | 20.7% | 345.6 | 24.7% | 19.0% | 56.3% |
| 0.90 | 61.1% | 13.9% | 329.8 | 13.1% | 27.5% | 59.3% |
| 0.92 | 61.1% | 12.5% | 324.9 | 10.9% | 28.6% | 60.4% |
| 0.95 | 61.1% | 11.7% | 320.9 | 10.0% | 27.7% | 62.3% |

FPS measured single-sample (batch=1) on an RTX 4060 Laptop GPU — a proxy for the target
edge deployment, not the Jetson/FPGA numbers themselves (hardware validation is planned,
see Roadmap).

### Comparison with lightweight baselines (same clean data, N=5, identical protocol)

| Model | mAP@0.5 | mAP@0.5:0.95 | Recall | Params | FLOPs |
|---|---|---|---|---|---|
| EMA-Det (final) | 56.64±2.44% | 25.18±2.08% | **96.33±2.54%** | 1.52M | 423.1M |
| YOLOv8n (scratch) | **79.45±0.69%** | **49.59±0.45%** | 73.75±1.86% | 3.01M | 8.2G |
| MobileNetV3-Small (scratch) | 57.04±4.51% | 26.35±3.41% | 94.05±1.39% | 0.93M | 38.0M |
| ShuffleNetV2 x0.5 (scratch) | 56.26±2.11% | 25.68±1.37% | 93.80±1.11% | 0.35M | 24.9M |

YOLOv8n's mAP is nearly double EMA-Det's, but its recall is capped ~23 points lower and
tight across seeds (not noise) — the clearest concrete illustration of this project's
central argument: since system recall is
`R_total = R_proposal × R_verifier`, a verifier optimized for mAP can look stronger on the
metric everyone reports while being the weaker link for actual two-stage system
performance. The two generic lightweight backbones land close to EMA-Det on both metrics
with fewer parameters, but without EMA-Det's early-exit compute savings or its stability
edge over CBAM.

## Corrections vs. the conference draft

Two methodological issues were found and fixed while preparing this extension:

1. **CBAM/training-budget confound.** The conference draft's headline comparison table
   was produced by a checkpoint that still had CBAM wired into blocks 3-6, while the
   ablation used to justify removing CBAM was run at a much smaller budget (N=1 seed, 50
   epochs) than the main result (N=5, 150 epochs). Both variants are now retrained under
   an identical protocol — see `src/train.py`.
2. **Train/test leakage in the RoI patch dataset.** The raw source photo pool used to
   build the original dataset was 52-69% internally duplicated (consistent with
   sequential video-frame extraction), and patches were shuffled and split at the *patch*
   level rather than the *source-photo* level. Empirically, 83.8% of test-set positive
   patches were bit-identical to a training patch (8×8 average-hash, Hamming distance 0).
   `src/build_dataset_clean.py` deduplicates raw photos first (Hamming ≤ 5), then splits
   at the source level, then crops — reducing the post-hoc leakage rate to 0.6% (residual
   consistent with hash collisions on low-detail images, not real duplication). The
   dataset shrank from 23,620 patches (heavily duplicated) to 4,136 genuinely unique
   patches as a direct consequence.

Both fixes are why the numbers in this README are lower than the conference draft's —
that's the leakage/confound being removed, not a regression.

## Repository layout

```
src/
  ema_det.py                    # model backbone (Fig. 1 architecture)
  losses.py                      # CIoU loss
  dataset.py                      # RoI dataset + augmentation
  train.py                         # N=5-seed controlled training / evaluation / early-exit analysis
  build_dataset_clean.py            # leakage-free dataset construction (dedup -> source split -> crop)
  check_leakage.py                   # empirical train/test near-duplicate check
  check_raw_duplicates.py             # empirical raw-source-pool duplicate check
  pr_curve.py                          # precision-recall curve for trained checkpoints
  baselines.py / train_baseline.py      # MobileNetV3-Small / ShuffleNetV2 x0.5 comparison models
  train_yolov8n_clean.py                 # YOLOv8n scratch retrain on the clean dataset
  export_onnx.py                          # ONNX export + PyTorch/ONNX parity + latency benchmark
  export_hls_weights.py                    # extracts trained weights for hardware/hls/
  make_architecture_figure.py               # generates paper/figures/architecture.png
hardware/                                    # HLS C++ (unverified — no toolchain here, see hardware/README.md)
paper/                                        # Springer SIVP LaTeX draft (see paper/README.md)
data/                                          # (gitignored) generated by build_dataset_clean.py
results/                                        # (gitignored .pt/.log/.onnx) metrics JSON, plots
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Build the leakage-free dataset from raw source photos
python src/build_dataset_clean.py

# 2. (optional) verify no leakage
python src/check_leakage.py

# 3. Train both configs, N=5 seeds each, with early-exit analysis
python src/train.py --only both

# 4. Precision-recall curves for the trained checkpoints
python src/pr_curve.py
```

`python src/ema_det.py` runs the model's self-test (shape/gradient/gate-logic checks)
standalone, without needing a dataset.

## Roadmap

- [ ] Cross-sensor (EO/thermal-IR) generalization — blocked on sourcing a thermal UAV
      dataset; no fabricated results will be reported.
- [x] ONNX export + latency benchmark (this machine, as a proxy for edge hardware)
- [ ] Vivado HLS synthesis of Gate 1/2 + Block 1 on Zynq-7020 (LUT/BRAM/power) — requires
      a machine with the Vivado HLS toolchain; C++ source is prepared but not yet
      synthesized in this repo (see `hardware/README.md`).
- [ ] NVIDIA Jetson on-device FPS validation
- [x] Additional lightweight baselines (MobileNetV3-Small, ShuffleNetV2 x0.5) alongside
      YOLOv8n, all retrained from scratch on the clean dataset, N=5 seeds each.

## Citation

```
@article{aydin2026emadet,
  title   = {EMA-Det: Hardware-Aware Edge Verification for Two-Stage UAV Detection},
  author  = {Ayd{\i}n, Enes Mert and Toklu{\u{g}}lu, Osman and {\c{C}}avu{\c{s}}, Enver},
  journal = {Signal, Image and Video Processing},
  year    = {2026},
  note    = {Extended version of a paper presented at SIU 2026}
}
```
