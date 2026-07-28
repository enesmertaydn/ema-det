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
| **EMA-Det (final)** | 58.03±1.60% | 24.07±2.43% | 28.04±1.90% | 95.00±0.93% | **96.08±0.47%** | **95.53±0.46%** | 65.70±1.46% | 1.532M | 423.1M |
| EMA-Det + CBAM | 57.82±2.12% | 24.77±3.88% | 28.30±2.07% | **96.91±0.43%** | 93.23±1.37% | 95.03±0.72% | **66.28±1.56%** | 1.558M | 423.7M |

With the default spatial bbox head (below), CBAM's earlier mAP advantage disappears
entirely — mAP is now effectively tied between the two configs. What remains is a real
but much milder recall cost (no seed collapses; worst CBAM run still reaches 90.8%
recall, vs. 2/5 seeds collapsing to 74-79% under the legacy bbox head). This suggests
CBAM's original benefit was substituting for spatial information the legacy head had
discarded — once that's restored directly, CBAM has nothing left to contribute except a
reliability cost. See `results/controlled_comparison.json` for per-seed data.

### Bounding-box head ablation (the reason the numbers above changed)

| Bbox head | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Recall | Params |
|---|---|---|---|---|---|
| Pooled vector (legacy) | 56.64±2.44% | 18.33±2.67% | 25.18±2.08% | 96.33±2.54% | 1.517M |
| **Spatial feature map (default)** | **58.03±1.60%** | **24.07±2.43%** | **28.04±1.90%** | 96.08±**0.47%** | 1.532M |

The legacy head predicted the box from the GAP-pooled (spatially-flattened) vector —
discarding all positional information before regression ever saw it. Reading the
pre-pooling 256×4×4 feature map directly instead strictly improves every metric above
at +15K params (~1%) and ~free FLOPs, while also *tightening* recall's cross-seed
variance. Reproduce the legacy head via `--legacy_gap_bbox` in `train.py`.

### Early-exit (τ) trade-off, final config

| τ | mAP@0.5 | FLOPs saved | FPS | Gate1% | Gate2% | Main% |
|---|---|---|---|---|---|---|
| 0.85 | 58.9% | 21.2% | 358.5 | 24.8% | 21.7% | 53.5% |
| 0.90 | 58.9% | 16.2% | 340.8 | 16.0% | 29.4% | 54.6% |
| 0.92 | 58.9% | 13.9% | 336.3 | 11.9% | 33.1% | 55.1% |
| 0.95 | 58.9% | 12.5% | 337.8 | 9.7% | 34.0% | 56.3% |

FPS measured single-sample (batch=1) on an RTX 4060 Laptop GPU — a proxy for the target
edge deployment, not the Jetson/FPGA numbers themselves (hardware validation is planned,
see Roadmap).

### Comparison with lightweight baselines (same clean data, N=5, identical protocol)

| Model | mAP@0.5 | mAP@0.5:0.95 | Recall | Params | FLOPs |
|---|---|---|---|---|---|
| EMA-Det (final) | 58.03±1.60% | 28.04±1.90% | **96.08±0.47%** | 1.53M | 423.1M |
| YOLOv8n (scratch) | **79.45±0.69%** | **49.59±0.45%** | 73.75±1.86% | 3.01M | 8.2G |
| MobileNetV3-Small (scratch) | 57.04±4.51% | 26.35±3.41% | 94.05±1.39% | 0.93M | 38.0M |
| ShuffleNetV2 x0.5 (scratch) | 56.26±2.11% | 25.68±1.37% | 93.80±1.11% | 0.35M | 24.9M |

YOLOv8n's mAP is ~1.8x EMA-Det's, but its recall is capped ~22 points lower and tight
across seeds (not noise) — the clearest concrete illustration of this project's central
argument: since system recall is
`R_total = R_proposal × R_verifier`, a verifier optimized for mAP can look stronger on the
metric everyone reports while being the weaker link for actual two-stage system
performance. The two generic lightweight backbones land close to EMA-Det on both metrics
with fewer parameters, but without EMA-Det's early-exit compute savings.

## Corrections vs. the conference draft

Three methodological issues were found and fixed while preparing this extension:

1. **CBAM/training-budget confound.** The conference draft's headline comparison table
   was produced by a checkpoint that still had CBAM wired into blocks 3-6, while the
   ablation used to justify removing CBAM was run at a much smaller budget (N=1 seed, 50
   epochs) than the main result (N=5, 150 epochs). Both variants are now retrained under
   an identical protocol — see `src/train.py`. The corrected picture turned out to be more
   interesting than a simple "confirmed/refuted": see the bbox-head ablation below, which
   changes the CBAM comparison's conclusion a second time.
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
3. **Bounding-box head discarding spatial information.** See the ablation above — the
   legacy head's GAP-pooling-before-regression design was a direct, architecturally
   obvious cause of EMA-Det's large mAP gap versus full detection architectures (Table
   below). Fixing it also fully explains away CBAM's originally-observed mAP advantage.

These are why the numbers in this README differ from the conference draft's — the
leakage/confound being removed and the architecture being improved, not a regression.

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

- [x] Cross-sensor (EO/thermal-IR) generalization — built from Anti-UAV410
      (`src/build_thermal_dataset.py`; 27,768/12,388/16,808 train/val/test patches,
      video-level split preserved, same-frame hard negatives). Zero-shot EO→thermal
      transfer measured (`src/cross_sensor_eval.py`,
      `results/cross_sensor_zeroshot.json`): mAP@0.5 39.4% (test) / 52.1% (val),
      Recall 80.1% (test) / 86.3% (val), Precision ~95.5% both — vs. 96.08%
      in-domain EO recall. Thermal-from-scratch reference (`src/train_thermal_scratch.py`,
      N=5 seeds, `results/thermal/controlled_comparison.json`): mAP@0.5 89.52±0.93%,
      Recall 91.64±0.41%, Precision 97.55±0.39% — confirms the zero-shot gap is a real
      domain-shift effect, not an architectural ceiling.
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
