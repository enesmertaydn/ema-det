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
  -> Verification Head: GAP -> Dropout(0.4) -> FC(1) cls
  -> Localization head (final model, --center_head): 16x16 grid fused from Block3 (64ch), Block4 (128ch)
     and Block6 maps -> centre heat map + per-cell offset (2) + size (2); box = peak cell,
     score = classification probability x heat-map peak
```

The earlier spatial regression head (FC(4) on the pre-pooling 256x4x4 map) and the pooled-vector head
remain as ablations (Table 1 of the paper); `--center_head` selects the final model.

Gate 1 / Gate 2 implement an asymptotic early-exit rule: a candidate halts (classified
background, no bbox refinement) once its confidence is below `1 - tau`; otherwise it
proceeds to the next block. See [`src/ema_det.py`](src/ema_det.py) for the full
implementation, including a self-test (`python src/ema_det.py`).

## Results (clean, leakage-free dataset, N=5 seeds, 150 epochs, identical protocol)

| Config | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Precision | Recall | F1 | Mean IoU | Params | FLOPs |
|---|---|---|---|---|---|---|---|---|---|
| **EMA-Det (final, centre-based head)** | 73.58±1.63% | 38.25±9.92% | 40.60±4.99% | 92.97±1.60% | 96.46±0.81% | 94.66±0.52% | 68.56±2.81% | 1.656M | 483.8M |
| Spatial regression head | 58.03±1.60% | 24.07±2.43% | 28.04±1.90% | 95.00±0.93% | 96.08±0.47% | 95.53±0.46% | 65.70±1.46% | 1.532M | 423.1M |
| Regression head + CBAM | 57.82±2.12% | 24.77±3.88% | 28.30±2.07% | 96.91±0.43% | 93.23±1.37% | 95.03±0.72% | 66.28±1.56% | 1.558M | 423.7M |
| Final model, depthwise Blocks 2-6 | 70.98±2.40% | 33.74±6.07% | 37.15±3.13% | 93.06±0.78% | 95.76±0.99% | 94.38±0.66% | 66.71±2.00% | 0.359M | 133M |
| Final model, checkpoint chosen by YOLO's rule | 74.97±1.85% | 47.25±2.47% | 45.74±2.13% | 94.73±0.54% | 95.44±0.97% | 95.08±0.36% | 71.65±1.04% | 1.656M | 483.8M |

With the spatial regression head, CBAM's earlier mAP advantage disappears entirely: mAP is
effectively tied between the two configs. What remains is a real but milder recall cost (no seed
collapses; worst CBAM run still reaches 90.8% recall, vs. 2/5 seeds collapsing to 74-79% under the
pooled-vector head). This suggests CBAM's original benefit was substituting for spatial information
the pooled head had discarded; once that is restored, CBAM adds no localization benefit. CBAM and
initialization are therefore compared with the regression head (controlled comparisons); the final model
leaves CBAM out. See `results/controlled_comparison.json` for per-seed data.

### Localization head ablation (the reason the numbers changed)

| Head | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Recall | Params |
|---|---|---|---|---|---|
| Pooled vector (legacy) | 56.64±2.44% | 18.33±2.67% | 25.18±2.08% | 96.33±2.54% | 1.517M |
| Spatial regression head | 58.03±1.60% | 24.07±2.43% | 28.04±1.90% | 96.08±0.47% | 1.532M |
| **Centre-based head (final)** | **73.58±1.63%** | **38.25±9.92%** | **40.60±4.99%** | 96.46±0.81% | 1.656M |

The legacy head predicted the box from the GAP-pooled vector, discarding positional information
before regression. Reading the pre-pooling 256x4x4 map instead improves every metric at +15K params.
The centre-based head (`src/ema_det.py::CenterHead`, `src/losses.py::center_head_loss`) is CenterNet-style: a heat map
plus per-cell offset and size on a 16x16 grid fused from Blocks 3, 4 and 6; the detection score is the classification
probability times the heat-map peak, so it reflects localization quality. Against the regression head over five seeds:
mAP@0.5 58.03±1.60 -> 73.58±1.63% (p=0.0001), mAP@0.5:0.95 28.04±1.90 -> 40.60±4.99% (p=0.005), box recall (IoU>=0.5)
71.7 -> 77.6% (p=0.003); classification recall and AUROC unchanged. The strict-IoU metrics are unstable across seeds
(mAP@0.5:0.95 from 32.9 to 44.8%). YOLOv8n from scratch reaches 79.5% / 49.6%, COCO-pretrained 84.0% / 54.1%: the gap narrows but is not closed.
Reproduce the legacy head with `--legacy_gap_bbox`, the final model with `--center_head`.

The head was designed after `src/map_decomposition.py` (run on the test set) showed that the loss comes from box quality and from
scores that ignore it; the variant was selected on the validation set, but many variants were tried (selection bias), and the test set
was seen during that analysis. Independent checks (not used in the design): EO validation split, mAP@0.5 67.2 -> 78.4% (+11.2 points,
p=0.001, `src/eo_val_check.py`); Anti-UAV410 thermal data, five seeds trained from scratch (`scratch_runner3.sh`,
`src/thermal_variant_eval.py`): mAP@0.5:0.95 51.82±1.62 -> 56.59±0.90% (p=0.003), mAP@0.75 53.4 -> 61.8% (p=0.006), mean IoU
79.7 -> 80.7% (p=0.03), mAP@0.5 89.5 -> 90.7% (p=0.14, near its ceiling). A 32x32 fine-grid head (`--center_grid 32`) gave no further
gain. `src/domain_gap_analysis.py` explains the different gains: thermal boxes are already accurate (91% of positives have IoU>=0.5, EO 73%).

Two variants of the final model (five seeds each, `src/new_variants_eval.py`, `results/new_variants_eval.json`, paired tests against the final model):
- depthwise-separable Blocks 2-6 (`--ds_all`): 0.36M parameters, 133M FLOPs (3.6x fewer); mAP@0.5 -2.6 points (p=0.23), mAP@0.5:0.95 -3.5 points (p=0.37),
  recall -0.7 points (p=0.004). An efficiency variant, not the final model.
- external EO check (`src/build_dut_dataset.py`, `src/external_eval.py`, `results/external_dut_eval.json`): the five final seeds, zero-shot, on the
  DUT Anti-UAV detection test split (Zhao et al., IEEE T-ITS 2022; 2,043 positive + 2,043 hard-negative patches, never used in training or selection):
  recall 94.47±1.17, recall@5%FPR 90.4±1.6, AUROC 0.980±0.001, mAP@0.5 55.7±5.2, mAP@0.5:0.95 20.4±3.9, box recall 67.6±1.5 (classification holds). The mAP drop is a size effect, not a general domain shift: `src/external_eval_by_scale.py` gives mAP@0.5 2.2 / 67.3 / 89.6 / 90.4 for drones spanning <0.18 / 0.18-0.24 / 0.24-0.27 / >=0.27 of the patch side (EO training median 0.45, 10th percentile 0.27); with a 32 px instead of 128 px minimum context window (`python src/build_dut_dataset.py 32`; diagnostic chosen after this analysis, `results/external_dut_eval_floor32.json`) mAP@0.5 84.0±1.8, mAP@0.5:0.95 36.7±4.2, box recall 85.7±1.3, AUROC 0.982
- checkpoint selection by YOLOv8n's rule (`--select fitness`: 0.1 mAP@0.5 + 0.9 mAP@0.5:0.95 on the validation split instead of mAP@0.5):
  mAP@0.5:0.95 45.74±2.13% (+5.1 points, p=0.10, not significant over five seeds), mAP@0.75 47.25±2.47%, mean IoU 71.65±1.04%, box recall
  unchanged (p=0.82), recall 1.0 point lower (95.44±0.97%, p=0.012). Still below YOLOv8n (49.6% from scratch).

### Early-exit (τ) trade-off, final model

| τ | mAP@0.5 | Recall | FLOPs saved | FPS | Gate1% | Gate2% | Main% |
|---|---|---|---|---|---|---|---|
| full network | 74.0% | 97.5% | 0.0% | -- | 0.0% | 0.0% | 100.0% |
| 0.85 | 73.3% | 97.2% | 21.9% | 186.1 | 22.2% | 19.8% | 58.1% |
| 0.90 | 73.4% | 97.5% | 16.2% | 175.6 | 11.1% | 29.4% | 59.5% |
| 0.92 | 73.4% | 97.5% | 15.6% | 167.4 | 10.4% | 28.6% | 60.9% |
| 0.95 | 73.4% | 97.5% | 13.9% | 165.7 | 8.9% | 27.1% | 64.1% |

mAP and recall are those of the best-validation checkpoint (seed 7). Over the five seeds, recall goes from
96.46±0.81% (full network) to 95.95±1.01% (τ=0.85) and 96.20±1.08% (τ=0.95), and mAP@0.5 from 73.58±1.63% to 72.91±1.33% and 73.20±1.34%, i.e. early exit
costs 0.3-0.5 recall points for 13.9-21.9% fewer FLOPs (`src/earlyexit_center.py`, `results/early_exit_center_recall.json`,
`results/early_exit_center_table.json`). FLOPs savings rise and FPS rises as τ falls (a lower τ raises the halting threshold 1−τ,
so more candidates exit early).

FPS measured single-sample (batch=1) on an RTX 4060 Laptop GPU while idle — a proxy for the target
edge deployment, not the Jetson/FPGA numbers themselves (hardware validation is planned, see Roadmap).

### Comparison with lightweight baselines (same clean data, N=5, identical protocol)

Recall† = target patches accepted at score >= 0.5 (box not checked); Recall‡ = accepted with a box of IoU >= 0.5.

| Model | mAP@0.5 | mAP@0.5:0.95 | Recall† | Recall‡ | Params | FLOPs |
|---|---|---|---|---|---|---|
| **EMA-Det (final, centre-based head)** | 73.58±1.63% | 40.60±4.99% | **96.46±0.81%** | **77.59±1.29%** | 1.66M | 483.8M |
| EMA-Det, centre head, depthwise Blocks 2-6 | 70.98±2.40% | 37.15±3.13% | 95.76±0.99% | 75.25±2.38% | 0.36M | 133M |
| EMA-Det, regression head | 58.03±1.60% | 28.04±1.90% | 96.08±0.47% | 71.71±1.41% | 1.53M | 423.1M |
| YOLOv8n (scratch) | 79.45±0.69% | 49.59±0.45% | 69.94±1.64% | 62.78±1.40% | 3.01M | 8.2G |
| YOLOv8n (COCO-pretrained) | **84.02±0.61%** | **54.10±1.48%** | 77.91±2.82% | 70.76±2.26% | 3.01M | 8.2G |
| MobileNetV3-Small (scratch) | 57.04±4.51% | 26.35±3.41% | 94.05±1.39% | 68.99±2.74% | 0.93M | 38.0M |
| ShuffleNetV2 x0.5 (scratch) | 56.26±2.11% | 25.68±1.37% | 93.80±1.11% | 67.09±1.18% | 0.35M | 24.9M |

**Correction (October 2026).** Earlier versions of this README and of the manuscript compared EMA-Det's recall
(classification level, score >= 0.5, no box check, `src/train.py`) with YOLOv8n's recall as printed by Ultralytics
(box level, IoU 0.5, best-F1 threshold: 73.75±1.86% scratch, 78.17±0.57% COCO-pretrained). These are not the same
quantity. Measured under one definition and across score thresholds (`src/recall_definitions.py`,
`results/recall_definitions.json`), YOLOv8n's low recall at the default threshold is partly a threshold effect (at 0.1 it accepts
96.0% / 97.5% of the targets, but at a precision of 86.2% / 88.4% against 93.0% for EMA-Det at 0.5). At a fixed 5% false-positive rate
(`src/threshold_free_cls.py`, `src/yolo_threshold_free.py`, `src/main_model_numbers.py`), EMA-Det accepts 94.8±1.9% of the targets, against 88.4±2.8%
(COCO-pretrained YOLOv8n), 79.5±3.1% (scratch), 90.4±2.1% (MobileNetV3-Small) and 88.2±2.9% (ShuffleNetV2); AUROC
0.986 vs. 0.978 / 0.968 / 0.980 / 0.975 (p=0.07 against the COCO-pretrained model, otherwise p<0.01). With a box-overlap requirement,
EMA-Det (77.6%) is ahead of the COCO-pretrained YOLOv8n (70.8%) at a score threshold of 0.5 but behind it at 0.1 (78.7% vs. 83.2%), and its
mAP@0.5:0.95 (40.6% vs. 54.1%) is lower. So EMA-Det is the better *verifier* at a fixed false-alarm rate and a competitive but
weaker *detector*, with 17x fewer FLOPs than YOLOv8n; it needs 13x / 19x the FLOPs of MobileNetV3-Small / ShuffleNetV2.

Size ablations with the regression head (5 seeds each, `src/variants_eval.py`, `results/variants_eval.json`): depthwise-separable Blocks 4-6 (0.30M params),
Block 6 narrowed to 128 channels (0.93M) and a multi-scale box head (1.72M) show no significant difference from the regression head in any metric
(paired t-test, all p > 0.05). The final model keeps standard convolutions in Blocks 2-6 for the hardware reasons given in the paper; the depthwise
variant above trades 2.6-3.5 mAP points and 0.7 recall points for 3.6x fewer FLOPs.

### Initialization ablation: Laplacian/Sobel vs. random first layer (N=5, regression head)

| Metric (test) | Laplacian/Sobel (trainable) | Random init | p | 12 kernels frozen | p |
|---|---|---|---|---|---|
| mAP@0.5 | 58.03±1.60% | 58.08±1.78% | 0.98 | 59.11±1.37% | 0.44 |
| mAP@0.5:0.95 | 28.04±1.90% | 29.09±2.09% | 0.62 | 29.02±1.18% | 0.53 |
| Recall | 96.08±0.47% | 96.14±0.97% | 0.93 | 96.77±0.97% | 0.26 |

No measurable difference in accuracy or convergence speed: the physical prior is kept as an
interpretable starting point, not as a performance claim. Freezing the 12 fixed Laplacian/Sobel
kernels (the other 4 channels stay trainable) costs no accuracy either, so those channels could use
shifts and additions instead of multipliers (accuracy only: the multiplier-free Block 1 is not synthesized). Run with `python src/train.py --only final --random_init`,
then `src/init_ablation_exact.py` and `src/init_ablation_analysis.py` (results in `results/init_ablation*.json`).
Frozen kernels: `python src/train.py --only final --freeze_kernels`, then `python src/frozen_ablation.py eval` and `... stats`.

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
  train_yolov8n_pretrained.py             # same protocol, COCO-pretrained YOLOv8n (needs ultralytics + yolov8n.pt)
  earlyexit_recall.py                      # recall (and mAP) across the early-exit thresholds, best seed + 5 seeds
  plot_featuremaps.py                       # Block 1 feature maps (paper Fig. 2) and kernel-drift statistic
  init_ablation_exact.py / init_ablation_analysis.py   # Laplacian/Sobel vs. random-init ablation
  frozen_ablation.py                                    # frozen vs. trainable fixed kernels (eval + stats)
  recall_definitions.py / lightweight_box_recall.py      # recall under one definition: score thresholds, box overlap
  threshold_free_cls.py / yolo_threshold_free.py          # AUROC and recall at a fixed false-positive rate (EMA-Det, lightweight, YOLOv8n)
  variants_eval.py / legacy_fill.py                        # size variants (5 seeds, paired tests); legacy-head precision/F1/IoU
  export_onnx.py                          # ONNX export + PyTorch/ONNX parity + latency benchmark
  export_hls_weights.py                    # extracts trained weights for hardware/hls/
  make_architecture_figure.py               # generates paper/architecture.png
hardware/                                    # HLS C++; Block 1 synthesized for real (Vitis 2025.2), see hardware/README.md
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

# 5. Final model (centre-based head) and revision experiments
python src/train.py --only final --center_head          # final EMA-Det, 5 seeds (add --ds_all or --select fitness for the two variants)
python src/new_variants_eval.py                        # paired tests of both variants against the final model
python src/earlyexit_center.py recall && python src/earlyexit_center.py table   # early exit of the final model
python src/earlyexit_recall.py                       # recall vs. early-exit threshold
python src/train_yolov8n_pretrained.py                # COCO-pretrained YOLOv8n, 5 seeds
python src/train.py --only final --random_init        # random first-layer init, 5 seeds
python src/init_ablation_exact.py && python src/init_ablation_analysis.py
python src/plot_featuremaps.py                        # paper Fig. 2
python src/recall_definitions.py                      # recall at several thresholds / with and without box overlap
python src/threshold_free_cls.py && python src/yolo_threshold_free.py   # AUROC, recall at 5% false-positive rate
python src/train.py --only final --ds_deep --seeds 42   # size variants: also --c6 128, --fusion_bbox (one seed per run)
python src/variants_eval.py                            # 5-seed evaluation and paired tests of the variants
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
      in-domain EO recall (regression head). With the final model (five EO-trained seeds,
      `src/cross_sensor_center.py`, `results/cross_sensor_zeroshot_center.json`): recall
      79.57±2.73% (test) / 86.53±1.17% (val), precision 93.1% / 93.3%, mAP@0.5 38.34±4.93% / 51.52±5.70%,
      vs. 96.46±0.81% in-domain EO recall. Thermal-from-scratch reference of the final model
      (`scratch_runner3.sh`, `src/train.py --only final --center_head` on the thermal data, N=5 seeds): mAP@0.5 90.73±0.42%, mAP@0.5:0.95 56.59±0.90%,
      recall 91.89±0.29%, precision 97.43±0.53% — the zero-shot gap points to domain
      shift, not an architectural ceiling.
- [x] ONNX export + latency benchmark (this machine, as a proxy for edge hardware)
- [x] Vivado/Vitis HLS synthesis of Block 1 on Zynq-7020 (`xc7z020clg484-1`, Vitis
      Unified IDE 2025.2) — real csynth resource numbers for both float and
      `ap_fixed<16,6>` representations (BRAM 0%, DSP 65%, FF 10%, LUT 26% fixed-point;
      see `hardware/README.md`). C/RTL co-simulation for Block 1 and any synthesis of
      Block 2 onward (needs a time-multiplexed MAC-array redesign, not yet attempted)
      remain open.
- [ ] NVIDIA Jetson on-device FPS validation
- [x] Additional lightweight baselines (MobileNetV3-Small, ShuffleNetV2 x0.5) alongside
      YOLOv8n, all retrained from scratch on the clean dataset, N=5 seeds each.

## Citation

```
@article{aydin2026emadet,
  title   = {EMA-Det: Hardware-Aware Edge Verification for Two-Stage UAV Detection},
  author  = {Ayd{\i}n, Enes Mert and Tokluo{\u{g}}lu, Osman and {\c{C}}avu{\c{s}}, Enver},
  journal = {Signal, Image and Video Processing},
  year    = {2026},
  note    = {Extended version of a paper presented at SIU 2026}
}
```
