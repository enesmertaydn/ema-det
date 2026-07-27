# Paper draft status

`main.tex` is a content draft of the Springer SIVP extended version, written
directly from this repository's own verified results
(`results/controlled_comparison.json`, `results/pr_curve.json`,
`results/onnx_benchmark_*.json`) — nothing is carried over from the SIU
conference draft without being independently re-checked.

## What's solid
- Architecture description, loss function, corrected hyperparameters
  (weight_decay, lambda_bbox — both corrected from the conference draft's
  stated values to the code-verified ones).
- Section 3.1 (leakage bug + fix) and Section 4.1 (CBAM stability finding):
  fully written, numbers pulled directly from this repo's result files.
- Early-exit table (Section 4.3), PR curve section (4.4), hardware status
  section (4.5) — all describe only measurements that actually exist.

## What's explicitly marked TODO in the source (search for "TODO")
- **Table III** (Section 3.2/4.2): MobileNetV3-Small / ShuffleNetV2 x0.5 /
  YOLOv8n rows — training in progress at draft time
  (`results/baseline_*_results.json` once complete).
- **Abstract**: references the still-pending Table III comparison.

## What's NOT in this draft, on purpose
- No cross-sensor (EO/thermal-IR) section — no thermal dataset was
  available; the original conference plan's claim is not repeated here
  without data to back it.
- No FPGA synthesis numbers (LUT/BRAM/latency/power) — not measured, see
  `hardware/README.md`.

## Before submission (not done here)
- Swap `\documentclass{article}` for Springer Nature's actual `sn-jnl.cls`
  (obtain from the SIVP submission portal).
- Fill in author affiliations/ORCID/funding/acknowledgments.
- Fill Table III once baseline training finishes.
- Add the architecture figure (Fig. 1) — referenced but not embedded here.
- A full proofread/native-English pass — this draft prioritizes getting the
  corrected technical content right over prose polish.
