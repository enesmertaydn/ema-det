# Paper draft status

`main.tex` is a content draft of the Springer SIVP extended version, written
directly from this repository's own verified results
(`results/controlled_comparison.json`, `results/pr_curve.json`,
`results/onnx_benchmark_*.json`, `results/baseline_*_results.json`) —
nothing is carried over from the SIU conference draft without being
independently re-checked.

## What's solid
- Architecture description, loss function, corrected hyperparameters
  (weight_decay, lambda_bbox — both corrected from the conference draft's
  stated values to the code-verified ones), and Fig. 1 (generated from the
  actual implementation, see `figures/architecture.png`).
- Section 3.1 (leakage bug + fix), Section 4.1 (CBAM stability finding),
  and Section 4.2/Table III (lightweight-baseline comparison, all N=5,
  same protocol) — fully written, numbers pulled directly from this repo's
  result files. Table III's YOLOv8n row (high mAP, capped ~74% recall) is
  now the paper's clearest concrete illustration of its central
  recall-vs-mAP argument.
- Early-exit table (Section 4.3), PR curve section (4.4), hardware status
  section (4.5) — all describe only measurements that actually exist.

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
- Replace the schematic Fig. 1 with a professionally typeset version.
- A full proofread/native-English pass — this draft prioritizes getting the
  corrected technical content right over prose polish.
- Decide whether to source a thermal/IR dataset for cross-sensor
  generalization, or drop that claim from the paper's contribution list
  permanently (currently just omitted, not resolved either way).
