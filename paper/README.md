# Paper draft status

`main.tex` is a content draft of the Springer SIVP extended version, written
directly from this repository's own verified results
(`results/controlled_comparison.json`, `results/pr_curve.json`,
`results/onnx_benchmark_*.json`, `results/baseline_*_results.json`) —
nothing is carried over from the SIU conference draft without being
independently re-checked.

## What's solid — all numbers real, no TODOs remaining in the source

- Architecture description, loss function, corrected hyperparameters
  (weight_decay, lambda_bbox — both corrected from the conference draft's
  stated values to the code-verified ones), and Fig. 1 (generated from the
  actual implementation, see `figures/architecture.png`).
- **Spatial bbox head ablation** (Section on `sec:spatialbbox`, new this
  round): decoupling the box head from the pooled classification feature so
  it reads the pre-GAP 256x4x4 map instead strictly improved every tracked
  metric at negligible cost (N=5). This is now the default architecture
  (`use_spatial_bbox=True` in `ema_det.py`); the legacy pooled-vector head
  is still reproducible via `--legacy_gap_bbox`.
- **CBAM comparison, re-run against the new default** (Table in
  `sec:cbam`): with the spatial bbox head in place, CBAM's mAP advantage
  from the legacy-head comparison *disappears entirely* (mAP@0.5:0.95
  effectively tied), while a real but much milder recall cost remains (no
  seed collapses under the new head, vs. 2/5 collapsing to 74-79% under the
  old one). Read together with the bbox-head ablation, this suggests CBAM's
  original benefit was substituting for spatial information the pooled head
  had discarded — a materially different (and more mechanistically
  satisfying) explanation than either of the two earlier corrections gave.
- Section 3.1 (leakage bug + fix), Section 4.2/Table III (lightweight-
  baseline comparison) — unaffected by the bbox-head change, still current.
- Early-exit table, PR curves (AP 0.9859/0.9886, both configs, both current
  checkpoints), hardware status section (ONNX numbers for both configs) —
  all regenerated against the final checkpoints, no stale numbers left.

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

## Note on a mid-run CUDA crash

The CBAM N=5 retrain hit a transient CUDA "unspecified launch failure" at
epoch 62/150 of seed 2024 (GPU responded normally afterward via
`nvidia-smi` — consistent with a driver hiccup under sustained load, not a
code bug or persistent hardware fault). Recovered via
`src/recover_cbam_run.py`: the 3 seeds that finished cleanly were
re-evaluated from their saved checkpoints (not re-trained, since they
completed correctly) to recover the full metric set, and the 2 affected
seeds were retrained from scratch. Results merged into the same
`controlled_comparison.json` used throughout.
