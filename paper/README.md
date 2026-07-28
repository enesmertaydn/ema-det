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
  actual implementation, see `architecture.png`).

**Submission file layout:** Springer's authoring template (`user-manual.pdf`
in `sn-template/`) explicitly says not to put image files in subfolders —
"the manuscript submission system will be unable to find them." Both
figures (`architecture.png`, `pr_curve.png`) live directly in `paper/`,
next to `main.tex`, not in a `figures/` subfolder (this was wrong in an
earlier draft and has been fixed). `pr_curve.png` is a manual copy of
`results/pr_curve.png` — re-copy it here if you re-run `src/pr_curve.py`
and the plot changes. `architecture.png` is written directly to this flat
location by `src/make_architecture_figure.py`, so re-running that script
needs no extra copy step.
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

- **Cross-sensor generalization section added and complete** (`sec:crosssensor`,
  no TODOs remaining): built a leakage-free thermal RoI dataset from
  Anti-UAV410 (video-level split preserved, one cross-split video excluded,
  same-frame hard negatives — 27,768/12,388/16,808 train/val/test patches,
  see `src/build_thermal_dataset.py`). Zero-shot EO→thermal transfer of the
  existing EO-trained checkpoint is measured and real
  (`results/cross_sensor_zeroshot.json`): mAP@0.5 drops from the EO
  in-domain 58.03% to 39.4%/52.1% (test/val), recall from 96.08% to
  80.1%/86.3%, while precision stays ~95.5%. The thermal-from-scratch
  reference (`src/train_thermal_scratch.py`, N=5 seeds,
  `results/thermal/controlled_comparison.json`) reaches mAP@0.5
  89.52±0.93%, recall 91.64±0.41%, precision 97.55±0.39% — same
  architecture/params/FLOPs as the EO run, confirming the zero-shot gap is
  a genuine domain-shift effect rather than an architectural ceiling, and
  suggesting thermal-IR verification is if anything an easier task than EO
  at this resolution.
- Real Springer `sn-jnl.cls` (v3.1, Dec 2024) is now in use, not the
  placeholder `article` class — see the header comment in `main.tex`.
- **Dataset-source citations resolved**: cross-checked against the SIU
  conference PDF (`Yayınlarım/emadet.pdf`, Section III-A), which explicitly
  names the two raw sources — Kaggle (`dasmehdixtr_kaggle`, rotary/fixed-wing
  UAV photos, 80% of the raw pool) and Roboflow Universe
  (`mohsen_roboflow`, bird/empty-sky hard negatives, 20%). Both are now
  cited in `sec:leakage` and the Declarations/Data-availability bullet;
  `refs.bib` has zero unused entries.

## What's NOT in this draft, on purpose

- No FPGA synthesis numbers (LUT/BRAM/latency/power) — not measured, see
  `hardware/README.md`.

## Before submission (not done here)

- Fill in author affiliations/ORCID/funding/acknowledgments (Declarations
  section placeholders in `main.tex`).
- Replace the schematic Fig. 1 with a professionally typeset version.
- A full proofread/native-English pass — this draft prioritizes getting the
  corrected technical content right over prose polish.
- Compile end-to-end (pdflatex → bibtex → pdflatex → pdflatex) on a machine
  with a LaTeX toolchain — not available in this dev environment, so
  `main.tex` has not been test-compiled.

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
