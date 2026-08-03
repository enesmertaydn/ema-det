# -*- coding: utf-8 -*-
"""
plot_earlyexit_tradeoff.py — Fig. for the paper's Early-Exit Trade-off
section (Table 4 / tab:earlyexit in main.tex).

Two panels from the same tau sweep, reusing the exact numbers already
reported in Table 4 (results/*.json is the source of record; the values
below are transcribed from that table, not re-measured here):
  (a) FLOPs saved (%) and inference FPS vs. tau, dual y-axis line plot.
  (b) Gate1 / Gate2 / Main-path exit-share stacked bar per tau.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\paper"

# Transcribed from Table 4 (tab:earlyexit), EMA-Det (final, no CBAM), test set.
TAU = [0.85, 0.90, 0.92, 0.95]
FLOPS_SAVED = [21.2, 16.2, 13.9, 12.5]
FPS = [358.5, 340.8, 336.3, 337.8]
GATE1 = [24.8, 16.0, 11.9, 9.7]
GATE2 = [21.7, 29.4, 33.1, 34.0]
MAIN = [53.5, 54.6, 55.1, 56.3]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 3.6))

    # Panel (a): FLOPs saved and FPS vs. tau, dual y-axis.
    color_flops, color_fps = "#1f77b4", "#d62728"
    ax1.plot(TAU, FLOPS_SAVED, "o-", color=color_flops, label="FLOPs saved (%)")
    ax1.set_xlabel(r"Early-exit threshold $\tau$")
    ax1.set_ylabel("FLOPs saved (%)", color=color_flops)
    ax1.tick_params(axis="y", labelcolor=color_flops)
    ax1.set_xticks(TAU)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(TAU, FPS, "s--", color=color_fps, label="FPS")
    ax2.set_ylabel("FPS (RTX 4060, batch=1)", color=color_fps)
    ax2.tick_params(axis="y", labelcolor=color_fps)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("(a) Compute/throughput trade-off", fontsize=10)

    # Panel (b): Gate1/Gate2/Main exit-share stacked bar.
    x = np.arange(len(TAU))
    width = 0.55
    ax3.bar(x, GATE1, width, label="Gate 1 exit", color="#f7d9a0")
    ax3.bar(x, GATE2, width, bottom=GATE1, label="Gate 2 exit", color="#e8a33d")
    bottom2 = [g1 + g2 for g1, g2 in zip(GATE1, GATE2)]
    ax3.bar(x, MAIN, width, bottom=bottom2, label="Main path", color="#4472c4")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(t) for t in TAU])
    ax3.set_xlabel(r"Early-exit threshold $\tau$")
    ax3.set_ylabel("Share of test-set candidates (%)")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8)
    ax3.set_title("(b) Exit-path composition", fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "earlyexit_tradeoff.png")
    pdf_path = os.path.join(OUT_DIR, "earlyexit_tradeoff.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved -> {png_path}\nSaved -> {pdf_path}")


if __name__ == "__main__":
    main()
