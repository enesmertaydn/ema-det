# -*- coding: utf-8 -*-
"""make_architecture_figure.py — Fig. 1 schematic for the paper (matplotlib, no external deps)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_PATH = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\paper\figures\architecture.png"

MAIN_X, MAIN_W = 2.5, 5.0
BOX_H = 1.0
STEP = 1.55  # top-to-top spacing between consecutive main-path boxes


def box(ax, xy, w, h, text, color="#dbe7f5", fontsize=9):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                    linewidth=1.2, edgecolor="black", facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, xy1, xy2, color="black"):
    ax.annotate("", xy=xy2, xytext=xy1, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    stages = [
        ("Input: 1x128x128 (grayscale/thermal)", "#f5f5f5", None),
        ("Block 1 (Laplacian/Sobel init)\nConv 1->16, 3x3 -> 16x64x64", "#cfe8cf", None),
        ("Block 2: Conv 16->32 -> 32x32x32", "#dbe7f5", "Gate 1\n1x1,GAP,FC"),
        ("Block 3: Conv 32->64 -> 64x16x16", "#dbe7f5", None),
        ("Block 4: Conv 64->128 -> 128x8x8", "#dbe7f5", "Gate 2\n1x1,GAP,FC"),
        ("Block 5: multi-scale (d=1 & d=2)\nfusion -> 128x4x4", "#dbe7f5", None),
        ("Block 6: Conv 128->256 -> 256x4x4", "#dbe7f5", None),
        ("Verification Head\nGAP -> Dropout(0.4)\nFC->1 (cls) | FC->4 (bbox)", "#f5e0e0", None),
        ("Output: P(UAV) in [0,1] + box refinement", "#f5f5f5", None),
    ]

    n = len(stages)
    top_y = (n - 1) * STEP
    fig_h = 1.6 + n * STEP * 0.42
    fig, ax = plt.subplots(figsize=(6.5, fig_h))

    y = top_y
    for i, (text, color, gate_text) in enumerate(stages):
        fontsize = 8.5 if "\n" in text else 9
        box(ax, (MAIN_X, y), MAIN_W, BOX_H, text, color, fontsize)
        if gate_text is not None:
            box(ax, (8.0, y), 1.7, BOX_H, gate_text, "#f7d9a0", fontsize=8)
            arrow(ax, (MAIN_X + MAIN_W, y + BOX_H / 2), (8.0, y + BOX_H / 2), color="#c07a00")
        if i > 0:
            arrow(ax, (MAIN_X + MAIN_W / 2, y + BOX_H + (STEP - BOX_H)), (MAIN_X + MAIN_W / 2, y + BOX_H))
        y -= STEP

    ax.set_xlim(0, 10)
    ax.set_ylim(y + STEP - 0.9, top_y + BOX_H + 1.0)
    ax.axis("off")
    ax.text(5, top_y + BOX_H + 0.5, "EMA-Det Architecture (Fig. 1)",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(5, y + STEP - 0.5,
            "Orange dashed paths: early-exit routes. A candidate halting at Gate 1/2\n"
            "is classified background and skips the remaining blocks (Eq. 4).",
            ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
