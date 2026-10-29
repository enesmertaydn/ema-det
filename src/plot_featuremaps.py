# -*- coding: utf-8 -*-
"""
plot_featuremaps.py -- first-layer (Laplacian/Sobel-initialized) feature maps
for the paper (revision, Reviewer 3). Shows, for one UAV and one hard-negative
test patch, the raw 3x3-conv responses of Block 1 of the trained EMA-Det
(final, centre-head model, seed 42) for the Laplacian, Sobel-x and Sobel-y channels, next to a
channel of an identically shaped randomly (Kaiming-uniform) initialised layer.
Also reports how far the trained kernels stay from their physical priors.
"""
import json, os, sys
import numpy as np, torch, torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ema_det import EMADet, make_edge_kernels  # noqa

TEST = os.path.join(ROOT, "data", "shuffled_dataset_detect_clean", "test")
RES = os.path.join(ROOT, "results")
SEEDS = [42, 7, 123, 2024, 31415]

def imread(p):
    return np.array(Image.open(p).convert("L"))

def load(seed):
    m = EMADet(center_head=True)   # final model: centre-based head
    m.load_state_dict(torch.load(os.path.join(RES, f"emadet_final_centerhead_seed{seed}.pt"), map_location="cpu"))
    return m.eval()

# ---- kernel drift: cosine similarity of trained vs. initial kernel (channels 0-11)
torch.manual_seed(0)
init = make_edge_kernels(16)[:12].reshape(12, 9)
cos = []
for s in SEEDS:
    w = load(s).block1.conv.weight.detach().reshape(16, 9)[:12]
    cos.append(torch.nn.functional.cosine_similarity(w, init, dim=1).numpy())
cos = np.array(cos)
stats = {"cos_mean_over_channels_and_seeds": float(cos.mean()),
         "cos_min": float(cos.min()), "per_channel_mean": cos.mean(0).tolist()}
json.dump(stats, open(os.path.join(RES, "kernel_drift.json"), "w"), indent=2)
print(json.dumps(stats, indent=1))

# ---- pick one positive and one negative test patch (fixed, first of each)
# fixed, visually inspected examples: a quadcopter UAV and a bird (hard negative)
pos, neg = "test_000041", "test_000019"
print("pos", pos, "neg", neg)

model = load(42)
W = model.block1.conv.weight.detach()
rand = nn.Conv2d(1, 16, 3, padding=1, bias=False)  # default Kaiming-uniform
torch.manual_seed(1); rand.reset_parameters()
cols = [("Input", None), ("Laplacian (ch 0)", 0), ("Sobel-x (ch 4)", 4),
        ("Sobel-y (ch 6)", 6), ("Random init", "rand")]
fig, axes = plt.subplots(2, 5, figsize=(3.45, 1.45))
for r, name in enumerate([pos, neg]):
    img = imread(os.path.join(TEST, "images", name + ".jpg"))
    img = np.array(Image.fromarray(img).resize((128, 128), Image.BILINEAR))
    x = torch.tensor(img.astype(np.float32) / 255.0)[None, None]
    for c, (title, ch) in enumerate(cols):
        ax = axes[r, c]
        if ch is None:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            with torch.no_grad():
                y = (rand(x)[0, 3] if ch == "rand" else
                     nn.functional.conv2d(x, W, padding=1)[0, ch])
            y = y.abs().numpy()
            ax.imshow(y, cmap="magma", vmin=0, vmax=np.percentile(y, 99.5) + 1e-9)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(title, fontsize=4.6, pad=2)
    axes[r, 0].set_ylabel("UAV" if r == 0 else "Bird (neg.)", fontsize=5)
plt.subplots_adjust(wspace=0.03, hspace=0.04, left=0.07, right=0.995, top=0.9, bottom=0.005)
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(ROOT, "paper", f"featuremaps.{ext}"), dpi=220, bbox_inches="tight")
print("saved")
