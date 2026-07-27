# -*- coding: utf-8 -*-
"""
build_dataset_clean.py — Leakage-free RoI dataset construction for EMA-Det.

Supersedes emadetforsiu/datasetorganizer.py. That script pooled every raw
source photo, cropped 1 positive + N hard-negative patches per photo, then
shuffled ALL resulting patches together and split by ratio. Two problems,
both confirmed empirically (see check_leakage.py / check_raw_duplicates.py):

  1. The raw source pool itself is heavily duplicated — 52.1% of raw photos
     have a bit-identical near-duplicate elsewhere in the pool (likely
     sequential video-frame exports: e.g. rotor/0032.jpg and rotor/0033.jpg
     hash identically). This was never deduplicated.
  2. Patches were shuffled and split at the PATCH level, not the SOURCE
     level, so sibling patches (and near-duplicate-source patches) from the
     same underlying photo could land in different splits. Measured result:
     83.8% of test-set drone patches were bit-identical to a training patch.

Fix, in order:
  (a) Deduplicate raw photos via 8x8 average-hash (Hamming distance <= 5 ==
      same underlying photo) BEFORE any cropping.
  (b) Split the deduplicated photo pool into train/val/test at the SOURCE
      level (70/15/15, fixed seed).
  (c) Only then crop positive + hard-negative patches, strictly within each
      photo's assigned split — so no patch derived from one photo can ever
      appear in two splits.
  (d) Balance positive/negative counts per split (not globally).

Output layout is unchanged (YOLO-style train/val/test/{images,labels}), so
dataset.py / train.py need no changes beyond pointing dataset_root here.
"""
import glob
import math
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# SETTINGS
# ==============================================================================
RAW_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforsiu\raw"
ROTOR_DIR = os.path.join(RAW_ROOT, "rotor")
FIXED_DIRS = [
    os.path.join(RAW_ROOT, "fixed", "train"),
    os.path.join(RAW_ROOT, "fixed", "val"),
    os.path.join(RAW_ROOT, "fixed", "pred"),
]
SKY_DIR = os.path.join(RAW_ROOT, "sky")

OUTPUT_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\shuffled_dataset_detect_clean"

PATCH_SIZE = 128
CONTEXT_MULT = 3.0
HARD_NEG_PER_POS = 1
MIN_DRONE_PIX = 8

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

DEDUP_HAMMING_THRESH = 5  # <=5/64 bits different => treat as the same underlying photo

random.seed(SEED)
np.random.seed(SEED)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ==============================================================================
# HELPERS (ported unchanged from datasetorganizer.py)
# ==============================================================================
def find_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXTS))


def read_yolo_labels(lbl_path):
    if not lbl_path or not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return []
    bboxes = []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    bboxes.append([int(parts[0])] + [float(p) for p in parts[1:5]])
                except ValueError:
                    continue
    return bboxes


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def crop_and_relabel(img, bbox_pixel, img_w, img_h, context_mult=CONTEXT_MULT):
    x1, y1, x2, y2 = bbox_pixel
    drone_w, drone_h = x2 - x1, y2 - y1
    if drone_w < MIN_DRONE_PIX or drone_h < MIN_DRONE_PIX:
        return None

    cx_pix, cy_pix = (x1 + x2) // 2, (y1 + y2) // 2
    half = int(max(drone_w, drone_h) * context_mult / 2)
    half = max(half, PATCH_SIZE // 2)

    px1, py1 = max(cx_pix - half, 0), max(cy_pix - half, 0)
    px2, py2 = min(cx_pix + half, img_w), min(cy_pix + half, img_h)
    patch_w, patch_h = px2 - px1, py2 - py1
    if patch_w < 16 or patch_h < 16:
        return None

    patch = cv2.resize(img[py1:py2, px1:px2], (PATCH_SIZE, PATCH_SIZE))
    new_cx = float(np.clip((cx_pix - px1) / patch_w, 0.01, 0.99))
    new_cy = float(np.clip((cy_pix - py1) / patch_h, 0.01, 0.99))
    new_w = float(np.clip(drone_w / patch_w, 0.01, 0.99))
    new_h = float(np.clip(drone_h / patch_h, 0.01, 0.99))
    return patch, new_cx, new_cy, new_w, new_h


def generate_hard_negatives(img, bboxes_pixel, img_w, img_h, n=1):
    negatives = []
    attempts, max_attempts = 0, n * 20
    while len(negatives) < n and attempts < max_attempts:
        attempts += 1
        cx = random.randint(PATCH_SIZE // 2, max(img_w - PATCH_SIZE // 2, PATCH_SIZE // 2))
        cy = random.randint(PATCH_SIZE // 2, max(img_h - PATCH_SIZE // 2, PATCH_SIZE // 2))
        overlaps = False
        for x1, y1, x2, y2 in bboxes_pixel:
            dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = math.sqrt((cx - dcx) ** 2 + (cy - dcy) ** 2)
            if dist < max(x2 - x1, y2 - y1) * 2:
                overlaps = True
                break
        if overlaps:
            continue
        px1, py1 = max(cx - PATCH_SIZE // 2, 0), max(cy - PATCH_SIZE // 2, 0)
        px2, py2 = min(cx + PATCH_SIZE // 2, img_w), min(cy + PATCH_SIZE // 2, img_h)
        if (px2 - px1) < PATCH_SIZE // 2 or (py2 - py1) < PATCH_SIZE // 2:
            continue
        negatives.append(cv2.resize(img[py1:py2, px1:px2], (PATCH_SIZE, PATCH_SIZE)))
    return negatives


# ==============================================================================
# DEDUPLICATION
# ==============================================================================
def ahash64(img: np.ndarray) -> np.uint64:
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint64).flatten()
    h = np.uint64(0)
    for b in bits:
        h = (h << np.uint64(1)) | b
    return h


def popcount64(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint8).reshape(x.shape[0], 8)
    return POPCOUNT_TABLE[b].sum(axis=1)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def dedup_pool(paths, has_label_flags, chunk=500):
    """
    Cluster near-duplicate images (Hamming <= DEDUP_HAMMING_THRESH) and keep
    one representative per cluster, preferring a labeled member if the
    cluster has one. Returns the list of representative indices (into
    `paths`), sorted.
    """
    n = len(paths)
    hashes = np.zeros(n, dtype=np.uint64)
    for i, p in enumerate(tqdm(paths, desc="Hashing for dedup")):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        hashes[i] = ahash64(img) if img is not None else np.uint64(0)

    uf = UnionFind(n)
    for i in range(0, n, chunk):
        block = hashes[i:i + chunk]
        xor = block[:, None] ^ hashes[None, :]
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        for local_idx in range(block.shape[0]):
            global_idx = i + local_idx
            matches = np.where(dist[local_idx] <= DEDUP_HAMMING_THRESH)[0]
            for j in matches:
                if j != global_idx:
                    uf.union(global_idx, int(j))

    clusters = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    reps = []
    for members in clusters.values():
        labeled = [m for m in members if has_label_flags[m]]
        reps.append(labeled[0] if labeled else members[0])
    reps.sort()
    print(f"  {n} raw images -> {len(clusters)} unique photos after dedup "
          f"({n - len(clusters)} duplicates removed)")
    return reps


# ==============================================================================
# SOURCE COLLECTION
# ==============================================================================
def collect_labeled_sources():
    sources = []
    for img_path in find_images(ROTOR_DIR):
        stem = os.path.splitext(img_path)[0]
        lbl = stem + ".txt"
        sources.append((img_path, lbl if os.path.exists(lbl) and os.path.getsize(lbl) > 0 else None))

    for split_dir in FIXED_DIRS:
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        if not os.path.isdir(img_dir):
            img_dir, lbl_dir = split_dir, split_dir
        for img_path in find_images(img_dir):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            sources.append((img_path, lbl if os.path.exists(lbl) and os.path.getsize(lbl) > 0 else None))

    print(f"Labeled-candidate raw pool: {len(sources)} images")
    return sources


def collect_sky_sources():
    sky_imgs = find_images(SKY_DIR)
    print(f"Sky raw pool: {len(sky_imgs)} images")
    return sky_imgs


# ==============================================================================
# SOURCE-LEVEL SPLIT
# ==============================================================================
def split_sources(items, ratios=(TRAIN_RATIO, VAL_RATIO)):
    items = list(items)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


# ==============================================================================
# PATCH GENERATION (per split, so no cross-split contamination is possible)
# ==============================================================================
def generate_patches_for_split(labeled_items, sky_items):
    positives, negatives = [], []
    skipped = 0

    for img_path, lbl_path in labeled_items:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        bboxes = read_yolo_labels(lbl_path)

        if not bboxes:
            negatives.append(cv2.resize(img, (PATCH_SIZE, PATCH_SIZE)))
            continue

        bboxes_pixel = []
        for bbox in bboxes:
            _, cx, cy, w, h = bbox
            pix = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
            bboxes_pixel.append(pix)
            result = crop_and_relabel(img, pix, img_w, img_h)
            if result is None:
                skipped += 1
                continue
            patch, ncx, ncy, nw, nh = result
            positives.append((patch, f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"))

        negatives.extend(generate_hard_negatives(
            img, bboxes_pixel, img_w, img_h, n=HARD_NEG_PER_POS * len(bboxes)))

    for sky_path in sky_items:
        img = cv2.imread(sky_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        for _ in range(2):
            if img_w >= PATCH_SIZE and img_h >= PATCH_SIZE:
                x = random.randint(0, img_w - PATCH_SIZE)
                y = random.randint(0, img_h - PATCH_SIZE)
                negatives.append(img[y:y + PATCH_SIZE, x:x + PATCH_SIZE])
            else:
                negatives.append(cv2.resize(img, (PATCH_SIZE, PATCH_SIZE)))

    return positives, negatives, skipped


def save_split(split_name, positives, negatives):
    target = min(len(positives), len(negatives))
    pos_sel = random.sample(positives, target)
    neg_sel = random.sample(negatives, target)
    samples = [(img, lbl) for img, lbl in pos_sel] + [(img, None) for img in neg_sel]
    random.shuffle(samples)

    os.makedirs(os.path.join(OUTPUT_ROOT, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, split_name, "labels"), exist_ok=True)

    for i, (patch_img, label_str) in enumerate(tqdm(samples, desc=f"Saving {split_name}")):
        name = f"{split_name}_{i:06d}"
        cv2.imwrite(os.path.join(OUTPUT_ROOT, split_name, "images", name + ".jpg"),
                    patch_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(OUTPUT_ROOT, split_name, "labels", name + ".txt"), "w") as f:
            if label_str:
                f.write(label_str + "\n")

    print(f"  {split_name:5s}: {len(samples)} patches "
          f"(+{len(pos_sel)} / -{len(neg_sel)}, balanced from "
          f"{len(positives)} pos / {len(negatives)} neg candidates)")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  Leakage-free dataset build")
    print("=" * 70)

    labeled_sources = collect_labeled_sources()
    sky_sources = collect_sky_sources()

    print("\nDeduplicating labeled-candidate pool...")
    has_label = [lbl is not None for _, lbl in labeled_sources]
    labeled_paths = [p for p, _ in labeled_sources]
    keep_idx = dedup_pool(labeled_paths, has_label)
    labeled_unique = [labeled_sources[i] for i in keep_idx]

    print("\nDeduplicating sky pool...")
    sky_keep_idx = dedup_pool(sky_sources, [False] * len(sky_sources))
    sky_unique = [sky_sources[i] for i in sky_keep_idx]

    print(f"\nUnique labeled-candidate sources: {len(labeled_unique)} "
          f"(of {len(labeled_sources)})")
    print(f"Unique sky sources: {len(sky_unique)} (of {len(sky_sources)})")

    print("\nSplitting at SOURCE level (train/val/test)...")
    labeled_splits = split_sources(labeled_unique)
    sky_splits = split_sources(sky_unique)
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {len(labeled_splits[s])} labeled-candidate + "
              f"{len(sky_splits[s])} sky sources")

    if os.path.exists(OUTPUT_ROOT):
        print(f"\nRemoving existing output at {OUTPUT_ROOT}...")
        shutil.rmtree(OUTPUT_ROOT)

    print("\nGenerating + saving patches per split...")
    for split_name in ("train", "val", "test"):
        print(f"\n--- {split_name} ---")
        positives, negatives, skipped = generate_patches_for_split(
            labeled_splits[split_name], sky_splits[split_name])
        print(f"  candidate patches: +{len(positives)} / -{len(negatives)}  "
              f"(skipped too-small boxes: {skipped})")
        save_split(split_name, positives, negatives)

    print(f"\nDone -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
