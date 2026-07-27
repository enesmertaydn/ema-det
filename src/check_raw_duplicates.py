# -*- coding: utf-8 -*-
"""
check_raw_duplicates.py — Is duplication already present in the RAW source
pool (before any patch cropping), or does it only emerge from the crop step?

Hashes every raw labeled-drone image (rotor + fixed/train + fixed/val +
fixed/pred) with the same 8x8 average-hash used in check_leakage.py, then
looks for near-duplicates within this raw pool itself.
"""
import glob
import os

import cv2
import numpy as np

RAW_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforsiu\raw"
SOURCES = {
    "rotor": os.path.join(RAW_ROOT, "rotor"),
    "fixed_train": os.path.join(RAW_ROOT, "fixed", "train", "images"),
    "fixed_val": os.path.join(RAW_ROOT, "fixed", "val", "images"),
    "fixed_pred": os.path.join(RAW_ROOT, "fixed", "pred", "images"),
}

POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


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


def main():
    all_paths, all_src = [], []
    for src_name, folder in SOURCES.items():
        # Windows filesystems are case-insensitive, so globbing both "*.jpg"
        # and "*.JPG" would match the same files twice. List the directory
        # once and filter by lowercased suffix instead.
        paths = []
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(folder, fname))
        print(f"{src_name}: {len(paths)} images")
        all_paths += paths
        all_src += [src_name] * len(paths)

    print(f"\nTotal raw images: {len(all_paths)}")
    hashes = np.zeros(len(all_paths), dtype=np.uint64)
    for i, p in enumerate(all_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        hashes[i] = ahash64(img)

    n = len(hashes)
    chunk = 500
    near_dup_count_5 = 0
    near_dup_count_0 = 0
    example_pairs = []

    for i in range(0, n, chunk):
        block = hashes[i:i + chunk]
        xor = block[:, None] ^ hashes[None, :]
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        for local_idx in range(block.shape[0]):
            global_idx = i + local_idx
            row = dist[local_idx].copy()
            row[global_idx] = 255  # exclude self
            min_dist = row.min()
            if min_dist <= 5:
                near_dup_count_5 += 1
            if min_dist == 0:
                near_dup_count_0 += 1
            if min_dist == 0 and len(example_pairs) < 15:
                j = int(np.argmin(row))
                example_pairs.append((all_paths[global_idx], all_src[global_idx],
                                       all_paths[j], all_src[j]))

    print(f"\nImages with an exact-hash (0-bit) near-duplicate elsewhere in the raw pool: "
          f"{near_dup_count_0} / {n}  ({near_dup_count_0/n*100:.1f}%)")
    print(f"Images with a <=5-bit near-duplicate elsewhere in the raw pool: "
          f"{near_dup_count_5} / {n}  ({near_dup_count_5/n*100:.1f}%)")

    print("\nExample exact-hash-match pairs (path1, source1, path2, source2):")
    for a, sa, b, sb in example_pairs:
        print(f"  {os.path.basename(a)} [{sa}]  <->  {os.path.basename(b)} [{sb}]")


if __name__ == "__main__":
    main()
