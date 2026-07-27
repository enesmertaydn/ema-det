# -*- coding: utf-8 -*-
"""
check_leakage.py — Empirical near-duplicate check across train/val/test.

datasetorganizer.py pools positive + hard-negative patches from ALL source
images first, then shuffles and splits by ratio (see save_dataset()). That
means patches cropped from the same raw photo can land in different splits.
This script checks, via a cheap perceptual hash (8x8 average hash, 64 bits),
whether test images have near-identical neighbors in train — direct evidence
of leakage, independent of reading the dataset-construction code.
"""
import glob
import os

import cv2
import numpy as np

DATASET_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\shuffled_dataset_detect_clean"

POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ahash64(img: np.ndarray) -> np.uint64:
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint64).flatten()
    h = np.uint64(0)
    for b in bits:
        h = (h << np.uint64(1)) | b
    return h


def label_of(img_path: str, label_dir: str) -> int:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lp = os.path.join(label_dir, stem + ".txt")
    return 1 if (os.path.exists(lp) and os.path.getsize(lp) > 0) else 0


def load_split(split: str):
    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    hashes = np.zeros(len(paths), dtype=np.uint64)
    labels = np.zeros(len(paths), dtype=np.int8)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        hashes[i] = ahash64(img)
        labels[i] = label_of(p, lbl_dir)
    return paths, hashes, labels


def popcount64(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint8).reshape(x.shape[0], 8)
    return POPCOUNT_TABLE[b].sum(axis=1)


def nearest_distances(query_hashes: np.ndarray, ref_hashes: np.ndarray, chunk: int = 200) -> np.ndarray:
    """For each query hash, min Hamming distance to any ref hash."""
    n = query_hashes.shape[0]
    out = np.zeros(n, dtype=np.int32)
    for i in range(0, n, chunk):
        q = query_hashes[i:i + chunk]
        xor = q[:, None] ^ ref_hashes[None, :]
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        out[i:i + chunk] = dist.min(axis=1)
    return out


def report(name: str, dists: np.ndarray, thresholds=(0, 3, 5, 8, 10)):
    print(f"\n{name}  (n={len(dists)})")
    print(f"  min={dists.min()}  median={int(np.median(dists))}  mean={dists.mean():.1f}  max={dists.max()}")
    for t in thresholds:
        pct = (dists <= t).mean() * 100
        print(f"  <= {t:2d} bits different: {pct:5.1f}%  ({int((dists <= t).sum())} images)")


def main():
    print("Loading + hashing train/val/test (this reads every image once)...")
    train_paths, train_hash, train_lbl = load_split("train")
    val_paths, val_hash, val_lbl = load_split("val")
    test_paths, test_hash, test_lbl = load_split("test")
    print(f"train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

    print("\n=== TEST vs TRAIN (all labels) ===")
    d_all = nearest_distances(test_hash, train_hash)
    report("test -> nearest train neighbor, all images", d_all)

    pos_mask_test = test_lbl == 1
    pos_mask_train = train_lbl == 1
    if pos_mask_test.sum() > 0 and pos_mask_train.sum() > 0:
        print("\n=== TEST-POSITIVE vs TRAIN-POSITIVE only (drone patches) ===")
        d_pos = nearest_distances(test_hash[pos_mask_test], train_hash[pos_mask_train])
        report("test-positive -> nearest train-positive neighbor", d_pos)

    neg_mask_test = test_lbl == 0
    neg_mask_train = train_lbl == 0
    if neg_mask_test.sum() > 0 and neg_mask_train.sum() > 0:
        print("\n=== TEST-NEGATIVE vs TRAIN-NEGATIVE only (background patches) ===")
        d_neg = nearest_distances(test_hash[neg_mask_test], train_hash[neg_mask_train])
        report("test-negative -> nearest train-negative neighbor", d_neg)

    print("\n=== VAL vs TRAIN (all labels), for comparison ===")
    d_val = nearest_distances(val_hash, train_hash)
    report("val -> nearest train neighbor, all images", d_val)


if __name__ == "__main__":
    main()
