# -*- coding: utf-8 -*-
"""
build_thermal_dataset.py — RoI-verification patches from the Anti-UAV410
thermal-IR benchmark, for the cross-sensor generalization experiment.

Anti-UAV410 already ships a video-level train/val/test split (each video's
1500-3000 frames stay together), which is exactly the source-level
splitting discipline build_dataset_clean.py had to retrofit for the EO
data — so no leakage fix is needed here beyond one exception: video
"02_6319" was found split into a train chunk (frames 0-1499) and a test
chunk (frames 1500-2999) -- same source video, non-overlapping frame
ranges. Excluded entirely from both splits out of caution, even though the
frame ranges themselves don't overlap.

Annotation format (annos/<split>/<video>.txt): one line per frame,
"x,y,w,h" in pixel coordinates; "0,0,0,0" marks the drone absent/occluded
in that frame, used as a natural negative source. Drone-absent frames are
rare relative to the number of positive candidates, so same-frame hard
negatives (a region near the drone but not overlapping it, cropped from
every positive frame) are also generated, mirroring
build_dataset_clean.py's generate_hard_negatives() for the EO pipeline.

Frames are subsampled (every FRAME_STRIDE-th) within each video -- adjacent
frames in a 1500-frame clip are near-duplicates, so using every single one
would both waste time and not add real diversity.
"""
import glob
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

ANNOS_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\Anti-UAV410-main\Anti-UAV410-main\annos"
FRAMES_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\Anti-UAV410"
OUTPUT_ROOT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\data\antiuav410_clean"

PATCH_SIZE = 128
CONTEXT_MULT = 3.0
FRAME_STRIDE = 15          # sample every 15th frame per video
MIN_DRONE_PIX = 4          # thermal drones are often just a few pixels
SEED = 42

EXCLUDE_VIDEO_IDS = {"02_6319"}  # split across train/test, see module docstring

random.seed(SEED)
np.random.seed(SEED)


def video_id(stem: str) -> str:
    """'02_6319_0000-1499' -> '02_6319'; '20190925_101846_1_4' -> itself (no frame-range suffix)."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and "-" in parts[1]:
        return parts[0]
    return stem


def read_annotations(path: str):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 4:
                boxes.append(None)
                continue
            x, y, w, h = (float(p) for p in parts)
            if w <= 0 or h <= 0:
                boxes.append(None)  # "0,0,0,0" (or similar) -> absent
            else:
                boxes.append((x, y, w, h))
    return boxes


def crop_positive(img, box, img_w, img_h):
    x, y, w, h = box
    if w < MIN_DRONE_PIX or h < MIN_DRONE_PIX:
        return None
    cx, cy = x + w / 2, y + h / 2
    half = max(max(w, h) * CONTEXT_MULT / 2, PATCH_SIZE / 2)

    px1 = int(max(cx - half, 0)); py1 = int(max(cy - half, 0))
    px2 = int(min(cx + half, img_w)); py2 = int(min(cy + half, img_h))
    patch_w, patch_h = px2 - px1, py2 - py1
    if patch_w < 16 or patch_h < 16:
        return None

    patch = cv2.resize(img[py1:py2, px1:px2], (PATCH_SIZE, PATCH_SIZE))
    new_cx = float(np.clip((cx - px1) / patch_w, 0.01, 0.99))
    new_cy = float(np.clip((cy - py1) / patch_h, 0.01, 0.99))
    new_w = float(np.clip(w / patch_w, 0.01, 0.99))
    new_h = float(np.clip(h / patch_h, 0.01, 0.99))
    return patch, new_cx, new_cy, new_w, new_h


def crop_negative(img, img_w, img_h):
    if img_w < PATCH_SIZE or img_h < PATCH_SIZE:
        return cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
    x = random.randint(0, img_w - PATCH_SIZE)
    y = random.randint(0, img_h - PATCH_SIZE)
    return img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]


def crop_hard_negative(img, box, img_w, img_h, max_attempts=20):
    """Same-frame hard negative: a patch near the drone but not overlapping
    it -- matches build_dataset_clean.py's generate_hard_negatives() for
    the EO pipeline. Without this, negatives are limited to the rare
    drone-absent frames ("0,0,0,0"), which starves the dataset of usable
    negatives (observed: 13,884 positive candidates vs only 338 true-absent
    frames in the train split alone)."""
    x, y, w, h = box
    dcx, dcy = x + w / 2, y + h / 2
    for _ in range(max_attempts):
        cx = random.randint(PATCH_SIZE // 2, max(img_w - PATCH_SIZE // 2, PATCH_SIZE // 2))
        cy = random.randint(PATCH_SIZE // 2, max(img_h - PATCH_SIZE // 2, PATCH_SIZE // 2))
        dist = ((cx - dcx) ** 2 + (cy - dcy) ** 2) ** 0.5
        if dist < max(w, h) * 2:
            continue
        px1 = max(cx - PATCH_SIZE // 2, 0); py1 = max(cy - PATCH_SIZE // 2, 0)
        px2 = min(cx + PATCH_SIZE // 2, img_w); py2 = min(cy + PATCH_SIZE // 2, img_h)
        if (px2 - px1) < PATCH_SIZE // 2 or (py2 - py1) < PATCH_SIZE // 2:
            continue
        return cv2.resize(img[py1:py2, px1:px2], (PATCH_SIZE, PATCH_SIZE))
    return None


def collect_split(split: str):
    anno_dir = os.path.join(ANNOS_ROOT, split)
    anno_files = sorted(glob.glob(os.path.join(anno_dir, "*.txt")))

    positives, negatives = [], []
    skipped_videos = 0

    for anno_path in tqdm(anno_files, desc=f"{split} videos"):
        stem = os.path.splitext(os.path.basename(anno_path))[0]
        if video_id(stem) in EXCLUDE_VIDEO_IDS:
            skipped_videos += 1
            continue

        frame_dir = os.path.join(FRAMES_ROOT, split, stem)
        if not os.path.isdir(frame_dir):
            continue

        boxes = read_annotations(anno_path)
        for frame_idx in range(0, len(boxes), FRAME_STRIDE):
            frame_path = os.path.join(frame_dir, f"{frame_idx + 1:06d}.jpg")
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_h, img_w = img.shape

            box = boxes[frame_idx]
            if box is None:
                negatives.append(crop_negative(img, img_w, img_h))
                continue

            result = crop_positive(img, box, img_w, img_h)
            if result is None:
                continue
            patch, ncx, ncy, nw, nh = result
            positives.append((patch, f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"))

            hard_neg = crop_hard_negative(img, box, img_w, img_h)
            if hard_neg is not None:
                negatives.append(hard_neg)

    print(f"  {split}: excluded {skipped_videos} video(s) (train/test split overlap)")
    return positives, negatives


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


def main():
    print("=" * 70)
    print("  Anti-UAV410 thermal RoI dataset build")
    print("=" * 70)

    if os.path.exists(OUTPUT_ROOT):
        print(f"Removing existing output at {OUTPUT_ROOT}...")
        shutil.rmtree(OUTPUT_ROOT)

    for split in ("train", "val", "test"):
        print(f"\n--- {split} ---")
        positives, negatives = collect_split(split)
        print(f"  candidate patches: +{len(positives)} / -{len(negatives)}")
        save_split(split, positives, negatives)

    print(f"\nDone -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
