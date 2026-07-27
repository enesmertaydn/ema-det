# -*- coding: utf-8 -*-
"""
dataset.py — RoI dataset for EMA-Det.

Expects the standard YOLO layout under `root_dir`:
    root_dir/images/*.jpg|*.png
    root_dir/labels/<same-stem>.txt      # "<class> <cx> <cy> <w> <h>", normalized
A missing or empty label file means the crop is a negative (background) sample.

Augmentations are restricted to the set the manuscript specifies as safe for
the 128x128 micro-input regime (Sec. III-A): Mosaic/MixUp are deliberately
excluded because they destroy small-target shape at this resolution. Only
flip, brightness/contrast, motion blur, Gaussian noise, small-angle rotation
and coarse dropout (occlusion) are applied, matching bbox coordinates
transformed alongside the image where relevant.
"""

import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 128


class DroneDetectDataset(Dataset):
    """
    Returns per sample:
        img   : (1, H, W) float tensor in [0, 1]
        label : scalar float — 1.0 positive (UAV present), 0.0 negative
        bbox  : (4,) float tensor (cx, cy, w, h), normalized; zeros if negative
    """

    def __init__(self, root_dir: str, augment: bool = False, img_size: int = IMG_SIZE,
                 cache_images: bool = True):
        self.img_paths = sorted(
            glob.glob(os.path.join(root_dir, "images", "*.jpg")) +
            glob.glob(os.path.join(root_dir, "images", "*.png")) +
            glob.glob(os.path.join(root_dir, "images", "*.jpeg"))
        )
        if not self.img_paths:
            raise FileNotFoundError(f"No images found under {os.path.join(root_dir, 'images')}")
        self.label_dir = os.path.join(root_dir, "labels")
        self.augment   = augment
        self.img_size  = img_size

        # Labels are cheap to parse once; always precomputed regardless of cache_images.
        self.labels = [self._read_label(p) for p in self.img_paths]

        # Images: 128x128 uint8 grayscale is ~16KB each, so even a 25K-image
        # dataset fits in a few hundred MB of RAM. Caching removes the
        # cv2.imread + resize disk/CPU cost from every epoch (it otherwise
        # repeats identically every single epoch since these ops are
        # deterministic and independent of augmentation).
        self.cache_images = cache_images
        self._cache = None
        if cache_images:
            self._cache = [self._load_resized(p) for p in self.img_paths]

    def _load_resized(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        return cv2.resize(img, (self.img_size, self.img_size))

    def __len__(self) -> int:
        return len(self.img_paths)

    def _read_label(self, img_path: str):
        """Single-object RoI verification: only the first labeled box is used."""
        stem  = os.path.splitext(os.path.basename(img_path))[0]
        lbl_p = os.path.join(self.label_dir, stem + ".txt")

        if not os.path.exists(lbl_p) or os.path.getsize(lbl_p) == 0:
            return 0.0, None

        with open(lbl_p, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return 0.0, None

        parts = lines[0].split()
        if len(parts) < 5:
            return 0.0, None

        try:
            bbox = np.array([float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])], dtype=np.float32)
            if (bbox < 0).any() or (bbox > 1).any():
                return 0.0, None
            return 1.0, bbox
        except ValueError:
            return 0.0, None

    def _augment(self, img: np.ndarray, bbox):
        h, w = img.shape

        if random.random() < 0.5:  # horizontal flip
            img = cv2.flip(img, 1)
            if bbox is not None:
                bbox = bbox.copy()
                bbox[0] = 1.0 - bbox[0]

        if random.random() < 0.6:  # brightness / contrast
            img = np.clip(
                img.astype(np.float32) * random.uniform(0.5, 1.5)
                + random.randint(-40, 40), 0, 255
            ).astype(np.uint8)

        if random.random() < 0.5:  # directional motion blur
            sz  = random.choice([3, 5, 7, 9])
            ang = random.uniform(0, 180)
            k   = np.zeros((sz, sz), dtype=np.float32)
            k[sz // 2, :] = 1.0
            M   = cv2.getRotationMatrix2D((sz / 2, sz / 2), ang, 1)
            k   = cv2.warpAffine(k, M, (sz, sz))
            k  /= k.sum() + 1e-8
            img = cv2.filter2D(img, -1, k)

        if random.random() < 0.4:  # Gaussian noise
            img = np.clip(
                img.astype(np.float32) + np.random.randn(h, w) * random.uniform(5, 20),
                0, 255
            ).astype(np.uint8)

        if random.random() < 0.4:  # small-angle rotation (bbox center rotated too)
            ang = random.uniform(-15, 15)
            M   = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            if bbox is not None:
                ang_r = math.radians(-ang)
                cx, cy = bbox[0] - 0.5, bbox[1] - 0.5
                new_cx = cx * math.cos(ang_r) - cy * math.sin(ang_r) + 0.5
                new_cy = cx * math.sin(ang_r) + cy * math.cos(ang_r) + 0.5
                bbox = bbox.copy()
                bbox[0] = float(np.clip(new_cx, 0.01, 0.99))
                bbox[1] = float(np.clip(new_cy, 0.01, 0.99))

        if random.random() < 0.3:  # coarse dropout / occlusion (bbox unchanged)
            for _ in range(random.randint(1, 2)):
                cx_, cy_ = random.randint(0, w), random.randint(0, h)
                sz_ = random.randint(8, 20)
                img[max(0, cy_ - sz_):cy_ + sz_, max(0, cx_ - sz_):cx_ + sz_] = 0

        return img, bbox

    def __getitem__(self, idx: int):
        label, bbox = self.labels[idx]

        # .copy() is required here: several augmentation branches (occlusion
        # in particular) mutate the array in place, and the cache/label list
        # is shared and re-read every epoch.
        img = self._cache[idx].copy() if self.cache_images else self._load_resized(self.img_paths[idx])

        if self.augment:
            img, bbox = self._augment(img, bbox)

        img_t   = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0)
        label_t = torch.tensor(label, dtype=torch.float32)
        bbox_t  = torch.tensor(bbox if bbox is not None else np.zeros(4, dtype=np.float32),
                                dtype=torch.float32)
        return img_t, label_t, bbox_t
