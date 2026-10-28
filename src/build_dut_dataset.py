# -*- coding: utf-8 -*-
"""
build_dut_dataset.py -- RoI-verification patches from the DUT Anti-UAV detection test split (visible light), used only as an
external, never-trained-on EO test set (Zhao et al., IEEE T-ITS 2022; github.com/wangdongdut/DUT-Anti-UAV, Apache-2.0).
Same patch protocol as build_dataset_clean.py / build_thermal_dataset.py: one positive patch per image (3x context factor,
128x128) and one same-image hard-negative patch near the drone but not overlapping it. No dataset image is used for training
or checkpoint selection, so this set cannot share a source photograph with the EO training pool.
Input: <DUT_ROOT>/test/img/*.jpg and test/xml/*.xml (Pascal VOC). Output: data/dut_antiuav_test/{images,labels} (YOLO labels).
"""
import glob
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUT_ROOT = os.path.join(os.path.dirname(ROOT), "external_dut_test", "x", "test")
OUT = os.path.join(ROOT, "data", "dut_antiuav_test")
PATCH_SIZE = 128
CONTEXT_MULT = 3.0
MIN_DRONE_PIX = 4
random.seed(42)
np.random.seed(42)


def read_boxes(path):
    boxes = []
    for o in ET.parse(path).getroot().iter("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def crop_positive(img, box):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box
    if w < MIN_DRONE_PIX or h < MIN_DRONE_PIX:
        return None
    cx, cy = x + w / 2, y + h / 2
    half = max(max(w, h) * CONTEXT_MULT / 2, PATCH_SIZE / 2)
    px1, py1 = int(max(cx - half, 0)), int(max(cy - half, 0))
    px2, py2 = int(min(cx + half, w_img)), int(min(cy + half, h_img))
    pw, ph = px2 - px1, py2 - py1
    if pw < 16 or ph < 16:
        return None
    patch = cv2.resize(img[py1:py2, px1:px2], (PATCH_SIZE, PATCH_SIZE))
    c = lambda v: float(np.clip(v, 0.01, 0.99))
    return patch, c((cx - px1) / pw), c((cy - py1) / ph), c(w / pw), c(h / ph)


def crop_hard_negative(img, boxes, max_attempts=40):
    h_img, w_img = img.shape[:2]
    for _ in range(max_attempts):
        cx = random.randint(PATCH_SIZE // 2, max(w_img - PATCH_SIZE // 2, PATCH_SIZE // 2))
        cy = random.randint(PATCH_SIZE // 2, max(h_img - PATCH_SIZE // 2, PATCH_SIZE // 2))
        if any(((cx - (x + w / 2)) ** 2 + (cy - (y + h / 2)) ** 2) ** 0.5 < max(w, h) * 2 + PATCH_SIZE / 2 for x, y, w, h in boxes):
            continue
        px1, py1 = max(cx - PATCH_SIZE // 2, 0), max(cy - PATCH_SIZE // 2, 0)
        px2, py2 = min(cx + PATCH_SIZE // 2, w_img), min(cy + PATCH_SIZE // 2, h_img)
        if (px2 - px1) < PATCH_SIZE // 2 or (py2 - py1) < PATCH_SIZE // 2:
            continue
        return cv2.resize(img[py1:py2, px1:px2], (PATCH_SIZE, PATCH_SIZE))
    return None


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)
    pos_n = neg_n = 0
    for xp in sorted(glob.glob(os.path.join(DUT_ROOT, "xml", "*.xml"))):
        stem = os.path.splitext(os.path.basename(xp))[0]
        img = cv2.imread(os.path.join(DUT_ROOT, "img", stem + ".jpg"))
        boxes = read_boxes(xp)
        if img is None or not boxes:
            continue
        big = max(boxes, key=lambda b: b[2] * b[3])
        res = crop_positive(img, big)
        neg = crop_hard_negative(img, boxes)
        if res is None or neg is None:
            continue
        patch, ncx, ncy, nw, nh = res
        for name, im, lbl in (("p_" + stem, patch, "0 %.6f %.6f %.6f %.6f\n" % (ncx, ncy, nw, nh)), ("n_" + stem, neg, "")):
            cv2.imwrite(os.path.join(OUT, "images", name + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(os.path.join(OUT, "labels", name + ".txt"), "w").write(lbl)
        pos_n += 1
        neg_n += 1
    print("DUT test patches: +%d / -%d -> %s" % (pos_n, neg_n, OUT))


if __name__ == "__main__":
    main()
