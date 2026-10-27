# -*- coding: utf-8 -*-
"""
thermal_watch.py -- watches scratch_runner3.sh (centre-head training on the thermal data) and warns early.
Every few minutes: warns when the runner has stopped or a log shows a crash; when a seed finishes it evaluates that seed's centre-head
checkpoint and the default checkpoint of the same seed on the thermal test split (CPU) and prints the difference, with a WARNING if the
localization gain is small (mAP@0.5:0.95 gain < 2 points) or negative. Output goes to stdout (append to results/thermal/watch.log).
"""
import os
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results", "thermal")
TEST = os.path.join(ROOT, "data", "antiuav410_clean", "test")
SEEDS = [42, 7, 123, 2024, 31415]


def runner_alive():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'scratch_runner3|src[\\\\/]train.py' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return int(out) > 0
    except ValueError:
        return True


def log_crashed(seed):
    p = os.path.join(RES, "seedrun_center_%d.log" % seed)
    if not os.path.exists(p):
        return False
    t = open(p, encoding="utf-8", errors="ignore").read()
    return ("Traceback" in t or "AcceleratorError" in t) and "Test mAP@.5:" not in t


def main():
    import train
    from ema_det import EMADet
    from dataset import DroneDetectDataset
    ds = None
    reported, warned_crash = set(), set()
    print("watcher started", time.strftime("%H:%M"), flush=True)
    while True:
        for sd in SEEDS:
            if sd in reported:
                continue
            if os.path.exists(os.path.join(RES, "emadet_final_centerhead_seed%d.DONE" % sd)):
                if ds is None:
                    ds = DroneDetectDataset(TEST, augment=False, img_size=128, cache_images=True)
                res = {}
                for name, tag, kw in (("default", "final", {}), ("centre", "final_centerhead", dict(center_head=True))):
                    m = EMADet(**kw)
                    m.load_state_dict(torch.load(os.path.join(RES, "emadet_%s_seed%d.pt" % (tag, sd)), map_location="cpu"))
                    m.eval()
                    r = train.evaluate(m, TEST, torch.device("cpu"), 128, 0.5, dataset=ds)
                    res[name] = r
                d = {k: getattr(res["centre"], k) - getattr(res["default"], k) for k in ("map50", "map50_95", "mean_iou", "recall")}
                line = ("RESULT seed %d (thermal test): mAP@0.5 %.1f -> %.1f (%+.1f), mAP@0.5:0.95 %.1f -> %.1f (%+.1f), mean IoU %.1f -> %.1f (%+.1f), recall %+.1f"
                        % (sd, res["default"].map50, res["centre"].map50, d["map50"], res["default"].map50_95, res["centre"].map50_95, d["map50_95"],
                           res["default"].mean_iou, res["centre"].mean_iou, d["mean_iou"], d["recall"]))
                if d["map50_95"] < 2.0:
                    line = "WARNING small or no localization gain! " + line
                print(line, flush=True)
                reported.add(sd)
            elif log_crashed(sd) and sd not in warned_crash:
                print("WARNING crash detected in the log of seed %d (the runner retries up to 3 times)" % sd, flush=True)
                warned_crash.add(sd)
        if os.path.exists(os.path.join(RES, "CENTER_THERMAL.DONE")):
            print("all thermal centre-head seeds finished", flush=True)
            return
        if not runner_alive():
            print("WARNING the training runner is no longer running (PC restart or crash?) - restart with: bash scratch_runner3.sh", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
