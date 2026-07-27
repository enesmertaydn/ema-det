# -*- coding: utf-8 -*-
"""
export_onnx.py — ONNX export + latency benchmark for the trained EMA-Det checkpoint.

NOTE on scope: this benchmarks on whatever hardware runs this script (in
practice an RTX 4060 laptop GPU + host CPU during development). It is a
*proxy* for the target edge deployment, not a substitute for it — Table IV
of the paper still needs real Jetson/FPGA numbers (see README Roadmap). Do
not present these figures as Jetson/FPGA measurements.
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ema_det import EMADet


class ExportWrapper(torch.nn.Module):
    """Training-mode forward (tau=None) returns a NamedTuple; ONNX export
    wants plain tensors, so unpack explicitly."""

    def __init__(self, model: EMADet):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.cls_logit, out.bbox, out.gate1_logit, out.gate2_logit


def export(model: EMADet, onnx_path: str, img_size: int = 128, opset: int = 13):
    model.eval()
    wrapper = ExportWrapper(model)
    dummy = torch.zeros(1, 1, img_size, img_size)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["input"],
        output_names=["cls_logit", "bbox", "gate1_logit", "gate2_logit"],
        dynamic_axes={"input": {0: "batch"},
                       "cls_logit": {0: "batch"}, "bbox": {0: "batch"},
                       "gate1_logit": {0: "batch"}, "gate2_logit": {0: "batch"}},
        opset_version=opset,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported + validated ONNX graph -> {onnx_path}")


def check_parity(model: EMADet, onnx_path: str, img_size: int = 128, n_trials: int = 20, atol: float = 1e-4):
    model.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.rand(1, 1, img_size, img_size)
            torch_out = model(x)
            onnx_out = sess.run(None, {"input": x.numpy()})
            for a, b in zip(
                    [torch_out.cls_logit, torch_out.bbox, torch_out.gate1_logit, torch_out.gate2_logit],
                    onnx_out):
                diff = float(np.abs(a.numpy() - b).max())
                max_diff = max(max_diff, diff)
    ok = max_diff < atol
    print(f"PyTorch vs ONNX max abs diff over {n_trials} random inputs: {max_diff:.2e} "
          f"({'OK' if ok else 'MISMATCH'}, atol={atol:.0e})")
    return ok, max_diff


def benchmark_pytorch(model: EMADet, device: torch.device, img_size: int, n_warmup: int, n_iters: int):
    model.to(device).eval()
    x = torch.rand(1, 1, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return n_iters / elapsed


def benchmark_onnx(onnx_path: str, provider: str, img_size: int, n_warmup: int, n_iters: int):
    """
    Returns (fps, actual_provider). onnxruntime silently falls back to
    CPUExecutionProvider if the requested provider's native libs are
    missing (e.g. cuBLAS/cuDNN version mismatch) — sess.get_providers()[0]
    is the only reliable way to know what actually executed, so results
    must be labeled with that, not with the provider that was requested.
    """
    sess = ort.InferenceSession(onnx_path, providers=[provider])
    actual_provider = sess.get_providers()[0]
    if actual_provider != provider:
        print(f"  [warn] requested {provider} but session is actually running on {actual_provider}")

    x = np.random.rand(1, 1, img_size, img_size).astype(np.float32)
    for _ in range(n_warmup):
        sess.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sess.run(None, {"input": x})
    elapsed = time.perf_counter() - t0
    return n_iters / elapsed, actual_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\emadet_final_best.pt")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--out_dir", default=r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_iters", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "cbam" if args.use_attention else "final"
    onnx_path = os.path.join(args.out_dir, f"emadet_{tag}.onnx")

    model = EMADet(use_attention=args.use_attention)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    export(model, onnx_path, args.img_size)
    ok, max_diff = check_parity(model, onnx_path, args.img_size)

    results = {"config": tag, "checkpoint": args.ckpt, "onnx_parity_ok": ok, "onnx_parity_max_diff": max_diff,
               "n_params": sum(p.numel() for p in model.parameters()), "note": "proxy hardware, not Jetson/FPGA"}

    if torch.cuda.is_available():
        fps = benchmark_pytorch(model, torch.device("cuda"), args.img_size, args.n_warmup, args.n_iters)
        results["pytorch_cuda_fps_batch1"] = fps
        print(f"PyTorch CUDA FPS (batch=1): {fps:.1f}")

    cpu_fps = benchmark_pytorch(model, torch.device("cpu"), args.img_size, args.n_warmup, args.n_iters)
    results["pytorch_cpu_fps_batch1"] = cpu_fps
    print(f"PyTorch CPU FPS (batch=1): {cpu_fps:.1f}")

    seen_actual_providers = set()
    for provider in ort.get_available_providers():
        try:
            fps, actual_provider = benchmark_onnx(onnx_path, provider, args.img_size, args.n_warmup, args.n_iters)
            if actual_provider in seen_actual_providers:
                print(f"ONNXRuntime {provider} FPS (batch=1): {fps:.1f}  "
                      f"[SKIPPED IN RESULTS — fell back to {actual_provider}, already recorded]")
                continue
            seen_actual_providers.add(actual_provider)
            results[f"onnx_{actual_provider}_fps_batch1"] = fps
            results[f"onnx_{actual_provider}_requested_as"] = provider
            print(f"ONNXRuntime {actual_provider} FPS (batch=1): {fps:.1f}"
                  + ("" if actual_provider == provider else f"  (requested {provider}, fell back)"))
        except Exception as e:
            print(f"[skip] {provider}: {e}")

    out_json = os.path.join(args.out_dir, f"onnx_benchmark_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
