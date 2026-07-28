// hls_types.h
//
// STATUS: real, measured evidence now exists (2026-07-28, Vitis Unified IDE
// 2025.2, block1_streaming_forward's C-simulation) that accumulating a 3x3
// conv's MAC sum directly in FixedT (ap_fixed<16,6>) loses real accuracy:
// ~5.6%/10.5% of output values (two test images) exceeded a 1e-2 absolute
// tolerance, max|diff| ~1.9e-2, versus the float build's ~3e-6 (rounding
// noise only). Root cause: `FixedT acc = bias; acc += a*b;` truncates the
// running sum back to 16,6 after every one of the 9 terms, compounding
// rounding error each step. AccT below fixes this by accumulating in a
// wider type and truncating to FixedT only once, at the very end.
//
// Two build modes, selected by HLS_TARGET:
//   - Portable/host build (HLS_TARGET undefined): FixedT = AccT = float.
//     Lets the testbench be compiled with any standard C++ compiler to
//     numerically validate the translation before touching real HLS
//     tooling.
//   - Vivado HLS build (HLS_TARGET defined, via project settings):
//     FixedT = ap_fixed<16,6> (Q6.10 signed fixed point) for
//     storage/activations, AccT = ap_fixed<32,12> (Q12.20) for
//     MAC-accumulation only. Neither width has been swept/optimized yet —
//     these are starting points (10/20 fractional bits chosen to
//     comfortably cover this network's inspected activation range, roughly
//     [-16, 16)) for the paper's planned INT8/INT4 quantization-aware
//     study, not the final word on it. Re-run the *_streaming_tb / main_tb
//     testbenches under HLS co-simulation and re-check tolerance after any
//     width change.
#pragma once

#ifdef HLS_TARGET
#include "ap_fixed.h"
typedef ap_fixed<16, 6> FixedT;
typedef ap_fixed<32, 12> AccT;
#else
typedef float FixedT;
typedef float AccT;
#endif
