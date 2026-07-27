// hls_types.h
//
// STATUS: written but NOT compiled — this development environment has no
// C/C++ toolchain (g++/clang++/cl all absent) and no Vivado/Vitis HLS
// installation. Build and validate against reference_io.csv (see
// main_tb.cpp) on a machine with a compiler before trusting this code, and
// run actual `vivado_hls`/`vitis_hls` csynth before citing any LUT/BRAM/
// latency/power numbers in the paper.
//
// Two build modes, selected by HLS_TARGET:
//   - Portable/host build (HLS_TARGET undefined): FixedT = float. Lets the
//     testbench be compiled with any standard C++ compiler to numerically
//     validate the translation against reference_io.csv before anyone
//     touches real HLS tooling.
//   - Vivado HLS build (HLS_TARGET defined, via project settings):
//     FixedT = ap_fixed<16,6> (Q6.10 signed fixed point) — chosen as a
//     starting point (10 fractional bits should comfortably cover this
//     network's activation range, given trained weights/activations were
//     inspected to stay within roughly [-16, 16)) but has NOT been
//     accuracy-validated in fixed point. Re-run main_tb comparing against
//     reference_io.csv under HLS co-simulation before trusting it, and
//     narrow the type once real per-layer dynamic ranges are profiled —
//     this is a starting point for the paper's planned INT8/INT4
//     quantization-aware study, not the final word on it.
#pragma once

#ifdef HLS_TARGET
#include "ap_fixed.h"
typedef ap_fixed<16, 6> FixedT;
#else
typedef float FixedT;
#endif
