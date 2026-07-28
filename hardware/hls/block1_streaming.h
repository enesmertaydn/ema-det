// block1_streaming.h — line-buffer (streaming) implementation of EMA-Det's
// Block1 (Laplacian/Sobel-initialized conv, 1->16, 3x3, "same" padding, BN
// folded, LeakyReLU(0.1)).
//
// STATUS: NOT compiled/synthesized in this dev environment (no C/C++
// toolchain, no Vivado/Vitis HLS here). See ../README.md before trusting
// anything from this file.
//
// WHY THIS EXISTS: hardware/README.md's static BRAM tally
// (src/estimate_hls_bram.py) found the flat translation in
// ema_det_gates.cpp allocates one full-size static buffer per layer --
// Block1's output alone is 512KB (30.3% of the naive 1688KB total) against
// the Zynq-7020's 630KB budget. This file is the line-buffer restructuring
// that tally called for, adapted from a working streaming pattern
// (Bitirme/project_v2/preprocessing_filter/filter.cpp's 5-row Gaussian
// line-buffer) but with real trained Block1 weights (weights.h,
// BatchNorm-folded) instead of a fixed kernel, and "same" (zero) padding
// instead of that filter's "valid" convolution -- EMA-Det's Block1 must
// preserve spatial size (128x128 in, 128x128 out), not shrink it.
//
// Only Block1 is restructured here. Feeding this into Block2 (and the rest
// of the network) as a fused dataflow pipeline is future work -- this
// module is validated standalone against a PyTorch-computed golden
// reference (block1_ref/*.bin, from export_hls_weights.py).
#pragma once

#include "hls_stream.h"
#include "hls_types.h"

#define B1_H 128
#define B1_W 128
#define B1_COUT 16
#define B1_K 3

// One spatial position's full 16-channel output -- channel-parallel,
// spatial-serial, the standard HLS image-pipeline granularity.
struct Block1OutPixel {
    FixedT ch[B1_COUT];
};

typedef hls::stream<FixedT> Block1InStream;
typedef hls::stream<Block1OutPixel> Block1OutStream;

// Reads exactly B1_H*B1_W pixels from in_stream (row-major, matching
// ema_det.py's DroneDetectDataset preprocessing) and writes exactly
// B1_H*B1_W Block1OutPixel values to out_stream, same raster order.
void block1_streaming_forward(Block1InStream &in_stream, Block1OutStream &out_stream);
