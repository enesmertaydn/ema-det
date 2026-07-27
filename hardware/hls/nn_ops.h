// nn_ops.h — generic, template-parameterized NN building blocks.
//
// STATUS: NOT compiled/synthesized in this dev environment (no C++
// toolchain, no Vivado/Vitis HLS present here). See hls_types.h and
// ../README.md before trusting or citing anything from this code.
//
// All shapes are compile-time template parameters (required for Vivado
// HLS — no dynamic allocation, no std::vector in the datapath). Padding is
// handled by zero-extension inline rather than materializing a padded
// buffer, to keep BRAM usage down.
#pragma once

#include "hls_types.h"

// 2D convolution, stride 1, "same" padding = (K-1)/2 on each side.
// in:  [Cin][H][W]      weight: [Cout][Cin][K][K]      bias: [Cout]
// out: [Cout][H][W]
template <int Cin, int Cout, int H, int W, int K>
void conv2d_same(const FixedT in[Cin][H][W],
                  const FixedT weight[Cout][Cin][K][K],
                  const FixedT bias[Cout],
                  FixedT out[Cout][H][W]) {
    const int pad = (K - 1) / 2;
CONV_OC:
    for (int oc = 0; oc < Cout; oc++) {
CONV_OY:
        for (int oy = 0; oy < H; oy++) {
CONV_OX:
            for (int ox = 0; ox < W; ox++) {
#pragma HLS PIPELINE II=1
                FixedT acc = bias[oc];
            CONV_IC:
                for (int ic = 0; ic < Cin; ic++) {
                CONV_KY:
                    for (int ky = 0; ky < K; ky++) {
                    CONV_KX:
                        for (int kx = 0; kx < K; kx++) {
                            int iy = oy + ky - pad;
                            int ix = ox + kx - pad;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                acc += in[ic][iy][ix] * weight[oc][ic][ky][kx];
                            }
                        }
                    }
                }
                out[oc][oy][ox] = acc;
            }
        }
    }
}

// Same as conv2d_same, but K=1 (the gate convs): no padding needed, direct
// per-pixel dot product over channels.
template <int Cin, int Cout, int H, int W>
void conv1x1(const FixedT in[Cin][H][W],
             const FixedT weight[Cout][Cin][1][1],
             const FixedT bias[Cout],
             FixedT out[Cout][H][W]) {
    for (int oc = 0; oc < Cout; oc++) {
        for (int oy = 0; oy < H; oy++) {
            for (int ox = 0; ox < W; ox++) {
#pragma HLS PIPELINE II=1
                FixedT acc = bias[oc];
                for (int ic = 0; ic < Cin; ic++) {
                    acc += in[ic][oy][ox] * weight[oc][ic][0][0];
                }
                out[oc][oy][ox] = acc;
            }
        }
    }
}

template <int C, int H, int W>
void leaky_relu(FixedT x[C][H][W], FixedT alpha = FixedT(0.1)) {
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x_ = 0; x_ < W; x_++) {
#pragma HLS PIPELINE II=1
                FixedT v = x[c][y][x_];
                x[c][y][x_] = (v > FixedT(0)) ? v : FixedT(alpha * v);
            }
}

template <int C, int H, int W>
void relu(FixedT x[C][H][W]) {
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x_ = 0; x_ < W; x_++) {
#pragma HLS PIPELINE II=1
                if (x[c][y][x_] < FixedT(0)) x[c][y][x_] = FixedT(0);
            }
}

// 2x2 max pooling, stride 2. Requires even H, W.
template <int C, int H, int W>
void maxpool2x2(const FixedT in[C][H][W], FixedT out[C][H / 2][W / 2]) {
    for (int c = 0; c < C; c++) {
        for (int oy = 0; oy < H / 2; oy++) {
            for (int ox = 0; ox < W / 2; ox++) {
#pragma HLS PIPELINE II=1
                FixedT a = in[c][2 * oy][2 * ox];
                FixedT b = in[c][2 * oy][2 * ox + 1];
                FixedT cc = in[c][2 * oy + 1][2 * ox];
                FixedT d = in[c][2 * oy + 1][2 * ox + 1];
                FixedT m1 = (a > b) ? a : b;
                FixedT m2 = (cc > d) ? cc : d;
                out[c][oy][ox] = (m1 > m2) ? m1 : m2;
            }
        }
    }
}

// Global average pool: [C][H][W] -> [C]
template <int C, int H, int W>
void global_avg_pool(const FixedT in[C][H][W], FixedT out[C]) {
    for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE II=1
        FixedT acc = FixedT(0);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                acc += in[c][y][x];
        out[c] = acc / FixedT(H * W);
    }
}

// Fully connected: in[Cin] -> out[Cout], weight[Cout][Cin], bias[Cout]
template <int Cin, int Cout>
void fully_connected(const FixedT in[Cin], const FixedT weight[Cout][Cin],
                      const FixedT bias[Cout], FixedT out[Cout]) {
    for (int oc = 0; oc < Cout; oc++) {
#pragma HLS PIPELINE II=1
        FixedT acc = bias[oc];
        for (int ic = 0; ic < Cin; ic++)
            acc += in[ic] * weight[oc][ic];
        out[oc] = acc;
    }
}
