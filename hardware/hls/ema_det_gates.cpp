// ema_det_gates.cpp — see ema_det_gates.h and ../README.md for status.
#include "ema_det_gates.h"
#include "nn_ops.h"
#include "weights.h"

void ema_det_gates_forward(const FixedT input[1][128][128],
                            FixedT &gate1_logit,
                            FixedT &gate2_logit) {
#pragma HLS INTERFACE m_axi port=input depth=16384 bundle=gmem
#pragma HLS INTERFACE s_axilite port=gate1_logit bundle=ctrl
#pragma HLS INTERFACE s_axilite port=gate2_logit bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // --- Block 1: Laplacian/Sobel-initialized conv, 1->16, 128x128 ---
    static FixedT block1_out[16][128][128];
    conv2d_same<1, 16, 128, 128, 3>(input, BLOCK1_W, BLOCK1_B, block1_out);
    leaky_relu<16, 128, 128>(block1_out);

    static FixedT pool1_out[16][64][64];
    maxpool2x2<16, 128, 128>(block1_out, pool1_out);

    // --- Block 2: 16->32, 64x64 ---
    static FixedT block2a_out[32][64][64];
    conv2d_same<16, 32, 64, 64, 3>(pool1_out, BLOCK2A_W, BLOCK2A_B, block2a_out);
    leaky_relu<32, 64, 64>(block2a_out);

    static FixedT block2b_out[32][64][64];
    conv2d_same<32, 32, 64, 64, 3>(block2a_out, BLOCK2B_W, BLOCK2B_B, block2b_out);
    leaky_relu<32, 64, 64>(block2b_out);

    static FixedT pool2_out[32][32][32];
    maxpool2x2<32, 64, 64>(block2b_out, pool2_out);

    // --- Gate 1: conv1x1(32->16) + ReLU + GAP + FC(16->1) ---
    static FixedT gate1_conv_out[16][32][32];
    conv1x1<32, 16, 32, 32>(pool2_out, GATE1_CONV_W, GATE1_CONV_B, gate1_conv_out);
    relu<16, 32, 32>(gate1_conv_out);

    FixedT gate1_pooled[16];
    global_avg_pool<16, 32, 32>(gate1_conv_out, gate1_pooled);

    FixedT gate1_out[1];
    fully_connected<16, 1>(gate1_pooled, GATE1_FC_W, GATE1_FC_B, gate1_out);
    gate1_logit = gate1_out[0];

    // Eq. 4 gating decision itself is left to the caller (see
    // ema_det_gates.h) — this kernel always continues to Gate 2 so the
    // full path can be validated in one shot against reference_io.csv.

    // --- Block 3: 32->64, 32x32 ---
    static FixedT block3a_out[64][32][32];
    conv2d_same<32, 64, 32, 32, 3>(pool2_out, BLOCK3A_W, BLOCK3A_B, block3a_out);
    leaky_relu<64, 32, 32>(block3a_out);

    static FixedT block3b_out[64][32][32];
    conv2d_same<64, 64, 32, 32, 3>(block3a_out, BLOCK3B_W, BLOCK3B_B, block3b_out);
    leaky_relu<64, 32, 32>(block3b_out);

    static FixedT pool3_out[64][16][16];
    maxpool2x2<64, 32, 32>(block3b_out, pool3_out);

    // --- Block 4: 64->128, 16x16 ---
    static FixedT block4a_out[128][16][16];
    conv2d_same<64, 128, 16, 16, 3>(pool3_out, BLOCK4A_W, BLOCK4A_B, block4a_out);
    leaky_relu<128, 16, 16>(block4a_out);

    static FixedT block4b_out[128][16][16];
    conv2d_same<128, 128, 16, 16, 3>(block4a_out, BLOCK4B_W, BLOCK4B_B, block4b_out);
    leaky_relu<128, 16, 16>(block4b_out);

    static FixedT pool4_out[128][8][8];
    maxpool2x2<128, 16, 16>(block4b_out, pool4_out);

    // --- Gate 2: conv1x1(128->64) + ReLU + GAP + FC(64->1) ---
    static FixedT gate2_conv_out[64][8][8];
    conv1x1<128, 64, 8, 8>(pool4_out, GATE2_CONV_W, GATE2_CONV_B, gate2_conv_out);
    relu<64, 8, 8>(gate2_conv_out);

    FixedT gate2_pooled[64];
    global_avg_pool<64, 8, 8>(gate2_conv_out, gate2_pooled);

    FixedT gate2_out[1];
    fully_connected<64, 1>(gate2_pooled, GATE2_FC_W, GATE2_FC_B, gate2_out);
    gate2_logit = gate2_out[0];
}
