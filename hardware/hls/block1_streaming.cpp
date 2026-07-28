// block1_streaming.cpp — see block1_streaming.h for design notes.
//
// Build note: unlike ema_det_gates.cpp (plain arrays, no HLS-specific
// headers needed for a host float build), this file uses hls::stream
// (following Bitirme/project_v2's existing filter.cpp/npu.cpp pattern),
// which requires Xilinx's standalone HLS headers even for a plain g++
// build: add "-I<Vitis install>/include" (e.g.
// "-ID:/Vivado/2025.2/Vitis/include") to the g++ command line, alongside
// this directory ("-I.").
#include "block1_streaming.h"
#include "weights.h"  // BLOCK1_W[16][1][3][3], BLOCK1_B[16] (BN already folded)

void block1_streaming_forward(Block1InStream &in_stream, Block1OutStream &out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 3-row, single-channel line buffer -- 3*128 elements (768B at
    // ap_fixed<16,6>) versus the 512KB block1_out full-frame buffer this
    // replaces in ema_det_gates.cpp's flat translation.
    static FixedT line_buf[B1_K][B1_W];
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    // Zero-initialize: represents the implicit top zero-padding (rows
    // before row 0 of a "same"-padding conv) before any real row arrives.
INIT_LINEBUF:
    for (int r = 0; r < B1_K; r++)
#pragma HLS UNROLL
        for (int c = 0; c < B1_W; c++)
            line_buf[r][c] = FixedT(0);

    // H+1 iterations: the extra one (y == B1_H) feeds a zero row so the
    // last real output row (B1_H-1) gets its bottom zero-padding, exactly
    // mirroring conv2d_same()'s explicit iy/ix boundary checks (nn_ops.h)
    // but applied incrementally per row instead of over a fully
    // materialized [Cin][H][W] buffer.
ROW_LOOP:
    for (int y = 0; y < B1_H + 1; y++) {
    SHIFT_AND_READ:
        for (int c = 0; c < B1_W; c++) {
#pragma HLS PIPELINE II=1
            line_buf[0][c] = line_buf[1][c];
            line_buf[1][c] = line_buf[2][c];
            FixedT new_val = FixedT(0);
            if (y < B1_H) {
                new_val = in_stream.read();
            }
            line_buf[2][c] = new_val;
        }

        // After the shift above, line_buf holds rows (y-2, y-1, y) in
        // slots [0][1][2] respectively (with negative indices meaning the
        // zero top-pad). The row whose full 3x3 receptive field is now
        // available is (y-1): its neighborhood is exactly (y-2, y-1, y).
        int out_y = y - 1;
        if (out_y >= 0 && out_y < B1_H) {
        COL_LOOP:
            for (int ox = 0; ox < B1_W; ox++) {
#pragma HLS PIPELINE II=1
                Block1OutPixel px;
            CH_LOOP:
                for (int oc = 0; oc < B1_COUT; oc++) {
#pragma HLS UNROLL
                    FixedT acc = FixedT(BLOCK1_B[oc]);
                KY_LOOP:
                    for (int ky = 0; ky < B1_K; ky++) {
                    KX_LOOP:
                        for (int kx = 0; kx < B1_K; kx++) {
                            int ix = ox + kx - 1;  // pad=1, matching conv2d_same
                            if (ix >= 0 && ix < B1_W) {
                                acc += line_buf[ky][ix] * FixedT(BLOCK1_W[oc][0][ky][kx]);
                            }
                        }
                    }
                    // LeakyReLU(0.1) -- BatchNorm is already folded into
                    // BLOCK1_W/BLOCK1_B by export_hls_weights.py, matching
                    // LaplacianSobelInit.forward()'s Conv->BN->LeakyReLU.
                    px.ch[oc] = (acc > FixedT(0)) ? acc : FixedT(acc * FixedT(0.1));
                }
                out_stream.write(px);
            }
        }
    }
}
