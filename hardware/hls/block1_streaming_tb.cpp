// block1_streaming_tb.cpp — validates block1_streaming_forward() against a
// PyTorch-computed golden reference (block1_ref/*.bin, from
// export_hls_weights.py's Block1-only dump).
//
// STATUS: NOT compiled or run in this dev environment (no C++ toolchain --
// see ../README.md). Build and run this BEFORE trusting
// block1_streaming.cpp's translation.
//
// Build (host float test; needs Xilinx's standalone HLS headers even
// outside Vivado/Vitis, for hls::stream -- see block1_streaming.cpp):
//   g++ -O2 -std=c++11 -I. -I<Vitis install>/include \
//       block1_streaming.cpp block1_streaming_tb.cpp -o block1_test
//   ./block1_test block1_ref
//
// Compares every one of the 16*128*128 output values per test image
// against PyTorch's model.block1(x) output, tolerance chosen the same way
// as main_tb.cpp's (loose for the float build; re-derive for ap_fixed<16,6>
// once that build exists).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "block1_streaming.h"

static bool load_floats(const std::string &path, std::vector<float> &out, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }
    out.resize(expected_count);
    f.read(reinterpret_cast<char *>(out.data()), expected_count * sizeof(float));
    if (!f) {
        std::cerr << "Short read on " << path << "\n";
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    // Defaults match this repo's fixed layout (Vitis's GUI C-simulation
    // argument passing doesn't cleanly support multiple space-separated
    // argv entries -- see hardware/README.md's Vitis 2025.2 notes), so no
    // command-line arguments are required. Override argv[1]/argv[2:] if
    // you need to point at a different reference dir or stem list.
    std::string dir = (argc > 1) ? argv[1]
        : "C:/Users/enesm/Desktop/emadet/emadetforspringer/hardware/hls/block1_ref";

    std::vector<std::string> stems;
    if (argc > 2) {
        for (int i = 2; i < argc; i++) stems.push_back(argv[i]);
    } else {
        // The two stems export_hls_weights.py dumps by default
        // (test_imgs[:2], sorted .jpg filenames from the clean test split).
        stems = {"test_000000", "test_000001"};
    }

    const float tol = 1e-2f;
    int n_images_ok = 0;

    for (const auto &stem : stems) {
        std::vector<float> input, golden;
        if (!load_floats(dir + "/" + stem + "_input.bin", input, B1_H * B1_W)) continue;
        if (!load_floats(dir + "/" + stem + "_block1_out.bin", golden, B1_COUT * B1_H * B1_W)) continue;

        Block1InStream in_stream;
        Block1OutStream out_stream;
        for (size_t i = 0; i < input.size(); i++) in_stream.write(FixedT(input[i]));

        block1_streaming_forward(in_stream, out_stream);

        int n_mismatch = 0;
        double max_abs_diff = 0.0;
        for (int y = 0; y < B1_H; y++) {
            for (int x = 0; x < B1_W; x++) {
                Block1OutPixel px = out_stream.read();
                for (int c = 0; c < B1_COUT; c++) {
                    // PyTorch (C,H,W) row-major: index = c*H*W + y*W + x.
                    float ref = golden[(size_t)c * B1_H * B1_W + (size_t)y * B1_W + x];
                    double got = double(px.ch[c]);
                    double diff = std::fabs(got - ref);
                    max_abs_diff = std::max(max_abs_diff, diff);
                    if (diff > tol) n_mismatch++;
                }
            }
        }

        bool ok = (n_mismatch == 0);
        n_images_ok += ok ? 1 : 0;
        std::printf("%-20s  max|diff|=%.4e  mismatches=%d/%d  %s\n",
                     stem.c_str(), max_abs_diff, n_mismatch, B1_COUT * B1_H * B1_W,
                     ok ? "OK" : "MISMATCH");
    }

    std::printf("\n%d / %zu images passed (tol=%.0e)\n", n_images_ok, stems.size(), tol);
    return (n_images_ok == (int)stems.size()) ? 0 : 1;
}
