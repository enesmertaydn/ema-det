// main_tb.cpp — host-side (or Vitis HLS C-sim) testbench.
//
// STATUS: NOT compiled or run in this dev environment (no C++ toolchain
// available — see ../README.md). Build and run this BEFORE trusting
// ema_det_gates.cpp's translation, and again under HLS C/RTL co-simulation
// before trusting any synthesized numbers.
//
// Build (portable host test, once a compiler is available):
//   g++ -O2 -std=c++11 -I. ema_det_gates.cpp main_tb.cpp -o gate_test
//   ./gate_test
//
// Reads the 8 reference inputs dumped by export_hls_weights.py
// (input_test_*.bin: raw float32, row-major 128x128, normalized to [0,1],
// matching ema_det.py's DroneDetectDataset preprocessing exactly) and
// compares this C++ translation's Gate1/Gate2 logits against the PyTorch
// reference values in reference_io.csv.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ema_det_gates.h"
#include "hls_types.h"

struct RefRow {
    std::string input_bin;
    double gate1_logit;
    double gate2_logit;
};

static std::vector<RefRow> load_reference_csv(const std::string &path) {
    std::vector<RefRow> rows;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Cannot open " << path << "\n";
        return rows;
    }
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        RefRow r;
        std::string tok;
        std::getline(ss, r.input_bin, ',');
        std::getline(ss, tok, ',');
        r.gate1_logit = std::stod(tok);
        std::getline(ss, tok, ',');
        r.gate2_logit = std::stod(tok);
        rows.push_back(r);
    }
    return rows;
}

static bool load_input(const std::string &dir, const std::string &fname,
                        FixedT out[1][128][128]) {
    std::ifstream f(dir + "/" + fname, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open " << fname << "\n";
        return false;
    }
    float buf[128 * 128];
    f.read(reinterpret_cast<char *>(buf), sizeof(buf));
    if (!f) {
        std::cerr << "Short read on " << fname << "\n";
        return false;
    }
    int idx = 0;
    for (int y = 0; y < 128; y++)
        for (int x = 0; x < 128; x++)
            out[0][y][x] = FixedT(buf[idx++]);
    return true;
}

int main(int argc, char **argv) {
    std::string dir = (argc > 1) ? argv[1] : ".";
    auto rows = load_reference_csv(dir + "/reference_io.csv");
    if (rows.empty()) {
        std::cerr << "No reference rows loaded — check the path to reference_io.csv.\n";
        return 1;
    }

    // Logit tolerance is loose deliberately: this compares the *float*
    // build's numerics against PyTorch's float32, where small differences
    // in summation order are expected and harmless. Under HLS_TARGET
    // (ap_fixed<16,6>) this tolerance is almost certainly too tight —
    // re-derive it from the network's actual fixed-point error profile
    // once that build has been run; do not just loosen it blindly to make
    // the test pass.
    const double tol = 1e-2;
    int n_pass = 0;

    for (const auto &r : rows) {
        static FixedT input[1][128][128];
        if (!load_input(dir, r.input_bin, input)) continue;

        FixedT g1, g2;
        ema_det_gates_forward(input, g1, g2);

        double g1_d = double(g1);
        double g2_d = double(g2);
        double d1 = std::fabs(g1_d - r.gate1_logit);
        double d2 = std::fabs(g2_d - r.gate2_logit);
        bool ok = (d1 < tol) && (d2 < tol);
        n_pass += ok ? 1 : 0;

        std::printf("%-28s  gate1: cpp=%+.4f ref=%+.4f (|d|=%.2e)  "
                     "gate2: cpp=%+.4f ref=%+.4f (|d|=%.2e)  %s\n",
                     r.input_bin.c_str(), g1_d, r.gate1_logit, d1,
                     g2_d, r.gate2_logit, d2, ok ? "OK" : "MISMATCH");
    }

    std::printf("\n%d / %zu passed (tol=%.0e)\n", n_pass, rows.size(), tol);
    return (n_pass == (int)rows.size()) ? 0 : 1;
}
