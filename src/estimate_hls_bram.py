# -*- coding: utf-8 -*-
"""
estimate_hls_bram.py -- static arithmetic tally of the buffer sizes declared
in hardware/hls/ema_det_gates.cpp, at the ap_fixed<16,6> width used under
HLS_TARGET (hardware/hls/hls_types.h).

This is NOT a synthesis result -- no Vivado/Vitis HLS is available in this
environment to run csynth_design and get real BRAM/LUT utilization. It is
just element-count x bit-width arithmetic on the declared `static` array
shapes, meant to quantify (with actual numbers, not a vague "too big")
which buffers dominate the known BRAM overflow noted in hardware/README.md,
to prioritize the planned AXI4-Stream/line-buffer restructuring.
"""

BITS = 16  # ap_fixed<16, 6>
BYTES_PER_ELEM = BITS / 8

# (name, channels, height, width) for every `static FixedT ...` buffer in
# ema_det_gates.cpp, in declaration order.
BUFFERS = [
    ("block1_out",      16, 128, 128),
    ("pool1_out",       16,  64,  64),
    ("block2a_out",     32,  64,  64),
    ("block2b_out",     32,  64,  64),
    ("pool2_out",       32,  32,  32),
    ("gate1_conv_out",  16,  32,  32),
    ("block3a_out",     64,  32,  32),
    ("block3b_out",     64,  32,  32),
    ("pool3_out",       64,  16,  16),
    ("block4a_out",    128,  16,  16),
    ("block4b_out",    128,  16,  16),
    ("pool4_out",      128,   8,   8),
    ("gate2_conv_out",  64,   8,   8),
]

# XC7Z020 (Zynq-7020): 140 x 36Kb BRAM blocks.
ZYNQ7020_BRAM_KB = 140 * 36 * 1024 / 8 / 1024  # = 630 KB


def main():
    rows = []
    total_kb = 0.0
    for name, c, h, w in BUFFERS:
        n_elem = c * h * w
        kb = n_elem * BYTES_PER_ELEM / 1024
        total_kb += kb
        rows.append((name, c, h, w, n_elem, kb))

    rows.sort(key=lambda r: -r[5])

    print(f"{'buffer':<18}{'shape':<16}{'elements':>10}{'KB @16b':>10}{'% of total':>12}")
    for name, c, h, w, n_elem, kb in rows:
        print(f"{name:<18}{f'{c}x{h}x{w}':<16}{n_elem:>10,}{kb:>10.1f}{kb/total_kb*100:>11.1f}%")

    print(f"\nSum of all static buffers (if concurrently live): {total_kb:.1f} KB")
    print(f"XC7Z020 total BRAM capacity:                        {ZYNQ7020_BRAM_KB:.1f} KB")
    print(f"Ratio: {total_kb / ZYNQ7020_BRAM_KB:.2f}x over budget "
          f"(before counting weights.h, FIFOs, or any double-buffering)")


if __name__ == "__main__":
    main()
