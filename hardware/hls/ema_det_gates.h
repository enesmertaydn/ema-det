// ema_det_gates.h
//
// STATUS: NOT compiled/synthesized in this dev environment — see
// ../README.md. This is a FUNCTIONAL / golden-reference implementation
// (straightforward per-layer buffers), not a resource-optimized one: a
// literal translation of ema_det.py's Block1->Gate1->Block4->Gate2 path
// using real trained weights (BatchNorm folded), for numerically
// validating the C++ translation against reference_io.csv.
//
// It is NOT yet suitable for Zynq-7020 synthesis as-is: e.g. Block1's
// output buffer alone (16 x 128 x 128) is ~512KB even at ap_fixed<16,6>,
// against the Zynq-7020's ~613KB total BRAM — every layer buffer here is
// currently sized for correctness, not for the on-chip memory budget. The
// paper's planned AXI4-Stream / line-buffer dataflow restructuring is
// exactly the next step, informed by real csynth resource reports once
// this functional version has been validated — do not skip straight to
// synthesizing this file and reporting its resource usage.
#pragma once

#include "hls_types.h"

// Runs Block1 -> pool1 -> Block2 -> pool2 -> Gate1, then continues
// Block3 -> pool3 -> Block4 -> pool4 -> Gate2. Both gate logits are always
// computed (this is the training-mode/full-forward path used to validate
// against reference_io.csv) — the Eq. 4 early-exit decision itself
// (`gate_logit -> sigmoid -> compare to 1-tau`) is a trivial scalar
// comparison left to the caller (see main_tb.cpp), since it has no
// meaningful HLS resource cost of its own; the value of synthesizing this
// path is in measuring how much of Block2-4's compute can actually be
// *skipped* in hardware when Gate 1 fires, which requires the dataflow
// restructuring noted above, not just this functional kernel.
void ema_det_gates_forward(const FixedT input[1][128][128],
                            FixedT &gate1_logit,
                            FixedT &gate2_logit);
