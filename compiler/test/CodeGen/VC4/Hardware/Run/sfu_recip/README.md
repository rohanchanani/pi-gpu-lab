# sfu_recip

`compiler/test/CodeGen/VC4/Hardware/Run/sfu_recip` is a hardware-grounded VC4 codegen test for the first Special Functions Unit (SFU) operation in the corpus.

## Purpose

This test checks a single incremental concept: issuing an SFU reciprocal operation from a scheduled QPU program and consuming the result after the required r4 hazard window.

The reference kernel computes, for one 16-lane vector per active QPU:

```text
out[i] = recip(x[i])
```

The harness supplies exact power-of-two inputs, so the CPU-side expected reciprocal values are exactly representable in `f32`. The VC4 SFU reciprocal datapath is approximate on hardware, and the oracle therefore uses an absolute tolerance of `0.0003`, matching the observed maximum error from the first hardware run (`0.000244`). The result checksum is locked to the observed SFU output stream rather than the ideal CPU reciprocal stream.

## Why this is the right next step

The existing hardware-run corpus already covers QPU launch/completion, uniform ABI packing, TMU direct memory reads, VPM/VDW writeback, scalar broadcast arithmetic, overlapped TMU receives, and a runtime-N loop. SFU is the next independent datapath not yet exercised by a runnable codegen golden.

## Hardware path exercised

The trusted reference bundle exercises:

- TMU0 direct memory lookup for the input vector,
- accumulator r4 delivery from `ldtmu0`,
- write to the SFU `recip` register,
- two following instructions that do not touch r4,
- reading the SFU result from r4 in the third following instruction,
- VPM staging and VDW DMA writeback to host-visible memory.

The MLIR input is final-stage scheduled `vc4.qpu.*` sink IR and includes launcher ABI metadata for:

```text
[0] x base address
[1] out base address
[2] n element count
[3] qpu_id
[4] num_qpus
```

`qpu_id` and `num_qpus` are conceptual execution builtins, physically carried in the uniform suffix for this reference bundle.

## What this test does not prove yet

This test does not prove `recipsqrt`, `exp`, or `log`; it does not prove tail masking; it does not prove a loop; it does not use VDR input DMA; and it does not claim candidate/codegen coverage until candidate-side generation exists.

## Test policy

The kernel processes exactly one 16-lane vector per active QPU. The handwritten harness enforces:

```text
n == active_qpus * vc4_runtime_lane_width()
```

The generated-style launcher hides physical uniform packing and does not expose `qpu_id`, `num_qpus`, raw uniform arrays, GPU bus addresses, VPM rows, or V3D scheduler internals.
