
conv1d_3tap

Purpose: validate a tail-safe 1D three-tap float convolution kernel using TMU direct global loads, uniform scalar coefficients, clamp-to-edge boundary handling in qasm, and VPM/VDW stores.

This is the right next incremental step after SAXPY and TMU load/store tests because it is the smallest canonical stencil-like signal-processing kernel beyond elementwise math. It exercises multiple neighboring global loads per output element, scalar coefficient uniforms, boundary rules, and dynamic tail-depth stores, while intentionally avoiding shared-memory tiling and barriers.

The hardware path exercised is:

TMU0 direct memory lookups for left/current/right samples.

Scalar coefficient uniforms consumed by QPU ALU floating-point operations.

Clamp-to-edge boundary arithmetic in qasm for the first and last logical elements.

VPM staging plus VDW DMA stores for coalesced vector output.

Dynamic VDW DEPTH for final partial-vector stores.

Global mutex protection around VPM/VDW setup and store sequences.

This test does not prove optimized convolution, shared-memory convolution, 2D image stencils, barriers, or VPM reuse.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until the reference hardware run passes and the run log validates against expected.json.

Files

Authored material files:

compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/
  README.md
  input.mlir
  expected.json
  candidate/README.md
  reference/.gitignore
  reference/conv1d_3tap_harness.c
  reference/conv1d_3tap.qasm
  reference/conv1d_3tap_launch.c
  reference/conv1d_3tap_launch.h

The mechanical prompt at repo root is:

conv1d_3tap_codex_mechanical_prompt.md

It copies/adapts Makefile, run.sh, mailbox.c, mailbox.h, and the share/ tree from the saxpy_full donor test.

