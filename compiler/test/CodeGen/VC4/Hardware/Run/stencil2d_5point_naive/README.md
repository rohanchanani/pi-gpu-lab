
stencil2d_5point_naive

Purpose: validate a naive 2D five-point float stencil using direct TMU global loads and coalesced VPM/VDW stores.

This is the right next incremental step after 1D convolution and TMU-backed vector kernels because it tests 2D indexing, row pitch, multiple neighbor loads, clamp-to-edge boundary handling, and tail-safe row stores without adding shared-memory tiling or barriers.

The hardware path exercised is:

TMU0 direct memory lookups for center/up/down/left/right samples.

Scalar coefficient uniforms consumed by QPU floating-point ALU operations.

Clamp-to-edge row and column boundary arithmetic in qasm.

VPM staging plus VDW DMA stores for coalesced output rows.

Dynamic VDW DEPTH for odd row widths and final row-tail stores.

Global mutex protection around VPM/VDW setup and store sequences.

This test does not use shared memory, VPM halos, barriers, optimized row reuse, or 2D image-specific texture modes. It is explicitly the naive direct-load stencil baseline.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until the reference hardware run passes and the run log validates against expected.json.

Files

Authored material files:

compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/
  README.md
  input.mlir
  expected.json
  candidate/README.md
  reference/.gitignore
  reference/stencil2d_5point_naive_harness.c
  reference/stencil2d_5point_naive.qasm
  reference/stencil2d_5point_naive_launch.c
  reference/stencil2d_5point_naive_launch.h

The mechanical prompt at repo root is:

stencil2d_5point_naive_codex_mechanical_prompt.md

It copies/adapts Makefile, run.sh, mailbox.c, mailbox.h, and the share/ tree from the saxpy_full donor test.

