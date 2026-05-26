# matmul_blocked candidate side

This candidate is generated from `../input.mlir` and run by the shared
`run_candidate_codegen_test.sh` fixture path.

The harness launches the generated VC4 kernel on the real device.  Host-side
`kmalloc` buffers are only staging/oracle memory; GPU-visible buffers are
allocated through the generated VC4 runtime helper.  Verification compares the
device result bit-for-bit against a CPU matmul oracle and checks compact and
padded sentinels so out-of-range writes are caught.

The generated kernel is a cooperative-block VC4 implementation: each tile-wave
uses 12 warps, stages a B tile in shared VPM rows, synchronizes with the
four-semaphore barrier, accumulates with QPU fmul/fadd, and stores through
VPM/VDW.
